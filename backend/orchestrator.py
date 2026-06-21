"""Background pipeline: photos -> vision -> [user review] -> Gizmo scene -> compose -> map -> rollout.

Split into two background tasks around a human-in-the-loop checkpoint:
  - run_describe(run): runs the vision stage, then emits "awaiting_confirmation"
    and STOPS so the user can review/edit the scene description.
  - run_generate_onward(run): kicked off by POST /confirm-scene with the (possibly
    edited) prompt; runs Gizmo generation -> compose -> map -> rollout.
Both push SSE events onto the run's event bus (runs.py). "awaiting_confirmation"
is non-terminal so the already-open SSE stream survives the pause.
"""

from __future__ import annotations

import asyncio
from typing import Any

from backend.gizmo_client import GizmoClient, GizmoError
from backend.improve import train_further
from backend.rollout import run_rollout
from backend.runs import ActivityState, RunState, emit
from backend.scene_compose import SceneComposeError, compose_scene
from backend.task_mapping import classify_all
from backend.vision import VisionError, describe_images
from backend.vla_pipeline import generate_vla_scene
from backend.vla_scene import VlaSceneError


async def run_describe(run: RunState) -> None:
    """Phase 1: describe the photos, then pause for user review of the prompt."""
    try:
        await _stage_describe(run)
        run.stage = "awaiting_confirmation"
        await emit(run, "awaiting_confirmation", {
            "scene_prompt": run.scene_prompt,
            "objects": run.object_hints,
        })
    except VisionError as exc:
        run.status, run.error = "failed", str(exc)
        await emit(run, "error", {"message": str(exc), "stage": run.stage})
    except Exception as exc:  # don't let an unexpected bug hang the SSE stream forever
        run.status, run.error = "failed", f"unexpected error: {exc}"
        await emit(run, "error", {"message": run.error, "stage": run.stage})


async def run_generate_onward(run: RunState) -> None:
    """Phase 2: from the confirmed prompt through Gizmo, compose, map, and rollouts."""
    run.status = "running"
    try:
        await _stage_generate_scene(run)
        await _stage_compose(run)
        await _stage_map_activities(run)
        await _stage_rollouts(run)
        run.stage, run.status = "done", "done"
        await emit(run, "done", {})
    except (VisionError, GizmoError, SceneComposeError) as exc:
        run.status, run.error = "failed", str(exc)
        await emit(run, "error", {"message": str(exc), "stage": run.stage})
    except Exception as exc:  # don't let an unexpected bug hang the SSE stream forever
        run.status, run.error = "failed", f"unexpected error: {exc}"
        await emit(run, "error", {"message": run.error, "stage": run.stage})


async def run_vla_pipeline(run: RunState) -> None:
    """Phase 2 (VLA branch): from the confirmed prompt, generate a Franka scene, adapt +
    validate it for the pi0.5 loop, and emit a TaskSpec. The alternative to
    run_generate_onward — the floating-gripper best-of-N path — selected for the VLA
    training flow. Relays the pipeline's events onto the run's SSE bus; the final "done"
    carries the TaskSpec (scene_id/target/instruction/lift) the train/loop.py round needs.
    """
    run.stage, run.status = "generating_vla_scene", "running"
    try:
        async def on_event(name: str, data: dict[str, Any]) -> None:
            await emit(run, name, data)

        first_activity = run.activities[0] if run.activities else None
        spec = await generate_vla_scene(
            scene_prompt=run.scene_prompt or "",
            scene_id=run.run_id,  # scenes/<run_id>/ holds the adapted scene
            objects=run.object_hints,
            instruction=first_activity,
            target_hint=first_activity,
            on_event=on_event,
        )
        run.scene_id = spec.scene_id
        run.stage, run.status = "done", "done"
        await emit(run, "done", {"task_spec": spec.as_dict()})
    except (VisionError, GizmoError, VlaSceneError) as exc:
        run.status, run.error = "failed", str(exc)
        await emit(run, "error", {"message": str(exc), "stage": run.stage})
    except Exception as exc:
        run.status, run.error = "failed", f"unexpected error: {exc}"
        await emit(run, "error", {"message": run.error, "stage": run.stage})


async def _stage_describe(run: RunState) -> None:
    run.stage = "describing_photos"
    await emit(run, "stage", {"stage": run.stage, "status": "started"})

    description = await describe_images(run.image_paths, run.activities)
    if not description["scene_prompt"]:
        raise VisionError("the vision model returned an empty scene description")

    run.scene_prompt = description["scene_prompt"]
    run.object_hints = description["objects"]
    await emit(run, "stage", {"stage": run.stage, "status": "completed", "detail": description})


async def _stage_generate_scene(run: RunState) -> None:
    run.stage = "generating_scene"
    await emit(run, "stage", {
        "stage": run.stage, "status": "started", "detail": {"prompt": run.scene_prompt},
    })

    client = GizmoClient()
    job = await client.generate_scene(run.scene_prompt)
    job_id, scene_id = job["job_id"], job["scene_id"]
    run.scene_id = scene_id

    async for evt in client.stream_job_events(job_id):
        await emit(run, "gizmo", evt)
        if evt["type"] == "error":
            raise GizmoError(f"Gizmo job {job_id} reported an error: {evt['data']}")
        if evt["type"] == "done":
            break

    final = await client.get_job(job_id, include_result=False)
    status = (final.get("job") or {}).get("status")
    if status not in ("succeeded", None):
        raise GizmoError(f"Gizmo job {job_id} ended with status={status!r}")

    await emit(run, "stage", {
        "stage": run.stage, "status": "completed",
        "detail": {"scene_id": scene_id, "job_id": job_id},
    })


async def _stage_compose(run: RunState) -> None:
    run.stage = "composing_scene"
    await emit(run, "stage", {"stage": run.stage, "status": "started"})

    client = GizmoClient()
    zip_bytes = await client.export_scene(run.scene_id, fmt="mjcf")
    # compose_scene is sync/blocking (ElementTree + mujoco load) - keep it off the event loop.
    result = await asyncio.to_thread(compose_scene, zip_bytes, run.scene_id, run.scene_prompt)
    run.objects = result["objects"]

    await emit(run, "stage", {
        "stage": run.stage, "status": "completed",
        "detail": {"objects": list(run.objects), "cameras": result["cameras"]},
    })


async def _stage_map_activities(run: RunState) -> None:
    run.stage = "mapping_activities"
    await emit(run, "stage", {"stage": run.stage, "status": "started"})

    mappings = classify_all(run.activities, run.objects, run.object_hints)
    run.activity_states = [
        ActivityState(activity_index=i, activity=activity, mapping=mapping)
        for i, (activity, mapping) in enumerate(zip(run.activities, mappings, strict=True))
    ]

    await emit(run, "stage", {
        "stage": run.stage, "status": "completed",
        "detail": {"mappings": [
            {"activity": m.activity, "task": m.task, "target": m.target} for m in mappings
        ]},
    })


async def _stage_rollouts(run: RunState) -> None:
    run.stage = "running_rollouts"
    await emit(run, "stage", {"stage": run.stage, "status": "started"})

    for a in run.activity_states:
        a.status = "running"
        await emit(run, "rollout", run.activity_result(a))
        try:
            result = await run_rollout(run.scene_id, a.mapping)
            a.status = "completed"
            a.reward, a.success, a.content = result["reward"], result["success"], result["content"]
        except Exception as exc:
            a.status = "failed"
            a.content = str(exc)
        await emit(run, "rollout", run.activity_result(a))

    await emit(run, "stage", {"stage": run.stage, "status": "completed"})


async def run_train_further(run: RunState, activity_index: int) -> None:
    """Background task for POST /api/runs/{id}/train-further.

    Uses its own "train_further_done"/"train_further_error" event names rather
    than the pipeline's "done"/"error" - those are in runs.TERMINAL_EVENTS and
    would end the whole SSE stream over a single per-activity training failure,
    even though the main pipeline (or another activity's training) may still be
    in progress.
    """
    a = next((x for x in run.activity_states if x.activity_index == activity_index), None)
    if a is None or a.mapping is None or run.scene_id is None:
        await emit(run, "train_further_error", {
            "activity_index": activity_index,
            "message": f"no such activity_index {activity_index} to train further",
        })
        return

    async def on_round(record: dict[str, Any]) -> None:
        await emit(run, "train_round", {"activity_index": activity_index, **record})

    try:
        await train_further(run.scene_id, a.mapping, on_round=on_round)
        await emit(run, "train_further_done", {"activity_index": activity_index})
    except Exception as exc:
        await emit(run, "train_further_error", {"activity_index": activity_index, "message": str(exc)})
