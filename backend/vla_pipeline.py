"""The VLA branch: a natural-language description -> a validated, VLA-ready Franka scene.

Ties the pieces together for the pi0.5 path (distinct from scene_compose.py's
floating-gripper LLM path):

    bias prompt -> Gizmo generate -> export(robot_profile="franka_panda")
    -> adapt_vla_scene (inject eef + cameras, discover target)
    -> validate_vla_scene (no-GPU smoke test) -> regenerate if it fails
    -> return a TaskSpec the train/loop.py round consumes (--scene/--target/--instruction/--lift).

`on_event` is an async callback `(event_name, data)` so the orchestrator can relay
progress onto its SSE stream. The validation gate is what makes regeneration safe:
a broken/unreachable scene is caught here, locally, before any Modal A100 time.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Any

from backend.gizmo_client import GIZMO_TERMINAL_FAIL, GIZMO_TERMINAL_OK, GizmoClient
from backend.vision import bias_prompt_for_vla
from backend.vla_scene import TaskSpec, VlaSceneError, adapt_vla_scene, validate_vla_scene

OnEvent = Callable[[str, dict[str, Any]], Awaitable[None]]


async def _noop_event(_name: str, _data: dict[str, Any]) -> None:
    pass


async def _generate_and_export(
    client: GizmoClient, biased_prompt: str, on_event: OnEvent
) -> tuple[str, bytes]:
    """Generate one scene, stream its job events, and export it with the Franka embedded.
    Returns (gizmo_scene_id, mjcf_zip_bytes)."""
    job = await client.generate_scene(biased_prompt)
    gizmo_scene_id, job_id = job["scene_id"], job["job_id"]
    await on_event("vla_generate", {"status": "started", "scene_id": gizmo_scene_id, "job_id": job_id})

    async for evt in client.stream_job_events(job_id):
        t = evt.get("type")
        if t == "ping":
            continue
        await on_event("gizmo", evt)
        if t in GIZMO_TERMINAL_FAIL:
            raise VlaSceneError(f"Gizmo job {job_id} {t}: {evt.get('data')}")
        if t in GIZMO_TERMINAL_OK:
            break

    final = await client.get_job(job_id, include_result=False)
    status = (final.get("job") or {}).get("status")
    if status not in ("succeeded", None):
        raise VlaSceneError(f"Gizmo job {job_id} ended status={status!r}")

    zip_bytes = await client.export_scene(gizmo_scene_id, fmt="mjcf", robot_profile="franka_panda")
    await on_event("vla_generate", {"status": "exported", "scene_id": gizmo_scene_id})
    return gizmo_scene_id, zip_bytes


async def generate_vla_scene(
    *,
    scene_prompt: str,
    scene_id: str,
    objects: list[str] | None = None,
    instruction: str | None = None,
    target_hint: str | None = None,
    snap_target: bool = False,
    max_attempts: int = 2,
    on_event: OnEvent | None = None,
) -> TaskSpec:
    """NL description -> a validated VLA-ready scene under scenes/{scene_id}/.

    Regenerates (up to max_attempts) if a generated scene fails the validation gate.
    Returns the TaskSpec for the training loop. Raises VlaSceneError if every attempt
    fails validation.
    """
    on_event = on_event or _noop_event
    client = GizmoClient()
    biased = bias_prompt_for_vla(scene_prompt, objects)
    await on_event("vla_prompt", {"biased_prompt": biased})

    last_error = "no attempts run"
    for attempt in range(1, max_attempts + 1):
        await on_event("vla_attempt", {"attempt": attempt, "max_attempts": max_attempts})
        try:
            _, zip_bytes = await _generate_and_export(client, biased, on_event)
            # adapt is sync/blocking (ElementTree + MuJoCo load) - keep it off the loop.
            spec = await asyncio.to_thread(
                adapt_vla_scene, zip_bytes, scene_id,
                instruction=instruction, target_hint=target_hint, snap_target=snap_target,
            )
            await on_event("vla_adapt", {"status": "completed", **spec.as_dict()})

            verdict = await validate_vla_scene(spec)
            await on_event("vla_validate", verdict)
            if verdict["ok"]:
                await on_event("vla_ready", {
                    **spec.as_dict(),
                    "train_command": (
                        f"python -m train.loop --round 0 --scene {spec.scene_id} "
                        f"--target {spec.target_object} --instruction \"{spec.instruction}\" "
                        f"--lift {spec.lift_height:.2f}"
                    ),
                })
                return spec
            last_error = verdict.get("error", "validation failed")
        except VlaSceneError as exc:
            last_error = str(exc)
            await on_event("vla_attempt_failed", {"attempt": attempt, "error": last_error})

    raise VlaSceneError(f"no scene passed validation after {max_attempts} attempts: {last_error}")


def _main() -> None:
    ap = argparse.ArgumentParser(description="Generate a VLA-ready Franka scene from a prompt.")
    ap.add_argument("--prompt", required=True, help="natural-language scene description")
    ap.add_argument("--scene-id", required=True, help="output scene id (scenes/<id>/)")
    ap.add_argument("--objects", nargs="*", default=None, help="object-name hints for biasing")
    ap.add_argument("--target-hint", default=None, help="substring to pick the target object")
    ap.add_argument("--instruction", default=None)
    ap.add_argument("--snap-target", action="store_true", help="force the target within reach")
    ap.add_argument("--attempts", type=int, default=2)
    args = ap.parse_args()

    async def _run() -> None:
        async def printer(name: str, data: dict[str, Any]) -> None:
            print(f"[{name}] {json.dumps(data)[:200]}", flush=True)

        spec = await generate_vla_scene(
            scene_prompt=args.prompt, scene_id=args.scene_id, objects=args.objects,
            instruction=args.instruction, target_hint=args.target_hint,
            snap_target=args.snap_target, max_attempts=args.attempts, on_event=printer,
        )
        print("\nTASK SPEC:\n" + json.dumps(spec.as_dict(), indent=2))

    asyncio.run(_run())


if __name__ == "__main__":
    _main()
