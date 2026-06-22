"""Drive the real Modal training loop (eval -> curate -> finetune) with live SSE.

`train/loop.py::run_round` is synchronous and drives Modal A100s. This module
bridges it onto the async run/event bus so the frontend can chart reward live:

  - a per-stage `progress(event, data)` callback and a per-rollout reward callback
    are marshalled from the worker thread onto the main event loop via
    `run_coroutine_threadsafe`, and
  - a `dry_run` path replays reward from already-recorded episodes (or, if none
    exist yet, synthesizes a plausible learning curve) so the whole chart can be
    demoed without spending GPU time.

Events emitted (all carry `activity_index`):
  train_stage   {round, stage, status, detail?}   stage in serve|eval|curate|finetune
  eval_rollout  {round, index, reward, success}    one per collected rollout (live)
  eval_summary  {round, mean_reward, success_rate, rewards}
  curate        {round, threshold, selected, available, mean_selected_reward}
  train_modal_done  {summary}
  train_modal_error {message}
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
from pathlib import Path
from typing import Any

from backend.rollout_video import generate_rollout_video
from backend.runs import RunState, emit
from train.config import TrainConfig

# Demo curve shape when there are no recorded episodes to replay yet.
_DEMO_GROUP = 6
_DEMO_ROUNDS = 3
_DEMO_SUCCESS_AT = 0.6


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def _demo_reward(round_idx: int, index: int, run_idx: int = 0) -> float:
    """Deterministic upward-drifting reward. run_idx offsets the baseline upward
    so successive fine-tune button clicks show real improvement, not the same curve."""
    h = (round_idx * 1103515245 + index * 12345 + 7) % 1000
    jitter = (h / 1000.0 - 0.5) * 0.3
    return _clamp01(0.18 + 0.18 * run_idx + 0.18 * round_idx + jitter)


def _summary(rewards: list[float], threshold: float) -> dict[str, Any]:
    n = len(rewards) or 1
    selected = [r for r in rewards if r >= threshold]
    return {
        "rewards": [round(r, 4) for r in rewards],
        "mean_reward": round(sum(rewards) / n, 4),
        "success_rate": round(sum(1 for r in rewards if r >= _DEMO_SUCCESS_AT) / n, 4),
        "selected": len(selected),
        "available": len(rewards),
        "mean_selected_reward": round(sum(selected) / (len(selected) or 1), 4),
    }


def _activity(run: RunState, activity_index: int):
    return next((x for x in run.activity_states if x.activity_index == activity_index), None)


def _build_cfg(run: RunState, activity) -> TrainConfig:
    """Point a TrainConfig at this run's scene/activity (overrides env defaults)."""
    from backend.reward_compiler import compile_reward_heuristic

    base = TrainConfig()
    target = (activity.mapping.target if activity and activity.mapping else None) or base.target_object
    instruction = (activity.activity if activity else base.instruction)
    objects = list(run.objects.keys()) if run.objects else []
    program = compile_reward_heuristic(instruction, objects).program.as_dict()
    lift_height = base.lift_height
    if activity and activity.mapping and "lift_height" in activity.mapping.kwargs:
        lift_height = float(activity.mapping.kwargs["lift_height"])
    return dataclasses.replace(
        base,
        scene_id=run.scene_id or base.scene_id,
        instruction=instruction,
        target_object=target,
        lift_height=lift_height,
        reward_program=program,
    )


async def run_modal_training(
    run: RunState,
    activity_index: int,
    *,
    dry_run: bool = False,
    rounds: int = 1,
) -> None:
    """Background task for POST /api/runs/{id}/train-modal."""
    activity = _activity(run, activity_index)
    if activity is None:
        await emit(run, "train_modal_error", {
            "activity_index": activity_index,
            "message": f"no such activity_index {activity_index}",
        })
        return

    cfg = _build_cfg(run, activity)
    await emit(run, "train_stage", {
        "activity_index": activity_index, "round": 0, "stage": "init", "status": "started",
        "detail": {"scene_id": cfg.scene_id, "instruction": cfg.instruction,
                   "target": cfg.target_object, "dry_run": dry_run},
    })

    try:
        summary = (
            await _demo_loop(run, activity_index, cfg)
            if dry_run else
            await _modal_loop(run, activity_index, cfg, rounds=rounds)
        )
        await generate_rollout_video(run, activity_index, force=True)
        await emit(run, "train_modal_done", {"activity_index": activity_index, "summary": summary})
    except Exception as exc:  # never hang the SSE stream on a training failure
        await emit(run, "train_modal_error", {"activity_index": activity_index, "message": str(exc)})


# ── real path: drive train/loop.py round(s) on a worker thread ──────────────────


async def _modal_loop(
    run: RunState, activity_index: int, cfg: TrainConfig, *, rounds: int
) -> dict[str, Any]:
    from train.loop import run_round

    loop = asyncio.get_running_loop()

    def progress(event: str, data: dict[str, Any]) -> None:
        # Called from the worker thread / its eval loop -> hop back to the main loop.
        asyncio.run_coroutine_threadsafe(
            emit(run, event, {"activity_index": activity_index, **data}), loop,
        )

    on_rollout = lambda d: progress("eval_rollout", d)  # noqa: E731

    # Resume from the best checkpoint produced by a previous fine-tune run on this
    # activity, so repeated clicks on "Fine-tune (VLA)" compound rather than restart.
    activity = _activity(run, activity_index)
    checkpoint = (activity.best_checkpoint if activity and activity.best_checkpoint else None) or cfg.base_checkpoint
    summaries: list[dict[str, Any]] = []
    for round_idx in range(rounds):
        summary = await asyncio.to_thread(
            run_round,
            round_idx=round_idx,
            checkpoint=checkpoint,
            repo_id="hudathon/vla-pick",
            steps=1000,
            batch_size=4,
            dry_run=False,
            cfg=cfg,
            progress=progress,
            on_rollout=on_rollout,
        )
        summaries.append(summary)
        if summary.get("stop"):
            break
        checkpoint = summary.get("new_checkpoint", checkpoint)
    # Persist the best checkpoint so the next fine-tune run starts from here.
    if activity and checkpoint != cfg.base_checkpoint:
        activity.best_checkpoint = checkpoint
    return {"rounds": summaries}


# ── dry path: replay recorded episodes, else synthesize a learning curve ────────


async def _demo_loop(run: RunState, activity_index: int, cfg: TrainConfig) -> dict[str, Any]:
    # Count how many fine-tune runs this activity has already had so the baseline rises.
    activity = _activity(run, activity_index)
    run_idx = int(activity.best_checkpoint or "0") if (
        activity and activity.best_checkpoint and activity.best_checkpoint.isdigit()
    ) else 0

    recorded = _recorded_rounds(cfg)
    if recorded:
        for round_idx, rewards in enumerate(recorded):
            await _emit_round(run, activity_index, round_idx, rewards, cfg.curation_threshold)
        new_run_idx = run_idx + 1
        if activity:
            activity.best_checkpoint = str(new_run_idx)
        return {"rounds": len(recorded), "source": "recorded"}

    for round_idx in range(_DEMO_ROUNDS):
        rewards = [_demo_reward(round_idx, i, run_idx) for i in range(_DEMO_GROUP)]
        await _emit_round(run, activity_index, round_idx, rewards, cfg.curation_threshold, delay=0.25)

    if activity:
        activity.best_checkpoint = str(run_idx + 1)
    return {"rounds": _DEMO_ROUNDS, "source": "synthetic"}


def _recorded_rounds(cfg: TrainConfig) -> list[list[float]]:
    """Read reward lists from any eval-round-* dirs already on disk (newest demo input)."""
    out: list[list[float]] = []
    root = cfg.dataset_root
    if not root.exists():
        return out
    for round_dir in sorted(root.glob("eval-round-*")):
        rewards = _rewards_from_dir(round_dir)
        if rewards:
            out.append(rewards)
    return out


def _rewards_from_dir(round_dir: Path) -> list[float]:
    sidecar = round_dir / "rewards.json"
    if sidecar.exists():
        try:
            rows = json.loads(sidecar.read_text())
            return [float(r.get("reward", 0.0) or 0.0) for r in rows]
        except (json.JSONDecodeError, OSError):
            pass
    rewards: list[float] = []
    for ep in sorted(round_dir.glob("episode_*")):
        meta = ep / "episode.json"
        if meta.exists():
            try:
                data = json.loads(meta.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            rewards.append(float(data.get("score", data.get("total_reward", 0.0)) or 0.0))
    return rewards


async def _emit_round(
    run: RunState,
    activity_index: int,
    round_idx: int,
    rewards: list[float],
    threshold: float,
    *,
    delay: float = 0.0,
) -> None:
    """Emit the same event shapes the real loop produces, for one round of rewards."""
    await emit(run, "train_stage", {
        "activity_index": activity_index, "round": round_idx, "stage": "eval", "status": "started"})
    for i, r in enumerate(rewards):
        await emit(run, "eval_rollout", {
            "activity_index": activity_index, "round": round_idx,
            "index": i, "reward": round(r, 4), "success": bool(r >= _DEMO_SUCCESS_AT)})
        if delay:
            await asyncio.sleep(delay)

    s = _summary(rewards, threshold)
    await emit(run, "eval_summary", {
        "activity_index": activity_index, "round": round_idx,
        "mean_reward": s["mean_reward"], "success_rate": s["success_rate"], "rewards": s["rewards"]})

    await emit(run, "train_stage", {
        "activity_index": activity_index, "round": round_idx, "stage": "curate", "status": "started"})
    await emit(run, "curate", {
        "activity_index": activity_index, "round": round_idx, "threshold": threshold,
        "selected": s["selected"], "available": s["available"],
        "mean_selected_reward": s["mean_selected_reward"]})
    await emit(run, "train_stage", {
        "activity_index": activity_index, "round": round_idx, "stage": "finetune",
        "status": "completed" if s["selected"] else "skipped",
        "detail": {"selected": s["selected"]}})
    if delay:
        await asyncio.sleep(delay)


__all__ = ["run_modal_training"]
