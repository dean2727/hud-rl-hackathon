"""Generate rollout MP4s: serve fine-tuned policy on Modal, run the activity, encode video."""

from __future__ import annotations

import asyncio
import dataclasses
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.runs import RunState, emit
from sim.rollout_video import encode_gif, encode_mp4
from train.config import TrainConfig
from train.video_rollout import run_modal_video_rollout

VIDEOS_ROOT = Path(__file__).resolve().parent / "data" / "videos"
VIDEO_FPS = int(__import__("os").environ.get("HUDATHON_VIDEO_FPS", "10"))


def _build_cfg(run: RunState, activity) -> TrainConfig:
    base = TrainConfig()
    target = (activity.mapping.target if activity and activity.mapping else None) or base.target_object
    return dataclasses.replace(
        base,
        scene_id=run.scene_id or base.scene_id,
        instruction=(activity.activity if activity else base.instruction),
        target_object=target,
        group=1,
        max_concurrent=1,
    )


def _video_paths(run_id: str, activity_index: int) -> tuple[Path, Path, Path]:
    base = VIDEOS_ROOT / run_id / str(activity_index)
    return base / "frames", base / "rollout.mp4", base / "rollout.gif"


def _resolve_checkpoint(activity, cfg: TrainConfig) -> str:
    """Map activity state to a checkpoint Modal can serve."""
    ckpt = (activity.best_checkpoint if activity else None) or cfg.base_checkpoint
    if not ckpt or ckpt.isdigit():
        return cfg.base_checkpoint
    if ckpt.startswith("volume:") or ckpt.startswith("lerobot/"):
        return ckpt
    return cfg.base_checkpoint


def _encode_frames(frames_dir: Path, mp4_path: Path, gif_path: Path) -> dict[str, Any]:
    n = len(list(frames_dir.glob("frame_*.png")))
    if n < 2:
        raise RuntimeError(
            f"only {n} frame(s) captured — expected a full rollout "
            f"(check scene cameras and Modal policy connection)"
        )
    encode_mp4(frames_dir, mp4_path, fps=VIDEO_FPS)
    gif = encode_gif(frames_dir, gif_path, fps=VIDEO_FPS)
    return {
        "mp4": str(mp4_path),
        "gif": str(gif) if gif else None,
        "frames": n,
        "duration_s": round(n / VIDEO_FPS, 2),
    }


async def generate_rollout_video(
    run: RunState,
    activity_index: int,
    *,
    force: bool = False,
) -> None:
    """Serve fine-tuned policy on Modal, run the user's activity, encode MP4."""
    activity = next((a for a in run.activity_states if a.activity_index == activity_index), None)
    if activity is None:
        await emit(run, "video_error", {
            "activity_index": activity_index,
            "message": f"no such activity_index {activity_index}",
        })
        return

    frames_dir, mp4_path, gif_path = _video_paths(run.run_id, activity_index)
    cfg = _build_cfg(run, activity)
    checkpoint = _resolve_checkpoint(activity, cfg)

    if not force and mp4_path.is_file():
        n = len(list(frames_dir.glob("frame_*.png"))) if frames_dir.is_dir() else 0
        if n >= 2:
            await emit(run, "video_ready", {
                "activity_index": activity_index,
                "url": f"/api/runs/{run.run_id}/videos/{activity_index}",
                "mp4": str(mp4_path),
                "frames": n,
                "duration_s": round(n / VIDEO_FPS, 2),
                "source": "cached",
            })
            return

    await emit(run, "video_generating", {
        "activity_index": activity_index,
        "detail": {
            "checkpoint": checkpoint,
            "instruction": cfg.instruction,
            "scene_id": cfg.scene_id,
        },
    })

    try:
        if frames_dir.exists():
            shutil.rmtree(frames_dir)
        if mp4_path.exists():
            mp4_path.unlink()

        rollout = await asyncio.to_thread(
            run_modal_video_rollout,
            checkpoint=checkpoint,
            cfg=cfg,
            frames_dir=frames_dir,
        )
        meta = await asyncio.to_thread(_encode_frames, frames_dir, mp4_path, gif_path)
        activity.video_path = str(mp4_path)
        await emit(run, "video_ready", {
            "activity_index": activity_index,
            "url": f"/api/runs/{run.run_id}/videos/{activity_index}",
            "checkpoint": checkpoint,
            "instruction": cfg.instruction,
            "reward": rollout.get("reward"),
            **meta,
        })
    except Exception as exc:
        await emit(run, "video_error", {
            "activity_index": activity_index,
            "message": str(exc),
        })


__all__ = ["VIDEOS_ROOT", "generate_rollout_video"]
