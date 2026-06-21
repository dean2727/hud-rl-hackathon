"""Run one VLA rollout against a Modal-served policy and capture rollout video frames."""

from __future__ import annotations

import asyncio
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from environment.vla_env import vla_pick  # noqa: E402
from rewards.spec import RewardSpec  # noqa: E402
from train.config import TrainConfig  # noqa: E402
from train.modal_serve import SERVE_WAIT_S, modal_policy_server  # noqa: E402


async def _single_rollout(
    *,
    remote: str,
    cfg: TrainConfig,
    frames_dir: Path,
    seed: int = 42,
) -> dict[str, Any]:
    from hud import LocalRuntime, Taskset

    from agents.vla_agent import RemoteAgent

    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    spec = RewardSpec.pick(
        instruction=cfg.instruction,
        target_object=cfg.target_object,
        lift_height=cfg.lift_height,
    )
    host, _, port = remote.rpartition(":")
    agent = RemoteAgent(host=host or "localhost", port=int(port))
    task = vla_pick(
        scene_id=cfg.scene_id,
        instruction=spec.instruction,
        target_object=spec.target_object or cfg.target_object,
        lift_height=cfg.lift_height,
        seed=seed,
        max_steps=cfg.max_steps,
        reward_spec=spec.as_dict(),
        video_dir=str(frames_dir),
    )
    job = await Taskset("hudathon-vla", [task]).run(
        agent,
        runtime=LocalRuntime(str(ROOT / "environment" / "vla_env.py")),
        max_concurrent=1,
    )
    run = job.runs[-1] if job.runs else None
    n_frames = len(list(frames_dir.glob("frame_*.png")))
    return {
        "frames": n_frames,
        "reward": float(getattr(run, "reward", 0.0) or 0.0) if run else 0.0,
        "remote": remote,
    }


def run_modal_video_rollout(
    *,
    checkpoint: str,
    cfg: TrainConfig,
    frames_dir: Path,
    wait_s: int = SERVE_WAIT_S,
) -> dict[str, Any]:
    """Serve *checkpoint* on Modal, run the activity once, capture frames locally."""
    with modal_policy_server(checkpoint, policy_family=cfg.policy_family, wait_s=wait_s) as remote:
        summary = asyncio.run(_single_rollout(remote=remote, cfg=cfg, frames_dir=frames_dir))
    summary["checkpoint"] = checkpoint
    return summary


__all__ = ["run_modal_video_rollout"]
