"""Run graded VLA rollouts and record episodes for curation."""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Callable
import json
import os
import sys
from pathlib import Path
from typing import Any

from hud import LocalRuntime, Taskset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from environment.vla_env import vla_pick  # noqa: E402
from rewards.spec import RewardSpec  # noqa: E402
from train.config import TrainConfig  # noqa: E402


def _remote_agent(remote: str):
    from agents.vla_agent import RemoteAgent

    host, _, port = remote.rpartition(":")
    return RemoteAgent(host=host or "localhost", port=int(port))


def write_reward_sidecar(job: Any, record_dir: Path) -> Path:
    runs = list(getattr(job, "runs", []))
    rows: list[dict[str, Any]] = []
    for index, run in enumerate(runs):
        rows.append({
            "episode_index": index,
            "episode_dir": f"episode_{index:04d}",
            "reward": float(getattr(run, "reward", 0.0) or 0.0),
            "trace_id": getattr(run, "trace_id", None),
        })
    path = record_dir / "rewards.json"
    path.write_text(json.dumps(rows, indent=2) + "\n")
    return path


async def _watch_episodes(
    record_dir: Path,
    on_rollout: Callable[[dict[str, Any]], None],
    stop: asyncio.Event,
    *,
    poll_s: float = 0.5,
) -> None:
    """Emit one event per episode as the bridge finishes writing its episode.json.

    HUD's Taskset.run() returns all rollouts at once, but the env-side bridge dumps
    each episode to disk the instant it completes - so tailing the record dir is how
    we stream per-rollout reward live during collection (rather than only at the end).
    """
    seen: set[str] = set()

    def scan() -> None:
        for ep in sorted(record_dir.glob("episode_*")):
            if ep.name in seen:
                continue
            meta = ep / "episode.json"
            if not meta.exists():
                continue  # still being written
            try:
                data = json.loads(meta.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            seen.add(ep.name)
            try:
                index = int(ep.name.rsplit("_", 1)[-1])
            except ValueError:
                index = len(seen) - 1
            on_rollout({
                "index": index,
                "reward": float(data.get("score", data.get("total_reward", 0.0)) or 0.0),
                "success": bool(data.get("success", False)),
            })

    while not stop.is_set():
        scan()
        try:
            await asyncio.wait_for(stop.wait(), timeout=poll_s)
        except asyncio.TimeoutError:
            pass
    scan()  # final drain for episodes written just before stop


async def run_eval_batch(
    *,
    remote: str,
    record_dir: Path,
    config: TrainConfig | None = None,
    reward_spec: RewardSpec | None = None,
    on_rollout: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    cfg = config or TrainConfig()
    record_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HUDATHON_RECORD_DIR"] = str(record_dir)

    spec = reward_spec or RewardSpec.pick(
        instruction=cfg.instruction,
        target_object=cfg.target_object,
        lift_height=cfg.lift_height,
    )
    tasks = [
        vla_pick(
            scene_id=cfg.scene_id,
            instruction=spec.instruction,
            target_object=spec.target_object or cfg.target_object,
            lift_height=float(spec.params.get("lift_height", cfg.lift_height)),
            seed=i,
            max_steps=cfg.max_steps,
            reward_spec=spec.as_dict(),
        )
        for i in range(cfg.group)
    ]
    agent = _remote_agent(remote)
    # HUD's robot recorder reads this on newer SDKs; the bridge also records via
    # HUDATHON_RECORD_DIR for this template.
    setattr(agent, "save", True)

    # Tail the record dir so callers can stream per-rollout reward as episodes land.
    stop = asyncio.Event()
    watcher = (
        asyncio.create_task(_watch_episodes(record_dir, on_rollout, stop))
        if on_rollout is not None
        else None
    )
    try:
        job = await Taskset("hudathon-vla", tasks).run(
            agent,
            runtime=LocalRuntime(str(ROOT / "environment" / "vla_env.py")),
            max_concurrent=cfg.max_concurrent,
        )
    finally:
        if watcher is not None:
            stop.set()
            await watcher
    sidecar = write_reward_sidecar(job, record_dir)
    rewards = [float(getattr(run, "reward", 0.0) or 0.0) for run in getattr(job, "runs", [])]
    successes = [r >= cfg.success_threshold for r in rewards]
    return {
        "record_dir": str(record_dir),
        "sidecar": str(sidecar),
        "rewards": rewards,
        "mean_reward": sum(rewards) / (len(rewards) or 1),
        "success_rate": sum(successes) / (len(successes) or 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Run a reward-recording VLA eval batch.")
    ap.add_argument("--remote", required=True, metavar="HOST:PORT")
    ap.add_argument("--record-dir", required=True, type=Path)
    args = ap.parse_args()
    summary = asyncio.run(run_eval_batch(remote=args.remote, record_dir=args.record_dir))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
