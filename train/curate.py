"""Curate recorded episodes into a training dataset.

This operates on the template's env-side recording layout:
record_dir/episode_0000/{episode.json, steps.jsonl, images/...}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Any


def _load_rewards(record_dir: Path) -> dict[str, float]:
    sidecar = record_dir / "rewards.json"
    if sidecar.exists():
        rows = json.loads(sidecar.read_text())
        return {str(row["episode_dir"]): float(row["reward"]) for row in rows}
    rewards: dict[str, float] = {}
    for ep in sorted(record_dir.glob("episode_*")):
        meta = ep / "episode.json"
        if meta.exists():
            data = json.loads(meta.read_text())
            rewards[ep.name] = float(data.get("score", data.get("total_reward", 0.0)))
    return rewards


def curate_episodes(
    *,
    source_dir: Path,
    dest_dir: Path,
    threshold: float,
    top_k: int | None = None,
) -> dict[str, Any]:
    rewards = _load_rewards(source_dir)
    candidates = sorted(rewards.items(), key=lambda item: item[1], reverse=True)
    selected = [(name, reward) for name, reward in candidates if reward >= threshold]
    if top_k is not None:
        selected = selected[:top_k]

    dest_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    for out_idx, (name, reward) in enumerate(selected):
        src = source_dir / name
        if not src.is_dir():
            continue
        dst = dest_dir / f"episode_{out_idx:04d}"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        manifest.append({"source_episode": name, "episode_dir": dst.name, "reward": reward})

    (dest_dir / "curation.json").write_text(json.dumps({
        "source_dir": str(source_dir),
        "threshold": threshold,
        "top_k": top_k,
        "selected": manifest,
    }, indent=2) + "\n")
    return {
        "source_dir": str(source_dir),
        "dest_dir": str(dest_dir),
        "selected": len(manifest),
        "available": len(candidates),
        "mean_selected_reward": (
            sum(item["reward"] for item in manifest) / len(manifest)
            if manifest else 0.0
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Curate high-reward recorded episodes.")
    ap.add_argument("--source-dir", required=True, type=Path)
    ap.add_argument("--dest-dir", required=True, type=Path)
    ap.add_argument("--threshold", type=float, default=0.25)
    ap.add_argument("--top-k", type=int, default=None)
    args = ap.parse_args()
    print(json.dumps(curate_episodes(
        source_dir=args.source_dir,
        dest_dir=args.dest_dir,
        threshold=args.threshold,
        top_k=args.top_k,
    ), indent=2))


if __name__ == "__main__":
    main()
