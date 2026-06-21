"""Bootstrap successful demonstrations for round 0.

For the hackathon path, the practical demo source is a warm-start checkpoint that
already has some LIBERO skill. This module records its rollouts through the same
VLA environment so curation/training sees the same task distribution as eval.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
from train.config import ROOT, TrainConfig


def build_demo_command(*, record_dir: Path, config: TrainConfig | None = None) -> list[str]:
    cfg = config or TrainConfig()
    return [
        "python",
        "run_vla.py",
        "--checkpoint",
        cfg.base_checkpoint,
        "--group",
        str(cfg.group),
        "--max-steps",
        str(cfg.max_steps),
        "--max-concurrent",
        str(cfg.max_concurrent),
        "--record",
        str(record_dir),
        "--instruction",
        cfg.instruction,
        "--target-object",
        cfg.target_object,
        "--lift-height",
        str(cfg.lift_height),
    ]


def record_warmstart_demos(*, record_dir: Path, dry_run: bool = False) -> dict:
    record_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_demo_command(record_dir=record_dir)
    summary = {"cmd": cmd, "record_dir": str(record_dir), "dry_run": dry_run}
    (record_dir / "demo_request.json").write_text(json.dumps(summary, indent=2) + "\n")
    if not dry_run:
        subprocess.run(cmd, cwd=str(ROOT), check=True)
    return summary


def main() -> None:
    cfg = TrainConfig()
    ap = argparse.ArgumentParser(description="Record round-0 warm-start demonstrations.")
    ap.add_argument("--record-dir", type=Path, default=cfg.dataset_dir("demos-round-000"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    print(json.dumps(record_warmstart_demos(record_dir=args.record_dir, dry_run=args.dry_run), indent=2))


if __name__ == "__main__":
    main()
