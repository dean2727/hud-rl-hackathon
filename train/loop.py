"""Orchestrate the demo-first training loop.

This file wires the stages together without hiding the important data boundary:
`lerobot-train` needs a LeRobot dataset repo/path. If the run only has this
template's env-side episode folders, curate them first and convert/push them
before fine-tuning.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
from train.config import TrainConfig
from train.curate import curate_episodes
from train.demos import record_warmstart_demos
from train.finetune import FinetuneRequest, run_lerobot_train


def bootstrap_round(*, dataset_repo_id: str | None, dry_run: bool) -> dict:
    cfg = TrainConfig()
    demo_dir = cfg.dataset_dir("demos-round-000")
    curated_dir = cfg.dataset_dir("curated-round-000")
    demo = record_warmstart_demos(record_dir=demo_dir, dry_run=dry_run)
    curated = curate_episodes(
        source_dir=demo_dir,
        dest_dir=curated_dir,
        threshold=cfg.curation_threshold,
    )
    result: dict = {"demo": demo, "curated": curated}
    if dataset_repo_id:
        finetune = run_lerobot_train(
            FinetuneRequest(
                dataset_repo_id=dataset_repo_id,
                output_dir=cfg.checkpoint_dir("round-000"),
                policy_path=cfg.base_checkpoint,
                job_name="round-000",
            ),
            dry_run=dry_run,
        )
        result["finetune"] = finetune
    else:
        result["next_step"] = (
            "Convert/push curated episodes to a LeRobot dataset, then rerun with "
            "--dataset-repo-id HF_USER/dataset."
        )
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the round-0 demo-first training workflow.")
    ap.add_argument("--dataset-repo-id", default=None, help="LeRobot dataset repo/path for lerobot-train")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    print(json.dumps(bootstrap_round(dataset_repo_id=args.dataset_repo_id, dry_run=args.dry_run), indent=2))


if __name__ == "__main__":
    main()
