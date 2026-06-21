"""Modal/local wrapper for pi0.5/VLA fine-tuning with LeRobot.

The input must be a real LeRobot dataset repo id or path accepted by lerobot-train.
The env-side episode folders are useful for curation/debugging, but may need a
conversion step if HUD's agent-side LeRobot recorder was not used.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
from train.config import TrainConfig


@dataclass(frozen=True)
class FinetuneRequest:
    dataset_repo_id: str
    output_dir: Path
    policy_path: str
    # For a dataset built by train/convert.py and stored on the Volume (not pushed to
    # the HF Hub), point lerobot-train at the on-disk copy via dataset.root.
    dataset_root: Path | None = None
    policy_repo_id: str | None = None
    steps: int = 1000
    batch_size: int = 4
    job_name: str = "hudathon-pi05"
    device: str = "cuda"
    dtype: str = "bfloat16"


def build_lerobot_train_cmd(req: FinetuneRequest) -> list[str]:
    cmd = [
        "lerobot-train",
        f"--dataset.repo_id={req.dataset_repo_id}",
        f"--output_dir={req.output_dir}",
        f"--job_name={req.job_name}",
        f"--policy.path={req.policy_path}",
        f"--policy.dtype={req.dtype}",
        f"--policy.device={req.device}",
        f"--steps={req.steps}",
        f"--batch_size={req.batch_size}",
    ]
    if req.dataset_root:
        cmd.append(f"--dataset.root={req.dataset_root}")
    if req.policy_repo_id:
        cmd.append(f"--policy.repo_id={req.policy_repo_id}")
    return cmd


def run_lerobot_train(req: FinetuneRequest, *, dry_run: bool = False) -> dict:
    req.output_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_lerobot_train_cmd(req)
    summary = {"cmd": cmd, "output_dir": str(req.output_dir), "dry_run": dry_run}
    (req.output_dir / "finetune_request.json").write_text(json.dumps(summary, indent=2) + "\n")
    if not dry_run:
        subprocess.run(cmd, check=True)
    return summary


def main() -> None:
    cfg = TrainConfig()
    ap = argparse.ArgumentParser(description="Run or print a lerobot-train fine-tune command.")
    ap.add_argument("--dataset-repo-id", required=True, help="HF repo id or dataset path accepted by lerobot-train")
    ap.add_argument("--dataset-root", type=Path, default=None, help="on-disk dataset root (for a locally-built dataset)")
    ap.add_argument("--output-dir", type=Path, default=cfg.checkpoint_dir("round-000"))
    ap.add_argument("--policy-path", default=cfg.base_checkpoint)
    ap.add_argument("--policy-repo-id", default=None)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    req = FinetuneRequest(
        dataset_repo_id=args.dataset_repo_id,
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        policy_path=args.policy_path,
        policy_repo_id=args.policy_repo_id,
        steps=args.steps,
        batch_size=args.batch_size,
    )
    print(json.dumps(run_lerobot_train(req, dry_run=args.dry_run), indent=2))


if __name__ == "__main__":
    main()
