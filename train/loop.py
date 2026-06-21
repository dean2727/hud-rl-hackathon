"""Round-0 orchestrator for the Modal <-> HUD pi0.5 behavior-cloning loop.

One round, end to end:

    serve(checkpoint) on a Modal A100   ->  eval rollouts locally (Newton sim on CPU,
                                            driving the served policy via RemoteAgent)
    -> curate high-reward episodes      ->  convert them to a LeRobot dataset
    -> upload the dataset to the Modal checkpoint Volume
    -> fine_tune on a Modal A100        ->  new checkpoint committed to the Volume.

The Modal steps (serve + fine_tune) run on GPUs and cost money, and this box has no
GPU, so they are driven through the modal SDK. `serve_policy` publishes its public
tunnel address to a named `modal.Queue`; this orchestrator reads it, points the local
eval at it, then cancels the server. Use --dry-run to exercise the full control flow
(curate + convert run for real against any episodes already on disk; the GPU/Modal
steps are skipped) without spending.

Manual fallback — every stage is runnable on its own if the orchestration misbehaves:

    modal run train/modal_app.py::serve_policy --checkpoint lerobot/pi05_libero_finetuned_v044
    python run_vla.py --remote HOST:PORT --record datasets/eval-round-000 --group 8
    python -m train.curate  --source-dir datasets/eval-round-000 --dest-dir datasets/curated-round-000
    python -m train.convert --source-dir datasets/curated-round-000 --root datasets/lerobot-round-000
    modal volume put hudathon-policy-checkpoints datasets/lerobot-round-000 datasets/round-000
    modal run train/modal_app.py::fine_tune --dataset-repo-id hudathon/vla-pick --dataset-root datasets/round-000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
from train.config import TrainConfig
from train.convert import convert_episodes
from train.curate import curate_episodes

# A cold A100 plus the first-time checkpoint download into the HF-cache Volume can take
# several minutes before the tunnel address is published.
SERVE_WAIT_S = 900


def _round_name(idx: int) -> str:
    return f"round-{idx:03d}"


def _serve_and_eval(*, checkpoint: str, record_dir: Path, cfg: TrainConfig, wait_s: int) -> dict[str, Any]:
    """Spawn the Modal policy server, wait for its tunnel address, run a local eval
    batch against it, then stop the server. Returns the eval summary."""
    import modal
    from train.modal_app import SERVE_ADDR_QUEUE, app, serve_policy
    from train.eval import run_eval_batch

    with app.run():
        addr_q = modal.Queue.from_name(SERVE_ADDR_QUEUE, create_if_missing=True)
        try:
            addr_q.clear()  # drop any stale address from a previous run
        except Exception:
            pass
        call = serve_policy.spawn(checkpoint=checkpoint, policy_family=cfg.policy_family)
        try:
            remote = addr_q.get(timeout=wait_s)  # "host:port" published by serve_policy
            print(f"[loop] policy serving at {remote}; running {cfg.group} eval rollouts", flush=True)
            summary = asyncio.run(run_eval_batch(remote=remote, record_dir=record_dir, config=cfg))
        finally:
            call.cancel()  # free the A100 as soon as eval is done
    return summary


def _upload_dataset(local_dir: Path, vol_subdir: str) -> str:
    """Copy a locally-built LeRobot dataset onto the checkpoint Volume at vol_subdir."""
    from train.modal_app import checkpoint_vol

    with checkpoint_vol.batch_upload(force=True) as batch:
        batch.put_directory(str(local_dir), vol_subdir)
    return vol_subdir


def _finetune(*, repo_id: str, dataset_vol_subdir: str, output_name: str,
              policy_path: str, steps: int, batch_size: int) -> dict[str, Any]:
    from train.modal_app import fine_tune

    return fine_tune.remote(
        dataset_repo_id=repo_id,
        dataset_root=dataset_vol_subdir,
        output_name=output_name,
        policy_path=policy_path,
        steps=steps,
        batch_size=batch_size,
        dry_run=False,
    )


def run_round(
    *,
    round_idx: int,
    checkpoint: str,
    repo_id: str,
    steps: int,
    batch_size: int,
    dry_run: bool,
    episodes_dir: Path | None = None,
    cfg: TrainConfig | None = None,
) -> dict[str, Any]:
    cfg = cfg or TrainConfig()
    rname = _round_name(round_idx)
    raw_dir = episodes_dir or cfg.dataset_dir(f"eval-{rname}")
    curated_dir = cfg.dataset_dir(f"curated-{rname}")
    ds_dir = cfg.dataset_dir(f"lerobot-{rname}")
    vol_subdir = f"datasets/{rname}"
    result: dict[str, Any] = {"round": round_idx, "checkpoint": checkpoint, "dry_run": dry_run}

    if dry_run:
        result["plan"] = [
            f"serve  : Modal serve_policy(checkpoint={checkpoint!r}) -> tunnel address",
            f"eval   : run_eval_batch(remote=..., record_dir={raw_dir}, group={cfg.group})",
            f"curate : curate_episodes(threshold={cfg.curation_threshold}) -> {curated_dir}",
            f"convert: convert_episodes(repo_id={repo_id!r}) -> {ds_dir}",
            f"upload : checkpoint Volume <- {ds_dir}  at  {vol_subdir}",
            f"finetune: Modal fine_tune(dataset_root={vol_subdir}, policy_path={checkpoint!r}, steps={steps}) -> volume:{rname}",
        ]
        # If episodes already exist on disk, prove the local (non-GPU) stages for real.
        if raw_dir.exists() and any(raw_dir.glob("episode_*")):
            result["curated"] = curate_episodes(
                source_dir=raw_dir, dest_dir=curated_dir, threshold=cfg.curation_threshold)
            if result["curated"]["selected"]:
                result["convert"] = convert_episodes(
                    source_dir=curated_dir, repo_id=repo_id, root=ds_dir,
                    default_task=cfg.instruction)
        else:
            result["note"] = f"no episodes under {raw_dir}; run a real eval first to test curate/convert"
        return result

    # 1) serve on Modal + eval locally against it
    result["eval"] = _serve_and_eval(checkpoint=checkpoint, record_dir=raw_dir, cfg=cfg, wait_s=SERVE_WAIT_S)

    # 2) curate the recorded episodes by reward
    result["curated"] = curate_episodes(
        source_dir=raw_dir, dest_dir=curated_dir, threshold=cfg.curation_threshold)
    if result["curated"]["selected"] == 0:
        result["stop"] = "no episodes cleared the curation threshold; nothing to fine-tune"
        return result

    # 3) convert curated episodes -> LeRobot dataset
    result["convert"] = convert_episodes(
        source_dir=curated_dir, repo_id=repo_id, root=ds_dir, default_task=cfg.instruction)

    # 4) upload the dataset to the checkpoint Volume so the Modal finetune can read it
    result["dataset_volume_path"] = _upload_dataset(ds_dir, vol_subdir)

    # 5) fine-tune on Modal, committing the new checkpoint to the Volume
    result["finetune"] = _finetune(
        repo_id=repo_id, dataset_vol_subdir=vol_subdir, output_name=rname,
        policy_path=checkpoint, steps=steps, batch_size=batch_size)
    # lerobot-train writes its loadable weights under output_dir/checkpoints/last/pretrained_model;
    # confirm the exact subdir from the finetune output before serving it next round.
    result["new_checkpoint"] = f"volume:{rname}"
    return result


def main() -> None:
    cfg = TrainConfig()
    ap = argparse.ArgumentParser(description="Run one round of the pi0.5 BC loop end to end.")
    ap.add_argument("--checkpoint", default=cfg.base_checkpoint,
                    help="checkpoint to serve+fine-tune from ('volume:round-NNN' for a Volume checkpoint)")
    ap.add_argument("--round", type=int, default=0)
    ap.add_argument("--repo-id", default="hudathon/vla-pick", help="LeRobot dataset repo id for the built dataset")
    ap.add_argument("--steps", type=int, default=1000, help="lerobot-train steps")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--episodes-dir", type=Path, default=None,
                    help="(dry-run) existing episode_* dir to test curate/convert against")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    summary = run_round(
        round_idx=args.round,
        checkpoint=args.checkpoint,
        repo_id=args.repo_id,
        steps=args.steps,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
        episodes_dir=args.episodes_dir,
        cfg=cfg,
    )
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
