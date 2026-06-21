"""Modal entrypoints for serving and fine-tuning the default pi0.5/VLA policy.

Examples:
    modal run train/modal_app.py::serve_policy
    modal run train/modal_app.py::fine_tune --dataset-repo-id HF_USER/my_dataset

NOTE: Must run "uv run modal secret create huggingface HF_TOKEN=<TOKEN>" first
"""

from __future__ import annotations

from pathlib import Path
import sys

import modal

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
from train.config import ROOT, TrainConfig

CFG = TrainConfig()
APP_ROOT = "/root/hudathon"
HF_CACHE = "/cache"
CHECKPOINTS = "/checkpoints"
# Named queue the serving function uses to hand its public tunnel address back to a
# local orchestrator (train/loop.py); harmless for a plain `modal run ::serve_policy`.
SERVE_ADDR_QUEUE = "hudathon-serve-addr"

# lerobot[pi]: the pi0.5 policy extra. It owns its whole ML stack (torch, transformers,
# scipy, safetensors, huggingface-hub, einops, pillow, ...) with tested version pins, so we
# don't re-list those here. Re-listing them unpinned lets pip's resolver backtrack into
# source-only releases that fail to build on debian_slim (sentencepiece, scipy meson/OpenBLAS).
_LEROBOT = "lerobot[pi] @ git+https://github.com/huggingface/lerobot.git@b8ad81bf397d59dda69ccfc7e74e847f0a9d4fbf"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "ffmpeg")
    .pip_install(
        # Plain hud-python, NOT hud-python[robot]. The [robot] extra pulls openpi-client,
        # which pins numpy<2.0 -- a hard conflict with lerobot's numpy>=2.0. With the
        # extra in the solve, pip silently backslides hud-python to an old release whose
        # extra omits openpi-client, so `import openpi_client` fails at runtime. The base
        # package still ships hud.agents.robot.model (it imports only numpy).
        "hud-python>=0.6.5",          # hud.agents.robot.model.LeRobotModel (serve path)
        _LEROBOT,                     # PI05Policy + pre/post processors + its pinned ML deps
        "accelerate>=1.10.0,<2.0.0",  # transformers model loading; not in lerobot[pi], pin to its range
        "websockets",                 # policy server transport + openpi-client codec dep
        "msgpack",                    # openpi/0 wire codec dep (openpi-client below is --no-deps)
    )
    # openpi/0 wire codec, installed WITHOUT its deps: openpi-client's numpy<2.0 pin is
    # overly conservative (msgpack_numpy.py is numpy-2 safe) and would otherwise reconflict
    # with lerobot's numpy>=2.0. numpy/msgpack/pillow already come from the resolve above.
    .pip_install("openpi-client>=0.1.2", extra_options="--no-deps")
    .add_local_dir(
        str(ROOT), APP_ROOT, copy=True,
        # Exclude .git (its FETCH_HEAD churns mid-build -> "modified during build process"),
        # caches, and large/volatile trees the image never reads: datasets/checkpoints live
        # in Modal Volumes; frontend/wheels/telemetry-dump are eval-machine/local-only.
        ignore=[
            ".git", "**/__pycache__", "**/*.pyc",
            "datasets", "checkpoints", "wheels", "frontend", "telemetry-dump",
        ],
    )
    .env({"HF_HOME": HF_CACHE, "PYTHONPATH": APP_ROOT})
)

app = modal.App("hudathon-train")
cache_vol = modal.Volume.from_name("hudathon-hf-cache", create_if_missing=True)
checkpoint_vol = modal.Volume.from_name(CFG.modal_volume_name, create_if_missing=True)
# HF auth for higher rate limits + faster checkpoint downloads. Create the secret once:
#   modal secret create huggingface HF_TOKEN=hf_xxx
# huggingface_hub reads HF_TOKEN from the env this injects. (Public repos still work
# without it — just rate-limited; the secret must exist once it's referenced here.)
hf_secret = modal.Secret.from_name("huggingface")


@app.function(
    image=image,
    gpu="A100",
    timeout=24 * 3600,
    volumes={HF_CACHE: cache_vol, CHECKPOINTS: checkpoint_vol},
    secrets=[hf_secret],
)
def serve_policy(
    checkpoint: str = CFG.base_checkpoint,
    policy_family: str = CFG.policy_family,
    addr_queue_name: str = SERVE_ADDR_QUEUE,
) -> None:
    import os

    # pi0.5 lazily torch.compiles its action expert; on the first inference TorchInductor
    # autotunes hundreds of matmuls (~5-8s each + shared-mem OOM retries) -> a 10-20 min
    # cold start before a single action comes back. For eval/serving we don't need compiled
    # kernels: disable dynamo so the first inference runs eager in seconds (steady-state eager
    # on an A100 is fine for rollouts). Must be set before torch is imported below.
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

    import asyncio
    import sys

    sys.path.insert(0, APP_ROOT)
    from serve.policy_server import build_lerobot_infer, serve_openpi

    resolved = str(Path(CHECKPOINTS) / checkpoint.removeprefix("volume:").lstrip("/")) if checkpoint.startswith("volume:") else checkpoint
    infer = build_lerobot_infer(resolved, device="cuda", policy_family=policy_family)
    with modal.forward(8000, unencrypted=True) as tunnel:
        host, port = tunnel.tcp_socket
        print(f"[serve] policy ready - use --remote {host}:{port}", flush=True)
        # Hand the address to a local orchestrator (train/loop.py) if one is waiting.
        try:
            modal.Queue.from_name(addr_queue_name, create_if_missing=True).put(f"{host}:{port}")
        except Exception as exc:  # never let address-publishing kill the server
            print(f"[serve] address publish skipped: {exc}", flush=True)
        asyncio.run(serve_openpi(
            "0.0.0.0",
            8000,
            infer,
            metadata={"checkpoint": resolved, "policy_family": policy_family},
        ))


@app.function(
    image=image,
    gpu="A100",
    timeout=24 * 3600,
    volumes={HF_CACHE: cache_vol, CHECKPOINTS: checkpoint_vol},
    secrets=[hf_secret],
)
def fine_tune(
    dataset_repo_id: str,
    dataset_root: str | None = None,
    output_name: str = "round-000",
    policy_path: str = CFG.base_checkpoint,
    steps: int = 1000,
    batch_size: int = 4,
    dry_run: bool = False,
) -> dict:
    import sys

    sys.path.insert(0, APP_ROOT)
    from train.finetune import FinetuneRequest, run_lerobot_train

    # A volume-relative dataset dir (e.g. "datasets/round-000") resolves under the
    # mounted checkpoint Volume so lerobot-train reads the locally-built dataset.
    root = None
    if dataset_root:
        root = Path(dataset_root)
        if not root.is_absolute():
            root = Path(CHECKPOINTS) / dataset_root
    if policy_path.startswith("volume:"):
        policy_path = str(Path(CHECKPOINTS) / policy_path.removeprefix("volume:").lstrip("/"))

    req = FinetuneRequest(
        dataset_repo_id=dataset_repo_id,
        dataset_root=root,
        output_dir=Path(CHECKPOINTS) / output_name,
        policy_path=policy_path,
        steps=steps,
        batch_size=batch_size,
        job_name=output_name,
    )
    result = run_lerobot_train(req, dry_run=dry_run)
    checkpoint_vol.commit()
    return result


@app.local_entrypoint()
def main(
    action: str = "serve",
    dataset_repo_id: str | None = None,
    dataset_root: str | None = None,
    dry_run: bool = False,
) -> None:
    if action == "serve":
        serve_policy.remote()
    elif action == "finetune":
        if not dataset_repo_id:
            raise ValueError("--dataset-repo-id is required when action=finetune")
        print(fine_tune.remote(
            dataset_repo_id=dataset_repo_id, dataset_root=dataset_root, dry_run=dry_run,
        ))
    else:
        raise ValueError(f"unknown action: {action}")
