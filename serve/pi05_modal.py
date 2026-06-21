"""Serve a LeRobot pi0/pi0.5 policy on a Modal GPU.

The zero-infrastructure way to run the policy on a remote GPU box: no machine to
rent or SSH into. This runs the same `serve/policy_server.py` server on a Modal
A100, forwards a public TCP tunnel, and prints the `ws://HOST:PORT` to pass to the
eval machine:

    pip install modal && modal token new          # one-time Modal setup
    modal run serve/pi05_modal.py                  # prints ws://HOST:PORT, stays up

    # then on the (CPU-only) eval machine:
    python run_vla.py --remote HOST:PORT --group 10

The checkpoint downloads once into a Modal Volume and is cached across runs. Stop
the server with Ctrl-C (or let `--timeout` expire).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import modal

CHECKPOINT = os.environ.get("HUDATHON_POLICY_CHECKPOINT", "lerobot/pi05_libero_finetuned_v044")
POLICY_FAMILY = os.environ.get("HUDATHON_POLICY_FAMILY", "pi05")
PORT = 8000
CACHE = "/cache"  # HF cache (checkpoint + processors), Volume-backed so it persists
CHECKPOINTS = "/checkpoints"

# lerobot is pinned to a git commit (0.5.2 isn't on PyPI; PyPI's 0.5.1 lacks pi05).
# The [pi] extra owns its whole ML stack (torch, transformers, scipy, safetensors,
# huggingface-hub, einops, pillow, ...) with tested pins, so we don't re-list those:
# re-listing them unpinned lets pip backtrack into source-only releases that fail to
# build on debian_slim (sentencepiece; scipy via meson/OpenBLAS).
_LEROBOT = "lerobot[pi] @ git+https://github.com/huggingface/lerobot.git@b8ad81bf397d59dda69ccfc7e74e847f0a9d4fbf"

# Mount this package's serve/ dir so the container imports the SAME server code the
# GPU box would run; only meaningful locally (the container hydrates the image).
_SERVE_DIR = Path(__file__).resolve().parent

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
        str(_SERVE_DIR), "/root/serve", copy=True,
        # Skip caches so a stray .pyc write can't trip "modified during build process".
        ignore=["**/__pycache__", "**/*.pyc"],
    )
    .env({"HF_HOME": CACHE, "PYTHONPATH": "/root"})
)

app = modal.App("hudathon-policy-serve")
cache_vol = modal.Volume.from_name("hudathon-pi05-cache", create_if_missing=True)
checkpoint_vol = modal.Volume.from_name("hudathon-policy-checkpoints", create_if_missing=True)


@app.function(
    image=image,
    gpu="A100",
    timeout=24 * 3600,
    volumes={CACHE: cache_vol, CHECKPOINTS: checkpoint_vol},
)
def serve() -> None:
    import asyncio

    sys.path.insert(0, "/root/serve")
    from policy_server import build_lerobot_infer, serve_openpi

    checkpoint = CHECKPOINT
    if checkpoint.startswith("volume:"):
        checkpoint = str(Path(CHECKPOINTS) / checkpoint.removeprefix("volume:").lstrip("/"))
    infer = build_lerobot_infer(checkpoint, device="cuda", policy_family=POLICY_FAMILY)
    with modal.forward(PORT, unencrypted=True) as tunnel:
        host, port = tunnel.tcp_socket
        print(f"[serve] policy ready - run: python run_vla.py --remote {host}:{port}", flush=True)
        asyncio.run(serve_openpi(
            "0.0.0.0",
            PORT,
            infer,
            metadata={"checkpoint": checkpoint, "policy_family": POLICY_FAMILY},
        ))


@app.local_entrypoint()
def main() -> None:
    serve.remote()
