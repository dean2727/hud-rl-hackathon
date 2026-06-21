"""Encode rollout frame PNGs into MP4 (and optional GIF).

Frames are written as frame_0000.png, frame_0001.png, … under a directory.
Uses ffmpeg when available; raises a clear error otherwise.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path


def encode_mp4(
    frames_dir: Path,
    output: Path,
    *,
    fps: int = 10,
    pattern: str = "frame_%04d.png",
) -> Path:
    """Encode sequential PNGs in *frames_dir* to *output* (H.264, yuv420p)."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found on PATH — install ffmpeg to encode rollout videos")

    frames = sorted(frames_dir.glob("frame_*.png"))
    if not frames:
        raise RuntimeError(f"no frames found under {frames_dir}")

    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(frames_dir / pattern),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ],
        check=True,
        capture_output=True,
    )
    return output


def encode_gif(
    frames_dir: Path,
    output: Path,
    *,
    fps: int = 10,
    pattern: str = "frame_%04d.png",
) -> Path | None:
    """Optional GIF fallback (100 ms per frame when fps=10)."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return None
    frames = sorted(frames_dir.glob("frame_*.png"))
    if not frames:
        return None
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(frames_dir / pattern),
            "-vf",
            f"fps={fps},scale=640:-1:flags=lanczos",
            str(output),
        ],
        check=True,
        capture_output=True,
    )
    return output


def frames_from_episode_images(
    episode_dir: Path,
    dest: Path,
    *,
    stride: int = 5,
) -> int:
    """Copy subsampled agentview PNGs from a recorded VLA episode into frame_XXXX layout."""
    src_dir = episode_dir / "images"
    if not src_dir.is_dir():
        return 0
    dest.mkdir(parents=True, exist_ok=True)
    sources = sorted(src_dir.glob("agentview_*.png"))[:: max(1, stride)]
    for i, src in enumerate(sources):
        shutil.copy2(src, dest / f"frame_{i:04d}.png")
    return len(sources)


def best_episode_dir(record_dir: Path) -> Path | None:
    """Return the highest-reward episode folder under a VLA eval record dir."""
    best: Path | None = None
    best_score = -1.0
    for ep in sorted(record_dir.glob("episode_*")):
        meta = ep / "episode.json"
        images = ep / "images"
        if not meta.is_file() or not images.is_dir():
            continue
        if not any(images.glob("agentview_*.png")):
            continue
        try:
            data = json.loads(meta.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        score = float(data.get("score", data.get("total_reward", 0.0)) or 0.0)
        if score >= best_score:
            best, best_score = ep, score
    return best


def prepare_frames_from_record_dir(
    record_dir: Path,
    dest: Path,
    *,
    stride: int = 5,
) -> int:
    """Build frame_XXXX.png sequence from the best episode in an eval record dir."""
    episode = best_episode_dir(record_dir)
    if episode is None:
        return 0
    if dest.exists():
        shutil.rmtree(dest)
    return frames_from_episode_images(episode, dest, stride=stride)


__all__ = [
    "encode_mp4",
    "encode_gif",
    "frames_from_episode_images",
    "best_episode_dir",
    "prepare_frames_from_record_dir",
]
