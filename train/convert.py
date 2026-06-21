"""Convert env-side recorded episodes into a LeRobot v3 dataset for ``lerobot-train``.

This is the data boundary the rest of train/ kept flagging: the VLA bridge
(`sim/franka_bridge.py`) records, per episode, under ``HUDATHON_RECORD_DIR``:

    episode_XXXX/
        images/agentview_{t:04d}.png, wrist_{t:04d}.png
        steps.jsonl    # one line per control step: {"t", "state":[8], "action":[7]}
        episode.json   # {instruction, target_object, reward_spec, score, success, ...}

`lerobot-train` instead needs a LeRobot dataset (``observation.images.*``,
``observation.state``, ``action``, ``task``). This module builds exactly that from
the curated episode folders, using the franka-libero / pi0.5 feature names so a
``--policy.path=<pi05 checkpoint>`` fine-tune sees in-distribution keys/shapes.

If ``lerobot-train`` later complains about a feature-name or shape mismatch with the
checkpoint, adjust ``--image-keys`` / ``--state-dim`` / ``--action-dim`` here — they
default to the LIBERO convention `lerobot/pi05_libero_finetuned_v044` was trained on.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

# Contract defaults (see contracts/franka_libero.json + scenes/franka-libero-v1).
IMAGE_SHAPE = (256, 256, 3)
STATE_DIM = 8
ACTION_DIM = 7
DEFAULT_FPS = 20  # matches the contract's 20 Hz control rate
# env-side image file stem -> LeRobot feature key (LIBERO/pi0.5 convention).
DEFAULT_IMAGE_KEYS: dict[str, str] = {
    "agentview": "observation.images.image",
    "wrist": "observation.images.wrist_image",
}


def build_features(
    *,
    state_dim: int = STATE_DIM,
    action_dim: int = ACTION_DIM,
    image_shape: tuple[int, int, int] = IMAGE_SHAPE,
    image_keys: dict[str, str] | None = None,
    use_videos: bool = True,
) -> dict[str, dict[str, Any]]:
    image_keys = image_keys or DEFAULT_IMAGE_KEYS
    img_dtype = "video" if use_videos else "image"
    # lerobot's validate_frame compares value.shape (a tuple) against feature["shape"]
    # for equality, so these must be tuples — a list [7] != (7,) and fails validation.
    features: dict[str, dict[str, Any]] = {}
    for key in image_keys.values():
        features[key] = {
            "dtype": img_dtype,
            "shape": tuple(image_shape),
            "names": ["height", "width", "channels"],
        }
    features["observation.state"] = {
        "dtype": "float32",
        "shape": (state_dim,),
        "names": ["state"],
    }
    features["action"] = {
        "dtype": "float32",
        "shape": (action_dim,),
        "names": ["action"],
    }
    return features


def _episode_dirs(source_dir: Path) -> list[Path]:
    return [p for p in sorted(source_dir.glob("episode_*")) if (p / "steps.jsonl").exists()]


def _read_steps(ep_dir: Path) -> list[dict[str, Any]]:
    lines = (ep_dir / "steps.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def convert_episodes(
    *,
    source_dir: Path,
    repo_id: str,
    root: Path,
    fps: int = DEFAULT_FPS,
    use_videos: bool = True,
    image_keys: dict[str, str] | None = None,
    default_task: str = "pick up the red block",
    overwrite: bool = True,
) -> dict[str, Any]:
    """Build a LeRobot dataset at ``root`` from ``source_dir``'s episode folders."""
    import numpy as np
    from PIL import Image
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    image_keys = image_keys or DEFAULT_IMAGE_KEYS
    episodes = _episode_dirs(source_dir)
    if not episodes:
        raise ValueError(f"no episode_* folders with steps.jsonl found under {source_dir}")

    root = Path(root)
    if root.exists():
        if not overwrite:
            raise FileExistsError(f"dataset root already exists: {root} (pass overwrite=True)")
        shutil.rmtree(root)

    features = build_features(image_keys=image_keys, use_videos=use_videos)
    ds = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=features,
        root=root,
        robot_type="franka_panda_libero",
        use_videos=use_videos,
    )

    total_frames = 0
    kept_episodes = 0
    for ep_dir in episodes:
        meta_path = ep_dir / "episode.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        task = str(meta.get("instruction") or default_task)
        steps = _read_steps(ep_dir)
        if not steps:
            continue
        for step in steps:
            t = int(step["t"])
            frame: dict[str, Any] = {
                "observation.state": np.asarray(step["state"], dtype=np.float32),
                "action": np.asarray(step["action"], dtype=np.float32),
            }
            for stem, key in image_keys.items():
                img_path = ep_dir / "images" / f"{stem}_{t:04d}.png"
                frame[key] = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.uint8)
            ds.add_frame(frame, task=task)
        ds.save_episode()
        total_frames += len(steps)
        kept_episodes += 1

    return {
        "source_dir": str(source_dir),
        "dataset_root": str(root),
        "repo_id": repo_id,
        "episodes": kept_episodes,
        "frames": total_frames,
        "fps": fps,
        "use_videos": use_videos,
        "features": list(features),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert recorded episodes into a LeRobot dataset.")
    ap.add_argument("--source-dir", required=True, type=Path, help="dir of episode_* folders (curated)")
    ap.add_argument("--repo-id", default="hudathon/vla-pick", help="LeRobot dataset repo id (local path id)")
    ap.add_argument("--root", required=True, type=Path, help="output dataset root directory")
    ap.add_argument("--fps", type=int, default=DEFAULT_FPS)
    ap.add_argument("--no-videos", action="store_true", help="store image frames instead of mp4 (no ffmpeg)")
    ap.add_argument("--task", default="pick up the red block", help="fallback task string if episode.json lacks one")
    args = ap.parse_args()
    summary = convert_episodes(
        source_dir=args.source_dir,
        repo_id=args.repo_id,
        root=args.root,
        fps=args.fps,
        use_videos=not args.no_videos,
        default_task=args.task,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
