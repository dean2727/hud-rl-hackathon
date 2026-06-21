"""Shared config for the demo-first training loop."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class TrainConfig:
    """Conservative defaults for hackathon iteration."""

    # Use the LIBERO-tuned pi0.5 checkpoint as the default teacher/checkpoint.
    # It matches the current Franka/LIBERO scene and avoids the raw-pi0 cold start.
    base_checkpoint: str = os.environ.get("HUDATHON_BASE_CHECKPOINT", "lerobot/pi05_libero_finetuned_v044")
    policy_family: str = os.environ.get("HUDATHON_POLICY_FAMILY", "pi05")
    dataset_root: Path = Path(os.environ.get("HUDATHON_DATASET_ROOT", str(ROOT / "datasets")))
    checkpoints_root: Path = Path(os.environ.get("HUDATHON_CHECKPOINT_ROOT", str(ROOT / "checkpoints")))
    modal_volume_name: str = os.environ.get("HUDATHON_MODAL_VOLUME", "hudathon-policy-checkpoints")
    group: int = int(os.environ.get("HUDATHON_GROUP", "4"))
    max_steps: int = int(os.environ.get("HUDATHON_MAX_STEPS", "200"))
    max_concurrent: int = int(os.environ.get("HUDATHON_MAX_CONCURRENT", "1"))
    curation_threshold: float = float(os.environ.get("HUDATHON_CURATION_THRESHOLD", "0.25"))
    success_threshold: float = float(os.environ.get("HUDATHON_SUCCESS_THRESHOLD", "0.999"))
    rounds: int = int(os.environ.get("HUDATHON_ROUNDS", "2"))
    scene_id: str = os.environ.get("HUDATHON_SCENE", "franka-libero-v1")
    instruction: str = os.environ.get("HUDATHON_INSTRUCTION", "pick up the red block")
    target_object: str = os.environ.get("HUDATHON_TARGET_OBJECT", "block")
    lift_height: float = float(os.environ.get("HUDATHON_LIFT_HEIGHT", "0.55"))
    replan_horizon: int = int(os.environ.get("HUDATHON_REPLAN_HORIZON", "10"))

    def dataset_dir(self, name: str) -> Path:
        return self.dataset_root / name

    def checkpoint_dir(self, name: str) -> Path:
        return self.checkpoints_root / name
