"""Fixed reward archetype implementations.

The engine deliberately keeps reward logic in code we own. User tasks and LLM
reward design produce RewardSpec JSON, which selects an archetype and parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .spec import RewardSpec


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


@dataclass(frozen=True)
class RewardComputation:
    """A scalar reward plus named components for HUD subscores and diagnostics."""

    score: float
    progress: float
    success: bool
    progress_name: str
    success_name: str = "binary_success"
    details: dict[str, Any] | None = None

    def subscores(self, spec: RewardSpec) -> list[dict[str, float | str]]:
        return [
            {
                "name": self.progress_name,
                "weight": spec.weights.progress,
                "value": round(self.progress, 4),
            },
            {
                "name": self.success_name,
                "weight": spec.weights.success,
                "value": 1.0 if self.success else 0.0,
            },
        ]


def weighted_score(spec: RewardSpec, progress: float, success: bool) -> float:
    """Compute the self-consistent weighted score for a RewardSpec."""

    raw = spec.weights.progress * progress + spec.weights.success * (1.0 if success else 0.0)
    total = spec.weights.progress + spec.weights.success
    if total <= 0:
        raise ValueError("reward weights must sum to a positive value")
    return _clamp01(raw / total)


def compute_pick_reward(spec: RewardSpec, *, initial_z: float, final_z: float) -> RewardComputation:
    """Pick/lift reward: shaped lift progress plus binary success."""

    if spec.archetype != "pick":
        raise ValueError(f"compute_pick_reward expected archetype='pick', got {spec.archetype!r}")
    lift_height = float(spec.params.get("lift_height", 0.55))
    span = lift_height - initial_z
    progress = 1.0 if span <= 1e-3 else _clamp01((final_z - initial_z) / span)
    success = final_z >= lift_height
    return RewardComputation(
        score=round(weighted_score(spec, progress, success), 4),
        progress=round(progress, 4),
        success=success,
        progress_name="lift_progress",
        details={
            "initial_z": round(initial_z, 4),
            "final_z": round(final_z, 4),
            "lift_height": lift_height,
        },
    )
