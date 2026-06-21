"""Declarative reward specifications.

User-authored tasks should compile to this small JSON shape, not arbitrary Python.
The fixed reward engine interprets the archetype and parameters against sim state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Literal

RewardArchetype = Literal["pick", "place", "move", "open", "grasp"]


@dataclass(frozen=True)
class RewardWeights:
    """Weights for shaped progress and binary success."""

    progress: float = 0.5
    success: float = 0.5

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | None) -> "RewardWeights":
        if not data:
            return cls()
        return cls(
            progress=float(data.get("progress", cls.progress)),
            success=float(data.get("success", cls.success)),
        )

    def as_dict(self) -> dict[str, float]:
        return {"progress": self.progress, "success": self.success}


@dataclass(frozen=True)
class RewardSpec:
    """A safe, serializable task reward description.

    Examples:
        {"archetype": "pick", "target_object": "mug", "params": {"lift_height": 0.55}}
        {"archetype": "place", "target_object": "mug", "params": {"goal_xyz": [0, 0, 0.8]}}
    """

    archetype: RewardArchetype
    instruction: str
    target_object: str | None = None
    target_joint: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    weights: RewardWeights = field(default_factory=RewardWeights)

    @classmethod
    def pick(
        cls,
        *,
        instruction: str,
        target_object: str,
        lift_height: float,
        weights: RewardWeights | None = None,
    ) -> "RewardSpec":
        return cls(
            archetype="pick",
            instruction=instruction,
            target_object=target_object,
            params={"lift_height": float(lift_height)},
            weights=weights or RewardWeights(),
        )

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "RewardSpec":
        archetype = data.get("archetype")
        if archetype not in {"pick", "place", "move", "open", "grasp"}:
            raise ValueError(f"unsupported reward archetype: {archetype!r}")
        return cls(
            archetype=archetype,
            instruction=str(data.get("instruction", "")),
            target_object=data.get("target_object"),
            target_joint=data.get("target_joint"),
            params=dict(data.get("params") or {}),
            weights=RewardWeights.from_mapping(data.get("weights")),
        )

    @classmethod
    def from_json(cls, data: str) -> "RewardSpec":
        return cls.from_mapping(json.loads(data))

    @classmethod
    def parse(cls, data: str | dict[str, Any] | "RewardSpec") -> "RewardSpec":
        if isinstance(data, cls):
            return data
        if isinstance(data, str):
            return cls.from_json(data)
        return cls.from_mapping(data)

    def as_dict(self) -> dict[str, Any]:
        return {
            "archetype": self.archetype,
            "instruction": self.instruction,
            "target_object": self.target_object,
            "target_joint": self.target_joint,
            "params": self.params,
            "weights": self.weights.as_dict(),
        }

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), sort_keys=True)
