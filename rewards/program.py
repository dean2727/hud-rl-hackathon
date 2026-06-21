"""A composable reward program: the dynamic generalization of fixed archetypes.

A `RewardProgram` is a small, JSON-serializable composition of vetted predicates
(rewards/predicates.py). It is what the reward *compiler* (backend/reward_compiler.py)
emits for an arbitrary user activity - no per-skill Python, no `archetype`
enumeration to extend. The engine interprets it against sim state:

    score = (w_progress * progress + w_success * success) / (w_progress + w_success)

where `progress` is the weighted blend of the (continuous) term predicates and
`success` is the boolean over the success clauses. `score` is what the curation
filter ranks on; `success` is what gates an episode into the training set.

The JSON shape (what an LLM produces / what we persist in episode.json):

    {
      "instruction": "pick up the shirt and place it to the left of the can",
      "target_object": "shirt",
      "terms": [
        {"name": "grasp",      "weight": 0.15, "fn": "grasped",       "args": {"object": "shirt"}},
        {"name": "lift",       "weight": 0.15, "fn": "lifted",        "args": {"object": "shirt", "height": 0.1}},
        {"name": "left_of_can","weight": 0.4,  "fn": "relative_side", "args": {"object": "shirt", "reference": "can", "axis": "x", "sign": "negative", "full_margin": 0.15}},
        {"name": "on_desk",    "weight": 0.3,  "fn": "on_surface",    "args": {"object": "shirt", "tol": 0.05}}
      ],
      "success": [
        {"fn": "relative_side", "args": {"object": "shirt", "reference": "can", "axis": "x", "sign": "negative", "full_margin": 0.15}, "threshold": 0.2},
        {"fn": "near",          "args": {"object": "shirt", "reference": "can", "radius": 0.5}, "threshold": 0.2},
        {"fn": "on_surface",    "args": {"object": "shirt", "tol": 0.06}, "threshold": 0.6}
      ],
      "success_mode": "all",
      "weights": {"progress": 0.5, "success": 0.5}
    }
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any

from .engine import RewardComputation
from .predicates import OBJECT_ARG_KEYS, REGISTRY, Baseline, SimView
from .predicates import evaluate as _eval_predicate
from .spec import RewardSpec, RewardWeights


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def _validate_fn(fn: str) -> str:
    if fn not in REGISTRY:
        raise ValueError(f"unknown reward predicate {fn!r} (known: {sorted(REGISTRY)})")
    return fn


@dataclass(frozen=True)
class Term:
    """One shaped progress component: a predicate read continuously, weighted."""

    name: str
    weight: float
    fn: str
    args: dict[str, Any] = field(default_factory=dict)

    def value(self, view: SimView, base: Baseline) -> float:
        return _eval_predicate(self.fn, view, base, self.args)

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "weight": self.weight, "fn": self.fn, "args": self.args}

    @classmethod
    def from_mapping(cls, d: dict[str, Any]) -> "Term":
        return cls(
            name=str(d.get("name", d.get("fn", "term"))),
            weight=float(d.get("weight", 1.0)),
            fn=_validate_fn(str(d["fn"])),
            args=dict(d.get("args") or {}),
        )


@dataclass(frozen=True)
class SuccessClause:
    """A predicate read against a threshold to decide (part of) success."""

    fn: str
    args: dict[str, Any] = field(default_factory=dict)
    threshold: float = 0.999

    def value(self, view: SimView, base: Baseline) -> float:
        return _eval_predicate(self.fn, view, base, self.args)

    def satisfied(self, view: SimView, base: Baseline) -> bool:
        return self.value(view, base) >= self.threshold

    def as_dict(self) -> dict[str, Any]:
        return {"fn": self.fn, "args": self.args, "threshold": self.threshold}

    @classmethod
    def from_mapping(cls, d: dict[str, Any]) -> "SuccessClause":
        return cls(
            fn=_validate_fn(str(d["fn"])),
            args=dict(d.get("args") or {}),
            threshold=float(d.get("threshold", 0.999)),
        )


@dataclass(frozen=True)
class RewardProgram:
    instruction: str
    terms: tuple[Term, ...]
    success: tuple[SuccessClause, ...]
    weights: RewardWeights = field(default_factory=RewardWeights)
    success_mode: str = "all"  # "all" | "any"
    target_object: str | None = None

    # ── evaluation ───────────────────────────────────────────────────────────
    def progress(self, view: SimView, base: Baseline) -> float:
        total = sum(t.weight for t in self.terms)
        if total <= 0:
            return 0.0
        return _clamp01(sum(t.weight * t.value(view, base) for t in self.terms) / total)

    def is_success(self, view: SimView, base: Baseline) -> bool:
        if not self.success:
            return False
        checks = (c.satisfied(view, base) for c in self.success)
        return any(checks) if self.success_mode == "any" else all(checks)

    def evaluate(self, view: SimView, base: Baseline) -> RewardComputation:
        progress = self.progress(view, base)
        success = self.is_success(view, base)
        wp, ws = self.weights.progress, self.weights.success
        denom = wp + ws
        if denom <= 0:
            raise ValueError("reward weights must sum to a positive value")
        score = _clamp01((wp * progress + ws * (1.0 if success else 0.0)) / denom)

        subscores: list[dict[str, float | str]] = [
            {"name": t.name, "weight": round(t.weight, 4), "value": round(t.value(view, base), 4)}
            for t in self.terms
        ]
        subscores.append({"name": "success", "weight": round(ws, 4), "value": 1.0 if success else 0.0})

        return RewardComputation(
            score=round(score, 4),
            progress=round(progress, 4),
            success=success,
            progress_name="progress",
            details={
                "subscores": subscores,
                "terms": {t.name: round(t.value(view, base), 4) for t in self.terms},
                "success_clauses": [
                    {**c.as_dict(), "value": round(c.value(view, base), 4),
                     "ok": c.satisfied(view, base)}
                    for c in self.success
                ],
            },
        )

    def referenced_objects(self) -> list[str]:
        """Body names the program reads - what scene-gen must ensure exist (objects
        not in the photo get added by Gizmo). Order-preserving, de-duplicated."""
        out: list[str] = []
        for src in (*self.terms, *self.success):
            for key in OBJECT_ARG_KEYS:
                v = src.args.get(key)
                if isinstance(v, str) and v and v not in out:
                    out.append(v)
        return out

    # ── serialization ────────────────────────────────────────────────────────
    def as_dict(self) -> dict[str, Any]:
        return {
            "instruction": self.instruction,
            "target_object": self.target_object,
            "terms": [t.as_dict() for t in self.terms],
            "success": [c.as_dict() for c in self.success],
            "success_mode": self.success_mode,
            "weights": self.weights.as_dict(),
        }

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), sort_keys=True)

    @classmethod
    def from_mapping(cls, d: dict[str, Any]) -> "RewardProgram":
        terms = tuple(Term.from_mapping(t) for t in (d.get("terms") or []))
        if not terms:
            raise ValueError("reward program needs at least one term")
        return cls(
            instruction=str(d.get("instruction", "")),
            terms=terms,
            success=tuple(SuccessClause.from_mapping(c) for c in (d.get("success") or [])),
            weights=RewardWeights.from_mapping(d.get("weights")),
            success_mode=str(d.get("success_mode", "all")),
            target_object=d.get("target_object"),
        )

    @classmethod
    def parse(cls, data: "str | dict[str, Any] | RewardProgram") -> "RewardProgram":
        if isinstance(data, cls):
            return data
        if isinstance(data, str):
            return cls.from_mapping(json.loads(data))
        return cls.from_mapping(data)

    @staticmethod
    def looks_like_program(data: Any) -> bool:
        """True if `data` is a program payload (vs. a legacy RewardSpec)."""
        if isinstance(data, RewardProgram):
            return True
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except (json.JSONDecodeError, TypeError):
                return False
        return isinstance(data, dict) and "terms" in data

    # ── legacy bridge: compile an archetype RewardSpec into a program ──────────
    @classmethod
    def from_spec(cls, spec: RewardSpec) -> "RewardProgram":
        """Make the fixed archetypes flow through the same interpreter, so old
        callers (and stored pick specs) keep working with one evaluation path."""
        obj = spec.target_object or "object"
        if spec.archetype == "pick":
            target_z = float(spec.params.get("lift_height", 0.55))
            return cls(
                instruction=spec.instruction,
                terms=(Term("lift_progress", 1.0, "lift_to", {"object": obj, "target_z": target_z}),),
                success=(SuccessClause("above_height", {"object": obj, "z": target_z}, 0.999),),
                weights=spec.weights,
                target_object=obj,
            )
        if spec.archetype == "grasp":
            return cls(
                instruction=spec.instruction,
                terms=(Term("grasp", 1.0, "grasped", {"object": obj}),),
                success=(SuccessClause("grasped", {"object": obj}, 0.999),),
                weights=spec.weights,
                target_object=obj,
            )
        if spec.archetype == "open":
            joint = spec.target_joint or obj
            frac = float(spec.params.get("open_frac", 0.8))
            return cls(
                instruction=spec.instruction,
                terms=(Term("open", 1.0, "joint_open", {"joint": joint, "open_frac": frac}),),
                success=(SuccessClause("joint_open", {"joint": joint, "open_frac": frac}, 0.999),),
                weights=spec.weights,
                target_object=obj,
            )
        if spec.archetype in ("place", "move"):
            goal = spec.params.get("goal_xyz") or spec.params.get("goal")
            tol = float(spec.params.get("tolerance", 0.05))
            if goal is not None:
                pt = [float(v) for v in goal]
                radius = max(tol, 0.05)
                return cls(
                    instruction=spec.instruction,
                    terms=(
                        Term("approach", 0.5, "reached", {"object": obj, "radius": 0.3}),
                        Term("placed", 0.5, "reached", {"point": pt, "radius": radius}),
                    ),
                    success=(SuccessClause("reached", {"point": pt, "radius": radius}, 0.5),),
                    weights=spec.weights,
                    target_object=obj,
                )
        raise ValueError(f"cannot compile archetype {spec.archetype!r} to a program")


def to_program(data: "str | dict[str, Any] | RewardProgram | RewardSpec") -> RewardProgram:
    """Coerce any reward payload into an executable program.

    Accepts a program (dict/json/obj), a legacy RewardSpec, or a legacy spec
    dict/json - so the bridge has a single entry point regardless of caller."""
    if isinstance(data, RewardProgram):
        return data
    if isinstance(data, RewardSpec):
        return RewardProgram.from_spec(data)
    if RewardProgram.looks_like_program(data):
        return RewardProgram.parse(data)  # type: ignore[arg-type]
    return RewardProgram.from_spec(RewardSpec.parse(data))  # type: ignore[arg-type]


__all__ = ["RewardProgram", "SuccessClause", "Term", "to_program"]
