"""Vetted reward predicates over an abstract sim view.

These are the *safe vocabulary* an LLM-authored reward program composes - never
arbitrary code. Each predicate is a pure function `(SimView, Baseline, args)`
returning a float in [0, 1]:

  - read continuously, it shapes progress (how close are we?),
  - read against a threshold, it gates success (did it happen?).

All sim access goes through the `SimView` protocol, so this module is free of
MuJoCo and unit-testable with a fake view. Missing/unreadable state yields 0.0
(we never reward what we can't measure - a conservative bias that protects the
curation filter from false positives).

Frame convention for directional predicates (`relative_side`): axes are world
axes; the reward *compiler* picks (axis, sign) per task. For the default Franka
tabletop (agentview behind the arm at -y looking toward +y, up +z) the viewer's
LEFT is -x and RIGHT is +x; CLOSER is -y and FARTHER is +y. The compiler encodes
those, so "left of the can" is never wrong-by-construction here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

Vec3 = tuple[float, float, float]
_AXIS = {"x": 0, "y": 1, "z": 2}


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


@runtime_checkable
class SimView(Protocol):
    """Read-only window onto the live sim. Implemented by sim/reward_view.py for
    the real Newton/MuJoCo sim and by a fake in tests."""

    def object_pos(self, name: str) -> Vec3 | None: ...
    def eef_pos(self) -> Vec3 | None: ...
    def gripper_opening(self) -> float: ...  # 0 = fully closed, 1 = fully open
    def in_contact(self, a: str, b: str) -> bool: ...  # b may be the literal "gripper"
    def surface_z(self, name: str | None) -> float | None: ...
    def joint_value(self, name: str) -> tuple[float, float, float] | None: ...  # (value, lo, hi)


@dataclass(frozen=True)
class Baseline:
    """Snapshot taken at reset: where things rested before the policy acted.

    Predicates reference this for "lifted off its start" / "back on the surface"
    so they don't depend on hard-coded world heights (which break when Gizmo
    regenerates the desk at a different z)."""

    initial_pos: dict[str, Vec3] = field(default_factory=dict)
    surface_z: float | None = None

    def initial_z(self, name: str) -> float | None:
        p = self.initial_pos.get(name)
        return None if p is None else float(p[2])


def _dist(a: Vec3, b: Vec3, plane: str) -> float:
    if plane == "xy":
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2) ** 0.5


# ── predicates (each: (view, baseline, args) -> float in [0,1]) ──────────────


def lifted(view: SimView, base: Baseline, args: dict[str, Any]) -> float:
    """Fraction of a target *relative* lift achieved, vs. the object's rest height.

    args: object, height (m, default 0.15), baseline ("initial"|"surface"|float)."""
    name = args["object"]
    pos = view.object_pos(name)
    if pos is None:
        return 0.0
    height = float(args.get("height", 0.15))
    ref = args.get("baseline", "initial")
    if isinstance(ref, (int, float)):
        z0: float | None = float(ref)
    elif ref == "surface":
        z0 = base.surface_z if base.surface_z is not None else base.initial_z(name)
    else:
        z0 = base.initial_z(name)
    if z0 is None or height <= 1e-6:
        return 0.0
    return _clamp01((pos[2] - z0) / height)


def lift_to(view: SimView, base: Baseline, args: dict[str, Any]) -> float:
    """Progress toward an *absolute* target height (matches the legacy pick reward).

    args: object, target_z (m)."""
    name = args["object"]
    pos = view.object_pos(name)
    if pos is None:
        return 0.0
    target_z = float(args["target_z"])
    z0 = base.initial_z(name)
    if z0 is None:
        return 0.0
    span = target_z - z0
    if span <= 1e-3:
        return 1.0
    return _clamp01((pos[2] - z0) / span)


def above_height(view: SimView, base: Baseline, args: dict[str, Any]) -> float:
    """Boolean (as 0/1): is the object at or above an absolute z?  args: object, z."""
    pos = view.object_pos(args["object"])
    if pos is None:
        return 0.0
    return 1.0 if pos[2] >= float(args["z"]) else 0.0


def grasped(view: SimView, base: Baseline, args: dict[str, Any]) -> float:
    """Boolean (as 0/1): gripper is touching the object and not fully open.

    args: object, open_max (default 0.98)."""
    name = args["object"]
    if not view.in_contact(name, "gripper"):
        return 0.0
    return 1.0 if view.gripper_opening() <= float(args.get("open_max", 0.98)) else 0.0


def relative_side(view: SimView, base: Baseline, args: dict[str, Any]) -> float:
    """How far the object sits on the desired side of a reference, along one axis.

    0 when on the wrong side (or level); ramps to 1 at `full_margin` past it.
    args: object, reference, axis ("x"|"y"|"z"), sign ("positive"|"negative"),
          full_margin (m, default 0.15)."""
    op = view.object_pos(args["object"])
    rp = view.object_pos(args["reference"])
    if op is None or rp is None:
        return 0.0
    ax = _AXIS[str(args.get("axis", "x"))]
    m = -1.0 if str(args.get("sign", "positive")) in ("negative", "less", "minus") else 1.0
    margin = float(args.get("full_margin", 0.15))
    if margin <= 1e-6:
        return 0.0
    return _clamp01(m * (op[ax] - rp[ax]) / margin)


def near(view: SimView, base: Baseline, args: dict[str, Any]) -> float:
    """Proximity of two objects: 1 when coincident, 0 at/beyond `radius`.

    args: object, reference, radius (m, default 0.2), plane ("xy"|"xyz")."""
    op = view.object_pos(args["object"])
    rp = view.object_pos(args["reference"])
    if op is None or rp is None:
        return 0.0
    radius = float(args.get("radius", 0.2))
    if radius <= 1e-6:
        return 0.0
    return _clamp01(1.0 - _dist(op, rp, str(args.get("plane", "xy"))) / radius)


def reached(view: SimView, base: Baseline, args: dict[str, Any]) -> float:
    """Proximity of the end-effector to an object (or a fixed point).

    args: object (name) OR point ([x,y,z]); radius (m, default 0.1), plane."""
    eef = view.eef_pos()
    if eef is None:
        return 0.0
    if "point" in args:
        tgt: Vec3 | None = tuple(float(v) for v in args["point"])  # type: ignore[assignment]
    else:
        tgt = view.object_pos(args["object"])
    if tgt is None:
        return 0.0
    radius = float(args.get("radius", 0.1))
    if radius <= 1e-6:
        return 0.0
    return _clamp01(1.0 - _dist(eef, tgt, str(args.get("plane", "xyz"))) / radius)


def on_surface(view: SimView, base: Baseline, args: dict[str, Any]) -> float:
    """1 when the object rests near its support height, falling off with distance.

    Uses a named surface's z when available, else the object's own rest height from
    the baseline - so "placed back down on the desk" reads correctly without a
    hard-coded table z.  args: object, surface (name, optional), tol (m, default 0.04)."""
    name = args["object"]
    pos = view.object_pos(name)
    if pos is None:
        return 0.0
    surf = view.surface_z(args.get("surface"))
    target_z = surf if surf is not None else base.initial_z(name)
    if target_z is None:
        return 0.0
    tol = float(args.get("tol", 0.04))
    if tol <= 1e-6:
        return 1.0 if abs(pos[2] - target_z) < 1e-6 else 0.0
    return _clamp01(1.0 - abs(pos[2] - target_z) / tol)


def joint_open(view: SimView, base: Baseline, args: dict[str, Any]) -> float:
    """Fraction of a joint's range opened, normalized by a target fraction.

    args: joint, open_frac (target fraction of range, default 0.8)."""
    jv = view.joint_value(args["joint"])
    if jv is None:
        return 0.0
    value, lo, hi = jv
    span = hi - lo
    if abs(span) <= 1e-6:
        return 0.0
    frac = (value - lo) / span
    target = float(args.get("open_frac", 0.8))
    if target <= 1e-6:
        return 0.0
    return _clamp01(frac / target)


REGISTRY: dict[str, Any] = {
    "lifted": lifted,
    "lift_to": lift_to,
    "above_height": above_height,
    "grasped": grasped,
    "relative_side": relative_side,
    "near": near,
    "reached": reached,
    "on_surface": on_surface,
    "joint_open": joint_open,
}

# args whose string values name a scene body (used to collect referenced objects).
OBJECT_ARG_KEYS = ("object", "reference", "surface")


def evaluate(fn: str, view: SimView, base: Baseline, args: dict[str, Any]) -> float:
    impl = REGISTRY.get(fn)
    if impl is None:
        raise KeyError(f"unknown reward predicate {fn!r} (known: {sorted(REGISTRY)})")
    return _clamp01(float(impl(view, base, args)))


__all__ = [
    "Baseline",
    "OBJECT_ARG_KEYS",
    "REGISTRY",
    "SimView",
    "Vec3",
    "evaluate",
]
