"""Tests for the dynamic reward system (predicates + program + compiler).

No MuJoCo: a FakeSimView stands in for the live sim, so the whole reward path is
exercised deterministically. Run with `python -m pytest tests/test_reward_program.py`
or directly: `python tests/test_reward_program.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rewards.engine import compute_pick_reward  # noqa: E402
from rewards.predicates import Baseline  # noqa: E402
from rewards.program import RewardProgram, to_program  # noqa: E402
from rewards.spec import RewardSpec  # noqa: E402
from backend.reward_compiler import compile_reward_heuristic  # noqa: E402


class FakeSimView:
    """Hand-set sim state for grading without MuJoCo."""

    def __init__(self, positions, *, eef=None, opening=1.0, contacts=(), joints=None, surfaces=None):
        self._pos = dict(positions)
        self._eef = eef
        self._opening = opening
        self._contacts = {frozenset(c) for c in contacts}
        self._joints = joints or {}
        self._surfaces = surfaces or {}

    def object_pos(self, name):
        return self._pos.get(name)

    def eef_pos(self):
        return self._eef

    def gripper_opening(self):
        return self._opening

    def in_contact(self, a, b):
        return frozenset((a, b)) in self._contacts

    def surface_z(self, name):
        return self._surfaces.get(name)

    def joint_value(self, name):
        return self._joints.get(name)


# ── the flagship task: pick up the shirt, place it to the LEFT of the can ──────

FLAGSHIP = "pick up the shirt and place it to the left of the can"


def _flagship_program():
    compiled = compile_reward_heuristic(FLAGSHIP, ["shirt", "can"])
    assert set(compiled.required_objects) == {"shirt", "can"}, compiled.required_objects
    return compiled.program


def _flagship_baseline():
    # shirt rests at x=0.4 (to the RIGHT of the can at x=0.2), on the desk at z=0.80.
    return Baseline(initial_pos={"shirt": (0.40, 0.10, 0.80), "can": (0.20, 0.10, 0.80)})


def test_flagship_success_when_shirt_left_of_can_and_resting():
    prog = _flagship_program()
    base = _flagship_baseline()
    # End state: shirt moved to the LEFT of the can (smaller x), back on the desk.
    view = FakeSimView({"shirt": (0.05, 0.10, 0.80), "can": (0.20, 0.10, 0.80)})
    rc = prog.evaluate(view, base)
    assert rc.success is True, rc.details
    assert rc.score > 0.7, rc.score


def test_flagship_fail_when_shirt_on_wrong_side():
    prog = _flagship_program()
    base = _flagship_baseline()
    # Shirt ends to the RIGHT of the can (wrong side) -> not success.
    view = FakeSimView({"shirt": (0.45, 0.10, 0.80), "can": (0.20, 0.10, 0.80)})
    rc = prog.evaluate(view, base)
    assert rc.success is False
    assert rc.score < 0.6, rc.score


def test_flagship_fail_when_left_but_floating():
    prog = _flagship_program()
    base = _flagship_baseline()
    # Correct side but still 30cm above the desk (not placed) -> on_surface fails.
    view = FakeSimView({"shirt": (0.05, 0.10, 1.10), "can": (0.20, 0.10, 0.80)})
    rc = prog.evaluate(view, base)
    assert rc.success is False, rc.details


def test_flagship_partial_progress_while_lifting():
    prog = _flagship_program()
    base = _flagship_baseline()
    # Grasped + lifted a bit, not yet moved left -> some progress, no success.
    view = FakeSimView(
        {"shirt": (0.40, 0.10, 0.86), "can": (0.20, 0.10, 0.80)},
        opening=0.3, contacts=[("shirt", "gripper")],
    )
    rc = prog.evaluate(view, base)
    assert rc.success is False
    assert 0.0 < rc.score < 0.6, rc.score


# ── direction sign mapping ─────────────────────────────────────────────────────


def test_right_of_uses_opposite_sign():
    prog = compile_reward_heuristic("place the shirt to the right of the can", ["shirt", "can"]).program
    base = Baseline(initial_pos={"shirt": (0.0, 0.0, 0.8), "can": (0.2, 0.0, 0.8)})
    # shirt to the RIGHT (larger x) of the can -> success
    view = FakeSimView({"shirt": (0.4, 0.0, 0.8), "can": (0.2, 0.0, 0.8)})
    assert prog.evaluate(view, base).success is True


# ── legacy archetype still works, via the same interpreter ─────────────────────


def test_legacy_pick_matches_compute_pick_reward():
    spec = RewardSpec.pick(instruction="pick up the block", target_object="block", lift_height=0.55)
    prog = to_program(spec)
    base = Baseline(initial_pos={"block": (0.0, 0.0, 0.40)})

    for final_z in (0.40, 0.475, 0.55, 0.60):
        view = FakeSimView({"block": (0.0, 0.0, final_z)})
        rc = prog.evaluate(view, base)
        ref = compute_pick_reward(spec, initial_z=0.40, final_z=final_z)
        assert abs(rc.score - ref.score) < 1e-6, (final_z, rc.score, ref.score)
        assert abs(rc.progress - ref.progress) < 1e-6, (final_z, rc.progress, ref.progress)
        assert rc.success == ref.success, (final_z, rc.success, ref.success)


# ── serialization round-trips (programs are persisted in episode.json) ─────────


def test_program_json_roundtrip():
    prog = _flagship_program()
    again = RewardProgram.parse(prog.to_json())
    assert again.as_dict() == prog.as_dict()


def test_unknown_predicate_rejected():
    bad = {"instruction": "x", "terms": [{"name": "t", "weight": 1.0, "fn": "teleport", "args": {}}]}
    try:
        RewardProgram.from_mapping(bad)
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown predicate fn")


# ── heuristic covers the other skills too ──────────────────────────────────────


def test_heuristic_pick_only():
    prog = compile_reward_heuristic("pick up the mug", ["mug"]).program
    assert prog.target_object == "mug"
    assert {t.fn for t in prog.terms} == {"grasped", "lifted"}


def test_heuristic_object_not_in_scene_is_still_required():
    # "water the plant" with no watering can in the photo: program references the
    # plant; scene-gen must ensure it exists.
    compiled = compile_reward_heuristic("move the stapler near the lamp", [])
    assert "stapler" in compiled.required_objects or "lamp" in compiled.required_objects


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  ok   {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  FAIL {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return failed


if __name__ == "__main__":
    raise SystemExit(1 if _run_all() else 0)
