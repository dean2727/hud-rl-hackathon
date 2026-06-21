"""Tests for the live training-loop SSE streaming (backend/train_modal.py).

Exercises the dry-run path (no Modal, no GPU) and asserts the event sequence the
frontend reward chart consumes. Run: `python tests/test_train_streaming.py` or
`python -m pytest tests/test_train_streaming.py`.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import backend.train_modal as tm  # noqa: E402
from backend.runs import ActivityState, RunState  # noqa: E402


def _run() -> RunState:
    run = RunState(run_id="testrun", activities=["pick up the shirt"], image_paths=[])
    run.scene_id = "scene-x"
    run.activity_states = [ActivityState(activity_index=0, activity="pick up the shirt")]
    return run


def _events(run: RunState, name: str) -> list[dict]:
    return [e["data"] for e in run.history if e["event"] == name]


def test_demo_replay_emits_expected_sequence(monkeypatch):
    # Force the replay path with two deterministic rounds (no sleeps, no disk).
    monkeypatch.setattr(tm, "_recorded_rounds", lambda cfg: [[0.1, 0.3, 0.7], [0.5, 0.8, 0.9]])
    run = _run()
    asyncio.run(tm.run_modal_training(run, 0, dry_run=True))

    names = [e["event"] for e in run.history]
    assert "train_stage" in names and "eval_rollout" in names
    assert "eval_summary" in names and "curate" in names
    assert names[-1] == "train_modal_done", names[-1]

    rollouts = _events(run, "eval_rollout")
    assert len(rollouts) == 6  # 3 + 3
    assert {r["round"] for r in rollouts} == {0, 1}
    # every rollout event carries what the chart plots
    for r in rollouts:
        assert {"activity_index", "round", "index", "reward", "success"} <= r.keys()

    summaries = _events(run, "eval_summary")
    assert len(summaries) == 2
    # mean of round 0 = (0.1+0.3+0.7)/3
    assert abs(summaries[0]["mean_reward"] - 0.3667) < 1e-3

    curates = _events(run, "curate")
    # default threshold 0.25: round 0 selects 0.3 & 0.7 -> 2 of 3
    assert curates[0]["selected"] == 2 and curates[0]["available"] == 3


def test_demo_synthetic_curve_when_no_recordings(monkeypatch):
    monkeypatch.setattr(tm, "_recorded_rounds", lambda cfg: [])
    monkeypatch.setattr(tm, "_DEMO_ROUNDS", 3)
    # don't actually sleep through the animation delays
    async def _no_sleep(*_a, **_k):
        return None
    monkeypatch.setattr(tm.asyncio, "sleep", _no_sleep)

    run = _run()
    asyncio.run(tm.run_modal_training(run, 0, dry_run=True))

    summaries = _events(run, "eval_summary")
    assert len(summaries) == 3
    # synthetic curve should trend upward across rounds
    means = [s["mean_reward"] for s in summaries]
    assert means[-1] > means[0], means
    assert _events(run, "train_modal_done")


def test_unknown_activity_emits_error():
    run = _run()
    asyncio.run(tm.run_modal_training(run, 99, dry_run=True))
    assert _events(run, "train_modal_error")
    assert not _events(run, "train_modal_done")


def test_demo_reward_is_deterministic():
    assert tm._demo_reward(0, 0) == tm._demo_reward(0, 0)
    assert 0.0 <= tm._demo_reward(2, 5) <= 1.0


def _run_all() -> int:
    # Minimal monkeypatch shim so this runs without pytest installed.
    class MP:
        def __init__(self):
            self._undo = []

        def setattr(self, obj, name, val):
            old = getattr(obj, name)
            self._undo.append((obj, name, old))
            setattr(obj, name, val)

        def undo(self):
            for obj, name, old in reversed(self._undo):
                setattr(obj, name, old)

    tests = [
        ("test_demo_replay_emits_expected_sequence", True),
        ("test_demo_synthetic_curve_when_no_recordings", True),
        ("test_unknown_activity_emits_error", False),
        ("test_demo_reward_is_deterministic", False),
    ]
    failed = 0
    for fn_name, needs_mp in tests:
        fn = globals()[fn_name]
        mp = MP()
        try:
            fn(mp) if needs_mp else fn()
            print(f"  ok   {fn_name}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  FAIL {fn_name}: {type(exc).__name__}: {exc}")
        finally:
            mp.undo()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return failed


if __name__ == "__main__":
    raise SystemExit(1 if _run_all() else 0)
