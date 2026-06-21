"""Concrete `SimView` over the live Newton/MuJoCo sim for reward predicates.

This is the only place reward evaluation touches MuJoCo. It reads exactly the
state the sim already exposes (the same accessors `sim/server.py` uses):

  - object pose      -> `data.xpos[body]`            (cf. get_object_state)
  - end-effector     -> `data.site_xpos[eef]`        (cf. control.site_pose)
  - gripper opening  -> finger joint qpos / range
  - contacts         -> `data.contact[:ncon]`        (cf. get_contact_forces)
  - joint value      -> `data.qpos[jnt_qposadr]`     (cf. get_joint_state)

The pure predicate/program layer (rewards/) stays MuJoCo-free and unit-testable;
this adapter bridges it to the running sim.
"""

from __future__ import annotations

import mujoco
import numpy as np

from rewards.predicates import Baseline, Vec3
from sim import control as ctl


def _model_data(sim):
    return (sim.mj_model or sim.solver.mj_model), (sim.mj_data or sim.solver.mj_data)


def _body_id(model, name: str) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)


def _geoms_of_body(model, body_id: int) -> set[int]:
    return {g for g in range(model.ngeom) if model.geom_bodyid[g] == body_id}


def _gripper_geoms(model) -> set[int]:
    """Geoms belonging to the Franka hand/fingers (any body named hand/*finger*)."""
    out: set[int] = set()
    for b in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        if name == "hand" or "finger" in name:
            out |= _geoms_of_body(model, b)
    return out


class BridgeSimView:
    """Read-only view bound to the sim singleton for one grading call."""

    def __init__(self, sim) -> None:
        self._sim = sim
        self._model, self._data = _model_data(sim)
        self._gripper_geoms: set[int] | None = None

    # ── positions ────────────────────────────────────────────────────────────
    def object_pos(self, name: str) -> Vec3 | None:
        bid = _body_id(self._model, name)
        if bid < 0:
            return None
        p = self._data.xpos[bid]
        return (float(p[0]), float(p[1]), float(p[2]))

    def eef_pos(self) -> Vec3 | None:
        sid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SITE, ctl.EEF_SITE)
        if sid < 0:
            return None
        p = self._data.site_xpos[sid]
        return (float(p[0]), float(p[1]), float(p[2]))

    # ── gripper ──────────────────────────────────────────────────────────────
    def gripper_opening(self) -> float:
        idx = getattr(self._sim, "robot_idx", None)
        if idx is None or getattr(idx, "finger_qadr", None) is None:
            return 1.0  # unknown -> assume open (grasped() then leans on contact)
        adr = np.asarray(idx.finger_qadr)
        vals = np.abs(self._data.qpos[adr])
        # Normalize by each finger joint's range so 0 = closed, 1 = fully open.
        spans = []
        for j in range(self._model.njnt):
            if self._model.jnt_qposadr[j] in set(int(a) for a in adr):
                lo, hi = self._model.jnt_range[j]
                spans.append(max(abs(hi - lo), 1e-6))
        denom = float(np.mean(spans)) if spans else 0.04
        return float(np.clip(np.mean(vals) / denom, 0.0, 1.0))

    # ── contacts ─────────────────────────────────────────────────────────────
    def in_contact(self, a: str, b: str) -> bool:
        bid_a = _body_id(self._model, a)
        if bid_a < 0:
            return False
        geoms_a = _geoms_of_body(self._model, bid_a)
        if b == "gripper":
            if self._gripper_geoms is None:
                self._gripper_geoms = _gripper_geoms(self._model)
            geoms_b = self._gripper_geoms
        else:
            bid_b = _body_id(self._model, b)
            if bid_b < 0:
                return False
            geoms_b = _geoms_of_body(self._model, bid_b)
        if not geoms_a or not geoms_b:
            return False
        for c in range(self._data.ncon):
            con = self._data.contact[c]
            g1, g2 = int(con.geom1), int(con.geom2)
            if (g1 in geoms_a and g2 in geoms_b) or (g2 in geoms_a and g1 in geoms_b):
                return True
        return False

    # ── surface / joints ─────────────────────────────────────────────────────
    def surface_z(self, name: str | None) -> float | None:
        if not name:
            return None
        bid = _body_id(self._model, name)
        if bid < 0:
            return None
        # Top of the body's tallest geom (best-effort support height).
        geoms = _geoms_of_body(self._model, bid)
        if not geoms:
            return float(self._data.xpos[bid][2])
        tops = []
        for g in geoms:
            cz = float(self._data.geom_xpos[g][2])
            half = float(self._model.geom_size[g][2]) if self._model.geom_size[g][2] > 0 else 0.0
            tops.append(cz + half)
        return max(tops)

    def joint_value(self, name: str) -> tuple[float, float, float] | None:
        jid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            return None
        adr = self._model.jnt_qposadr[jid]
        lo, hi = self._model.jnt_range[jid]
        return (float(self._data.qpos[adr]), float(lo), float(hi))


def snapshot_baseline(view: BridgeSimView, object_names, surface_name: str | None = None) -> Baseline:
    """Capture rest positions (and an optional support height) right after reset."""
    initial: dict[str, Vec3] = {}
    for name in object_names:
        pos = view.object_pos(name)
        if pos is not None:
            initial[name] = pos
    return Baseline(initial_pos=initial, surface_z=view.surface_z(surface_name))


__all__ = ["BridgeSimView", "snapshot_baseline"]
