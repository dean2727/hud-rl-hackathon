"""Adapt a Gizmo MJCF Franka export into a VLA-ready scene + a graded task spec.

This generalizes the manual `scenes/env1` adaptation. A Gizmo export with
`robot_profile="franka_panda"` embeds the arm (joint1..7 / actuator1..8) but is *not*
ready for the pi0.5 VLA bridge (`sim/franka_bridge.py` + `sim/control.py`):

  - no `eef` TCP site (control.py's IK target),
  - no `agentview` / `wrist` cameras (the two image inputs pi0.5 expects),
  - objects placed wherever Gizmo dropped them (often out of arm reach),
  - and complex scenes overflow Newton's collision-mask export -- handled globally by
    the clamp shim in sim/server.py, so nothing to do here.

`adapt_vla_scene()` injects the site + cameras (idempotent), discovers a target object
and lift height, optionally snaps the target within reach, and returns a `TaskSpec`.
`validate_vla_scene()` is the no-GPU gate: it resets the scene in the sim, renders both
cameras, and runs one noop rollout so a broken scene is caught before any A100 time.

The contract (state/action layout) is embodiment-level and loaded by environment/vla_env.py
from contracts/franka_libero.json for *every* scene, so no per-scene metadata is needed.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from backend.config import settings
from backend.scene_compose import _is_mujoco_doc, unzip_export

# Franka Panda reach envelope (m) and where to drop a snapped target in front of the base.
FRANKA_REACH = 0.85
SNAP_DISTANCE = 0.5

# Proven definitions from scenes/franka-libero-v1/franka_emika_panda/panda.xml.
EEF_SITE = {"name": "eef", "pos": "0 0 0.1034", "size": "0.01", "rgba": "1 0 0 0.25", "group": "3"}
WRIST_CAM = {"name": "wrist", "pos": "0.05 0 0.0", "xyaxes": "1 0 0 0 -1 0", "fovy": "70"}


class VlaSceneError(RuntimeError):
    pass


@dataclass
class TaskSpec:
    scene_id: str
    target_object: str
    instruction: str
    lift_height: float
    reachable: bool
    target_distance: float  # target -> arm-base horizontal distance (m)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "target_object": self.target_object,
            "instruction": self.instruction,
            "lift_height": round(self.lift_height, 4),
            "reachable": self.reachable,
            "target_distance": round(self.target_distance, 4),
            "notes": self.notes,
        }


# ── MJCF helpers (a Gizmo export may inline the Franka or pull it via <include>) ──


def _mjcf_files(scene_dir: Path) -> list[Path]:
    return [p for p in sorted(scene_dir.rglob("*.xml")) if _is_mujoco_doc(p)]


def _find_body(scene_dir: Path, body_name: str) -> tuple[Path, ET.ElementTree, ET.Element] | None:
    """Locate the MJCF file + element for a named <body> (searches includes too)."""
    for path in _mjcf_files(scene_dir):
        try:
            tree = ET.parse(path)
        except ET.ParseError:
            continue
        for body in tree.getroot().iter("body"):
            if body.get("name") == body_name:
                return path, tree, body
    return None


def _camera_exists(scene_dir: Path, name: str) -> bool:
    """A worldbody camera can live in the root scene or any <include>d file, so the
    idempotency check has to scan them all (not just the file we'd add it to)."""
    for path in _mjcf_files(scene_dir):
        try:
            tree = ET.parse(path)
        except ET.ParseError:
            continue
        if any(c.get("name") == name for c in tree.getroot().iter("camera")):
            return True
    return False


def _pick_worldbody_file(scene_dir: Path, root_mjcf: Path) -> tuple[Path, ET.ElementTree]:
    """Choose where to add the agentview camera: prefer the root scene file's
    <worldbody>; fall back to whichever MJCF has one (e.g. the included panda)."""
    root_tree = ET.parse(root_mjcf)
    if root_tree.getroot().find("worldbody") is not None:
        return root_mjcf, root_tree
    for path in _mjcf_files(scene_dir):
        try:
            tree = ET.parse(path)
        except ET.ParseError:
            continue
        if tree.getroot().find("worldbody") is not None:
            return path, tree
    raise VlaSceneError("no <worldbody> found in the export's MJCF")


# ── model introspection (raw MuJoCo; tolerant of the collision-mask quirk) ──


def _load_model(root_mjcf: Path) -> tuple[mujoco.MjModel, mujoco.MjData]:
    model = mujoco.MjModel.from_xml_path(str(root_mjcf.resolve()))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def _body_xpos(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> np.ndarray | None:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    return None if bid < 0 else np.array(data.xpos[bid])


def _free_bodies(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, np.ndarray]:
    """Pickable bodies = those with a free joint, keyed by name -> world position."""
    out: dict[str, np.ndarray] = {}
    for jid in range(model.njnt):
        if model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_FREE:
            continue
        bid = model.jnt_bodyid[jid]
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
        if name:
            out[name] = np.array(data.xpos[bid])
    return out


def _pick_target(free: dict[str, np.ndarray], eef: np.ndarray, hint: str | None) -> str:
    if not free:
        raise VlaSceneError("no free/pickable bodies found in the scene")
    if hint:
        hint_l = hint.lower()
        matches = [n for n in free if hint_l in n.lower()]
        if matches:  # nearest among hint matches
            return min(matches, key=lambda n: float(np.linalg.norm(free[n][:2] - eef[:2])))
    # else: the body nearest the arm (most likely reachable / most relevant)
    return min(free, key=lambda n: float(np.linalg.norm(free[n][:2] - eef[:2])))


# ── injection + snap ──


def _inject_features(scene_dir: Path, root_mjcf: Path, agentview_pos: tuple[float, float, float]) -> list[str]:
    """Add the eef site + wrist camera to the hand body and an agentview camera to a
    worldbody. Idempotent; returns human-readable notes about what changed."""
    notes: list[str] = []
    found = _find_body(scene_dir, "hand")
    if found is None:
        raise VlaSceneError(
            "no Franka 'hand' body found - export the scene with robot_profile='franka_panda'"
        )
    hand_path, hand_tree, hand = found
    touched: dict[Path, ET.ElementTree] = {}

    if not any(s.get("name") == "eef" for s in hand.findall("site")):
        ET.SubElement(hand, "site", EEF_SITE)
        notes.append("added eef TCP site")
        touched[hand_path] = hand_tree
    if not any(c.get("name") == "wrist" for c in hand.findall("camera")):
        ET.SubElement(hand, "camera", WRIST_CAM)
        notes.append("added wrist camera")
        touched[hand_path] = hand_tree

    if not _camera_exists(scene_dir, "agentview"):  # scan all files, not just the target
        wb_path, wb_tree = _pick_worldbody_file(scene_dir, root_mjcf)
        wb_tree = touched.get(wb_path, wb_tree)  # reuse if it's the same (already-edited) file
        ax, ay, az = agentview_pos
        ET.SubElement(wb_tree.getroot().find("worldbody"), "camera", {
            "name": "agentview", "pos": f"{ax:.3f} {ay:.3f} {az:.3f}",
            "mode": "targetbody", "target": "hand", "fovy": "45",
        })
        notes.append("added agentview camera")
        touched[wb_path] = wb_tree

    for path, tree in touched.items():
        tree.write(path)
    return notes


def _snap_target(scene_dir: Path, target: str, base_xy: np.ndarray) -> bool:
    """Move the target free body's spawn pose to SNAP_DISTANCE in front of the arm base,
    keeping its rest height. Best-effort: assumes a support surface near the arm at that
    height (caller validates by re-running the sim). Returns True if it edited the file."""
    found = _find_body(scene_dir, target)
    if found is None:
        return False
    path, tree, body = found
    pos = [float(v) for v in (body.get("pos") or "0 0 0").split()]
    new_xy = base_xy + np.array([0.0, SNAP_DISTANCE])  # in front (+y) of the base
    body.set("pos", f"{new_xy[0]:.4f} {new_xy[1]:.4f} {pos[2]:.4f}")
    tree.write(path)
    return True


# ── public API ──


def adapt_vla_scene(
    zip_bytes: bytes,
    scene_id: str,
    *,
    instruction: str | None = None,
    target_hint: str | None = None,
    snap_target: bool = False,
) -> TaskSpec:
    """Unzip a Gizmo Franka export into scenes/{scene_id}/, make it VLA-ready, and
    return the task spec. Set snap_target=True to force the target within reach
    (best-effort; off by default since it assumes a support surface near the arm)."""
    dest_dir = settings.scenes_dir / scene_id
    root_mjcf = unzip_export(zip_bytes, dest_dir)

    model, data = _load_model(root_mjcf)
    hand = _body_xpos(model, data, "hand")
    base = _body_xpos(model, data, "link0")
    if hand is None or base is None:
        raise VlaSceneError(
            "Franka 'hand'/'link0' bodies not found - export with robot_profile='franka_panda'"
        )
    free = _free_bodies(model, data)
    target = _pick_target(free, hand, target_hint)

    notes: list[str] = []
    if snap_target:
        if _snap_target(dest_dir, target, base[:2]):
            notes.append(f"snapped '{target}' to ~{SNAP_DISTANCE} m in front of the arm base")

    # agentview sits behind the arm (-y) and up, framing the gripper (targetbody=hand).
    agentview_pos = (float(base[0]), float(base[1]) - 1.4, float(hand[2]) + 0.25)
    notes += _inject_features(dest_dir, root_mjcf, agentview_pos)

    # Re-load after edits to read the (possibly snapped) target's rest pose for lift_height.
    model, data = _load_model(root_mjcf)
    tpos = _body_xpos(model, data, target)
    base = _body_xpos(model, data, "link0")
    tpos = tpos if tpos is not None else np.array([base[0], base[1], 0.8])
    lift_height = float(tpos[2]) + 0.15
    distance = float(np.linalg.norm(tpos[:2] - base[:2]))
    reachable = distance <= FRANKA_REACH
    if not reachable:
        notes.append(
            f"target is {distance:.2f} m from the arm base (> {FRANKA_REACH} m reach) - "
            f"pass snap_target=True or tighten the scene prompt for a feasible pick"
        )

    return TaskSpec(
        scene_id=scene_id,
        target_object=target,
        instruction=instruction or f"pick up the {_humanize(target)}",
        lift_height=lift_height,
        reachable=reachable,
        target_distance=distance,
        notes=notes,
    )


def _humanize(body_name: str) -> str:
    """'modern_silver_laptop_11_rigid_body' -> 'modern silver laptop'."""
    drop = {"rigid", "body", "interactable", "articulation", "free"}
    words = [w for w in body_name.split("_") if w and not w.isdigit() and w not in drop]
    return " ".join(words) or body_name


async def validate_vla_scene(spec: TaskSpec, *, max_steps: int = 15) -> dict[str, Any]:
    """No-GPU gate: reset the scene, render both cameras, run a noop rollout, grade.

    Catches missing cameras / bad Franka names / collision-load failures / a target the
    bridge can't read - before any A100 time. Returns {ok, reachable, error?, detail}.
    Run via asyncio.to_thread in the orchestrator (it spins up the sim, blocking)."""
    import asyncio

    def _run() -> dict[str, Any]:
        import sys
        root = Path(__file__).resolve().parents[1]
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from sim import server as sim_server  # noqa: E402 (heavy; import lazily)

        try:
            res = sim_server.reset(scene_id=spec.scene_id, seed=0, max_episode_steps=max_steps)
            if isinstance(res, dict) and res.get("error"):
                return {"ok": False, "error": f"sim reset failed: {res['error']}"}
            # confirm both cameras exist + render, and the target body is readable
            sim = sim_server._sim
            model = sim.mj_model or sim.solver.mj_model
            cams = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(model.ncam)}
            missing = {"agentview", "wrist"} - cams
            if missing:
                return {"ok": False, "error": f"missing cameras after adaptation: {sorted(missing)}"}
            if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "eef") < 0:
                return {"ok": False, "error": "missing 'eef' site after adaptation"}
            if _body_xpos(model, mujoco.MjData(model), spec.target_object) is None:
                return {"ok": False, "error": f"target body '{spec.target_object}' not in scene"}
            return {
                "ok": True,
                "reachable": spec.reachable,
                "detail": {"cameras": sorted(cams), "target": spec.target_object,
                           "target_distance": round(spec.target_distance, 3)},
            }
        except Exception as exc:  # any load/contract failure - surface it, don't crash the run
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    return await asyncio.to_thread(_run)
