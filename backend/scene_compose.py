"""Turn a Gizmo MJCF export into a scenes/{scene_id}/ folder sim/server.py can load.

Gizmo generates the static room/objects only (no manipulator, no knowledge of our
joint/actuator naming convention). This module:

  1. Unzips the export and locates its root MJCF document.
  2. Loads it once (pre-gripper) to discover manipulable objects/joints and compute
     a bounding box.
  3. Splices in templates/floating_gripper.xml (sized/positioned from that bbox) so
     the existing 4 graded tasks in environment/env.py work unmodified.
  4. Adds an overhead camera if needed, re-validates the composed model loads, and
     writes metadata.json matching scenes/tabletop-v1's shape.

Note: "gizmo.runtime" imported by sim/server.py is an unrelated *local* package (a
Newton/MJCF-loading helper) - not this external Gizmo API. Naming collision only.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET
import zipfile

import mujoco
import numpy as np

from backend.config import settings

TEMPLATE_PATH = Path(__file__).resolve().parent / "templates" / "floating_gripper.xml"
GRIPPER_BODY_NAMES = {"gripper_base", "finger_left", "finger_right"}


class SceneComposeError(RuntimeError):
    pass


# ── unzip + locate the root MJCF ────────────────────────────────────────────


def _is_mujoco_doc(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            return b"<mujoco" in f.read(4096)
    except OSError:
        return False


def _find_root_mjcf(extract_dir: Path) -> Path:
    """Mirrors sim/server.py::_find_scene_file's preference order, applied to the
    freshly-unzipped export dir."""
    for name in ("scene.xml", "scene.mjcf"):
        candidate = extract_dir / name
        if candidate.exists():
            return candidate
    candidates = sorted(
        (p for p in extract_dir.rglob("*.xml") if p.is_file() and _is_mujoco_doc(p)),
        key=lambda p: len(p.relative_to(extract_dir).parts),
    )
    if not candidates:
        raise SceneComposeError(f"no MJCF document found in Gizmo export under {extract_dir}")
    return candidates[0]


def unzip_export(zip_bytes: bytes, dest_dir: Path) -> Path:
    """Extracts the export into dest_dir and returns the path to its root MJCF file."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        zf.extractall(dest_dir)
    return _find_root_mjcf(dest_dir)


# ── object/joint discovery (pre-gripper) ────────────────────────────────────


def _discover_objects(worldbody: ET.Element) -> dict[str, dict[str, Any]]:
    """Recurses through every <body> in the tree (not just direct worldbody
    children) - articulated parts are commonly nested inside a parent fixture
    body, exactly like tabletop-v1's own "drawer" is nested inside "table"."""
    objects: dict[str, dict[str, Any]] = {}
    for body in worldbody.iter("body"):
        name = body.get("name")
        if not name or name in GRIPPER_BODY_NAMES:
            continue
        freejoint = body.find("freejoint")
        slide_or_hinge = next(
            (j for j in body.findall("joint") if j.get("type") in ("slide", "hinge")), None
        )
        if freejoint is not None:
            pos = [float(v) for v in (body.get("pos") or "0 0 0").split()]
            objects[name] = {"type": "free", "initial_position": pos}
        elif slide_or_hinge is not None:
            joint_range = slide_or_hinge.get("range", "0 1")
            lo, hi = (float(v) for v in joint_range.split())
            objects[name] = {
                "type": "articulated",
                "joint": slide_or_hinge.get("name"),
                "joint_type": slide_or_hinge.get("type"),
                "range": [lo, hi],
            }
    return objects


# ── bounding box (for sizing/placing the gripper) ───────────────────────────


def _scene_bbox(scene_xml_path: Path) -> tuple[np.ndarray, np.ndarray]:
    model = mujoco.MjModel.from_xml_path(str(scene_xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    mins = np.array([np.inf, np.inf, np.inf])
    maxs = np.array([-np.inf, -np.inf, -np.inf])
    for gid in range(model.ngeom):
        if model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        center = data.geom_xpos[gid]
        r = model.geom_rbound[gid]
        mins = np.minimum(mins, center - r)
        maxs = np.maximum(maxs, center + r)

    if not np.all(np.isfinite(mins)):  # no non-plane geoms - fall back to a small default volume
        return np.array([-0.3, -0.3, 0.0]), np.array([0.3, 0.3, 0.8])
    return mins, maxs


def _camera_names(scene_xml_path: Path) -> list[str]:
    model = mujoco.MjModel.from_xml_path(str(scene_xml_path))
    return [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(model.ncam)]


# ── splicing ─────────────────────────────────────────────────────────────


def _get_or_create(root: ET.Element, tag: str) -> ET.Element:
    el = root.find(tag)
    if el is None:
        el = ET.SubElement(root, tag)
    return el


def _prefix_file_paths(el: ET.Element, prefix: str) -> None:
    """Rewrites file="..." attributes so they stay valid if the root MJCF wasn't
    already at the extraction dir's top level (kept relative to its own folder)."""
    if not prefix:
        return
    for node in el.iter():
        f = node.get("file")
        if f and not f.startswith("/"):
            node.set("file", f"{prefix}/{f}")


def compose_scene(zip_bytes: bytes, scene_id: str, description: str = "") -> dict[str, Any]:
    """Writes scenes/{scene_id}/{scene.xml,metadata.json}; returns discovered objects/cameras."""
    dest_dir = settings.scenes_dir / scene_id
    root_mjcf = unzip_export(zip_bytes, dest_dir)

    tree = ET.parse(root_mjcf)
    root = tree.getroot()
    if root.tag != "mujoco":
        raise SceneComposeError(f"expected <mujoco> root in {root_mjcf}, got <{root.tag}>")

    rel_prefix = str(root_mjcf.parent.relative_to(dest_dir)) if root_mjcf.parent != dest_dir else ""
    if rel_prefix:
        _prefix_file_paths(root, rel_prefix)

    worldbody = _get_or_create(root, "worldbody")
    objects = _discover_objects(worldbody)

    scene_xml_path = dest_dir / "scene.xml"
    tree.write(scene_xml_path)
    if root_mjcf != scene_xml_path:
        root_mjcf.unlink(missing_ok=True)

    mins, maxs = _scene_bbox(scene_xml_path)
    _splice_gripper(tree, mins, maxs)

    asset = root.find("asset")
    if asset is not None and not any(c.get("name") == "overhead" for c in root.iter("camera")):
        center_xy = (mins[:2] + maxs[:2]) / 2
        ET.SubElement(
            worldbody,
            "camera",
            {
                "name": "overhead",
                "pos": f"{center_xy[0]:.3f} {center_xy[1] - 0.05:.3f} {maxs[2] + 1.2:.3f}",
                "xyaxes": "1 0 0 0 1 0",
            },
        )

    tree.write(scene_xml_path)

    try:
        cameras = _camera_names(scene_xml_path)
    except Exception as exc:  # mujoco raises plain Exception/ValueError on bad XML
        raise SceneComposeError(f"composed scene failed to load in MuJoCo: {exc}") from exc

    metadata = {
        "scene_id": scene_id,
        "description": description or f"Gizmo-generated scene '{scene_id}'.",
        "source": "gizmo",
        "format": "mjcf",
        "engine": "newton",
        "objects": objects,
        "gripper": {
            "actuators": ["act_x", "act_y", "act_z", "act_yaw", "act_finger_left", "act_finger_right"],
            "actuator_descriptions": {
                "act_x": "Move gripper left (-) / right (+)",
                "act_y": "Move gripper forward (+) / backward (-)",
                "act_z": "Move gripper up (+) / down (-)",
                "act_yaw": "Rotate gripper counter-clockwise (+) / clockwise (-)",
                "act_finger_left": "Left finger: 0 = closed, 0.04 = open",
                "act_finger_right": "Right finger: 0 = closed, 0.04 = open",
            },
        },
        "cameras": cameras,
        "sensors": {
            "contact_forces": "Use get_contact_forces(body_name) to query contact forces on any body",
            "depth_camera": "Use render_depth(camera) to get distance measurements from any camera",
        },
        "tasks": {},
    }
    (dest_dir / "metadata.json").write_text(__import__("json").dumps(metadata, indent=2))
    return {"scene_dir": str(dest_dir), "objects": objects, "cameras": cameras}


def _splice_gripper(tree: ET.ElementTree, mins: np.ndarray, maxs: np.ndarray) -> None:
    root = tree.getroot()
    rig = ET.parse(TEMPLATE_PATH).getroot()

    center_xy = (mins[:2] + maxs[:2]) / 2
    span_xy = (maxs[:2] - mins[:2]) / 2
    zmax, zmin = float(maxs[2]), float(mins[2])
    start_z = zmax + 0.3
    gripper_pos = (center_xy[0], center_xy[1] - (span_xy[1] + 0.25), start_z)
    range_x = span_xy[0] + 0.3
    range_y = span_xy[1] + 0.4
    range_z_low = -(start_z - max(zmin, 0.0) - 0.02)

    rig_body = rig.find("./worldbody/body[@name='gripper_base']")
    rig_body.set("pos", f"{gripper_pos[0]:.3f} {gripper_pos[1]:.3f} {gripper_pos[2]:.3f}")
    for jname, rng in (
        ("gripper_x", (-range_x, range_x)),
        ("gripper_y", (-range_y, range_y)),
        ("gripper_z", (range_z_low, 0.5)),
    ):
        joint = rig_body.find(f"joint[@name='{jname}']")
        joint.set("range", f"{rng[0]:.3f} {rng[1]:.3f}")

    asset = _get_or_create(root, "asset")
    for child in rig.find("asset"):
        asset.append(child)
    worldbody = _get_or_create(root, "worldbody")
    worldbody.append(rig_body)
    actuator = _get_or_create(root, "actuator")
    for child in rig.find("actuator"):
        actuator.append(child)
    sensor = _get_or_create(root, "sensor")
    for child in rig.find("sensor"):
        sensor.append(child)
