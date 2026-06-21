"""Map free-text user activities onto the 4 graded task templates in
environment/env.py (open_drawer, pick_object, move_object, force_grasp).

Keyword-rule classification + fuzzy/substring matching against the object and
joint names scene_compose.py actually discovered in the generated scene (the
authoritative source - Gizmo invents its own body names, we don't get to pick
them). No LLM call needed: this keeps the mapping fast, deterministic, and easy
to reason about for a hackathon demo.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import difflib
import re
from typing import Any

OPEN_DRAWER_KEYWORDS = ("open", "pull", "slide out", "unlock", "drawer", "door", "cabinet")
FORCE_GRASP_KEYWORDS = ("grip", "hold", "squeeze", "clamp", "grasp firmly", "firmly grasp")
MOVE_KEYWORDS = ("move", "push", "place", "put", "slide", "relocate", "bring")
PICK_KEYWORDS = ("pick", "lift", "grab", "take", "remove", "raise")


@dataclass
class TaskMapping:
    activity: str
    task: str  # "open-drawer" | "pick-object" | "move-object" | "force-grasp"
    target: str  # the matched object or joint name
    kwargs: dict[str, Any] = field(default_factory=dict)  # extra args for the env.py template call


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", text.lower())


def _best_match(activity: str, candidates: list[str]) -> tuple[str | None, float]:
    if not candidates:
        return None, 0.0
    text = activity.lower()
    words = _words(activity)
    best_name, best_score = None, 0.0
    for name in candidates:
        name_l = name.lower()
        if name_l in text or any(w in name_l or name_l in w for w in words if len(w) > 2):
            return name, 1.0  # direct substring hit - good enough, stop here
        for w in words:
            score = difflib.SequenceMatcher(None, w, name_l).ratio()
            if score > best_score:
                best_name, best_score = name, score
    return best_name, best_score


def _classify_task_type(activity: str, has_articulated: bool) -> str:
    text = activity.lower()
    if has_articulated and any(k in text for k in OPEN_DRAWER_KEYWORDS):
        return "open-drawer"
    if any(k in text for k in FORCE_GRASP_KEYWORDS):
        return "force-grasp"
    if any(k in text for k in MOVE_KEYWORDS):
        return "move-object"
    if any(k in text for k in PICK_KEYWORDS):
        return "pick-object"
    return "pick-object"  # safe generic default - exercises the grading pipeline either way


def classify_all(
    activities: list[str],
    objects: dict[str, dict[str, Any]],
    object_hints: list[str] | None = None,
) -> list[TaskMapping]:
    free_names = [n for n, o in objects.items() if o.get("type") == "free"]
    articulated = {n: o for n, o in objects.items() if o.get("type") == "articulated"}
    articulated_names = list(articulated)
    all_names = free_names + articulated_names

    mappings: list[TaskMapping] = []
    for activity in activities:
        task = _classify_task_type(activity, bool(articulated_names))

        if task == "open-drawer":
            name, score = _best_match(activity, articulated_names)
            if name is None:  # no articulated joint in this scene - fall back
                task = "pick-object"
            else:
                joint_info = articulated[name]
                lo, hi = joint_info.get("range", [0.0, 1.0])
                success_threshold = lo + 0.8 * (hi - lo)
                mappings.append(TaskMapping(
                    activity=activity, task=task, target=joint_info.get("joint") or name,
                    kwargs={"target_joint": joint_info.get("joint") or name, "success_threshold": success_threshold},
                ))
                continue

        name, _ = _best_match(activity, free_names or all_names)
        if name is None:
            name = "unknown"
        obj_info = objects.get(name, {})
        pos = obj_info.get("initial_position") or [0.0, 0.0, 0.8]

        if task == "force-grasp":
            mappings.append(TaskMapping(
                activity=activity, task=task, target=name,
                kwargs={"target_object": name, "min_grip_force": 0.5, "hold_steps": 100},
            ))
        elif task == "move-object":
            mappings.append(TaskMapping(
                activity=activity, task=task, target=name,
                kwargs={
                    "target_object": name,
                    "goal_x": pos[0] - 0.2, "goal_y": pos[1], "goal_z": pos[2],
                    "tolerance": 0.05,
                },
            ))
        else:  # pick-object
            mappings.append(TaskMapping(
                activity=activity, task=task, target=name,
                kwargs={"target_object": name, "lift_height": pos[2] + 0.15},
            ))
    return mappings
