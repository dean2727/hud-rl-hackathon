"""Compile an arbitrary user activity into a `RewardProgram`.

This is the "dynamic" half of the reward system: instead of hand-coding a reward
per skill, we turn natural language into a composition of vetted predicates
(rewards/predicates.py). Two backends:

  - LLM (default): the configured Gemma/Gemini (or Ollama) model emits a program
    JSON, validated against the predicate registry. Handles the long tail of
    skills and phrasings.
  - Heuristic (always available): a deterministic keyword parser that covers
    pick / place-relative / place-near / grasp / open. Used offline and as the
    fallback when the LLM is unreachable or emits an invalid program.

It also returns `required_objects`: every body the program references. Some may
not be in the photo (e.g. "water the plant" with no watering can) - scene-gen is
responsible for ensuring Gizmo places them, under these exact names.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from typing import Any

import httpx

from backend.config import settings
from rewards.predicates import REGISTRY
from rewards.program import RewardProgram

# Viewer-frame mapping for the default Franka tabletop (see predicates.py header).
# direction word -> (axis, sign) where the object should sit relative to the reference.
_DIRECTIONS: dict[str, tuple[str, str]] = {
    "left": ("x", "negative"),
    "right": ("x", "positive"),
    "front": ("y", "negative"),
    "in front": ("y", "negative"),
    "behind": ("y", "positive"),
    "back": ("y", "positive"),
    "above": ("z", "positive"),
    "on top": ("z", "positive"),
    "below": ("z", "negative"),
    "under": ("z", "negative"),
}

_PLACE_WORDS = ("place", "put", "move", "set", "drop", "position", "relocate", "bring")
_PICK_WORDS = ("pick", "lift", "grab", "raise", "hold up", "pick up")
_GRASP_WORDS = ("grasp", "grip", "squeeze", "clamp")
_OPEN_WORDS = ("open", "pull open", "slide open")


@dataclass
class CompiledReward:
    program: RewardProgram
    required_objects: list[str] = field(default_factory=list)
    source: str = "heuristic"  # "llm" | "heuristic"

    def as_dict(self) -> dict[str, Any]:
        return {
            "program": self.program.as_dict(),
            "required_objects": self.required_objects,
            "source": self.source,
        }


# ── public API ───────────────────────────────────────────────────────────────


async def compile_reward(
    instruction: str,
    objects: list[str] | None = None,
    *,
    prefer_llm: bool = True,
) -> CompiledReward:
    """Best path: try the LLM, fall back to the heuristic on any failure."""
    objects = objects or []
    if prefer_llm:
        try:
            return await _compile_llm(instruction, objects)
        except Exception:
            pass  # any failure -> deterministic fallback (never block a run on the LLM)
    return compile_reward_heuristic(instruction, objects)


def compile_reward_heuristic(instruction: str, objects: list[str] | None = None) -> CompiledReward:
    """Deterministic keyword compiler. Always succeeds (defaults to pick)."""
    objects = objects or []
    text = instruction.lower().strip()

    direction = _find_direction(text)
    target = _find_object(text, objects, prefer_first=True)
    has_place = any(w in text for w in _PLACE_WORDS)
    near_phrase = any(p in text for p in ("near", "next to", "beside", "by the", "close to"))

    reference = None
    if direction or has_place or near_phrase:
        reference = _find_object(text, objects, exclude=target, after=True)
        if reference == target:
            reference = None

    if direction and reference:
        prog = _place_relative_program(instruction, target, reference, direction)
    elif (near_phrase or has_place) and reference:
        prog = _place_near_program(instruction, target, reference)
    elif any(w in text for w in _OPEN_WORDS):
        prog = _open_program(instruction, target)
    elif any(w in text for w in _GRASP_WORDS) and not any(w in text for w in _PICK_WORDS):
        prog = _grasp_program(instruction, target)
    else:
        prog = _pick_program(instruction, target)

    return CompiledReward(program=prog, required_objects=prog.referenced_objects(), source="heuristic")


# ── heuristic program builders ────────────────────────────────────────────────


def _pick_program(instruction: str, target: str) -> RewardProgram:
    return RewardProgram.from_mapping({
        "instruction": instruction,
        "target_object": target,
        "terms": [
            {"name": "grasp", "weight": 0.3, "fn": "grasped", "args": {"object": target}},
            {"name": "lift", "weight": 0.7, "fn": "lifted",
             "args": {"object": target, "height": 0.15, "baseline": "initial"}},
        ],
        "success": [
            {"fn": "lifted", "args": {"object": target, "height": 0.15}, "threshold": 0.999},
        ],
        "weights": {"progress": 0.5, "success": 0.5},
        "target_object": target,
    })


def _grasp_program(instruction: str, target: str) -> RewardProgram:
    return RewardProgram.from_mapping({
        "instruction": instruction,
        "target_object": target,
        "terms": [{"name": "grasp", "weight": 1.0, "fn": "grasped", "args": {"object": target}}],
        "success": [{"fn": "grasped", "args": {"object": target}, "threshold": 0.999}],
        "weights": {"progress": 0.4, "success": 0.6},
        "target_object": target,
    })


def _open_program(instruction: str, joint: str) -> RewardProgram:
    return RewardProgram.from_mapping({
        "instruction": instruction,
        "target_object": joint,
        "terms": [{"name": "open", "weight": 1.0, "fn": "joint_open",
                   "args": {"joint": joint, "open_frac": 0.8}}],
        "success": [{"fn": "joint_open", "args": {"joint": joint, "open_frac": 0.8}, "threshold": 0.999}],
        "weights": {"progress": 0.5, "success": 0.5},
        "target_object": joint,
    })


def _place_relative_program(
    instruction: str, target: str, reference: str, direction: tuple[str, str]
) -> RewardProgram:
    axis, sign = direction
    side_args = {"object": target, "reference": reference, "axis": axis,
                 "sign": sign, "full_margin": 0.15}
    return RewardProgram.from_mapping({
        "instruction": instruction,
        "target_object": target,
        "terms": [
            {"name": "grasp", "weight": 0.15, "fn": "grasped", "args": {"object": target}},
            {"name": "lift", "weight": 0.15, "fn": "lifted",
             "args": {"object": target, "height": 0.1, "baseline": "initial"}},
            {"name": f"{_dir_word(direction)}_of_{reference}", "weight": 0.4,
             "fn": "relative_side", "args": side_args},
            {"name": "rest", "weight": 0.3, "fn": "on_surface",
             "args": {"object": target, "tol": 0.05}},
        ],
        "success": [
            {"fn": "relative_side", "args": side_args, "threshold": 0.2},
            {"fn": "near", "args": {"object": target, "reference": reference, "radius": 0.5},
             "threshold": 0.2},
            {"fn": "on_surface", "args": {"object": target, "tol": 0.06}, "threshold": 0.6},
        ],
        "success_mode": "all",
        "weights": {"progress": 0.5, "success": 0.5},
        "target_object": target,
    })


def _place_near_program(instruction: str, target: str, reference: str) -> RewardProgram:
    near_args = {"object": target, "reference": reference, "radius": 0.15}
    return RewardProgram.from_mapping({
        "instruction": instruction,
        "target_object": target,
        "terms": [
            {"name": "grasp", "weight": 0.2, "fn": "grasped", "args": {"object": target}},
            {"name": "near", "weight": 0.5, "fn": "near", "args": near_args},
            {"name": "rest", "weight": 0.3, "fn": "on_surface", "args": {"object": target, "tol": 0.05}},
        ],
        "success": [
            {"fn": "near", "args": near_args, "threshold": 0.5},
            {"fn": "on_surface", "args": {"object": target, "tol": 0.06}, "threshold": 0.6},
        ],
        "weights": {"progress": 0.5, "success": 0.5},
        "target_object": target,
    })


# ── parsing helpers ───────────────────────────────────────────────────────────


def _find_direction(text: str) -> tuple[str, str] | None:
    # Longer phrases first ("in front" before "front") so we match the most specific.
    for word in sorted(_DIRECTIONS, key=len, reverse=True):
        if re.search(rf"\b{re.escape(word)}\b", text):
            return _DIRECTIONS[word]
    return None


def _dir_word(direction: tuple[str, str]) -> str:
    for word, d in _DIRECTIONS.items():
        if d == direction:
            return word.replace(" ", "_")
    return "side"


def _find_object(
    text: str,
    objects: list[str],
    *,
    exclude: str | None = None,
    prefer_first: bool = False,
    after: bool = False,
) -> str:
    """Pick the scene object named in the text. `prefer_first` returns the earliest
    mention (the thing being manipulated); `after` returns a later one (the reference)."""
    hits: list[tuple[int, str]] = []
    for name in objects:
        if name == exclude:
            continue
        m = re.search(rf"\b{re.escape(name.lower())}\b", text)
        if m:
            hits.append((m.start(), name))
    if hits:
        hits.sort()
        return hits[-1][1] if after else hits[0][1]
    # No scene-object match: lift a noun out of the sentence so the program still
    # references *something* (scene-gen can then add it under this name).
    nouns = re.findall(r"\b[a-z]{3,}\b", text)
    stop = set(_PLACE_WORDS) | set(_PICK_WORDS) | set(_GRASP_WORDS) | set(_OPEN_WORDS) | {
        "the", "and", "then", "with", "into", "onto", "next", "near", "from", "out",
        "left", "right", "front", "behind", "back", "above", "below", "under", "top", "side",
        "robot", "arm", "please", "your", "this", "that", "them", "all",
    }
    nouns = [n for n in nouns if n not in stop and n != exclude]
    if not nouns:
        return exclude_fallback(exclude)
    return nouns[-1] if after else nouns[0]


def exclude_fallback(exclude: str | None) -> str:
    return "object" if exclude != "object" else "target"


# ── LLM backend ───────────────────────────────────────────────────────────────


def _predicate_doc() -> str:
    return "\n".join(f"  - {name}" for name in sorted(REGISTRY))


def _system_prompt(objects: list[str]) -> str:
    obj_line = ", ".join(objects) if objects else "(none detected; you may name new ones)"
    return (
        "You design reward programs for a Franka tabletop robot learning a skill in "
        "simulation. Output ONLY a JSON object describing a reward program built from a "
        "fixed library of predicates - never code.\n\n"
        f"Available predicates (fn names):\n{_predicate_doc()}\n\n"
        "Each predicate returns 0..1. Use them as weighted `terms` (shaped progress) and "
        "as thresholded `success` clauses (did the task complete).\n\n"
        "Frame: the camera is behind the arm looking forward. relative_side axis/sign - "
        "LEFT=(x,negative), RIGHT=(x,positive), CLOSER/IN FRONT=(y,negative), "
        "BEHIND=(y,positive), ABOVE=(z,positive).\n\n"
        f"Scene objects you may reference by name: {obj_line}. If the task needs an object "
        "not listed, still reference it by a short lowercase name (the scene will be made to "
        "contain it).\n\n"
        "JSON shape:\n"
        '{"instruction": str, "target_object": str, '
        '"terms": [{"name": str, "weight": float, "fn": str, "args": {...}}], '
        '"success": [{"fn": str, "args": {...}, "threshold": float}], '
        '"success_mode": "all"|"any", "weights": {"progress": float, "success": float}}'
    )


async def _compile_llm(instruction: str, objects: list[str]) -> CompiledReward:
    content = await _llm_json(
        system=_system_prompt(objects),
        user=f"Activity: {instruction}\nDesign the reward program JSON.",
    )
    data = _extract_json(content)
    program = RewardProgram.from_mapping(data)  # validates fn names against REGISTRY
    if not program.instruction:
        program = RewardProgram.from_mapping({**data, "instruction": instruction})
    return CompiledReward(program=program, required_objects=program.referenced_objects(), source="llm")


def _extract_json(content: str) -> dict[str, Any]:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", content, re.DOTALL)
        if not m:
            raise
        return json.loads(m.group(0))


async def _llm_json(system: str, user: str) -> str:
    """One text-in/JSON-out call via the configured provider (mirrors vision.py)."""
    if settings.vision_provider == "ollama":
        payload = {
            "model": settings.ollama_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "format": "json",
            "stream": False,
        }
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(f"{settings.ollama_host}/api/chat", json=payload)
        resp.raise_for_status()
        return resp.json().get("message", {}).get("content", "")

    if not settings.google_api_key:
        raise RuntimeError("GOOGLE_API_KEY not set")
    model = settings.google_vision_model
    url = f"{settings.google_base_url}/v1beta/models/{model}:generateContent"
    payload = {
        "contents": [{"role": "user", "parts": [{"text": f"{system}\n\n{user}"}]}],
        "generationConfig": {"temperature": 0.2, "responseMimeType": "application/json"},
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(url, headers={"x-goog-api-key": settings.google_api_key}, json=payload)
    resp.raise_for_status()
    data = resp.json()
    return data["candidates"][0]["content"]["parts"][0]["text"]


__all__ = ["CompiledReward", "compile_reward", "compile_reward_heuristic"]
