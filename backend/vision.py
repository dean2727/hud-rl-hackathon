"""Multimodal description step: photos + activities -> a Gizmo scene prompt.

Calls a local Ollama server running `gemma3:12b` (Ollama's vision-capable Gemma 3).
Setup: `ollama pull gemma3:12b` once; `ollama serve` (or the desktop app) running.
"""

from __future__ import annotations

import base64
from pathlib import Path
import json
from typing import Any

import httpx

from backend.config import settings

SYSTEM_PROMPT = (
    "You are looking at photos of a room or workspace where someone wants to place "
    "a robot arm. They've listed the activities they want the robot to learn. "
    "Describe the setting and the specific objects involved in those activities, "
    "then propose ONE concise natural-language scene description suitable for a "
    "text-to-3D-scene generator (similar in style to: 'A modern kitchen with a "
    "wooden table, a ceramic mug, and a cabinet with a pull-out drawer').\n\n"
    'Respond with ONLY a JSON object: {"scene_prompt": str, "objects": [str, ...]}. '
    "`objects` should list short, lowercase, singular names of the manipulable "
    "objects you see that are relevant to the listed activities (e.g. 'mug', "
    "'drawer', 'remote')."
)


class VisionError(RuntimeError):
    pass


def _encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _activities_block(activities: list[str]) -> str:
    return "\n".join(f"- {a}" for a in activities) if activities else "(none listed)"


async def describe_images(image_paths: list[Path], activities: list[str]) -> dict[str, Any]:
    """Returns {"scene_prompt": str, "objects": list[str]}."""
    images_b64 = [_encode_image(p) for p in image_paths]
    user_text = f"Activities the robot should learn:\n{_activities_block(activities)}"

    payload = {
        "model": settings.ollama_model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text, "images": images_b64},
        ],
        "format": "json",
        "stream": False,
    }

    try:
        async with httpx.AsyncClient(timeout=240.0) as client:
            resp = await client.post(f"{settings.ollama_host}/api/chat", json=payload)
    except httpx.ConnectError as exc:
        raise VisionError(
            f"Could not reach Ollama at {settings.ollama_host} - is `ollama serve` running?"
        ) from exc

    if resp.status_code == 404:
        raise VisionError(
            f"Model '{settings.ollama_model}' not found on Ollama - run "
            f"`ollama pull {settings.ollama_model}` first."
        )
    if resp.status_code != 200:
        raise VisionError(f"Ollama chat request failed ({resp.status_code}): {resp.text}")

    content = resp.json().get("message", {}).get("content", "")
    return _parse_description(content)


def _parse_description(content: str) -> dict[str, Any]:
    try:
        parsed = json.loads(content)
        scene_prompt = str(parsed.get("scene_prompt") or "").strip()
        objects = [str(o).strip().lower() for o in parsed.get("objects") or [] if str(o).strip()]
        if scene_prompt:
            return {"scene_prompt": scene_prompt, "objects": objects}
    except json.JSONDecodeError:
        pass
    # Fallback: the model didn't return valid JSON - use the raw text as the prompt.
    return {"scene_prompt": content.strip()[:500], "objects": []}
