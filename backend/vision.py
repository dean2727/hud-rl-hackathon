"""Multimodal description step: photos + activities -> a Gizmo scene prompt.

Two backends, chosen by settings.vision_provider:
  - "google" (default): Gemma-3-12B over the Google AI Studio (Gemini) API. Needs
    GOOGLE_API_KEY (or GEMINI_API_KEY) from aistudio.google.com.
  - "ollama": a local Ollama server running `gemma3:12b` (`ollama pull gemma3:12b`).
"""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
import json
from typing import Any

import httpx

from backend.config import settings

SYSTEM_PROMPT = (
    "You are looking at photos of a room or workspace where someone wants to place "
    "a robot arm. They've listed the activities they want the robot to learn.\n\n"
    "First identify EVERY distinct manipulable object sitting on the work surface "
    "(especially the small ones relevant to the listed activities), plus the main "
    "furniture/fixtures (table, shelf, drawer, cabinet). Then write ONE concise "
    "scene description for a text-to-3D-scene generator that NAMES each of those "
    "objects explicitly, with its color and material when visible.\n\n"
    "Hard rules for `scene_prompt`:\n"
    "- Name every relevant object individually. Good: 'a wooden table holding a red "
    "ceramic mug, a silver fork, and a small cardboard box'.\n"
    "- NEVER use vague catch-alls like 'various objects', 'several small items', "
    "'small objects', or 'objects scattered around'. Enumerate them instead.\n"
    "- One or two sentences, no more.\n\n"
    'Respond with ONLY a JSON object: {"scene_prompt": str, "objects": [str, ...]}. '
    "`objects` is the short, lowercase, singular names of those manipulable objects "
    "(e.g. 'mug', 'fork', 'box')."
)


class VisionError(RuntimeError):
    pass


def _encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _activities_block(activities: list[str]) -> str:
    return "\n".join(f"- {a}" for a in activities) if activities else "(none listed)"


def _user_text(activities: list[str]) -> str:
    return f"Activities the robot should learn:\n{_activities_block(activities)}"


async def describe_images(image_paths: list[Path], activities: list[str]) -> dict[str, Any]:
    """Returns {"scene_prompt": str, "objects": list[str]} via the configured provider."""
    if settings.vision_provider == "ollama":
        return await _describe_ollama(image_paths, activities)
    return await _describe_google(image_paths, activities)


async def _describe_google(image_paths: list[Path], activities: list[str]) -> dict[str, Any]:
    """Gemma-3-12B over the Google AI Studio (Gemini) API.

    Gemma models on this API don't accept a separate system instruction, so the system
    prompt is folded into the user turn; images go in as inline_data parts.
    """
    if not settings.google_api_key:
        raise VisionError(
            "GOOGLE_API_KEY is not set - get a key at aistudio.google.com and set it in "
            "hudathon/.env (or set HUDATHON_VISION_PROVIDER=ollama to use local Ollama)."
        )
    parts: list[dict[str, Any]] = [{"text": f"{SYSTEM_PROMPT}\n\n{_user_text(activities)}"}]
    for path in image_paths:
        mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        parts.append({"inline_data": {"mime_type": mime, "data": _encode_image(path)}})

    model = settings.google_vision_model
    url = f"{settings.google_base_url}/v1beta/models/{model}:generateContent"
    payload = {"contents": [{"role": "user", "parts": parts}],
               "generationConfig": {"temperature": 0.4}}

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                url, headers={"x-goog-api-key": settings.google_api_key}, json=payload
            )
    except httpx.ConnectError as exc:
        raise VisionError(f"Could not reach the Google AI Studio API at {settings.google_base_url}") from exc

    if resp.status_code == 404:
        raise VisionError(f"Model '{model}' not found on the Gemini API (check HUDATHON_VISION_MODEL).")
    if resp.status_code in (401, 403):
        raise VisionError(f"Google AI Studio auth failed ({resp.status_code}): check GOOGLE_API_KEY. {resp.text[:300]}")
    if resp.status_code != 200:
        raise VisionError(f"Gemini generateContent failed ({resp.status_code}): {resp.text[:300]}")

    data = resp.json()
    try:
        content = data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, TypeError) as exc:
        raise VisionError(f"unexpected Gemini response shape: {json.dumps(data)[:300]}") from exc
    return _parse_description(content)


async def _describe_ollama(image_paths: list[Path], activities: list[str]) -> dict[str, Any]:
    """Local Ollama gemma3:12b (HUDATHON_VISION_PROVIDER=ollama)."""
    images_b64 = [_encode_image(p) for p in image_paths]
    payload = {
        "model": settings.ollama_model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _user_text(activities), "images": images_b64},
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


def bias_prompt_for_vla(scene_prompt: str, objects: list[str] | None = None) -> str:
    """Steer the Gizmo prompt toward a VLA-trainable layout: one compact tabletop with
    the graspable objects clustered within a stationary arm's reach (not a large room).

    Gizmo has no placement controls, so this is the only lever we have over layout
    pre-generation. The post-export reachability snap/check in vla_scene.py is the
    deterministic guarantee; this just gives the generator a good starting point.
    """
    obj_phrase = ""
    if objects:
        named = ", ".join(objects[:6])
        obj_phrase = (
            f" Cluster the manipulable objects ({named}) together on one tabletop, "
            "within about half a meter of each other."
        )
    return (
        f"{scene_prompt.rstrip('. ')}. Arrange this as a single small tabletop "
        "manipulation workspace for a stationary robot arm mounted at the table edge: "
        "one table, a few graspable objects resting on its surface within the arm's "
        f"reach, no large room or distant furniture.{obj_phrase}"
    )
