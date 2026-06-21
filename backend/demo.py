"""Demo-mode helpers: stand in for the Gizmo network calls with a local scene.

When settings.demo_mode is on, the orchestrator does NOT POST to Gizmo. Instead it
emits mock stage events (so the UI progress bar still animates) and treats a local,
already-downloaded scene (settings.demo_scene, default scenes/env2) as the export ZIP
so the rest of the pipeline (compose -> map -> rollout) runs unchanged and offline.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

from backend.config import settings

# Fake gizmo event types whose names match ProgressTimeline's GIZMO_STAGES regexes,
# so the single meta-stage bar advances Queued -> ... -> Scene ready during the demo.
MOCK_GIZMO_EVENT_TYPES: list[str] = [
    "job_queued",
    "scene_director",
    "asset_card_created",
    "scripting_master_started",
    "structure_component_built",
    "scene_texture_image",
    "job_succeeded",
]


def demo_scene_dir() -> Path:
    return settings.scenes_dir / settings.demo_scene


def demo_scene_zip() -> bytes:
    """Zip the local demo scene dir into bytes shaped like a Gizmo MJCF export
    (files at the archive root), so compose_scene/adapt can consume it unchanged."""
    scene_dir = demo_scene_dir()
    if not scene_dir.is_dir():
        raise FileNotFoundError(
            f"demo scene not found: {scene_dir} - set HUDATHON_DEMO_SCENE to a folder under scenes/"
        )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(scene_dir.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(scene_dir))
    return buf.getvalue()
