"""Env-driven config for the backend: Gizmo, vision, and rollout defaults."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # hudathon/


def _load_dotenv() -> None:
    """Load hudathon/.env into os.environ before Settings reads it, so a plain
    `uvicorn backend.main:app` picks up GIZMO_API_KEY etc. without exporting them.
    Existing env vars win (so an explicit export still overrides .env). Uses
    python-dotenv if present, else a minimal KEY=VALUE parser (no hard dependency)."""
    env_path = ROOT / ".env"
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv(env_path, override=False)
        return
    except ModuleNotFoundError:
        pass
    if not env_path.exists():
        return
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.removeprefix("export ").strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv()


@dataclass(frozen=True)
class Settings:
    gizmo_base_url: str = os.environ.get("GIZMO_BASE_URL", "https://api.gizmo.antimlabs.com")
    gizmo_api_key: str = os.environ.get("GIZMO_API_KEY", "")

    # Vision (photo -> scene description). "google" = Gemma-3-12B over the Google AI
    # Studio (Gemini) API; "ollama" = local gemma3:12b.
    vision_provider: str = os.environ.get("HUDATHON_VISION_PROVIDER", "google")
    google_api_key: str = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY", "")
    google_base_url: str = os.environ.get(
        "GOOGLE_BASE_URL", "https://generativelanguage.googleapis.com"
    )
    google_vision_model: str = os.environ.get("HUDATHON_VISION_MODEL", "gemini-3-5-flash")

    ollama_host: str = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    ollama_model: str = os.environ.get("OLLAMA_MODEL", "gemma3:12b")

    agent_model: str = os.environ.get("HUDATHON_AGENT_MODEL", "claude-sonnet-4-5")

    scenes_dir: Path = ROOT / "scenes"
    upload_dir: Path = Path(__file__).resolve().parent / "data" / "uploads"
    env_module: Path = ROOT / "environment" / "env.py"

    train_further_rounds: int = int(os.environ.get("HUDATHON_TRAIN_ROUNDS", "3"))
    train_further_group: int = int(os.environ.get("HUDATHON_TRAIN_GROUP", "3"))


settings = Settings()
settings.upload_dir.mkdir(parents=True, exist_ok=True)
