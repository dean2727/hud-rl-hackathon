"""Env-driven config for the backend: Gizmo, Ollama, and rollout defaults."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # hudathon/


@dataclass(frozen=True)
class Settings:
    gizmo_base_url: str = os.environ.get("GIZMO_BASE_URL", "https://api.gizmo.antimlabs.com")
    gizmo_api_key: str = os.environ.get("GIZMO_API_KEY", "")

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
