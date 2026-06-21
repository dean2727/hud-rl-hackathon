"""Pydantic response models for the backend's REST API.

Request bodies are plain FastAPI `Form`/`File` params (see routes.py) since the
only POST that takes a body is a multipart upload. SSE event payloads are plain
dicts built in runs.py - typing them here would just be ceremony around values
that go straight to `json.dumps`.
"""

from __future__ import annotations

from pydantic import BaseModel


class RunCreatedResponse(BaseModel):
    run_id: str


class ActivityResult(BaseModel):
    activity_index: int
    activity: str
    task: str
    target: str
    status: str  # "pending" | "running" | "completed" | "failed"
    reward: float | None = None
    success: bool | None = None
    content: str | None = None
    can_train_further: bool = False  # true once this activity's initial rollout has completed


class RunStateResponse(BaseModel):
    run_id: str
    stage: str
    status: str  # "running" | "done" | "failed"
    scene_id: str | None = None
    scene_prompt: str | None = None
    results: list[ActivityResult] = []
    error: str | None = None


class TrainFurtherAccepted(BaseModel):
    ok: bool = True


class ConfirmSceneRequest(BaseModel):
    scene_prompt: str  # the user-reviewed (possibly edited) description to feed Gizmo


class SceneConfirmAccepted(BaseModel):
    ok: bool = True
