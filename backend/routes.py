"""REST + SSE API: upload photos/activities, stream pipeline progress, train further."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse

from backend.config import settings
from backend.orchestrator import run_describe, run_generate_onward, run_train_further
from backend.runs import store, subscribe, unsubscribe
from backend.schemas import (
    ConfirmSceneRequest,
    RunCreatedResponse,
    RunStateResponse,
    SceneConfirmAccepted,
    TrainFurtherAccepted,
)

router = APIRouter(prefix="/api")


@router.post("/runs", response_model=RunCreatedResponse)
async def create_run(
    images: list[UploadFile] = File(...),
    activities: list[str] = Form(...),
) -> RunCreatedResponse:
    if not (1 <= len(images) <= 3):
        raise HTTPException(400, "upload 1-3 photos")
    cleaned_activities = [a.strip() for a in activities if a.strip()]
    if not cleaned_activities:
        raise HTTPException(400, "list at least one activity")

    run = store.create(activities=cleaned_activities, image_paths=[])
    run_dir = settings.upload_dir / run.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    image_paths: list[Path] = []
    for i, image in enumerate(images):
        ext = Path(image.filename or "").suffix or ".jpg"
        dest = run_dir / f"photo_{i}{ext}"
        dest.write_bytes(await image.read())
        image_paths.append(dest)
    run.image_paths = image_paths

    asyncio.create_task(run_describe(run))
    return RunCreatedResponse(run_id=run.run_id)


@router.post("/runs/{run_id}/confirm-scene", response_model=SceneConfirmAccepted)
async def confirm_scene(run_id: str, body: ConfirmSceneRequest) -> SceneConfirmAccepted:
    run = store.get(run_id)
    if run is None:
        raise HTTPException(404, "run not found")
    if run.stage != "awaiting_confirmation":
        raise HTTPException(409, f"run is not awaiting scene confirmation (stage={run.stage})")
    prompt = body.scene_prompt.strip()
    if not prompt:
        raise HTTPException(400, "scene_prompt cannot be empty")
    run.scene_prompt = prompt
    asyncio.create_task(run_generate_onward(run))
    return SceneConfirmAccepted()


@router.get("/runs/{run_id}", response_model=RunStateResponse)
async def get_run(run_id: str) -> RunStateResponse:
    run = store.get(run_id)
    if run is None:
        raise HTTPException(404, "run not found")
    return RunStateResponse(**run.snapshot())


@router.get("/runs/{run_id}/events")
async def stream_events(run_id: str, request: Request) -> StreamingResponse:
    run = store.get(run_id)
    if run is None:
        raise HTTPException(404, "run not found")

    async def gen():
        q = subscribe(run)
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    item = await asyncio.wait_for(q.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
                    continue
                yield f"event: {item['event']}\ndata: {json.dumps(item['data'])}\n\n"
        finally:
            unsubscribe(run, q)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/runs/{run_id}/train-further", response_model=TrainFurtherAccepted)
async def train_further_route(run_id: str, activity_index: int) -> TrainFurtherAccepted:
    run = store.get(run_id)
    if run is None:
        raise HTTPException(404, "run not found")
    asyncio.create_task(run_train_further(run, activity_index))
    return TrainFurtherAccepted()
