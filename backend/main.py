"""FastAPI entrypoint: `uv run uvicorn backend.main:app --reload --port 8000`."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routes import router


def _quiet_loop_handler(loop: asyncio.AbstractEventLoop, context: dict) -> None:
    """Swallow one known-benign asyncio error; delegate everything else.

    The HUD SDK's byte-splice tunnel (hud/environment/utils.py::_pump) calls
    writer.write_eof() guarded only by `suppress(OSError)`. On uvloop, EOF on an
    already-closed transport raises RuntimeError("...the handler is closed"),
    which escapes that suppression and surfaces as a noisy "Unhandled exception in
    client_connected_cb" once per rollout connection teardown. The bytes already
    flowed (the rollout result is unaffected), so this is cosmetic - filter it so
    it doesn't flood the terminal we use to monitor runs.
    """
    exc = context.get("exception")
    if isinstance(exc, RuntimeError) and "the handler is closed" in str(exc):
        return
    loop.default_exception_handler(context)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    asyncio.get_running_loop().set_exception_handler(_quiet_loop_handler)
    yield


app = FastAPI(title="hudathon backend", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/api/health")
async def health() -> dict[str, bool]:
    return {"ok": True}
