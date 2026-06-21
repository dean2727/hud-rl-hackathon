"""Thin async client for the Gizmo scene-generation REST API.

https://docs.gizmo.antimlabs.com - base URL confirmed from the live OpenAPI spec
(`servers: [{"url": "https://api.gizmo.antimlabs.com"}]`). Endpoints used here:

    POST /v1/scenes                  {"prompt": str} -> 202 {scene_id, job_id, status}
    GET  /v1/jobs/{job_id}            poll: status queued -> running -> succeeded|failed|cancelled
    GET  /v1/jobs/{job_id}/events     SSE: stage_start, stage_complete, asset_ready, error, ping, done
    POST /v1/scenes/{scene_id}/export {"format": "mjcf"} -> raw application/zip bytes
"""

from __future__ import annotations

from collections.abc import AsyncIterator
import json
from typing import Any

import httpx

from backend.config import settings


class GizmoError(RuntimeError):
    """Raised on a non-2xx response or a job that finishes in a failed/cancelled state."""


class GizmoClient:
    def __init__(self, base_url: str | None = None, api_key: str | None = None) -> None:
        self.base_url = (base_url or settings.gizmo_base_url).rstrip("/")
        self.api_key = api_key or settings.gizmo_api_key

    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            raise GizmoError("GIZMO_API_KEY is not set - get a key at gizmo.antimlabs.com and set it in hudathon/.env")
        return {"authorization": f"Bearer {self.api_key}"}

    async def generate_scene(self, prompt: str) -> dict[str, Any]:
        """POST /v1/scenes - kicks off async generation, returns {scene_id, job_id, status}."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{self.base_url}/v1/scenes",
                headers=self._headers(),
                json={"prompt": prompt},
            )
        if resp.status_code != 202:
            raise GizmoError(f"generate_scene failed ({resp.status_code}): {resp.text}")
        return resp.json()

    async def get_job(self, job_id: str, *, include_result: bool = True) -> dict[str, Any]:
        """GET /v1/jobs/{job_id} - poll fallback / final-status check."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(
                f"{self.base_url}/v1/jobs/{job_id}",
                headers=self._headers(),
                params={"include_result": str(include_result).lower()},
            )
        if resp.status_code != 200:
            raise GizmoError(f"get_job failed ({resp.status_code}): {resp.text}")
        return resp.json()

    async def stream_job_events(self, job_id: str) -> AsyncIterator[dict[str, Any]]:
        """GET /v1/jobs/{job_id}/events - yields {"type": <event-name>, "data": <parsed-or-raw>}.

        The stream closes server-side once the job reaches a terminal state. Event
        payload shapes aren't part of the published spec, so we pass them through
        generically (JSON-parsed if possible, else the raw string) rather than
        assuming specific fields.
        """
        url = f"{self.base_url}/v1/jobs/{job_id}/events"
        async with httpx.AsyncClient(timeout=None) as client, client.stream(
            "GET", url, headers=self._headers()
        ) as resp:
            if resp.status_code != 200:
                raise GizmoError(f"stream_job_events failed ({resp.status_code}): {await resp.aread()}")

            event_type: str | None = None
            data_lines: list[str] = []
            async for line in resp.aiter_lines():
                if line == "":  # blank line: dispatch the buffered event
                    if data_lines:
                        raw = "\n".join(data_lines)
                        try:
                            data: Any = json.loads(raw)
                        except json.JSONDecodeError:
                            data = raw
                        yield {"type": event_type or "message", "data": data}
                    event_type, data_lines = None, []
                    continue
                if line.startswith(":"):  # comment/keepalive
                    continue
                if line.startswith("event:"):
                    event_type = line[len("event:") :].strip()
                elif line.startswith("data:"):
                    data_lines.append(line[len("data:") :].strip())
                # "id:" / "retry:" fields are ignored - no reconnect/resume support yet.

    async def export_scene(self, scene_id: str, fmt: str = "mjcf") -> bytes:
        """POST /v1/scenes/{scene_id}/export - returns the raw ZIP archive bytes."""
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{self.base_url}/v1/scenes/{scene_id}/export",
                headers=self._headers(),
                json={"format": fmt},
            )
        if resp.status_code != 200:
            raise GizmoError(f"export_scene failed ({resp.status_code}): {resp.text}")
        return resp.content
