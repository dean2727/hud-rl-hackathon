"""In-memory run store + a multi-subscriber SSE event bus per run.

No login, no persistence beyond process memory - matches the "no login, 1 page,
just watch it happen" brief. Each run keeps a buffered event history so a
reconnecting/late SSE client replays everything-so-far before tailing live events.

The SSE stream stays open until the client disconnects (see routes.py) rather than
auto-closing on a "done"/"error" event - a run can have multiple later
"train-further" actions on different activities, each streaming its own events on
the same connection.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import uuid

from backend.task_mapping import TaskMapping


@dataclass
class ActivityState:
    activity_index: int
    activity: str
    mapping: TaskMapping | None = None
    status: str = "pending"  # pending | running | completed | failed
    reward: float | None = None
    success: bool | None = None
    content: str | None = None


@dataclass
class RunState:
    run_id: str
    activities: list[str]
    image_paths: list[Path]
    stage: str = "queued"
    status: str = "running"  # running | done | failed
    scene_id: str | None = None
    scene_prompt: str | None = None
    objects: dict[str, Any] = field(default_factory=dict)
    object_hints: list[str] = field(default_factory=list)
    activity_states: list[ActivityState] = field(default_factory=list)
    error: str | None = None
    history: list[dict[str, Any]] = field(default_factory=list)
    subscribers: list[asyncio.Queue] = field(default_factory=list)

    def activity_result(self, a: ActivityState) -> dict[str, Any]:
        return {
            "activity_index": a.activity_index,
            "activity": a.activity,
            "task": a.mapping.task if a.mapping else "",
            "target": a.mapping.target if a.mapping else "",
            "status": a.status,
            "reward": a.reward,
            "success": a.success,
            "content": a.content,
            "can_train_further": a.status == "completed",
        }

    def snapshot(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "stage": self.stage,
            "status": self.status,
            "scene_id": self.scene_id,
            "scene_prompt": self.scene_prompt,
            "results": [self.activity_result(a) for a in self.activity_states],
            "error": self.error,
        }


class RunStore:
    def __init__(self) -> None:
        self._runs: dict[str, RunState] = {}

    def create(self, activities: list[str], image_paths: list[Path]) -> RunState:
        run_id = uuid.uuid4().hex[:12]
        run = RunState(run_id=run_id, activities=activities, image_paths=image_paths)
        self._runs[run_id] = run
        return run

    def get(self, run_id: str) -> RunState | None:
        return self._runs.get(run_id)


store = RunStore()


def subscribe(run: RunState) -> asyncio.Queue:
    q: asyncio.Queue = asyncio.Queue()
    for item in run.history:
        q.put_nowait(item)
    run.subscribers.append(q)
    return q


def unsubscribe(run: RunState, q: asyncio.Queue) -> None:
    if q in run.subscribers:
        run.subscribers.remove(q)


async def emit(run: RunState, event: str, data: dict[str, Any]) -> None:
    payload = {"event": event, "data": data}
    run.history.append(payload)
    for q in run.subscribers:
        await q.put(payload)
