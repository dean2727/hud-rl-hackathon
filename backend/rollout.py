"""Drive one environment/env.py task template with an LLM tool-agent.

Mirrors examples/example_agent.py's `--llm` path (`create_agent` + `LocalRuntime`
against environment/env.py), generalized to an arbitrary scene_id + TaskMapping so
it works on freshly Gizmo-generated scenes as well as the bundled ones.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hud import LocalRuntime  # noqa: E402
from hud.agents import create_agent  # noqa: E402

from backend.config import settings  # noqa: E402
from backend.task_mapping import TaskMapping  # noqa: E402

TASK_TEMPLATE_NAMES = {
    "open-drawer": "open_drawer",
    "pick-object": "pick_object",
    "move-object": "move_object",
    "force-grasp": "force_grasp",
}


async def run_rollout(scene_id: str, mapping: TaskMapping, model: str | None = None) -> dict[str, Any]:
    """Runs one graded rollout for `mapping`; returns {reward, success, content}."""
    from environment.env import force_grasp, move_object, open_drawer, pick_object  # noqa: E402

    template_fn = {
        "open_drawer": open_drawer,
        "pick_object": pick_object,
        "move_object": move_object,
        "force_grasp": force_grasp,
    }[TASK_TEMPLATE_NAMES[mapping.task]]

    task = template_fn(scene_id=scene_id, **mapping.kwargs)
    agent = create_agent(model or settings.agent_model)
    job = await task.run(agent, runtime=LocalRuntime(str(settings.env_module)))

    run = job.runs[-1] if job.runs else None
    content = run.grade.content if run else None
    return {
        "reward": job.reward,
        "success": bool(content and "SUCCESS" in content),
        "content": content,
    }
