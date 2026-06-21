"""Best-of-N policy-improvement loop ("train further").

Not gradient descent: train/loop.py's real fine-tuning pipeline (record demos ->
curate -> finetune) is wired specifically to the Franka/LIBERO VLA contract this
app doesn't use (we use the floating-gripper tool-task path instead). This loop is
repeated independent LLM rollouts of the same mapped task, scored by the exact same
graders rollout.py already drives, reporting the best/mean reward each round -
"improvement" via search/selection over LLM sampling variance, not weight updates.
Each round's `group` rollouts run concurrently (independent sim subprocesses on
ephemeral ports, read-only access to the same scene files - no conflict).
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from backend.config import settings
from backend.rollout import run_rollout
from backend.task_mapping import TaskMapping

OnRound = Callable[[dict[str, Any]], Awaitable[None]]


async def train_further(
    scene_id: str,
    mapping: TaskMapping,
    *,
    rounds: int | None = None,
    group: int | None = None,
    model: str | None = None,
    on_round: OnRound | None = None,
) -> list[dict[str, Any]]:
    """Runs `rounds` rounds of `group` concurrent rollouts; returns per-round records."""
    rounds = rounds or settings.train_further_rounds
    group = group or settings.train_further_group

    history: list[dict[str, Any]] = []
    best_reward = float("-inf")

    for round_idx in range(rounds):
        results = await asyncio.gather(
            *(run_rollout(scene_id, mapping, model=model) for _ in range(group))
        )
        rewards = [r["reward"] for r in results]
        best_reward = max(best_reward, *rewards)
        record = {
            "round": round_idx,
            "group_rewards": rewards,
            "best_reward": best_reward,
            "mean_reward": sum(rewards) / len(rewards),
        }
        history.append(record)
        if on_round is not None:
            await on_round(record)
    return history
