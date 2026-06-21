#!/usr/bin/env python
"""Watch a rollout live in the Newton/MuJoCo viewer.

Standalone CLI generalizing examples/example_agent.py to an arbitrary scene/task/
target, with the live 3D viewer forced on. sim/host.py already supports this: the
spawned sim process inherits the parent's environment and pops a viewer window on
its main thread whenever HUDATHON_VIEWER=1 is set - no sim-layer changes needed
here, just setting the env var before anything sim-related is imported.

Needs a display and the `viewer` extra: `uv sync --extra viewer`.

    # try it on a bundled scene first, no Gizmo dependency:
    python scripts/watch_live.py --scene-id tabletop-v1 --task pick-object --target mug

    # then on a freshly Gizmo-generated scene (after a backend run has produced one
    # under scenes/, with whatever object/joint names backend/scene_compose.py
    # discovered for it):
    python scripts/watch_live.py --scene-id <gizmo-scene-id> --task move-object --target cup
"""

from __future__ import annotations

import os

os.environ["HUDATHON_VIEWER"] = "1"

import argparse  # noqa: E402
import asyncio  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hud import LocalRuntime  # noqa: E402
from hud.agents import create_agent  # noqa: E402

from environment.env import force_grasp, move_object, open_drawer, pick_object  # noqa: E402

TEMPLATES = {
    "open-drawer": open_drawer,
    "pick-object": pick_object,
    "move-object": move_object,
    "force-grasp": force_grasp,
}


def _build_kwargs(args: argparse.Namespace) -> dict:
    if args.task == "open-drawer":
        kwargs: dict = {"target_joint": args.target}
        if args.success_threshold is not None:
            kwargs["success_threshold"] = args.success_threshold
        return kwargs
    if args.task == "pick-object":
        kwargs = {"target_object": args.target}
        if args.lift_height is not None:
            kwargs["lift_height"] = args.lift_height
        return kwargs
    if args.task == "move-object":
        kwargs = {"target_object": args.target, "tolerance": args.tolerance}
        if args.goal is not None:
            kwargs["goal_x"], kwargs["goal_y"], kwargs["goal_z"] = args.goal
        return kwargs
    return {  # force-grasp
        "target_object": args.target,
        "min_grip_force": args.min_grip_force,
        "hold_steps": args.hold_steps,
    }


async def main_async(args: argparse.Namespace) -> None:
    template_fn = TEMPLATES[args.task]
    task = template_fn(scene_id=args.scene_id, **_build_kwargs(args))

    if args.scripted:
        if args.task != "move-object":
            raise SystemExit("--scripted only supports --task move-object (the repo's scripted pusher)")
        print("note: ScriptedAgent always pushes toward its own hardcoded GOAL, "
              "ignoring --target/--goal - it's a fixed demo routine, not a general controller.")
        from examples.example_agent import ScriptedAgent
        agent = ScriptedAgent()
    else:
        agent = create_agent(args.model or "claude-sonnet-4-5")

    job = await task.run(agent, runtime=LocalRuntime(str(ROOT / "environment" / "env.py")))
    run = job.runs[-1] if job.runs else None
    print(f"reward: {job.reward}")
    if run is not None:
        print(f"content: {run.grade.content}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene-id", required=True, help="a scenes/ folder name, e.g. tabletop-v1 or a Gizmo-generated id")
    ap.add_argument("--task", required=True, choices=list(TEMPLATES))
    ap.add_argument("--target", required=True, help="object name (pick/move/force-grasp) or joint name (open-drawer)")
    ap.add_argument("--scripted", action="store_true", help="use the repo's deterministic mug-pusher instead of an LLM agent (move-object only)")
    ap.add_argument("--model", default=None, help="model id for the LLM agent (default claude-sonnet-4-5)")
    ap.add_argument("--lift-height", type=float, default=None)
    ap.add_argument("--goal", type=float, nargs=3, metavar=("X", "Y", "Z"), default=None)
    ap.add_argument("--tolerance", type=float, default=0.05)
    ap.add_argument("--min-grip-force", type=float, default=0.5)
    ap.add_argument("--hold-steps", type=int, default=100)
    ap.add_argument("--success-threshold", type=float, default=None)
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
