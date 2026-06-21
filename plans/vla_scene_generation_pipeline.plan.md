---
name: VLA scene-generation + adaptation pipeline
overview: "Frontend NL description -> Gizmo scene -> auto-adapt to a VLA-ready Franka scene -> feed the pi0.5 BC loop. Generalizes the manual env1 adaptation (cameras + eef site + collision-mask fix + reachable target) into one deterministic adapt_vla_scene() step, gated by a no-GPU validation smoke test before any A100 time is spent."
status: implemented (backend modules built + locally verified against env1; live Gizmo-key end-to-end + frontend/Modal-training stream still to run)
todos:
  - id: prompt-bias
    content: "Bias Gemma's scene_prompt toward VLA-friendly layouts (a Franka Panda on a tabletop with the relevant objects within arm's reach), without ignoring the user's photos."
    status: pending
  - id: export-franka
    content: "Backend VLA branch: export with POST /v1/scenes/{id}/export {format:mjcf, robot_profile:'franka_panda'} (Gizmo embeds the arm) instead of splicing the floating gripper."
    status: pending
  - id: adapt-module
    content: "New backend/vla_scene.py::adapt_vla_scene(zip_bytes, scene_id) -> TaskSpec. Generalizes the env1 edits: inject eef site + agentview + wrist cameras into the included panda.xml; defensive Franka name-remap; discover target object(s) + compute lift_height from rest z."
    status: pending
  - id: reachability
    content: "Guarantee the target is reachable: post-export snap the target free body's spawn pose to within ~0.7 m of the eef home on the work surface (+ optional prompt-bias). Decision pending: snap target vs. move arm vs. prompt-only."
    status: pending
  - id: validation-gate
    content: "No-GPU validation smoke test before training: reset scene in the sim, render both cameras, run a noop rollout, confirm it grades. Fail -> surface to frontend + regenerate, never burn A100 time on a broken scene."
    status: pending
  - id: orchestrator-wire
    content: "Wire the VLA branch into backend/orchestrator.py: export(robot_profile) -> adapt_vla_scene -> validation gate -> emit TaskSpec -> train/loop.py round (already takes --scene/--target/--instruction/--lift)."
    status: pending
isProject: false
---

# VLA scene-generation + adaptation pipeline

Come back to this after the franka-libero-v1 / Modal round-0 loop is proven. This plan
connects the `backend/` web app (Gizmo scene gen) to the `train/` pi0.5 BC loop so a
user's natural-language description becomes a trainable, VLA-ready Franka scene.

## What the Gizmo API actually supports (confirmed from docs.gizmo.antimlabs.com/api-reference)

- `POST /v1/scenes` — body is **only** `prompt`, `model`, `asset_pipeline` (auto|gizmo|cad),
  `persist`. No robot, no placement, no seed. **Scene layout is entirely prompt-driven.**
- `POST /v1/scenes/{id}/export` — body `{format: mjcf|usd|sdf, robot_profile: "franka_panda"}`.
  **`robot_profile` makes Gizmo embed the Franka for us** (this is how env1 got its arm) —
  so the VLA path does NOT splice its own robot (unlike the floating-gripper LLM path).
- No collision-group / contype control anywhere. The asset pipeline runs V-HACD convex
  decomposition -> many collision shapes -> many collision groups -> the `contype = 2**31`
  overflow is intrinsic to complex scenes; it must be handled downstream (the
  `sim/server.py` clamp shim already does, for every scene).
- `GET /v1/jobs/{id}` + SSE `GET /v1/jobs/{id}/events` for progress; statuses
  queued -> running -> succeeded|failed|cancelled.

## The flow

```
Frontend: photos + activities
   |
Backend vision (Gemma)  -> scene_prompt, biased toward a Franka-on-tabletop, objects-in-reach layout
   |
Gizmo POST /v1/scenes (prompt) -> job -> poll
   |
Gizmo POST /export {format: mjcf, robot_profile: "franka_panda"}     <- embed the arm
   |
adapt_vla_scene()    (NEW; generalizes the manual env1 work)
   1. inject eef site + agentview + wrist cameras into the embedded Franka's panda.xml
   2. discover target object(s); compute lift_height from rest z
   3. reachability fix (snap target within ~0.7 m of the eef)
   4. VALIDATION GATE: load in sim + noop rollout (render cameras + grade)
   5. emit TaskSpec {scene_id, target_object, instruction, lift_height}
   |
train/loop.py round: serve(base pi05) -> eval -> curate -> convert -> finetune -> repeat
   |
Frontend: live reward curve + MuJoCo viewer
```

The key change vs. today's `backend/scene_compose.py`: the VLA path exports with
`robot_profile="franka_panda"` instead of splicing the floating gripper.

## Each known failure mode -> deterministic fix (generalize the env1 work)

| Issue (hit on env1) | Root cause | Automated fix |
|---|---|---|
| No `eef` TCP site | robot_profile omits LIBERO's site | inject `<site name="eef" pos="0 0 0.1034">` into the `hand` body |
| No `agentview`/`wrist` cameras | not part of the robot profile | inject both (wrist on hand; agentview `mode="targetbody" target="hand"`) |
| `contype = 2**31` won't load | V-HACD -> >=31 collision groups | `sim/server.py` clamp shim (global, done) |
| Wrong Franka joint names | profile != joint1..7 | defensive assert + name-remap (env1 matched) |
| Target unreachable | Gizmo places objects freely | post-export snap (below) |
| Unknown unknowns | new scene quirks | the validation gate catches it pre-training |

## The two real decisions

1. **Reachability** (API gives no placement control):
   - *Prompt-bias* (cheap, partial): steer Gemma toward tabletop-scale, graspable layouts.
   - *Post-export snap* (reliable): snap the target free body's `pos` to within ~0.7 m of the
     eef home on the work surface — guarantees pickable targets so episodes can succeed and
     be curated. **Recommended: both.**
2. **Validation gate** is the safety net for "issues like the contype one." Every generated
   scene must pass a no-GPU smoke test (reset -> render both cameras -> noop rollout -> grade)
   before entering the loop. Runs locally in seconds; converts "A100 dies mid-serve" into
   "frontend: scene N failed validation, regenerating." Would have caught env1's contype
   overflow, missing cameras, and unreachable target automatically.

## How it plugs in

- New `backend/vla_scene.py::adapt_vla_scene(zip_bytes, scene_id) -> TaskSpec` — pure-Python,
  ElementTree on the included `panda.xml` + target-snap + validation call. Locally testable
  against env1 and a fresh export.
- `backend/orchestrator.py` gains a VLA branch (export with robot_profile -> adapt -> validate
  -> emit TaskSpec -> hand to a `train/loop.py` round).
- `train/loop.py` already accepts `--scene/--target/--instruction/--lift`, so the TaskSpec
  feeds it directly.

## Reference: the manual env1 adaptation this generalizes

- `scenes/env1/franka_emika_panda/panda.xml`: added `eef` site + `wrist` camera in the `hand`
  body, `agentview` camera in worldbody.
- `sim/server.py::_install_geom_mask_clamp`: clamps contype/conaffinity to int32.
- Threaded scene/target through `train/config.py` (scene_id), `train/eval.py`, `train/loop.py`,
  `run_vla.py` (`--scene`). See `train/README.md` "Running on a custom Gizmo scene".
- Open caveat: env1 parks the Franka ~1.1 m from laptop_11 (beyond ~0.85 m reach) — exactly
  the reachability problem the post-export snap solves.
