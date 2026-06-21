# backend/

FastAPI service that turns uploaded photos + a list of activities into graded RL
rollouts on a freshly generated scene, streaming progress over SSE.

## Flow

```
POST /api/runs (2-3 photos + activities[])
        │  (background task: backend/orchestrator.py)
        ▼
describing_photos   vision.py        gemma3:12b (local Ollama) → scene prompt + object hints
generating_scene    gizmo_client.py  POST /v1/scenes → relay GET /v1/jobs/{id}/events (SSE)
composing_scene     scene_compose.py export MJCF zip → splice floating gripper → scenes/{id}/
mapping_activities  task_mapping.py  each activity → {open-drawer|pick-object|move-object|force-grasp, target}
running_rollouts    rollout.py       create_agent + LocalRuntime on environment/env.py → reward
        │
        ▼
GET /api/runs/{id}/events  (SSE: stage | gizmo | rollout | done | error)

POST /api/runs/{id}/train-further?activity_index=N
        └─ improve.py: best-of-N graded rollouts → train_round events (best/mean reward)
```

## Modules

| File | Responsibility |
|------|----------------|
| `config.py` | env-driven settings (Gizmo, Ollama, agent model, train defaults) |
| `gizmo_client.py` | async Gizmo REST/SSE client |
| `vision.py` | Ollama `gemma3:12b` photo → scene prompt |
| `scene_compose.py` | Gizmo MJCF export → `scenes/{id}/` with the gripper spliced in |
| `task_mapping.py` | activity text → graded task template + target |
| `rollout.py` | one graded rollout via `create_agent` + `LocalRuntime` |
| `improve.py` | "train further" best-of-N search (no gradient updates) |
| `runs.py` | in-memory run store + per-run SSE event bus (history replay) |
| `orchestrator.py` | the background pipeline + train-further task |
| `routes.py` / `main.py` | REST + SSE endpoints, app entrypoint |

## Run

```bash
uv run uvicorn backend.main:app --reload --port 8000
```

Needs `GIZMO_API_KEY` (scene generation), a running Ollama with `gemma3:12b` pulled
(photo description), and `HUD_API_KEY` (the LLM tool-agent routes through the HUD
gateway). See `../.env.example`.

## Notes

- **"gizmo" name collision**: `sim/server.py` imports a *local* `gizmo.runtime`
  package (a Newton/MJCF loader, shipped in the Newton wheel) — unrelated to the
  external Gizmo SaaS API this backend calls. Different things, same word.
- **"Train further" is not fine-tuning**: the repo's real gradient pipeline
  (`train/loop.py`) is wired to the Franka/LIBERO VLA contract, which this app
  doesn't use. `improve.py` instead does best-of-N selection over LLM sampling
  variance against the same graders — labeled as such in the UI.
- **Scene compatibility caveat**: Newton's MJCF importer may reject exotic geometry
  from a Gizmo export; if `sim/server.py::reset()` can't load a composed scene, the
  orchestrator surfaces it as an `error` SSE event rather than failing silently.
