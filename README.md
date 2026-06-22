# Room2Robot

<p align="center">
  <img src="demo.gif" alt="Room2Robot demo" width="840">
</p>

<p align="center">
  Upload photos of a real space → Gizmo builds a 3D scene → HUD grades rollouts →
  fine-tune pi0.5 on Modal → watch the robot in sim.
</p>


> **describe** the room from photos  →  **generate** a Gizmo scene  →  **grade** LLM rollouts  →  **fine-tune** the VLA on Modal  →  **play back** a rollout video

## Web app

The one-page app in `frontend/` + `backend/` drives the full loop:

1. Upload 1–3 photos and list activities you want the arm to learn.
2. A vision model writes a scene prompt; you confirm or edit it.
3. [Gizmo](https://gizmo.antimlabs.com) generates a Newton scene; the backend splices in a Franka gripper.
4. Each activity maps to a graded manipulation task and runs an LLM tool-agent rollout (live SSE timeline).
5. **Fine-tune on Modal** runs eval → curate → `lerobot-train` on A100s and streams reward dots as episodes complete.
6. **Show Video of Final Robot** serves the fine-tuned checkpoint on Modal and records a sim rollout MP4.

### Setup

```bash
uv sync --extra robot --extra vla --extra serve
cp .env.example .env          # GIZMO_API_KEY, GOOGLE_API_KEY (or HUDATHON_VISION_PROVIDER=ollama)
hud set HUD_API_KEY=...       # LLM tool-agent rollouts route through the HUD gateway
modal token new
modal secret create huggingface HF_TOKEN=hf_...   # for Modal fine-tune + serve
```

Optional demo shortcuts (no Gizmo network, no Modal GPU):

```bash
# in .env
HUDATHON_DEMO_MODE=1
HUDATHON_DEMO_SCENE=env2
HUDATHON_TRAIN_MODAL_DRY_RUN=1   # synthetic fine-tune chart; no checkpoint volume writes
```

### Run

```bash
# terminal 1 — backend (REST + SSE on :8000)
uv run uvicorn backend.main:app --reload --port 8000

# terminal 2 — frontend (Vite on :5173, proxies /api → :8000)
cd frontend && npm install && npm run dev
# open http://localhost:5173
```

Modal fine-tune details and CLI equivalents live in [`train/README.md`](train/README.md).

## Newton sim + HUD SDK

Each folder under `scenes/` is a live Newton environment: reset, step, render, score.
Generate new scenes at [gizmo.antimlabs.com](https://gizmo.antimlabs.com) and drop
exports under `scenes/` (`scene.xml` + `metadata.json`). Set `HUDATHON_VIEWER=1` to
watch any rollout in a live 3D window.

| Task | Agent | Scene |
|------|-------|-------|
| `open-drawer`, `pick-object`, `move-object`, `force-grasp` | LLM tool API (`mcp`) | `tabletop-v1` or composed Gizmo scenes |
| `vla-pick` | pi0.5 VLA (`robot` / openpi) | `franka-libero-v1` |

## CLI eval (optional)

Python 3.12. From the repo root:

```bash
uv sync                              # bundled Newton wheel in wheels/
source .venv/bin/activate
hud set HUD_API_KEY=your-key-here

python scripts/check_setup.py        # boots sim + one scripted rollout (~1 min first run)
hud eval environment/tasks.py claude --all --group 3

# VLA — local GPU or remote policy server
python run_vla.py --group 10
modal run serve/pi05_modal.py        # prints ws://HOST:PORT
python run_vla.py --remote HOST:PORT --group 10

# watch live in 3D (needs display + --extra viewer)
HUDATHON_VIEWER=1 hud eval environment/tasks.py claude --group 1
python scripts/watch_live.py --scene-id tabletop-v1 --task pick-object --target mug
```

Bring your own VLA: scaffold in `agents/vla_agent.py`, then
`python run_vla.py --agent agents.vla_agent:CustomAgent --group 10`.

## Layout

```
hudathon/
├── demo.gif       animated preview (embedded above)
├── demo.mp4       full walkthrough
├── frontend/      React/Vite app: upload, timeline, fine-tune chart, rollout video
├── backend/       FastAPI: vision → Gizmo → compose → rollouts → Modal train (SSE)
├── train/         Modal eval → curate → convert → fine_tune loop (see train/README.md)
├── environment/   HUD envs + tasks (LLM env.py, VLA vla_env.py, tasks.py)
├── serve/         pi0.5 policy servers (local + Modal)
├── sim/           Newton sim, Franka bridge, rollout frame capture
├── scenes/        tabletop-v1, franka-libero-v1, + generated exports
├── agents/        pi0.5 baseline + custom VLA scaffold
├── run_vla.py     VLA eval runner
└── wheels/        bundled Newton engine
```
