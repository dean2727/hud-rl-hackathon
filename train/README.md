# train/ — Modal ⇄ HUD pi0.5 behavior-cloning loop

Fine-tune the `lerobot/pi05_libero_finetuned_v044` VLA on Modal GPUs from graded
rollouts in the Newton sim, and watch the reward improve. This is the *real*
weight-updating loop (per `plans/modal_hud_pi0_rl_loop.plan.md`), distinct from the
floating-gripper LLM-agent app under `backend/`.

```
serve(checkpoint) on Modal A100  ──tunnel──►  eval rollouts (local sim + RemoteAgent)
        ▲                                              │ records episode_* + rewards
        │ new checkpoint                               ▼
   fine_tune on Modal A100  ◄── upload ── convert ── curate (keep reward ≥ threshold)
        (lerobot-train)        to Volume   to LeRobot   episodes
```

## One-time setup

```bash
uv sync --extra vla            # local: lerobot + torch (CPU) for eval + dataset convert
modal token new                # authenticate Modal (needs an account + A100 quota)
```

- **HuggingFace token is optional.** Datasets are built locally and uploaded to the
  Modal Volume (`--dataset.root`), so nothing is pushed to the Hub. A token is only
  needed if the base checkpoint repo is private (`pi05_libero_finetuned_v044` is
  public). To use one anyway: `export HF_TOKEN=...` before `modal run`.
- The Newton sim runs **locally on CPU** for eval; only policy inference and
  `lerobot-train` run on Modal A100s.

Config knobs (all `HUDATHON_*` env vars, see `train/config.py`): `HUDATHON_GROUP`
(rollouts/round), `HUDATHON_CURATION_THRESHOLD`, `HUDATHON_MAX_STEPS`,
`HUDATHON_INSTRUCTION`, etc.

## Run round 0 (one command)

```bash
python -m train.loop --round 0 --steps 1000 --batch-size 4
```

This spawns the Modal server, waits for its tunnel address (published via a
`modal.Queue`), runs `--group` eval rollouts locally against it, curates the
high-reward episodes, converts them to a LeRobot dataset, uploads it to the
`hudathon-policy-checkpoints` Volume, and fine-tunes — committing
`volume:round-000` to the Volume.

Dry-run first to see the plan and exercise the local (non-GPU) stages — curate and
convert run for real against any episodes already under `datasets/eval-round-000`:

```bash
python -m train.loop --round 0 --dry-run
```

## Manual fallback (run any stage on its own)

If the orchestrator's Modal coordination misbehaves, each stage stands alone:

```bash
# 1) serve the policy on Modal (prints `--remote HOST:PORT`)
modal run train/modal_app.py::serve_policy --checkpoint lerobot/pi05_libero_finetuned_v044

# 2) eval against it locally, recording episodes + rewards
python run_vla.py --remote HOST:PORT --record datasets/eval-round-000 --group 8

# 3) curate by reward, then convert to a LeRobot dataset
python -m train.curate  --source-dir datasets/eval-round-000 --dest-dir datasets/curated-round-000
python -m train.convert --source-dir datasets/curated-round-000 --root datasets/lerobot-round-000

# 4) upload the dataset to the Volume, then fine-tune on Modal
modal volume put hudathon-policy-checkpoints datasets/lerobot-round-000 datasets/round-000
modal run train/modal_app.py::fine_tune --dataset-repo-id hudathon/vla-pick --dataset-root datasets/round-000
```

## Files

| File | Role |
|------|------|
| `config.py` | `HUDATHON_*` knobs: checkpoints, Volume name, group size, thresholds, task |
| `eval.py` | graded VLA rollouts via `RemoteAgent`; records episodes + `rewards.json` |
| `curate.py` | keep episodes with `reward ≥ threshold` (env-side `episode_*` layout) |
| `convert.py` | **episode_* → LeRobot v3 dataset** (the data boundary lerobot-train needs) |
| `finetune.py` | `lerobot-train` command wrapper (supports `--dataset.root` for local datasets) |
| `modal_app.py` | Modal app: `serve_policy` (tunnel + address queue) and `fine_tune` (A100) |
| `loop.py` | round-0 orchestrator wiring all of the above |
| `demos.py` | optional scripted/warm-start demo recorder (local-GPU path; not needed for round 0) |

## First-real-run checks (can't be verified without a GPU)

1. **Checkpoint subdir for the next round.** `lerobot-train` writes loadable weights
   under `output_dir/checkpoints/last/pretrained_model`. Confirm the exact path from
   the first `fine_tune` output and serve *that* (e.g.
   `volume:round-000/checkpoints/last/pretrained_model`) in round 1.
2. **Dataset feature keys.** `convert.py` uses the LIBERO convention
   (`observation.images.image`, `observation.images.wrist_image`,
   `observation.state[8]`, `action[7]`). If `lerobot-train` reports a feature/shape
   mismatch with the pi0.5 checkpoint, adjust `--image-keys`/`--state-dim` there.
3. **Reward spread before trusting curation.** With `HUDATHON_GROUP=8`, confirm the
   eval rewards aren't all identical (the grader has signal) before fine-tuning on
   the curated set.
