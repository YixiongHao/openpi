# OpenPI Inference Notes (LIBERO + Pi0.5)

## Architecture

```
Terminal 1: Policy Server (loads pi0.5 model, runs JAX inference on GPU 0)
     ↕  WebSocket (ws://0.0.0.0:8000)
Terminal 2: LIBERO Client  (runs MuJoCo sim on GPU 1, sends obs, receives actions, saves videos)
```

Two separate Python environments:
- Server: Python 3.11+ (main openpi venv, managed by `uv`)
- Client: Python 3.8 (separate venv at `examples/libero/.venv`)

## One-Time Setup

### 1. Submodules
```bash
git submodule update --init --recursive
```

### 2. Server environment (main openpi)
```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

### 3. Client environment (LIBERO, Python 3.8)
```bash
uv venv --python 3.8 examples/libero/.venv
source examples/libero/.venv/bin/activate
uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match
uv pip install -e packages/openpi-client
uv pip install -e third_party/libero

# Install wandb monitoring (use venv python directly since pip may point to system)
examples/libero/.venv/bin/python -m pip install wandb sentry_sdk
```

### 4. Wandb login (one-time, from any venv — saves to ~/.netrc)
```bash
wandb login
# Get API key from https://wandb.ai/authorize
```

## SLURM Allocation

Request 2 GPUs to avoid GPU contention (server inference + MuJoCo rendering):
```bash
salloc --gres=gpu:2 --mem=64G --time=4:00:00
```

## Running Inference

Use `tmux` inside the SLURM session to get two terminals:
```bash
tmux
# Ctrl+b then % to split vertically
# Ctrl+b then arrow keys to switch panes
# Ctrl+b then z to zoom/unzoom a pane
# Ctrl+b then [ to scroll, q to exit scroll
```

### Terminal 1 — Policy Server
```bash
uv run scripts/serve_policy.py --env LIBERO
```
This loads the `pi05_libero` config and checkpoint by default.
Checkpoints are cached to `~/scratch/openpi` (set in `serve_policy.py`).

### Terminal 2 — LIBERO Client
```bash
source examples/libero/.venv/bin/activate
python examples/libero/main.py --args.task-suite-name libero_spatial --args.num-trials-per-task 5
```

## Key Scripts

| Script | What it does |
|--------|-------------|
| `scripts/serve_policy.py` | Starts the WebSocket policy server. Loads model, runs inference on GPU. |
| `examples/libero/main.py` | LIBERO evaluation client. Runs sim, queries server, saves videos. |
| `examples/libero/convert_libero_data_to_lerobot.py` | Converts raw LIBERO data to LeRobot format (for training, not inference). |

## Configuring Task IDs

Edit the loop in `examples/libero/main.py` (line ~92):
```python
# Run specific tasks:
for task_id in tqdm.tqdm([0, 1, 5]):

# Run all tasks in the suite:
for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
```

## Videos

Saved to `data/libero/videos/` (configurable via `--args.video-out-path`).
Naming: `rollout_{task_description}_ep{episode_idx}_{success|failure}.mp4`

## What the pi05_libero Checkpoint Was Trained On

Trained on 4 suites (40 tasks total): libero_spatial, libero_object, libero_goal, libero_10.
**NOT trained on libero_90** — running on libero_90 tests generalization only.

## Task Suites & Max Steps

| Suite | # Tasks | Max Steps | CLI arg |
|-------|---------|-----------|---------|
| libero_spatial | 10 | 220 | `--args.task-suite-name libero_spatial` |
| libero_object | 10 | 280 | `--args.task-suite-name libero_object` |
| libero_goal | 10 | 300 | `--args.task-suite-name libero_goal` |
| libero_10 | 10 | 520 | `--args.task-suite-name libero_10` |
| libero_90 | 90 | 400 | `--args.task-suite-name libero_90` |

## Environment Variables (set automatically in scripts)

- `OPENPI_DATA_HOME=~/scratch/openpi` — checkpoint cache dir (set in `serve_policy.py`)
- `MUJOCO_GL=egl` — headless rendering backend (set in `main.py`)
- `MUJOCO_EGL_DEVICE_ID=1` — render on GPU 1 so server keeps GPU 0 (set in `main.py`)

## Activation Collection (from Teleop Data)

Captures VLM hidden states (all 18 layers, 2048-dim) during inference on recorded teleop demos.

### Terminal 1 — Server (add `--collect-activations`)
```bash
uv run scripts/serve_policy.py --env LIBERO --collect-activations
```

### Terminal 2 — Client
```bash
# Single file
uv run scripts/convert_teleop_for_inference.py \
    --args.data-dir steering-data/cobalt/cubby-back/some_file.hdf5 \
    --args.save-activations --args.activations-dir steering-data/cobalt/activations

# Whole directory
uv run scripts/convert_teleop_for_inference.py \
    --args.data-dir steering-data/cobalt/cubby-back \
    --args.save-activations --args.activations-dir steering-data/cobalt/activations
```

Output: one `.npz` per inference step at `{activations_dir}/{demo_stem}_t{step:04d}.npz`.
Each contains keys `layer_0`–`layer_17`, values are `(seq_len, 2048)` float16 arrays.

## Activation Steering

Adds steering vectors to VLM hidden states at chosen layers' post-MLP residual during inference.
Supports steering across a contiguous range of layers simultaneously.

### 1. Prepare a steering vector matrix
Save a 2-D array of shape `(18, 2048)` — one vector per layer — as `.pt` or `.npy`:
```python
import numpy as np
# sv_matrix[i] is the steering vector for layer i (0-indexed)
np.save("steer.npy", sv_matrix.astype(np.float32))  # shape: (18, 2048)
```

### 2. Launch server with steering
```bash
uv run scripts/serve_policy.py --env LIBERO \
    --steering-vector steer.npy \
    --steering-layers 10 12 \
    --steering-scale 1.0
```
This steers layers 10, 11, and 12 (inclusive range).

### 3. Run the client (normal pi0.5 inference)
The client side is unchanged — steering is entirely server-side. Run the client
exactly as you would for normal inference:
```bash
# LIBERO example
source examples/libero/.venv/bin/activate
python examples/libero/main.py --args.task-suite-name libero_spatial --args.num-trials-per-task 5 --args.video-out-path interp/videos
```
The client sends observations + a task prompt; the server applies steering transparently.

### Custom prompts
The client sends a `"prompt"` key with each observation (e.g. the task description in LIBERO).
To use a custom prompt:
- **Client-side (recommended):** Change the `"prompt"` value in the client's observation dict
  (e.g. in `examples/libero/main.py` line ~155).
- **Server-side fallback:** Use `--default-prompt "your prompt"` on the server. This only
  takes effect when the client does NOT send a `"prompt"` key in the observation data.

### CLI flags
| Flag | Default | Description |
|------|---------|-------------|
| `--steering-vector` | `None` | Path to `.pt`/`.npy` file, shape `(18, 2048)`. Omit to disable. |
| `--steering-layers` | `10 12` | Layer range `[start, end]` inclusive (0–17). |
| `--steering-scale` | `1.0` | Multiplier on the vector before adding. |
| `--steering-site` | `post_mlp_residual` | Where in the layer to intervene. |
| `--steering-component` | `vlm` | `vlm` (2048-dim) or `action_expert` (1024-dim). |

### How it works
The `(18, 2048)` matrix is passed as `steering_params` through `sample_kwargs` → `Pi0.sample_actions` → `gemma.Module` → `nn.scan(Block)`. Inside `Block.__call__`, the current layer's vector is indexed from the matrix, and a `jnp.where` conditional adds `scale * vector` only for layers within `[layer_start, layer_end]` and the target expert. When disabled, default no-op params (`layer_start=-1, layer_end=-1, scale=0`) ensure zero overhead.

## Troubleshooting

- **"Aborted" crash mid-episode**: GPU contention. Make sure you have 2 GPUs and `MUJOCO_EGL_DEVICE_ID=1`.
- **Stuck on "Starting episode"**: MuJoCo can't render. Check `MUJOCO_GL=egl` is set.
- **Slow first few inference calls**: Normal — JAX JIT compilation. Subsequent calls are fast.
- **osmesa errors**: `libOSMesa.so` not installed on cluster. Use `egl` instead.
- **pip installs to user site**: Use `examples/libero/.venv/bin/python -m pip install ...` directly.
