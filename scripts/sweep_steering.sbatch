#!/bin/bash
#SBATCH --job-name=steer-sweep
#SBATCH --output=logs/sweep_%A_%a.out
#SBATCH --error=logs/sweep_%A_%a.err
#SBATCH --gres=gpu:2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --array=0-539  # 36 layer pairs x 15 scales = 540 configs

# =============================================================================
# Steering Hyperparameter Sweep
# =============================================================================
# Each SLURM array task runs ONE (layers, scale) configuration.
# The policy server + libero client run sequentially within each task.
#
# Usage:
#   mkdir -p logs
#   sbatch scripts/sweep_steering.sbatch
#
# To run a subset:
#   sbatch --array=0-3 scripts/sweep_steering.sbatch
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# SWEEP CONFIGURATION
# ---------------------------------------------------------------------------
# Grid is built dynamically: all (start, end) layer pairs from 0–16 step 2
# where end > start, crossed with all scales below.

# LAYER_VALS=( 0 2 4 6 8 10 12 14 16 )
# SCALE_VALS=( 1 5 10 15 20 25 30 35 40 45 50 )

# Build flat config arrays from the grid
LAYER_STARTS=()
LAYER_ENDS=()
SCALES=()
for s in "${LAYER_VALS[@]}"; do
    for e in "${LAYER_VALS[@]}"; do
        if (( e > s )); then
            for sc in "${SCALE_VALS[@]}"; do
                LAYER_STARTS+=( "$s" )
                LAYER_ENDS+=( "$e" )
                SCALES+=( "$sc" )
            done
        fi
    done
done
# Total configs: 36 layer pairs x 15 scales = 540

# ---------------------------------------------------------------------------
# FIXED SETTINGS  — change as needed
# ---------------------------------------------------------------------------
STEERING_VECTOR="steering-data/vec-no-wrist/front-minus-back.npz"
TASK_SUITE="libero_custom"
NUM_TRIALS=20
TASK_IDS="0"                # space-separated task IDs (leave empty for all)
VIDEO_BASE="interp/videos"    # videos saved per-config under this dir
SAVE_VIDEOS=false             # set to true to save rollout mp4s
RESULTS_DIR="interp/results"  # per-config JSON results saved here
PORT=8000
SERVER_READY_TIMEOUT=300      # seconds to wait for server startup

# ---------------------------------------------------------------------------
# Derived variables
# ---------------------------------------------------------------------------
IDX=${SLURM_ARRAY_TASK_ID}
LAYER_START=${LAYER_STARTS[$IDX]}
LAYER_END=${LAYER_ENDS[$IDX]}
SCALE=${SCALES[$IDX]}
CONFIG_TAG="L${LAYER_START}-${LAYER_END}_S${SCALE}"
VIDEO_DIR="${VIDEO_BASE}/${CONFIG_TAG}"

mkdir -p logs "${VIDEO_DIR}"

echo "============================================"
echo "Array task  : ${IDX}"
echo "Config      : ${CONFIG_TAG}"
echo "Layers      : ${LAYER_START}–${LAYER_END}"
echo "Scale       : ${SCALE}"
echo "Vector      : ${STEERING_VECTOR}"
echo "Suite       : ${TASK_SUITE}"
echo "Trials/task : ${NUM_TRIALS}"
echo "Task IDs    : ${TASK_IDS:-all}"
echo "Video dir   : ${VIDEO_DIR}"
echo "Host        : $(hostname)"
echo "GPUs        : ${CUDA_VISIBLE_DEVICES:-not set}"
echo "============================================"

# ---------------------------------------------------------------------------
# 1. Launch policy server in background
# ---------------------------------------------------------------------------
echo "[$(date +%T)] Starting policy server on port ${PORT}..."

uv run scripts/serve_policy.py --env LIBERO \
    --steering-vector "${STEERING_VECTOR}" \
    --steering-layers ${LAYER_START} ${LAYER_END} \
    --steering-scale "${SCALE}" \
    --port "${PORT}" \
    &
SERVER_PID=$!

# Ensure server is killed on exit (success or failure)
cleanup() {
    echo "[$(date +%T)] Cleaning up server (PID ${SERVER_PID})..."
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# 2. Wait for server to be ready (probe WebSocket port)
# ---------------------------------------------------------------------------
echo "[$(date +%T)] Waiting for server on port ${PORT}..."
ELAPSED=0
while ! ss -tlnp | grep -q ":${PORT} "; do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "ERROR: Server process died before becoming ready."
        exit 1
    fi
    if (( ELAPSED >= SERVER_READY_TIMEOUT )); then
        echo "ERROR: Server did not start within ${SERVER_READY_TIMEOUT}s."
        exit 1
    fi
    sleep 5
    ELAPSED=$((ELAPSED + 5))
done
echo "[$(date +%T)] Server ready after ~${ELAPSED}s."

# ---------------------------------------------------------------------------
# 3. Run LIBERO client
# ---------------------------------------------------------------------------
echo "[$(date +%T)] Starting LIBERO client..."

# Build client args
RESULTS_FILE="${RESULTS_DIR}/${CONFIG_TAG}.json"
mkdir -p "${RESULTS_DIR}"

CLIENT_ARGS=(
    --args.task-suite-name "${TASK_SUITE}"
    --args.num-trials-per-task "${NUM_TRIALS}"
    --args.video-out-path "${VIDEO_DIR}"
    --args.port "${PORT}"
    --args.results-out "${RESULTS_FILE}"
)

if [[ -n "${TASK_IDS}" ]]; then
    CLIENT_ARGS+=( --args.task-ids ${TASK_IDS} )
fi

if [[ "${SAVE_VIDEOS}" != "true" ]]; then
    CLIENT_ARGS+=( --args.no-save-videos )
fi

source examples/libero/.venv/bin/activate
python examples/libero/main.py "${CLIENT_ARGS[@]}"
CLIENT_EXIT=$?

echo "[$(date +%T)] Client finished with exit code ${CLIENT_EXIT}."
echo "[$(date +%T)] Config ${CONFIG_TAG} complete."
exit ${CLIENT_EXIT}
