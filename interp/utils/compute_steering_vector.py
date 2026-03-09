"""Compute a steering vector from two folders of activation files.

For each demo, selects activations at timesteps closest to evenly-spaced
percentage points through the rollout (0%, 1%, 2%, ..., 100%), so that
demos of different lengths are aligned on a common progress axis.

The steering vector is: mean(folder_a activations) - mean(folder_b activations),
computed per layer and per progress point, then averaged over demos and progress.

Usage:
    python interp/utils/compute_steering_vector.py \
        --folder-a steering-data/act/cubby-back \
        --folder-b steering-data/act/cubby-front \
        --output steering-vectors/back_minus_front.npz

    # Optionally restrict to a subset of layers or progress range:
    python interp/utils/compute_steering_vector.py \
        --folder-a steering-data/act/cubby-back \
        --folder-b steering-data/act/cubby-front \
        --layers 8 9 10 11 12 \
        --progress-start 0.0 --progress-end 1.0 \
        --output steering-vectors/back_minus_front.npz
"""

import argparse
import glob
import pathlib

import numpy as np


NUM_LAYERS = 18


def load_activations(folder: str) -> list[dict]:
    """Load all .npz activation files from a folder.

    Returns a list of dicts, each with:
        - "timesteps": (N,) int array
        - "layer_0" .. "layer_17": (N, 2048) float arrays
    """
    files = sorted(glob.glob(str(pathlib.Path(folder) / "*.npz")))
    if not files:
        raise FileNotFoundError(f"No .npz files found in {folder}")
    demos = []
    for f in files:
        data = dict(np.load(f))
        demos.append(data)
    print(f"Loaded {len(demos)} demos from {folder}")
    return demos


def select_at_progress(demo: dict, progress_points: np.ndarray) -> dict[str, np.ndarray]:
    """Select activations at timesteps closest to the given progress fractions.

    Args:
        demo: dict with "timesteps" and "layer_*" keys.
        progress_points: 1-D array of fractions in [0, 1].

    Returns:
        dict mapping "layer_0".."layer_17" to (len(progress_points), 2048) arrays.
    """
    timesteps = demo["timesteps"]
    n = len(timesteps)
    # Map progress fractions to indices into the replan-step array
    # progress=0 -> index 0, progress=1 -> index n-1
    float_indices = progress_points * (n - 1)
    indices = np.clip(np.round(float_indices).astype(int), 0, n - 1)

    result = {}
    for layer_idx in range(NUM_LAYERS):
        key = f"layer_{layer_idx}"
        result[key] = demo[key][indices]  # (num_points, 2048)
    return result


def compute_steering_vector(
    folder_a: str,
    folder_b: str,
    *,
    num_points: int = 101,
    layers: list[int] | None = None,
    progress_start: float = 0.0,
    progress_end: float = 1.0,
) -> dict[str, np.ndarray]:
    """Compute mean(A) - mean(B) steering vectors per layer.

    Args:
        folder_a: path to "positive" activations folder.
        folder_b: path to "negative" activations folder.
        num_points: number of evenly-spaced progress points (default 101 = 0%..100%).
        layers: if provided, only compute for these layer indices.
        progress_start: start of progress range to average over.
        progress_end: end of progress range to average over.

    Returns:
        dict with:
            - "layer_0".."layer_17": (2048,) float32 steering vectors
            - "progress_points": (num_points,) the progress fractions used
            - "per_progress": dict of "layer_*" -> (num_points, 2048) mean diff per progress point
    """
    demos_a = load_activations(folder_a)
    demos_b = load_activations(folder_b)

    progress_points = np.linspace(0.0, 1.0, num_points)

    # Filter to requested progress range
    mask = (progress_points >= progress_start) & (progress_points <= progress_end)
    active_points = progress_points[mask]

    if layers is None:
        layers = list(range(NUM_LAYERS))

    # Accumulate per-progress-point activations across demos
    sum_a = {f"layer_{l}": np.zeros((len(active_points), 2048), dtype=np.float64) for l in layers}
    sum_b = {f"layer_{l}": np.zeros((len(active_points), 2048), dtype=np.float64) for l in layers}

    for demo in demos_a:
        selected = select_at_progress(demo, active_points)
        for l in layers:
            key = f"layer_{l}"
            sum_a[key] += selected[key].astype(np.float64)

    for demo in demos_b:
        selected = select_at_progress(demo, active_points)
        for l in layers:
            key = f"layer_{l}"
            sum_b[key] += selected[key].astype(np.float64)

    n_a, n_b = len(demos_a), len(demos_b)

    result = {"progress_points": progress_points}
    per_progress = {}
    for l in layers:
        key = f"layer_{l}"
        mean_a = sum_a[key] / n_a  # (num_active_points, 2048)
        mean_b = sum_b[key] / n_b
        diff = mean_a - mean_b  # (num_active_points, 2048)
        per_progress[key] = diff.astype(np.float32)
        # Average over progress points to get a single steering vector
        result[key] = diff.mean(axis=0).astype(np.float32)  # (2048,)
        norm = np.linalg.norm(result[key])
        print(f"  {key}: norm={norm:.4f}")

    result["per_progress"] = per_progress
    return result


def main():
    parser = argparse.ArgumentParser(description="Compute steering vector from two activation folders")
    parser.add_argument("--folder-a", required=True, help="Positive activations folder")
    parser.add_argument("--folder-b", required=True, help="Negative activations folder")
    parser.add_argument("--output", required=True, help="Output .npz path")
    parser.add_argument("--num-points", type=int, default=101, help="Number of progress points (default: 101)")
    parser.add_argument("--layers", type=int, nargs="+", default=None, help="Layer indices to compute (default: all)")
    parser.add_argument("--progress-start", type=float, default=0.0, help="Start of progress range")
    parser.add_argument("--progress-end", type=float, default=1.0, help="End of progress range")
    args = parser.parse_args()

    result = compute_steering_vector(
        args.folder_a,
        args.folder_b,
        num_points=args.num_points,
        layers=args.layers,
        progress_start=args.progress_start,
        progress_end=args.progress_end,
    )

    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Save flat: layer vectors + progress_points (skip nested per_progress for simplicity)
    save_dict = {k: v for k, v in result.items() if k != "per_progress"}
    np.savez(out_path, **save_dict)
    print(f"\nSaved steering vectors to {out_path}")
    print(f"Keys: {sorted(save_dict.keys())}")

    # Also save per-progress breakdown
    per_progress_path = out_path.with_name(out_path.stem + "_per_progress.npz")
    np.savez(per_progress_path, **result["per_progress"])
    print(f"Saved per-progress breakdown to {per_progress_path}")


if __name__ == "__main__":
    main()
