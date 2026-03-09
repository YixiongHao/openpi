"""Plot how activation differences evolve over rollout progress.

Reads a per-progress .npz file (from compute_steering_vector.py) and plots
the L2 norm of the difference vector at each progress point, per layer.

Usage:
    python interp/utils/plot_steering_progression.py \
        --input steering-vectors/back_minus_front_per_progress.npz \
        --output interp/plots/steering_progression.png

    # Only plot specific layers:
    python interp/utils/plot_steering_progression.py \
        --input steering-vectors/back_minus_front_per_progress.npz \
        --layers 8 9 10 11 12 \
        --output interp/plots/steering_progression.png
"""

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np


NUM_LAYERS = 18


def main():
    parser = argparse.ArgumentParser(description="Plot steering vector difference over rollout progress")
    parser.add_argument("--input", required=True, help="Path to *_per_progress.npz file")
    parser.add_argument("--output", default="interp/plots/steering_progression.png", help="Output plot path")
    parser.add_argument("--layers", type=int, nargs="+", default=None, help="Layer indices to plot (default: all)")
    parser.add_argument("--metric", choices=["l2", "cosine_shift", "mean_abs"], default="l2",
                        help="Metric to plot (default: l2 norm)")
    args = parser.parse_args()

    data = dict(np.load(args.input))

    layers = args.layers if args.layers is not None else list(range(NUM_LAYERS))
    progress = np.linspace(0, 100, len(data[f"layer_{layers[0]}"]))  # percentage

    fig, ax = plt.subplots(figsize=(12, 6))

    cmap = plt.cm.viridis
    colors = [cmap(i / max(len(layers) - 1, 1)) for i in range(len(layers))]

    for i, layer_idx in enumerate(layers):
        key = f"layer_{layer_idx}"
        diff = data[key]  # (num_progress_points, 2048)

        if args.metric == "l2":
            values = np.linalg.norm(diff, axis=1)
            ylabel = "L2 norm of difference"
        elif args.metric == "mean_abs":
            values = np.mean(np.abs(diff), axis=1)
            ylabel = "Mean absolute difference"
        elif args.metric == "cosine_shift":
            # Cosine similarity between consecutive progress points
            norms = np.linalg.norm(diff, axis=1, keepdims=True)
            normed = diff / np.maximum(norms, 1e-8)
            cos_sim = np.sum(normed[:-1] * normed[1:], axis=1)
            values = 1 - cos_sim  # cosine distance
            progress_trimmed = progress[:-1]
            ax.plot(progress_trimmed, values, label=f"Layer {layer_idx}", color=colors[i], alpha=0.8)
            continue

        ax.plot(progress, values, label=f"Layer {layer_idx}", color=colors[i], alpha=0.8)

    ax.set_xlabel("Rollout progress (%)")
    ax.set_ylabel(ylabel if args.metric != "cosine_shift" else "Cosine distance (consecutive)")
    ax.set_title("Activation difference: folder A − folder B")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 100)

    plt.tight_layout()
    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
