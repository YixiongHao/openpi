"""Activation steering for the VLM backbone (or action expert) during inference.

Usage from the inference server CLI::

    python scripts/serve_policy.py ... \
        --steering_vector /path/to/vector.pt \
        --steering_layers 10 12 \
        --steering_scale 1.0

The steering vector file should contain a 2-D tensor of shape ``(18, 2048)``
saved as either a ``.pt`` (PyTorch) or ``.npy`` (NumPy) file.  Each row is
the steering vector for the corresponding layer (0-indexed).

The vectors are added to every token's hidden state at the **post-MLP residual**
of each target layer (within the specified range) during the forward pass.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Literal

import torch
from torch import nn

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class SteeringConfig:
    """Configuration for a multi-layer activation-steering intervention."""

    # Path to the steering vector file (.pt or .npy), shape (18, 2048).
    vector_path: str
    # Start of layer range (0-indexed, inclusive).
    layer_start: int
    # End of layer range (0-indexed, inclusive).
    layer_end: int
    # Intervention site inside the layer.
    site: Literal["post_mlp_residual"] = "post_mlp_residual"
    # Multiplicative scale applied to the steering vector before addition.
    scale: float = 1.0
    # Which component to steer: the VLM backbone or the action expert.
    component: Literal["vlm", "action_expert"] = "vlm"


def load_steering_vector(path: str) -> torch.Tensor:
    """Load a steering vector matrix from disk.

    Supports ``.pt``, ``.npy``, and ``.npz`` (with keys ``layer_0``..``layer_17``) formats.
    Returns a 2-D CPU tensor of shape ``(18, hidden_dim)``.
    """
    if path.endswith(".pt") or path.endswith(".pth"):
        vec = torch.load(path, map_location="cpu", weights_only=True)
    elif path.endswith(".npy"):
        import numpy as np

        vec = torch.from_numpy(np.load(path))
    elif path.endswith(".npz"):
        import numpy as np

        data = np.load(path)
        vec = torch.from_numpy(
            np.stack([data[f"layer_{i}"] for i in range(18)])
        )
    else:
        raise ValueError(
            f"Unsupported steering vector format: {path!r}. Use .pt, .npy, or .npz."
        )

    if vec.ndim != 2 or vec.shape[0] != 18:
        raise ValueError(
            f"Steering vector must be 2-D with shape (18, hidden_dim), "
            f"got shape {tuple(vec.shape)}."
        )
    return vec


def _get_target_layers(model: nn.Module, component: str) -> nn.ModuleList:
    """Resolve the ``nn.ModuleList`` of transformer layers for *component*."""
    if component == "vlm":
        return model.paligemma_with_expert.paligemma.language_model.model.layers
    elif component == "action_expert":
        return model.paligemma_with_expert.gemma_expert.model.layers
    else:
        raise ValueError(
            f"Unknown steering component {component!r}. Use 'vlm' or 'action_expert'."
        )


def apply_steering(
    model: nn.Module,
    config: SteeringConfig,
) -> list[torch.utils.hooks.RemovableHandle]:
    """Register forward hooks that add steering vectors at the target layers.

    Args:
        model: A ``PI0Pytorch`` model instance.
        config: The steering configuration (with layer_start..layer_end range).

    Returns:
        A list of ``RemovableHandle`` handles that can be used to later remove the hooks.
    """
    sv_matrix = load_steering_vector(config.vector_path)  # (18, hidden_dim)
    layers = _get_target_layers(model, config.component)
    num_layers = len(layers)

    if config.layer_start < 0 or config.layer_end >= num_layers:
        raise ValueError(
            f"Steering layer range [{config.layer_start}, {config.layer_end}] is out of range "
            f"for {config.component} which has {num_layers} layers (valid: 0..{num_layers - 1})."
        )

    if config.site != "post_mlp_residual":
        raise ValueError(
            f"Unsupported steering site {config.site!r}. "
            "Currently only 'post_mlp_residual' is supported."
        )

    scale = config.scale
    handles = []

    for layer_idx in range(config.layer_start, config.layer_end + 1):
        target_layer = layers[layer_idx]
        vector = sv_matrix[layer_idx]  # (hidden_dim,)

        # Validate hidden dim.
        expected_dim = target_layer.hidden_size
        if vector.shape[0] != expected_dim:
            raise ValueError(
                f"Steering vector dim {vector.shape[0]} does not match "
                f"{config.component} hidden size {expected_dim}."
            )

        def _make_hook(sv: torch.Tensor) -> callable:
            def _post_mlp_residual_hook(
                module: nn.Module,
                input: tuple,
                output: tuple,
            ) -> tuple:
                hidden_states = output[0]
                v = sv.to(device=hidden_states.device, dtype=hidden_states.dtype)
                hidden_states = hidden_states + scale * v
                return (hidden_states,) + output[1:]
            return _post_mlp_residual_hook

        handle = target_layer.register_forward_hook(_make_hook(vector))
        handles.append(handle)
        logger.info(
            "Activation steering: registered hook on %s layer %d (site=%s, scale=%.4f, vector=%s)",
            config.component,
            layer_idx,
            config.site,
            config.scale,
            config.vector_path,
        )

    return handles
