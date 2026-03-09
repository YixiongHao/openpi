import os

os.environ["OPENPI_DATA_HOME"] = os.path.expanduser("~/scratch/openpi")

import dataclasses
import enum
import logging
import socket
from typing import Literal

import numpy as np
import tyro

from openpi.models_pytorch.activation_steering import SteeringConfig, apply_steering, load_steering_vector
from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.ALOHA_SIM

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)

    # --- Activation steering ---
    # Path to a steering vector file (.pt or .npy) of shape (18, 2048).
    # Each row is the steering vector for the corresponding layer (0-indexed).
    steering_vector: str | None = None
    # Layer range [start, end] (0-indexed, inclusive) at which to inject steering.
    # Both VLM and action expert have 18 layers (0..17). E.g. [10, 12] steers layers 10, 11, 12.
    steering_layers: list[int] = dataclasses.field(default_factory=lambda: [10, 12])
    # Intervention site inside the layer.
    steering_site: Literal["post_mlp_residual"] = "post_mlp_residual"
    # Multiplicative scale applied to the steering vector.
    steering_scale: float = 1.0
    # Which model component to steer.
    steering_component: Literal["vlm", "action_expert"] = "vlm"

    # --- Activation collection ---
    # If True, capture VLM hidden states from all 18 layers and include them
    # in the inference response under the key "vlm_activations".
    collect_activations: bool = False


# Default checkpoints that should be used for each environment.
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi05_aloha",
        dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi05_droid",
        dir="gs://openpi-assets/checkpoints/pi05_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi05_libero",
        dir="gs://openpi-assets/checkpoints/pi05_libero",
    ),
}


def _parse_steering_layers(layers: list[int]) -> tuple[int, int]:
    """Parse the steering layers range [start, end] (inclusive)."""
    if len(layers) != 2:
        raise ValueError(f"--steering-layers must be exactly 2 integers [start, end], got {layers}")
    start, end = layers
    if not (0 <= start <= end <= 17):
        raise ValueError(f"--steering-layers must satisfy 0 <= start <= end <= 17, got [{start}, {end}]")
    return start, end


def _build_sample_kwargs(args: Args) -> dict:
    """Build sample_kwargs dict, including steering_params if configured."""
    sample_kwargs: dict = {}
    if args.steering_vector is None:
        return sample_kwargs

    if args.steering_site != "post_mlp_residual":
        raise ValueError(f"Unsupported steering site {args.steering_site!r}. Only 'post_mlp_residual' is supported.")

    layer_start, layer_end = _parse_steering_layers(args.steering_layers)
    sv_matrix = load_steering_vector(args.steering_vector)  # (18, 2048)
    expert_idx = 0 if args.steering_component == "vlm" else 1
    logging.info(
        "Activation steering: vector=%s, layers=[%d, %d], scale=%.4f, component=%s",
        args.steering_vector, layer_start, layer_end, args.steering_scale, args.steering_component,
    )

    # For JAX: steering_params is a tuple of JAX arrays passed via sample_kwargs.
    # For PyTorch: hooks are registered separately after policy creation.
    import jax.numpy as jnp

    sv_np = sv_matrix.numpy() if hasattr(sv_matrix, 'numpy') else np.array(sv_matrix)
    sample_kwargs["steering_params"] = (
        jnp.array(sv_np),                # (18, hidden_dim)
        jnp.int32(layer_start),
        jnp.int32(layer_end),
        jnp.float32(args.steering_scale),
        jnp.int32(expert_idx),
    )
    return sample_kwargs


def create_default_policy(
    env: EnvMode, *, default_prompt: str | None = None, sample_kwargs: dict | None = None
) -> _policy.Policy:
    """Create a default policy for the given environment."""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config),
            checkpoint.dir,
            default_prompt=default_prompt,
            sample_kwargs=sample_kwargs,
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def create_policy(args: Args, sample_kwargs: dict | None = None) -> _policy.Policy:
    """Create a policy from the given arguments."""
    match args.policy:
        case Checkpoint():
            return _policy_config.create_trained_policy(
                _config.get_config(args.policy.config),
                args.policy.dir,
                default_prompt=args.default_prompt,
                sample_kwargs=sample_kwargs,
            )
        case Default():
            return create_default_policy(args.env, default_prompt=args.default_prompt, sample_kwargs=sample_kwargs)


class ActivationCollectorPolicy:
    """Thin wrapper that reads the global ActivationCollector after each infer() call
    and attaches the captured VLM hidden states to the response dict."""

    def __init__(self, inner_policy, collector):
        self._inner = inner_policy
        self._collector = collector
        self.metadata = getattr(inner_policy, "metadata", {})

    def infer(self, obs):
        result = self._inner.infer(obs)
        activations = self._collector.get_and_clear()
        if activations:
            result["vlm_activations"] = {
                f"layer_{k}": v[0, -1].astype(np.float16)  # remove batch dim, take last token
                for k, v in sorted(activations.items())
            }
        return result


def main(args: Args) -> None:
    sample_kwargs = _build_sample_kwargs(args)
    policy = create_policy(args, sample_kwargs=sample_kwargs or None)
    policy_metadata = policy.metadata

    # For PyTorch models, also register forward hooks (JAX uses sample_kwargs instead).
    if args.steering_vector is not None and policy._is_pytorch_model:
        layer_start, layer_end = _parse_steering_layers(args.steering_layers)
        steering_cfg = SteeringConfig(
            vector_path=args.steering_vector,
            layer_start=layer_start,
            layer_end=layer_end,
            site=args.steering_site,
            scale=args.steering_scale,
            component=args.steering_component,
        )
        apply_steering(policy._model, steering_cfg)

    # Collect VLM activations if requested (JAX only — uses jax.debug.callback).
    if args.collect_activations:
        from openpi.models.gemma import ActivationCollector, set_activation_collector

        collector = ActivationCollector()
        set_activation_collector(collector)
        policy = ActivationCollectorPolicy(policy, collector)
        logging.info("Activation collection enabled for all 18 VLM layers")

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
