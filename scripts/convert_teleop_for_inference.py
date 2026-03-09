"""Convert teleop HDF5 data to the dictionary format expected by the Pi inference server.

Loads HDF5 demos from a directory, constructs the observation dict the server expects,
and sends each timestep for inference to collect activations/actions.

Usage:
    # Dry run - just validate data without sending to server
    python scripts/convert_teleop_for_inference.py \
        --data-dir steering-data/cobalt/cubby-back \
        --dry-run

    # Run inference and save results
    python scripts/convert_teleop_for_inference.py \
        --data-dir steering-data/cobalt/cubby-back \
        --host 0.0.0.0 --port 8000 \
        --output-dir steering-data/cobalt/cubby-back-results
"""
import os
os.environ["PYTHONPATH"] = os.environ.get("PYTHONPATH", "") + ":" + os.path.join(os.getcwd(), "third_party/libero")
import pathlib
import dataclasses
import glob
import logging
import math
import h5py
import numpy as np
import tqdm
import tyro
import wandb


@dataclasses.dataclass
class Args:
    # Path to directory containing teleop HDF5 files
    data_dir: str = "steering-data/cobalt/cubby-back"
    # Task prompt to send to the model
    prompt: str = "pick up the book and place it in the back compartment of the caddy"
    # Replan interval (send observation every N steps, execute N actions from the chunk)
    replan_steps: int = 5

    # Server connection
    host: str = "0.0.0.0"
    port: int = 8000

    # Output
    output_dir: str = ""  # If empty, defaults to {data_dir}-results
    save_results: bool = False  # If True, save predicted vs ground-truth actions per demo
    dry_run: bool = False  # If True, just validate data without sending to server

    # Activation collection
    save_activations: bool = False  # If True, save VLM activations returned by server
    activations_dir: str = ""  # Directory to save activations; defaults to steering-data/cobalt/activations


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion (x, y, z, w) to axis-angle representation.

    Copied from robosuite to match the conversion used in examples/libero/main.py.
    """
    quat = quat.copy()
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def load_chunked_dataset(group: h5py.Group) -> np.ndarray:
    """Load a chunked HDF5 dataset by concatenating all numbered chunks in order."""
    chunk_names = sorted(group.keys())
    chunks = [np.array(group[name]) for name in chunk_names]
    return np.concatenate(chunks, axis=0)


def load_demo(filepath: str) -> dict:
    """Load a single HDF5 teleop demo and return raw data arrays."""
    with h5py.File(filepath, "r") as f:
        demo = f["data/demo_0"]

        # Load images (chunked directly under images/)
        images = load_chunked_dataset(demo["images"])

        # Load observations (each obs key is a chunked subgroup)
        eef_pos = load_chunked_dataset(demo["obs/robot0_eef_pos"])
        eef_quat = load_chunked_dataset(demo["obs/robot0_eef_quat"])
        gripper_qpos = load_chunked_dataset(demo["obs/robot0_gripper_qpos"])

        # Load actions (chunked)
        actions = load_chunked_dataset(demo["additional_data/robot_controls/actions"])

        # Load metadata
        metadata = demo.attrs.get("metadata", "{}")

    num_steps = len(images)
    assert len(eef_pos) == num_steps, f"eef_pos length {len(eef_pos)} != images {num_steps}"
    assert len(eef_quat) == num_steps, f"eef_quat length {len(eef_quat)} != images {num_steps}"
    assert len(gripper_qpos) == num_steps, f"gripper_qpos length {len(gripper_qpos)} != images {num_steps}"

    return {
        "images": images,            # (T, 224, 224, 3) uint8
        "eef_pos": eef_pos,          # (T, 3) float64
        "eef_quat": eef_quat,        # (T, 4) float64
        "gripper_qpos": gripper_qpos, # (T, 2) float64
        "actions": actions,          # (T, 7) float64
        "metadata": metadata,
        "num_steps": num_steps,
    }


def build_observation(
    image: np.ndarray,
    eef_pos: np.ndarray,
    eef_quat: np.ndarray,
    gripper_qpos: np.ndarray,
    prompt: str,
) -> dict:
    """Build the observation dict expected by the inference server.

    Expected format:
        {
            "observation/image": uint8 (224, 224, 3),
            "observation/wrist_image": uint8 (224, 224, 3),  # zeros - no wrist cam
            "observation/state": float32 (8,),  # [eef_pos(3), axisangle(3), gripper_qpos(2)]
            "prompt": str,
        }
    """
    # Rotate image 180 degrees to match training preprocessing
    # (the eval script in examples/libero/main.py does this)
    rotated_image = np.ascontiguousarray(image[::-1, ::-1])

    # Construct 8D state vector: [eef_pos(3), axis_angle(3), gripper_qpos(2)]
    axis_angle = quat2axisangle(eef_quat)
    state = np.concatenate([
        eef_pos.astype(np.float32),
        axis_angle.astype(np.float32),
        gripper_qpos.astype(np.float32),
    ])

    # Wrist image is zeros since teleop data doesn't have a wrist camera
    wrist_image = np.zeros_like(rotated_image)

    return {
        "observation/image": rotated_image,
        "observation/wrist_image": wrist_image,
        "observation/state": state,
        "prompt": prompt,
    }


def validate_observation(obs: dict) -> None:
    """Validate that an observation matches the expected format."""
    img = obs["observation/image"]
    assert img.shape == (224, 224, 3), f"Image shape {img.shape}, expected (224, 224, 3)"
    assert img.dtype == np.uint8, f"Image dtype {img.dtype}, expected uint8"

    wrist = obs["observation/wrist_image"]
    assert wrist.shape == (224, 224, 3), f"Wrist image shape {wrist.shape}"
    assert wrist.dtype == np.uint8, f"Wrist image dtype {wrist.dtype}"

    state = obs["observation/state"]
    assert state.shape == (8,), f"State shape {state.shape}, expected (8,)"
    assert state.dtype == np.float32, f"State dtype {state.dtype}, expected float32"

    assert isinstance(obs["prompt"], str), "Prompt must be a string"


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO)

    wandb.init(
        project="openpi-libero",
        name="activation-collection",
        config=dataclasses.asdict(args),
    )

    data_path = pathlib.Path(args.data_dir)
    if data_path.is_file() and data_path.suffix == ".hdf5":
        hdf5_files = [str(data_path)]
        default_output = f"{data_path.parent}-results"
    else:
        hdf5_files = sorted(glob.glob(str(data_path / "*.hdf5")))
        default_output = f"{args.data_dir}-results"
    if not hdf5_files:
        logging.error(f"No HDF5 files found at {data_path}")
        return

    logging.info(f"Found {len(hdf5_files)} HDF5 file(s)")

    output_dir = pathlib.Path(args.output_dir or default_output)

    # Connect to server (unless dry run)
    client = None
    if not args.dry_run:
        from openpi_client import websocket_client_policy as wcp
        client = wcp.WebsocketClientPolicy(args.host, args.port)
        if args.save_results:
            output_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Saving results to {output_dir}")
        logging.info(f"Connected to server at {args.host}:{args.port}")

    for file_idx, filepath in enumerate(hdf5_files):
        filename = os.path.basename(filepath)
        logging.info(f"\n[{file_idx+1}/{len(hdf5_files)}] Loading {filename}")

        demo = load_demo(filepath)
        num_steps = demo["num_steps"]
        logging.info(f"  {num_steps} timesteps")

        # Validate first observation
        obs = build_observation(
            demo["images"][0],
            demo["eef_pos"][0],
            demo["eef_quat"][0],
            demo["gripper_qpos"][0],
            args.prompt,
        )
        validate_observation(obs)

        if args.dry_run:
            # Just validate all timesteps
            logging.info(f"  Validating all {num_steps} timesteps...")
            for t in range(num_steps):
                obs = build_observation(
                    demo["images"][t],
                    demo["eef_pos"][t],
                    demo["eef_quat"][t],
                    demo["gripper_qpos"][t],
                    args.prompt,
                )
                validate_observation(obs)
            logging.info(f"  All timesteps valid.")
            continue

        # Run inference, replanning every replan_steps
        all_predicted_actions = []
        # Collect activations per layer: {layer_key: [vec_t0, vec_t1, ...]}
        demo_activations: dict[str, list[np.ndarray]] = {}
        replan_timesteps = list(range(0, num_steps, args.replan_steps))

        for t in tqdm.tqdm(replan_timesteps, desc=f"  Inference ({filename})"):
            obs = build_observation(
                demo["images"][t],
                demo["eef_pos"][t],
                demo["eef_quat"][t],
                demo["gripper_qpos"][t],
                args.prompt,
            )
            result = client.infer(obs)
            action_chunk = result["actions"]  # (10, 7)
            all_predicted_actions.append({
                "timestep": t,
                "predicted_actions": action_chunk,
            })

            # Accumulate VLM activations per layer
            if args.save_activations and "vlm_activations" in result:
                for key, vec in result["vlm_activations"].items():
                    demo_activations.setdefault(key, []).append(vec)

        # Save all activations for this demo in one file
        # Each layer key maps to shape (num_replan_steps, 2048)
        if args.save_activations and demo_activations:
            act_dir = pathlib.Path(
                args.activations_dir or "steering-data/cobalt/activations"
            )
            act_dir.mkdir(parents=True, exist_ok=True)
            demo_name = pathlib.Path(filename).stem
            np.savez(
                act_dir / f"{demo_name}.npz",
                timesteps=np.array(replan_timesteps),
                **{k: np.stack(v) for k, v in demo_activations.items()},
            )
            logging.info(f"  Saved activations to {act_dir / demo_name}.npz")

        # Save predicted vs ground-truth actions (optional)
        if args.save_results:
            result_path = output_dir / f"{pathlib.Path(filename).stem}_results.npz"
            np.savez(
                result_path,
                replan_timesteps=np.array([r["timestep"] for r in all_predicted_actions]),
                predicted_actions=np.array([r["predicted_actions"] for r in all_predicted_actions]),
                gt_actions=demo["actions"],
                eef_pos=demo["eef_pos"],
                eef_quat=demo["eef_quat"],
                gripper_qpos=demo["gripper_qpos"],
            )
            logging.info(f"  Saved results to {result_path}")

    logging.info("\nDone!")
    wandb.finish()


if __name__ == "__main__":
    tyro.cli(main)
