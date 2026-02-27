"""
Run sim evaluation episodes AND collect an on-policy LeRobot-format dataset.

Mirrors run_sim_eval.py but additionally saves observations + executed actions
into the LeRobot v2.0 format consumed by the training pipeline
(base_48_wan_fine_aug_relative.yaml).

Usage:
  # Launch policy server in a separate terminal, then:
  python eval_utils/run_sim_eval_and_collect_data.py \
      --episodes 10 --headless --output-dir ./data/on_policy_dataset \
      --host localhost --port 6000
"""

import argparse
import json
import uuid
from pathlib import Path

import av
import cv2
import numpy as np
import polars as pl
import torch
import tyro
from tqdm import tqdm

from openpi_client import image_tools
from sim_evals.inference.abstract_client import InferenceClient
from policy_client import WebsocketClientPolicy


# ---------------------------------------------------------------------------
# DreamZeroJointPosClient — copied from run_sim_eval.py
# ---------------------------------------------------------------------------

class DreamZeroJointPosClient(InferenceClient):
    def __init__(
        self,
        remote_host: str = "localhost",
        remote_port: int = 6000,
        open_loop_horizon: int = 8,
    ) -> None:
        self.client = WebsocketClientPolicy(remote_host, remote_port)
        self.open_loop_horizon = open_loop_horizon
        self.actions_from_chunk_completed = 0
        self.pred_action_chunk = None
        self.session_id = str(uuid.uuid4())

    def visualize(self, request: dict):
        curr_obs = self._extract_observation(request)
        right_img = image_tools.resize_with_pad(curr_obs["right_image"], 224, 224)
        wrist_img = image_tools.resize_with_pad(curr_obs["wrist_image"], 224, 224)
        left_img = image_tools.resize_with_pad(curr_obs["left_image"], 224, 224)
        combined = np.concatenate([right_img, wrist_img, left_img], axis=1)
        return combined

    def reset(self):
        self.actions_from_chunk_completed = 0
        self.pred_action_chunk = None
        self.session_id = str(uuid.uuid4())

    def infer(self, obs: dict, instruction: str) -> dict:
        curr_obs = self._extract_observation(obs)
        if (
            self.actions_from_chunk_completed == 0
            or self.actions_from_chunk_completed >= self.open_loop_horizon
        ):
            self.actions_from_chunk_completed = 0
            request_data = {
                "observation/exterior_image_0_left": image_tools.resize_with_pad(curr_obs["right_image"], 180, 320),
                "observation/exterior_image_1_left": image_tools.resize_with_pad(curr_obs["left_image"], 180, 320),
                "observation/wrist_image_left": image_tools.resize_with_pad(curr_obs["wrist_image"], 180, 320),
                "observation/joint_position": curr_obs["joint_position"].astype(np.float64),
                "observation/cartesian_position": np.zeros((6,), dtype=np.float64),
                "observation/gripper_position": curr_obs["gripper_position"].astype(np.float64),
                "prompt": instruction,
                "session_id": self.session_id,
            }

            actions = self.client.infer(request_data)
            assert len(actions.shape) == 2, f"Expected 2D array, got shape {actions.shape}"
            assert actions.shape[-1] == 8, f"Expected 8 action dimensions, got {actions.shape[-1]}"
            self.pred_action_chunk = actions

        action = self.pred_action_chunk[self.actions_from_chunk_completed]
        self.actions_from_chunk_completed += 1

        # binarize gripper action
        if action[-1].item() > 0.5:
            action = np.concatenate([action[:-1], np.ones((1,))])
        else:
            action = np.concatenate([action[:-1], np.zeros((1,))])

        return {"action": action, "raw_obs": curr_obs}

    def _extract_observation(self, obs_dict):
        right_image = obs_dict["policy"]["external_cam"][0].clone().detach().cpu().numpy()
        left_image = obs_dict["policy"]["external_cam_2"][0].clone().detach().cpu().numpy()
        wrist_image = obs_dict["policy"]["wrist_cam"][0].clone().detach().cpu().numpy()
        # Resize images to 320x180 to match the base DROID dataset resolution
        right_image = cv2.resize(right_image, (320, 180), interpolation=cv2.INTER_AREA)
        left_image = cv2.resize(left_image, (320, 180), interpolation=cv2.INTER_AREA)
        wrist_image = cv2.resize(wrist_image, (320, 180), interpolation=cv2.INTER_AREA)
        robot_state = obs_dict["policy"]
        joint_position = robot_state["arm_joint_pos"].clone().detach().cpu().numpy()
        gripper_position = robot_state["gripper_pos"].clone().detach().cpu().numpy()
        return {
            "right_image": right_image,
            "left_image": left_image,
            "wrist_image": wrist_image,
            "joint_position": joint_position,
            "gripper_position": gripper_position,
        }


# ---------------------------------------------------------------------------
# Video encoding — adapted from scripts/data/convert_droid.py
# ---------------------------------------------------------------------------

def encode_video(frames: np.ndarray, output_path: Path, fps: int) -> None:
    """Encode a sequence of uint8 RGB frames to an h264 mp4."""
    options = {
        "threads": "1",
        "thread_type": "slice",
        "preset": "ultrafast",
        "tune": "zerolatency",
        "crf": "23",
    }
    container = av.open(str(output_path), mode="w")
    stream = container.add_stream("h264", rate=fps, options=options)
    stream.width = frames.shape[2]
    stream.height = frames.shape[1]
    stream.pix_fmt = "yuv420p"

    video_frame = av.VideoFrame(width=stream.width, height=stream.height, format="rgb24")
    frame_array = video_frame.to_ndarray(format="rgb24")
    for frame in frames:
        frame_array[:] = frame
        packet = stream.encode(video_frame)
        container.mux(packet)
    packet = stream.encode(None)
    container.mux(packet)
    container.close()


# ---------------------------------------------------------------------------
# Episode saving
# ---------------------------------------------------------------------------

# Mapping from raw_obs keys → LeRobot video keys
IMAGE_KEYS_MAP = {
    "left_image": "exterior_image_1_left",
    "right_image": "exterior_image_2_left",
    "wrist_image": "wrist_image_left",
}


def save_episode(
    ep_idx: int,
    episode_steps: list[dict],
    output_path: Path,
    fps: int,
    task_index: int,
    task_string: str = "",
) -> dict:
    """Persist one episode as parquet + video files and return episode metadata."""
    num_frames = len(episode_steps)
    chunk_idx = ep_idx // 1000

    # Ensure directories exist
    (output_path / f"data/chunk-{chunk_idx:03d}").mkdir(parents=True, exist_ok=True)
    for lerobot_key in IMAGE_KEYS_MAP.values():
        (output_path / f"videos/chunk-{chunk_idx:03d}/observation.images.{lerobot_key}").mkdir(
            parents=True, exist_ok=True
        )

    # Collect columns
    states = []
    actions = []
    rewards = []
    dones = []

    # Collect video frames per camera
    video_frames = {k: [] for k in IMAGE_KEYS_MAP}

    for step in episode_steps:
        obs = step["obs"]
        joint_pos = obs["joint_position"].flatten().astype(np.float64)
        gripper_pos = obs["gripper_position"].flatten().astype(np.float64)
        state = np.concatenate([joint_pos, gripper_pos])  # (8,)
        states.append(state)
        actions.append(step["action"].astype(np.float64))
        rewards.append(step["reward"])
        dones.append(step["done"])
        for raw_key in IMAGE_KEYS_MAP:
            video_frames[raw_key].append(obs[raw_key])

    # Build parquet data
    ep_data = {
        "observation.state": [s.tolist() for s in states],
        "action": [a.tolist() for a in actions],
        "next.reward": [float(r) for r in rewards],
        "next.done": [bool(d) for d in dones],
        "is_terminal": [False] * (num_frames - 1) + [bool(dones[-1])],
        "is_first": [True] + [False] * (num_frames - 1),
        "discount": [1.0] * num_frames,
        "timestamp": [i / fps for i in range(num_frames)],
        "episode_index": [ep_idx] * num_frames,
        "frame_index": list(range(num_frames)),
        "task_index": [task_index] * num_frames,
        "annotation.language.language_instruction": [task_index] * num_frames,
        "annotation.language.language_instruction_2": [task_index] * num_frames,
        "annotation.language.language_instruction_3": [task_index] * num_frames,
    }

    df = pl.DataFrame(ep_data)
    parquet_path = output_path / f"data/chunk-{chunk_idx:03d}/episode_{ep_idx:06d}.parquet"
    df.write_parquet(parquet_path)

    # Encode videos
    for raw_key, lerobot_key in IMAGE_KEYS_MAP.items():
        frames_np = np.stack(video_frames[raw_key], axis=0)  # (T, H, W, 3) uint8
        video_path = (
            output_path
            / f"videos/chunk-{chunk_idx:03d}/observation.images.{lerobot_key}/episode_{ep_idx:06d}.mp4"
        )
        encode_video(frames_np, video_path, fps)

    return {
        "episode_index": ep_idx,
        "tasks": [task_string],
        "length": num_frames,
    }


# ---------------------------------------------------------------------------
# Meta-file generation
# ---------------------------------------------------------------------------

def write_meta_files(
    output_path: Path,
    episodes_data: list[dict],
    instruction: str,
    fps: int,
    image_shape: tuple[int, int, int],
) -> None:
    """Write all meta/ files after all episodes have been saved."""
    meta_dir = output_path / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    # --- modality.json ---
    modality_config = {
        "state": {
            "joint_position": {"start": 0, "end": 7},
            "gripper_position": {"start": 7, "end": 8},
        },
        "action": {
            "joint_position": {"start": 0, "end": 7},
            "gripper_position": {"start": 7, "end": 8},
        },
        "video": {
            k: {"original_key": f"observation.images.{k}"}
            for k in IMAGE_KEYS_MAP.values()
        },
        "annotation": {
            "language.language_instruction": {},
            "language.language_instruction_2": {},
            "language.language_instruction_3": {},
        },
    }
    with open(meta_dir / "modality.json", "w") as f:
        json.dump(modality_config, f, indent=4)

    # --- tasks.jsonl ---
    with open(meta_dir / "tasks.jsonl", "w") as f:
        f.write(json.dumps({"task_index": 0, "task": instruction}) + "\n")

    # --- episodes.jsonl ---
    with open(meta_dir / "episodes.jsonl", "w") as f:
        for ep in episodes_data:
            f.write(json.dumps(ep) + "\n")

    # --- info.json ---
    ds_length = len(episodes_data)
    num_chunks = (ds_length // 1000) + (1 if ds_length % 1000 else 0)
    image_keys = list(IMAGE_KEYS_MAP.values())
    info = {
        "codebase_version": "v2.0",
        "robot_type": "droid",
        "total_episodes": ds_length,
        "total_frames": sum(ep["length"] for ep in episodes_data),
        "total_tasks": 1,
        "total_videos": len(image_keys),
        "total_chunks": num_chunks,
        "chunks_size": 1000,
        "fps": fps,
        "splits": {"train": "0:100"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            **{
                f"observation.images.{k}": {
                    "dtype": "video",
                    "shape": list(image_shape),
                    "names": ["height", "width", "channel"],
                    "video_info": {
                        "video.fps": fps,
                        "video.codec": "h264",
                        "video.pix_fmt": "yuv420p",
                        "video.is_depth_map": False,
                        "has_audio": False,
                    },
                }
                for k in image_keys
            },
            "observation.state": {
                "dtype": "float64",
                "shape": [8],
                "names": ["joint_position", "gripper_position"],
            },
            "action": {
                "dtype": "float64",
                "shape": [8],
                "names": ["joint_position", "gripper_position"],
            },
            "timestamp": {"dtype": "float64", "shape": [1]},
            "task_index": {"dtype": "int64", "shape": [1]},
            "episode_index": {"dtype": "int64", "shape": [1]},
            "index": {"dtype": "int64", "shape": [1]},
            "next.reward": {"dtype": "float64", "shape": [1]},
            "next.done": {"dtype": "bool", "shape": [1]},
            "is_terminal": {"dtype": "bool", "shape": [1]},
            "is_first": {"dtype": "bool", "shape": [1]},
            "discount": {"dtype": "float64", "shape": [1]},
            "annotation.language.language_instruction": {"dtype": "int64", "shape": [1]},
            "annotation.language.language_instruction_2": {"dtype": "int64", "shape": [1]},
            "annotation.language.language_instruction_3": {"dtype": "int64", "shape": [1]},
        },
    }
    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=4)

    # --- stats.json ---
    compute_and_write_stats(output_path)


def compute_and_write_stats(output_path: Path) -> None:
    """Read all parquet files and compute per-dimension statistics for state & action."""
    import pandas as pd

    parquet_paths = sorted(output_path.glob("data/*/*.parquet"))
    all_states = []
    all_actions = []
    for p in parquet_paths:
        df = pd.read_parquet(p)
        all_states.extend(df["observation.state"].tolist())
        all_actions.extend(df["action"].tolist())

    stats = {}
    for name, data_list in [("observation.state", all_states), ("action", all_actions)]:
        arr = np.array(data_list, dtype=np.float64)  # (N, 8)
        stats[name] = {
            "mean": np.mean(arr, axis=0).tolist(),
            "std": np.std(arr, axis=0).tolist(),
            "min": np.min(arr, axis=0).tolist(),
            "max": np.max(arr, axis=0).tolist(),
            "q01": np.quantile(arr, 0.01, axis=0).tolist(),
            "q99": np.quantile(arr, 0.99, axis=0).tolist(),
        }

    with open(output_path / "meta" / "stats.json", "w") as f:
        json.dump(stats, f, indent=4)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    episodes: int = 10,
    scene: int = 1,
    headless: bool = True,
    host: str = "localhost",
    port: int = 6000,
    output_dir: str = "./data/on_policy_dataset",
    fps: int = 15,
):
    # Launch omniverse app (inside function to prevent overriding tyro)
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="Sim eval + data collection.")
    AppLauncher.add_app_launcher_args(parser)
    args_cli, _ = parser.parse_known_args()
    args_cli.enable_cameras = True
    args_cli.headless = headless
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    # IsaacLab imports after app launch
    import sim_evals.environments  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    # Set up environment
    env_cfg = parse_env_cfg("DROID", device=args_cli.device, num_envs=1, use_fabric=True)
    instruction = None
    match scene:
        case 1:
            instruction = "put the cube in the bowl"
        case 2:
            instruction = "pick up the can and put it in the mug"
        case 3:
            instruction = "put the banana in the bin"
        case _:
            raise ValueError(f"Scene {scene} not supported")

    env_cfg.set_scene(scene)

    import gymnasium as gym  # noqa: E402 — delayed so IsaacLab registers envs first

    env = gym.make("DROID", cfg=env_cfg)
    obs, _ = env.reset()
    obs, _ = env.reset()  # second render cycle for correct materials

    client = DreamZeroJointPosClient(remote_host=host, remote_port=port)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    max_steps = env.env.max_episode_length
    all_episodes_data = []
    image_shape = None

    with torch.no_grad():
        for ep in range(episodes):
            episode_steps = []
            for step_idx in tqdm(range(max_steps), desc=f"Episode {ep + 1}/{episodes}"):
                # Infer action (also returns raw_obs before env.step)
                ret = client.infer(obs, instruction)
                action = ret["action"]  # (8,) numpy
                raw_obs = ret["raw_obs"]

                if not headless:
                    viz = np.concatenate(
                        [
                            image_tools.resize_with_pad(raw_obs["right_image"], 224, 224),
                            image_tools.resize_with_pad(raw_obs["wrist_image"], 224, 224),
                            image_tools.resize_with_pad(raw_obs["left_image"], 224, 224),
                        ],
                        axis=1,
                    )
                    cv2.imshow("Camera Views", cv2.cvtColor(viz, cv2.COLOR_RGB2BGR))
                    cv2.waitKey(1)

                # Record image shape from first frame
                if image_shape is None:
                    image_shape = tuple(raw_obs["right_image"].shape)  # (H, W, 3)

                # Step environment
                action_tensor = torch.tensor(action)[None]
                obs, reward_t, term, trunc, _ = env.step(action_tensor)

                reward = float(reward_t) if np.isscalar(reward_t) else float(reward_t.item()) if hasattr(reward_t, 'item') else 0.0
                done = bool(term or trunc)

                episode_steps.append({
                    "obs": raw_obs,
                    "action": action,
                    "reward": reward,
                    "done": done,
                })

                if done:
                    break

            # Save episode to disk
            ep_meta = save_episode(ep, episode_steps, output_path, fps, task_index=0, task_string=instruction)
            all_episodes_data.append(ep_meta)
            print(f"Saved episode {ep} ({ep_meta['length']} frames)")

            # Reset for next episode
            client.reset()
            if ep < episodes - 1:
                obs, _ = env.reset()

    # Write meta files (modality, tasks, episodes, info, stats)
    write_meta_files(output_path, all_episodes_data, instruction, fps, image_shape)
    print(f"Dataset saved to {output_path}")
    print(f"  Episodes: {len(all_episodes_data)}")
    print(f"  Total frames: {sum(e['length'] for e in all_episodes_data)}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    tyro.cli(main)
