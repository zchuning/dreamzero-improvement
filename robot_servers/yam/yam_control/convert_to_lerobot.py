"""
Convert raw YAM data collection episodes to LeRobot v2 dataset format.

This is a standalone converter that reads the raw .npy + .mp4 episode files
produced by RecordEpisodeWrapper and outputs a LeRobot v2 dataset.

No external dependencies beyond the `convert` extra (av, pandas, pyarrow, tqdm).

Usage:
    python -m yam_control.convert_to_lerobot --input data/2025-01-15-10-30-00-YAM-00 --output lerobot_dataset

    # Process only the first 10 episodes (for testing)
    python -m yam_control.convert_to_lerobot --input data/2025-01-15-10-30-00-YAM-00 --output lerobot_dataset --max-episodes 10

Output LeRobot v2 layout:
    lerobot_dataset/
    ├── meta/
    │   ├── info.json
    │   ├── episodes.jsonl
    │   ├── tasks.jsonl
    │   └── stats.json
    ├── data/
    │   └── chunk-000/
    │       ├── episode_000000.parquet
    │       └── ...
    └── videos/
        └── chunk-000/
            ├── left_camera-images-rgb/
            │   ├── episode_000000.mp4
            │   └── ...
            ├── right_camera-images-rgb/
            │   └── ...
            └── top_camera-images-rgb/
                └── ...
"""

from dataclasses import dataclass
import json
from pathlib import Path
import sys

import av
import numpy as np
import pandas as pd
from tqdm import tqdm
import tyro


# --------------------------------------------------------------------------- #
#  Constants
# --------------------------------------------------------------------------- #

CHUNKS_SIZE = 1000
VIDEO_KEYS = [
    "left_camera-images-rgb",
    "right_camera-images-rgb",
    "top_camera-images-rgb",
]
# Observation state: left arm joints (6) + left gripper (1) + right arm joints (6) + right gripper (1)
STATE_NAMES = [
    "left_joint_0",
    "left_joint_1",
    "left_joint_2",
    "left_joint_3",
    "left_joint_4",
    "left_joint_5",
    "left_gripper",
    "right_joint_0",
    "right_joint_1",
    "right_joint_2",
    "right_joint_3",
    "right_joint_4",
    "right_joint_5",
    "right_gripper",
]
# Action: left arm joints (6) + left gripper (1) + right arm joints (6) + right gripper (1)
ACTION_NAMES = STATE_NAMES  # Same structure as state

REQUIRED_FILES = [
    "action-left-pos.npy",
    "action-right-pos.npy",
    "left-joint_pos.npy",
    "right-joint_pos.npy",
    "left-gripper_pos.npy",
    "right-gripper_pos.npy",
    "timestamp.npy",
    "metadata.json",
    "left_camera-images-rgb.mp4",
    "right_camera-images-rgb.mp4",
    "top_camera-images-rgb.mp4",
]


# --------------------------------------------------------------------------- #
#  Episode discovery
# --------------------------------------------------------------------------- #


def is_valid_episode(episode_dir: Path) -> bool:
    """Check if directory contains all required episode files."""
    if not episode_dir.is_dir():
        return False
    return all((episode_dir / f).is_file() for f in REQUIRED_FILES)


def discover_episodes(input_dir: Path) -> list[Path]:
    """Find and return sorted list of valid episode directories."""
    episodes = sorted(
        [d for d in input_dir.iterdir() if is_valid_episode(d)],
        key=lambda p: p.name,
    )
    return episodes


# --------------------------------------------------------------------------- #
#  Raw episode loading
# --------------------------------------------------------------------------- #


def load_episode_data(episode_dir: Path) -> dict:
    """Load raw .npy data from an episode directory.

    Returns dict with keys: state, action, timestamps, metadata, task_name, fps.
    """
    # Observations (state)
    left_joint = np.load(episode_dir / "left-joint_pos.npy")      # (N, 6)
    left_grip = np.load(episode_dir / "left-gripper_pos.npy")     # (N, 1)
    right_joint = np.load(episode_dir / "right-joint_pos.npy")    # (N, 6)
    right_grip = np.load(episode_dir / "right-gripper_pos.npy")   # (N, 1)

    # State: concat as [left_joint(6), left_gripper(1), right_joint(6), right_gripper(1)]
    state = np.concatenate([left_joint, left_grip, right_joint, right_grip], axis=1)  # (N, 14)

    # Actions (already concatenated as joint_pos + gripper_pos per arm)
    action_left = np.load(episode_dir / "action-left-pos.npy")    # (N, 7)
    action_right = np.load(episode_dir / "action-right-pos.npy")  # (N, 7)
    action = np.concatenate([action_left, action_right], axis=1)  # (N, 14)

    # Timestamps
    timestamps = np.load(episode_dir / "timestamp.npy")  # (N,)

    # Metadata
    with open(episode_dir / "metadata.json") as f:
        metadata = json.load(f)

    task_name = metadata.get("task_name", "unknown_task")
    fps = metadata.get("env_loop_frequency", 30.0)

    return {
        "state": state.astype(np.float32),
        "action": action.astype(np.float32),
        "timestamps": timestamps,
        "metadata": metadata,
        "task_name": task_name,
        "fps": fps,
    }


# --------------------------------------------------------------------------- #
#  Video re-encoding (mp4v -> h264)
# --------------------------------------------------------------------------- #


def reencode_video(src_path: Path, dst_path: Path, fps: float, expected_frames: int) -> dict:
    """Re-encode a video from mp4v to h264 and return video info.

    Returns video_info dict with codec, pix_fmt, shape, etc.
    """
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    # Read source frames
    frames = []
    with av.open(str(src_path)) as in_container:
        stream = in_container.streams.video[0]
        for frame in in_container.decode(stream):
            frames.append(frame.to_ndarray(format="rgb24"))

    if len(frames) != expected_frames:
        print(
            f"  Warning: {src_path.name} has {len(frames)} frames, expected {expected_frames}. "
            f"Using actual frame count."
        )

    if len(frames) == 0:
        raise ValueError(f"No frames decoded from {src_path}")

    height, width = frames[0].shape[:2]

    # Write h264 encoded video
    with av.open(str(dst_path), mode="w") as out_container:
        out_stream = out_container.add_stream("h264", rate=int(fps))
        out_stream.width = width
        out_stream.height = height
        out_stream.pix_fmt = "yuv420p"

        for frame_array in frames:
            av_frame = av.VideoFrame.from_ndarray(frame_array, format="rgb24")
            for packet in out_stream.encode(av_frame):
                out_container.mux(packet)

        # Flush
        for packet in out_stream.encode():
            out_container.mux(packet)

    video_info = {
        "video.fps": fps,
        "video.codec": "h264",
        "video.pix_fmt": "yuv420p",
        "video.is_depth_map": False,
        "has_audio": False,
    }
    return video_info, (height, width)


# --------------------------------------------------------------------------- #
#  LeRobot v2 dataset writing
# --------------------------------------------------------------------------- #


def write_parquet(
    output_dir: Path,
    episode_index: int,
    chunk_index: int,
    data: dict,
    task_index: int,
    global_frame_offset: int,
    fps: float,
) -> int:
    """Write a single episode's parquet file. Returns number of frames written."""
    state = data["state"]       # (N, 14) float32
    action = data["action"]     # (N, 14) float32
    n_frames = state.shape[0]

    # Build relative timestamps (seconds from episode start)
    timestamps_raw = data["timestamps"]
    relative_timestamps = (timestamps_raw - timestamps_raw[0]).astype(np.float32)

    # Build dataframe
    rows = {
        "observation.state": [state[i].tolist() for i in range(n_frames)],
        "action": [action[i].tolist() for i in range(n_frames)],
        "timestamp": relative_timestamps.tolist(),
        "frame_index": list(range(n_frames)),
        "episode_index": [episode_index] * n_frames,
        "index": list(range(global_frame_offset, global_frame_offset + n_frames)),
        "task_index": [task_index] * n_frames,
    }
    df = pd.DataFrame(rows)

    # Write parquet
    parquet_dir = output_dir / "data" / f"chunk-{chunk_index:03d}"
    parquet_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = parquet_dir / f"episode_{episode_index:06d}.parquet"
    df.to_parquet(parquet_path, index=False)

    return n_frames


def compute_stats(all_states: list[np.ndarray], all_actions: list[np.ndarray]) -> dict:
    """Compute dataset-level statistics for normalization."""
    states = np.concatenate(all_states, axis=0)  # (total_frames, 14)
    actions = np.concatenate(all_actions, axis=0)  # (total_frames, 14)

    def array_stats(arr: np.ndarray) -> dict:
        return {
            "mean": arr.mean(axis=0).tolist(),
            "std": arr.std(axis=0).tolist(),
            "min": arr.min(axis=0).tolist(),
            "max": arr.max(axis=0).tolist(),
        }

    return {
        "observation.state": array_stats(states),
        "action": array_stats(actions),
    }


def write_meta(
    output_dir: Path,
    total_episodes: int,
    total_frames: int,
    fps: float,
    tasks: list[str],
    episode_lengths: list[int],
    episode_tasks: list[str],
    stats: dict,
    video_info: dict,
    video_shape: tuple[int, int],
):
    """Write all meta/ files for the LeRobot v2 dataset."""
    meta_dir = output_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    # --- info.json ---
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": [14],
            "names": {"motors": STATE_NAMES},
        },
        "action": {
            "dtype": "float32",
            "shape": [14],
            "names": {"motors": ACTION_NAMES},
        },
        "timestamp": {
            "dtype": "float32",
            "shape": [1],
            "names": None,
        },
        "frame_index": {
            "dtype": "int64",
            "shape": [1],
            "names": None,
        },
        "episode_index": {
            "dtype": "int64",
            "shape": [1],
            "names": None,
        },
        "index": {
            "dtype": "int64",
            "shape": [1],
            "names": None,
        },
        "task_index": {
            "dtype": "int64",
            "shape": [1],
            "names": None,
        },
    }
    height, width = video_shape
    for vkey in VIDEO_KEYS:
        features[f"observation.images.{vkey}"] = {
            "dtype": "video",
            "shape": [height, width, 3],
            "names": ["height", "width", "channels"],
            "video_info": video_info,
        }

    info = {
        "codebase_version": "v2.1",
        "robot_type": "yam",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "fps": fps,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
        "chunks_size": CHUNKS_SIZE,
    }
    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    # --- tasks.jsonl ---
    task_to_index = {}
    with open(meta_dir / "tasks.jsonl", "w") as f:
        for i, task in enumerate(tasks):
            task_to_index[task] = i
            f.write(json.dumps({"task_index": i, "task": task}) + "\n")

    # --- episodes.jsonl ---
    with open(meta_dir / "episodes.jsonl", "w") as f:
        for ep_idx in range(total_episodes):
            ep_task = episode_tasks[ep_idx]
            entry = {
                "episode_index": ep_idx,
                "tasks": [ep_task],
                "length": episode_lengths[ep_idx],
            }
            f.write(json.dumps(entry) + "\n")

    # --- stats.json ---
    with open(meta_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)


# --------------------------------------------------------------------------- #
#  Main conversion
# --------------------------------------------------------------------------- #


@dataclass
class ConvertConfig:
    input: str
    """Path to raw data collection directory (contains episode subdirectories)."""

    output: str
    """Path for output LeRobot v2 dataset."""

    max_episodes: int | None = None
    """If set, only convert the first N episodes (useful for testing)."""


def convert_to_lerobot(input_dir: Path, output_dir: Path, max_episodes: int | None = None):
    """Convert raw YAM episodes to LeRobot v2 dataset format."""

    # Discover episodes
    episodes = discover_episodes(input_dir)
    if not episodes:
        print(f"No valid episodes found in {input_dir}")
        sys.exit(1)

    if max_episodes is not None:
        episodes = episodes[:max_episodes]

    print(f"Found {len(episodes)} valid episodes in {input_dir}")
    print(f"Output directory: {output_dir}")

    # Collect unique tasks and build task index mapping
    task_set: dict[str, int] = {}  # task_name -> task_index
    all_states: list[np.ndarray] = []
    all_actions: list[np.ndarray] = []
    episode_lengths: list[int] = []
    episode_tasks: list[str] = []
    global_frame_offset = 0
    fps = None
    video_info = None
    video_shape = None

    for ep_idx, ep_dir in enumerate(tqdm(episodes, desc="Converting episodes")):
        # Load raw data
        data = load_episode_data(ep_dir)
        n_frames = data["state"].shape[0]

        if fps is None:
            fps = data["fps"]

        # Register task
        task_name = data["task_name"]
        if task_name not in task_set:
            task_set[task_name] = len(task_set)
        task_index = task_set[task_name]

        # Chunk index
        chunk_index = ep_idx // CHUNKS_SIZE

        # Write parquet
        written = write_parquet(
            output_dir, ep_idx, chunk_index, data, task_index, global_frame_offset, fps
        )

        # Re-encode videos
        for vkey in VIDEO_KEYS:
            src_video = ep_dir / f"{vkey}.mp4"
            dst_video = (
                output_dir
                / "videos"
                / f"chunk-{chunk_index:03d}"
                / vkey
                / f"episode_{ep_idx:06d}.mp4"
            )
            vi, vs = reencode_video(src_video, dst_video, fps, n_frames)
            if video_info is None:
                video_info = vi
                video_shape = vs

        # Accumulate stats
        all_states.append(data["state"])
        all_actions.append(data["action"])
        episode_lengths.append(written)
        episode_tasks.append(task_name)
        global_frame_offset += written

    total_episodes = len(episodes)
    total_frames = global_frame_offset

    # Compute statistics
    print("Computing dataset statistics...")
    stats = compute_stats(all_states, all_actions)

    # Write metadata
    print("Writing metadata...")
    tasks = sorted(task_set.keys(), key=lambda t: task_set[t])
    write_meta(
        output_dir,
        total_episodes,
        total_frames,
        fps,
        tasks,
        episode_lengths,
        episode_tasks,
        stats,
        video_info,
        video_shape,
    )

    print(f"\nConversion complete!")
    print(f"  Episodes: {total_episodes}")
    print(f"  Total frames: {total_frames}")
    print(f"  Tasks: {tasks}")
    print(f"  FPS: {fps}")
    print(f"  Output: {output_dir}")


def main():
    cfg = tyro.cli(ConvertConfig)
    convert_to_lerobot(
        input_dir=Path(cfg.input),
        output_dir=Path(cfg.output),
        max_episodes=cfg.max_episodes,
    )


if __name__ == "__main__":
    main()
