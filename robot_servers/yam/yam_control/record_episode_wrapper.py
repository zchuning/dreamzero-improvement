from datetime import datetime
import json
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from typing import Any
import warnings

import cv2
import gymnasium as gym
import numpy as np


def _append_dictionaries(dest: dict[str, list[Any]], src: dict[str, Any]):
    """Appends a dictionary of items to a list of dictionaries.

    Ensures sure that the set of keys does not change after the first append.
    """
    if dest:
        assert set(dest.keys()) == set(
            src.keys()
        ), f"Set of keys changed from {dest.keys()} to {src.keys()}"
    else:
        for k, v in src.items():
            dest[k] = []
    for k, v in src.items():
        dest[k].append(v)


class RecordEpisodeWrapper(gym.Wrapper):
    """Gym wrapper that records YAM environment episodes to disk.

    Each call to `reset()` finalizes the ongoing episode and starts recording a new one.
    By default, every recorded episode is saved to disk.

    Available reset options:
    - discard_episode: Discard the ongoing episode without saving to disk.
    - skip_next_episode: After reset, the next episode will not be recorded.
    """

    def __init__(
        self,
        env: gym.Env,
        output_dir: str = "data/debug",
        operator: str | None = None,
        show_inline_timer: bool = True,
    ):
        super().__init__(env)
        self.output_dir = Path(output_dir)
        self.task_name = None
        self.operator = operator
        self.show_inline_timer = show_inline_timer

        # Recording state
        self.is_recording = False

        # Episode data
        self.tmp_episode_dir = None
        self.timestamps: list[float] = []
        self.observations: dict[str, list[np.ndarray]] = {}
        self.actions: dict[str, list[np.ndarray]] = {}
        self.action_sources: list[Any] = []
        self.video_writers: dict[str, cv2.VideoWriter] = {}

        # Previous step data
        self.prev_obs = None
        self.prev_action = None

        # Timer display state
        self._last_timer_update: float | None = None
        self._session_total_seconds: float = 0.0

    @property
    def recording_elapsed_seconds(self) -> float:
        if not self.timestamps:
            return 0.0
        return len(self.timestamps) * self.env.unwrapped.control_period

    # Keys that are robot action data (recorded via _append_dictionaries).
    # Everything else (e.g. "source") is metadata tracked separately.
    _ACTION_KEYS = frozenset([
        "left_joint_pos", "left_gripper_pos",
        "right_joint_pos", "right_gripper_pos",
    ])

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        if self.is_recording:
            # Record timestamp
            self.timestamps.append(time.time())

            # Live timer display (update at most once per second)
            if self.show_inline_timer:
                now = time.time()
                if self._last_timer_update is None or (now - self._last_timer_update) >= 1.0:
                    elapsed = int(len(self.timestamps) * self.env.unwrapped.control_period)
                    sys.stdout.write(f"\rRecording: {elapsed // 60:02d}:{elapsed % 60:02d}")
                    sys.stdout.flush()
                    self._last_timer_update = now

            # Record observations — only numpy arrays (skip string metadata
            # like "annotation.task" which may be injected by ViserUI).
            assert self.prev_obs is not None
            non_video_obs = {}
            for k, v in self.prev_obs.items():
                if not isinstance(v, np.ndarray):
                    continue
                if v.ndim == 3:
                    self._write_video_frame(k, v)
                else:
                    non_video_obs[k] = v
            _append_dictionaries(self.observations, non_video_obs)

            # Record actions -- separate robot keys from metadata
            robot_action = {k: v for k, v in action.items() if k in self._ACTION_KEYS}
            _append_dictionaries(self.actions, robot_action)
            self.action_sources.append(action.get("source"))

        self.prev_obs = obs
        self.prev_action = action

        return obs, reward, terminated, truncated, info

    def _write_video_frame(self, obs_key: str, frame: np.ndarray):
        if obs_key not in self.video_writers:
            assert self.tmp_episode_dir is not None
            h, w, _ = frame.shape
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_name = f"{obs_key.replace('_image', '-images-rgb')}.mp4"
            video_path = str(Path(self.tmp_episode_dir.name) / video_name)
            self.video_writers[obs_key] = cv2.VideoWriter(
                video_path, fourcc, self.env.unwrapped.policy_control_freq, (w, h)
            )
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        self.video_writers[obs_key].write(bgr_frame)

    def _reencode_videos_to_h264(self):
        """Re-encode mp4v videos to H.264 for broad playback compatibility."""
        tmp_dir = Path(self.tmp_episode_dir.name)
        for mp4_path in tmp_dir.glob("*.mp4"):
            h264_path = mp4_path.with_suffix(".h264.mp4")
            try:
                subprocess.run(
                    [
                        "ffmpeg", "-y", "-i", str(mp4_path),
                        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                        str(h264_path),
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=True,
                )
                mp4_path.unlink()
                h264_path.rename(mp4_path)
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                warnings.warn(f"H.264 re-encode failed for {mp4_path.name}, keeping mp4v: {e}")
                if h264_path.exists():
                    h264_path.unlink()

    def reset(self, seed=None, options=None):
        options = options if options is not None else {}

        # Finalize ongoing episode
        if self.is_recording:
            discard_episode = options.get("discard_episode")
            self._finalize_episode(discard_episode=discard_episode)
            self.is_recording = False

        # Start recording new episode
        skip_next_episode = options.get("skip_next_episode")
        if not skip_next_episode:
            self._start_new_episode()
            self.is_recording = True
            assert "task_name" in options, "task_name is required for new episode"
            self.task_name = options["task_name"]
        else:
            self.task_name = None

        # Note: We only call call super().reset() for the very first episode.
        # For subsequent episodes, the robot is already in the episode's initial state
        # when reset() is called, so super().reset() is not necessary.
        if self.prev_action is None or options.get("force_reset", False):
            obs, info = super().reset(seed=seed, options=options)
        else:
            obs, _, _, _, info = super().step(self.prev_action)

        self.prev_obs = obs

        return obs, info

    def _start_new_episode(self):
        self.tmp_episode_dir = tempfile.TemporaryDirectory()
        self.timestamps.clear()
        self.observations.clear()
        self.actions.clear()
        self.action_sources.clear()
        self.video_writers.clear()
        self._last_timer_update = None

    def _finalize_episode(self, discard_episode: bool = False):
        assert self.is_recording
        assert self.tmp_episode_dir is not None

        # Clear timer line
        sys.stdout.write("\r" + " " * 50 + "\r")
        sys.stdout.flush()

        # Close video writers and re-encode to H.264 for broad playback compatibility
        for writer in self.video_writers.values():
            writer.release()
        self._reencode_videos_to_h264()

        if not self.observations or not self.actions:
            print("No observations or actions recorded")
            discard_episode = True

        if discard_episode:
            print("Discarding episode", self.tmp_episode_dir.name)
            self.tmp_episode_dir.cleanup()
            return

        self._save_timestamps()
        self._save_observations()
        self._save_actions()
        self._save_metadata()

        self.output_dir.mkdir(parents=True, exist_ok=True)
        episode_dir = self.output_dir / datetime.now().strftime("%Y%m%dT%H%M%S%f")
        shutil.move(self.tmp_episode_dir.name, episode_dir)
        self.tmp_episode_dir.cleanup()
        num_episodes = len([x for x in self.output_dir.iterdir() if x.is_dir()])
        episode_duration = len(self.timestamps) * self.env.unwrapped.control_period
        self._session_total_seconds += episode_duration
        session_mins = self._session_total_seconds / 60
        print(f"Saved episode to {episode_dir} ({num_episodes} total - {session_mins:.1f} minutes)")

    MIN_REAL_ACTIONS_TO_SAVE = 3

    def close(self):
        if self.is_recording:
            real_actions = sum(1 for s in self.action_sources if s is not None)
            discard = real_actions < self.MIN_REAL_ACTIONS_TO_SAVE
            if discard:
                print(
                    f"[RecordEpisodeWrapper] Discarding in-progress episode on close "
                    f"({real_actions} real actions < {self.MIN_REAL_ACTIONS_TO_SAVE} minimum)"
                )
            self._finalize_episode(discard_episode=discard)
            self.is_recording = False

        self._print_session_summary()
        super().close()

    def _print_session_summary(self):
        """Print a summary of all saved episodes and intervention counts."""
        if not self.output_dir.exists():
            print("[RecordEpisodeWrapper] No episodes were saved.")
            return

        episode_dirs = sorted([x for x in self.output_dir.iterdir() if x.is_dir()])
        if not episode_dirs:
            print("[RecordEpisodeWrapper] No episodes were saved.")
            return

        print(f"\n{'=' * 60}")
        print("[RecordEpisodeWrapper] Session summary")
        print(f"  Output directory: {self.output_dir.resolve()}")
        print(f"  Total episodes saved: {len(episode_dirs)}")
        print(f"  Total session duration: {self._session_total_seconds / 60:.1f} minutes")
        print(f"{'─' * 60}")

        for i, ep_dir in enumerate(episode_dirs):
            action_source_path = ep_dir / "action-source.json"
            if action_source_path.exists():
                with open(action_source_path) as f:
                    sources = json.load(f)
                n_policy = sources.count("policy")
                n_human = sources.count("human")
                n_reverse_sync = sources.count("reverse_sync")
                print(
                    f"  Episode {i}: {ep_dir.name} "
                    f"({len(sources)} steps, "
                    f"policy={n_policy}, human={n_human}, reverse_sync={n_reverse_sync})"
                )
            else:
                print(f"  Episode {i}: {ep_dir.name} (no action-source.json)")

        print(f"{'=' * 60}\n")

    def _save_timestamps(self):
        np.save(Path(self.tmp_episode_dir.name) / "timestamp.npy", self.timestamps)

        # Show warning for unusually long episode
        duration_mins = len(self.timestamps) * self.env.unwrapped.control_period / 60.0
        if duration_mins > 10:  # 10 mins
            print(f"Warning: Unusually long episode ({duration_mins:.1f} mins)")

    def _save_observations(self):
        for k, v in self.observations.items():
            k = k.replace("left_", "left-")
            k = k.replace("right_", "right-")
            try:
                if v and isinstance(v[0], str):
                    np.save(Path(self.tmp_episode_dir.name) / f"{k}.npy", np.array(v, dtype=object))
                else:
                    np.save(
                        Path(self.tmp_episode_dir.name) / f"{k}.npy", np.array(v, dtype=np.float64)
                    )
            except (ValueError, TypeError):
                warnings.warn(f"Could not record key {k}")

    def _save_actions(self):
        left_action = np.hstack([self.actions["left_joint_pos"], self.actions["left_gripper_pos"]])
        right_action = np.hstack(
            [self.actions["right_joint_pos"], self.actions["right_gripper_pos"]]
        )

        np.save(Path(self.tmp_episode_dir.name) / "action-left-pos.npy", left_action)
        np.save(Path(self.tmp_episode_dir.name) / "action-right-pos.npy", right_action)

        # Save action source metadata if any non-None values exist.
        # During data collection TeleopPolicy sets "human"; during rollout
        # the ViserUI wrapper sets "policy"; home-pose waypoints have None.
        if any(s is not None for s in self.action_sources):
            with open(Path(self.tmp_episode_dir.name) / "action-source.json", "w") as f:
                json.dump(self.action_sources, f)

    def _save_metadata(self):
        metadata = self._get_metadata()
        with open(Path(self.tmp_episode_dir.name) / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    def _get_metadata(self):
        motion = self.task_name  # Placeholder
        motion_object = self.task_name  # Placeholder
        station_metadata = {
            "arm_type": "yam",
            "world_frame": "left_arm",
            "extrinsics": {
                "right_arm_extrinsic": {
                    "position": [0.0, -0.61, 0.0],
                    "rotation": [1.0, 0.0, 0.0, 0.0],
                }
            },
        }
        attributes = {
            "manipulation_surface": "white",
            "object": "object",  # Placeholder
        }
        camera_info = self._get_camera_metadata()
        metadata = {
            "task_name": self.task_name,
            "motion": motion,
            "motion_object": motion_object,
            "env_loop_frequency": self.env.unwrapped.policy_control_freq,
            "duration": self.timestamps[-1] - self.timestamps[0],
            "station_metadata": station_metadata,
            "attributes": attributes,
            "camera_info": camera_info,
        }

        # Custom metadata
        metadata["operator"] = self.operator
        metadata["hostname"] = socket.gethostname()

        return metadata

    def _get_camera_metadata(self):
        camera_info = {}
        if not hasattr(self.env.unwrapped, "cameras"):
            return camera_info

        for camera_name, camera in self.env.unwrapped.cameras.items():
            name = f"{camera_name}_camera"
            camera_info[name] = {
                "camera_type": "Intel RealSense D405",
                "device_id": camera.camera.device_id,
                "width": camera.camera.resolution[0],
                "height": camera.camera.resolution[1],
                "polling_fps": camera.camera.fps,
                "name": name,
                "image_transfer_time_offset_ms": None,
                "exposure_value": None,
                "auto_exposure": camera.camera.auto_exposure,
                "intrinsic_data": None,
                "extrinsics": None,
                "concat_image": False,
            }
        return camera_info
