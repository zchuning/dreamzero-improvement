"""Replay policy that loads actions from a LeRobotDataset."""

from pathlib import Path

import numpy as np
import pandas as pd

from yam_control.policy import Action, Info, Observation, Policy


class DatasetReplayPolicy(Policy):
    """Policy that replays actions from a LeRobotDataset episode.

    Action format (xdof, 46D):
        indices 0-6:   joint_pos_action_left (6D)
        indices 6-12:  joint_pos_action_right (6D)
        indices 12-28: ee_pose_action_left (16D) - not used
        indices 28-44: ee_pose_action_right (16D) - not used
        indices 44-45: gripper_pos_action_left (1D)
        indices 45-46: gripper_pos_action_right (1D)
    """

    def __init__(
        self,
        dataset_path: str | Path,
        episode_index: int = 0,
        chunk_size: int = 1,
    ):
        self.dataset_path = Path(dataset_path)
        self.episode_index = episode_index
        self.chunk_size = chunk_size

        self._load_episode_data()

        self.current_step = 0

    def _load_episode_data(self):
        data_dir = self.dataset_path / "data"

        episode_file = None
        for chunk_dir in sorted(data_dir.glob("chunk-*")):
            candidate = chunk_dir / f"episode_{self.episode_index:06d}.parquet"
            if candidate.exists():
                episode_file = candidate
                break

        if episode_file is None:
            raise FileNotFoundError(f"Episode {self.episode_index} not found in {data_dir}")

        df = pd.read_parquet(episode_file)

        actions = np.stack(df["action"].values)  # Shape: (T, 46)

        self.actions = actions
        self.num_steps = len(actions)

        print(
            f"[DatasetReplayPolicy] Loaded episode {self.episode_index} "
            f"with {self.num_steps} steps from {episode_file}"
        )

    def _extract_action(self, action_46d: np.ndarray) -> Action:
        return {
            "left_joint_pos": action_46d[0:6],
            "right_joint_pos": action_46d[6:12],
            "left_gripper_pos": action_46d[44:45],
            "right_gripper_pos": action_46d[45:46],
        }

    def _build_action_chunk(self, start_step: int, horizon: int) -> dict[str, np.ndarray]:
        end_step = min(start_step + horizon, self.num_steps)
        actual_horizon = end_step - start_step

        chunk_actions = self.actions[start_step:end_step]

        if actual_horizon < horizon:
            padding = np.tile(chunk_actions[-1:], (horizon - actual_horizon, 1))
            chunk_actions = np.concatenate([chunk_actions, padding], axis=0)

        return {
            "left_joint_pos": chunk_actions[:, 0:6],
            "right_joint_pos": chunk_actions[:, 6:12],
            "left_gripper_pos": chunk_actions[:, 44:45],
            "right_gripper_pos": chunk_actions[:, 45:46],
        }

    def get_action(self, observation: Observation) -> tuple[Action, Info]:
        if self.current_step >= self.num_steps:
            action = self._extract_action(self.actions[-1])
            info = {
                "action_chunk": self._build_action_chunk(self.num_steps - 1, self.chunk_size),
                "episode_done": True,
            }
            return action, info

        action = self._extract_action(self.actions[self.current_step])

        action_chunk = self._build_action_chunk(self.current_step, self.chunk_size)

        self.current_step += self.chunk_size

        info = {
            "action_chunk": action_chunk,
            "episode_done": self.current_step >= self.num_steps,
            "current_step": self.current_step - self.chunk_size,
            "num_steps": self.num_steps,
        }

        return action, info

    def reset(self) -> Info | None:
        self.current_step = 0
        return {"task_name": f"replay_episode_{self.episode_index}"}
