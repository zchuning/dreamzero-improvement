"""Simple replay policy for replaying recorded robot actions."""

from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np


class ReplayPolicy:
    """Simple replay policy that loads recorded actions from a directory."""

    def __init__(
        self,
        replay_dir,
        action_horizon=50,
        map_action: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    ):
        """Initialize replay policy by loading action data.

        Args:
            replay_dir: Path to directory containing action-left-pos.npy and action-right-pos.npy
            action_horizon: Number of actions to return per step (batch size). Default: 50
            map_action: Optional adapter function to map policy action to env action format
        """
        self.replay_dir = Path(replay_dir)
        self.action_horizon = action_horizon
        self.map_action = map_action

        self.left_actions = np.load(self.replay_dir / "action-left-pos.npy")  # shape: (T, 7)
        self.right_actions = np.load(self.replay_dir / "action-right-pos.npy")  # shape: (T, 7)

        self.num_steps = len(self.left_actions)
        self.current_step = 0

        self._current_chunk = None
        self._chunk_index = 0

        self.action_chunk_size = action_horizon
        self.embodiment_tag = None

    def step(self, vla_step_data=None, **kwargs):
        """Return next batch of actions from replay data (non-overlapping batches).

        Returns:
            Tuple of (action_dict, info_dict) where action_dict contains:
            - "left_joint_pos": (1, action_horizon, 6)
            - "left_gripper_pos": (1, action_horizon, 1)
            - "right_joint_pos": (1, action_horizon, 6)
            - "right_gripper_pos": (1, action_horizon, 1)
        """
        end_step = self.current_step + self.action_horizon

        if end_step <= self.num_steps:
            left_batch = self.left_actions[self.current_step : end_step]
            right_batch = self.right_actions[self.current_step : end_step]
        else:
            remaining = self.num_steps - self.current_step
            left_end = self.left_actions[self.current_step :]
            left_start = self.left_actions[: self.action_horizon - remaining]
            left_batch = np.concatenate([left_end, left_start], axis=0)

            right_end = self.right_actions[self.current_step :]
            right_start = self.right_actions[: self.action_horizon - remaining]
            right_batch = np.concatenate([right_end, right_start], axis=0)

        left_joints = left_batch[:, :6]
        left_grippers = left_batch[:, 6:7]
        right_joints = right_batch[:, :6]
        right_grippers = right_batch[:, 6:7]

        action_dict = {
            "left_joint_pos": left_joints[np.newaxis, :, :],
            "left_gripper_pos": left_grippers[np.newaxis, :, :],
            "right_joint_pos": right_joints[np.newaxis, :, :],
            "right_gripper_pos": right_grippers[np.newaxis, :, :],
        }

        self.current_step = (self.current_step + self.action_horizon) % self.num_steps

        return (action_dict, {})

    def reset(self):
        """Reset the replay policy to the beginning."""
        self.current_step = 0
        self._current_chunk = None
        self._chunk_index = 0

    def get_action(
        self, observation: Dict[str, Any]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Get next single action from the current chunk.

        Returns one action at a time from the current chunk. When the chunk is exhausted,
        fetches a new chunk via step().
        """
        if self._current_chunk is None or self._chunk_index >= self.action_horizon:
            chunk_dict, _ = self.step()

            self._current_chunk = {}
            for key, value in chunk_dict.items():
                self._current_chunk[key] = value[0, :, :]

            self._chunk_index = 0

        action = {}
        for key, value in self._current_chunk.items():
            action[key] = value[self._chunk_index, :]

        self._chunk_index += 1

        action_to_return = self.map_action(action) if self.map_action is not None else action
        action_chunk_to_return = (
            self.map_action(self._current_chunk)
            if self.map_action is not None
            else self._current_chunk
        )

        info = {"action_chunk": action_chunk_to_return}

        return action_to_return, info
