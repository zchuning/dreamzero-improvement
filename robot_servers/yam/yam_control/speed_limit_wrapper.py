"""Speed limit environment wrapper for safe robot control."""

from typing import Any

import gymnasium as gym
import numpy as np


class SpeedLimitWrapper(gym.Wrapper):
    """Gym wrapper that limits joint and gripper velocities for safety.

    Clips actions to ensure smooth, safe robot movement by limiting
    the maximum change in joint positions and gripper positions per step.

    Velocity limits (hardcoded):
    - Joint velocity: pi radians / 0.4 seconds (180 degrees in 400ms)
    - Gripper velocity: 1.0 / 0.2 seconds (full open/close in 200ms)
    """

    JOINT_KEYS = ("left_joint_pos", "right_joint_pos")
    GRIPPER_KEYS = ("left_gripper_pos", "right_gripper_pos")

    # Hardcoded velocity limits
    JOINT_180_TIME = 0.4  # seconds to rotate 180 degrees (pi radians)
    GRIPPER_FULL_TIME = 0.2  # seconds for full gripper open/close

    def __init__(self, env: gym.Env, control_freq: float):
        super().__init__(env)
        self.control_freq = control_freq

        # Compute max velocities
        self.max_joint_velocity = np.pi / self.JOINT_180_TIME  # rad/s
        self.max_gripper_velocity = 1.0 / self.GRIPPER_FULL_TIME  # /s

        # Compute per-step max deltas
        self.max_joint_delta = self.max_joint_velocity / control_freq
        self.max_gripper_delta = self.max_gripper_velocity / control_freq

        # State tracking
        self._last_state: dict[str, np.ndarray] | None = None

    def step(self, action: dict[str, np.ndarray]) -> tuple[dict, float, bool, bool, dict]:
        clipped_action = self._clip_action(action)
        obs, reward, terminated, truncated, info = self.env.step(clipped_action)
        self._update_state_from_obs(obs)
        return obs, reward, terminated, truncated, info

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict, dict]:
        obs, info = self.env.reset(seed=seed, options=options)
        self._update_state_from_obs(obs)
        return obs, info

    def _clip_action(self, action: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if self._last_state is None:
            return action

        clipped = {}
        for key, value in action.items():
            if key in self.JOINT_KEYS:
                delta = value - self._last_state[key]
                clipped_delta = np.clip(delta, -self.max_joint_delta, self.max_joint_delta)
                clipped[key] = self._last_state[key] + clipped_delta
            elif key in self.GRIPPER_KEYS:
                delta = value - self._last_state[key]
                clipped_delta = np.clip(delta, -self.max_gripper_delta, self.max_gripper_delta)
                clipped[key] = self._last_state[key] + clipped_delta
            else:
                clipped[key] = value
        return clipped

    def _update_state_from_obs(self, obs: dict[str, Any]) -> None:
        self._last_state = {
            "left_joint_pos": np.asarray(
                obs.get("left_joint_pos", np.zeros(6)), dtype=np.float32
            ),
            "left_gripper_pos": np.asarray(
                obs.get("left_gripper_pos", np.zeros(1)), dtype=np.float32
            ),
            "right_joint_pos": np.asarray(
                obs.get("right_joint_pos", np.zeros(6)), dtype=np.float32
            ),
            "right_gripper_pos": np.asarray(
                obs.get("right_gripper_pos", np.zeros(1)), dtype=np.float32
            ),
        }
