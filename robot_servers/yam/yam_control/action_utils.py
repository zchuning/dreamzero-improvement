"""Action utilities for observation/action manipulation."""

from typing import Any

import numpy as np


def hold_action_from_proprio(proprio: dict[str, Any]) -> dict[str, Any]:
    """Create a hold-in-place action from the current proprioceptive state.

    Tries common key naming conventions to extract joint and gripper positions.

    Args:
        proprio: Dict of proprioceptive state (joint positions, gripper positions)

    Returns:
        Action dict that holds the robot at the current position
    """
    try_keys = [
        ("left_joint_pos", "left_gripper_pos", "right_joint_pos", "right_gripper_pos"),
        (
            "joint_pos_obs_left",
            "gripper_pos_obs_left",
            "joint_pos_obs_right",
            "gripper_pos_obs_right",
        ),
    ]
    for lq, lg, rq, rg in try_keys:
        if lq in proprio and rq in proprio:
            return {
                "left_joint_pos": np.asarray(proprio[lq], dtype=np.float32).reshape(-1)[:6],
                "left_gripper_pos": np.asarray(
                    proprio.get(lg, np.zeros(1, np.float32)), dtype=np.float32
                ).reshape(-1)[:1],
                "right_joint_pos": np.asarray(proprio[rq], dtype=np.float32).reshape(-1)[:6],
                "right_gripper_pos": np.asarray(
                    proprio.get(rg, np.zeros(1, np.float32)), dtype=np.float32
                ).reshape(-1)[:1],
                "source": None,
            }
    raise ValueError("No valid keys found in proprio")
