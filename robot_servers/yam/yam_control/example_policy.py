"""Example policy that cycles through 5 predefined joint poses.

Useful for verifying the rollout pipeline without a trained model.
Each call to ``get_action`` produces a cosine-interpolated trajectory
from the robot's current position to the next target pose.  After
the last pose is reached the sequence wraps back to the first.

Usage::

    from yam_control.example_policy import PredefinedPosePolicy
    from yam_control.run_policy_rollout import run_rollout, RolloutConfig

    policy = PredefinedPosePolicy(num_interpolation_steps=30)
    run_rollout(policy, RolloutConfig(action_horizon=30, use_sim=True))
"""

import numpy as np

from yam_control.policy import Action, Info, Observation, Policy

# Five target poses (both arms + grippers).  Joint angles are in radians;
# gripper values range from 0 (closed) to 1 (open).
PREDEFINED_POSES: list[dict[str, np.ndarray]] = [
    {  # Pose 0 – home / neutral (all-zeros joint positions)
        "left_joint_pos": np.zeros(6),
        "right_joint_pos": np.zeros(6),
        "left_gripper_pos": np.array([1.0]),
        "right_gripper_pos": np.array([1.0]),
    },
    {  # Pose 1 – arms forward, grippers closed
        "left_joint_pos": np.array([0.3, 0.8, 1.2, 0.0, 0.2, 0.0]),
        "right_joint_pos": np.array([-0.3, 0.8, 1.2, 0.0, -0.2, 0.0]),
        "left_gripper_pos": np.array([0.0]),
        "right_gripper_pos": np.array([0.0]),
    },
    {  # Pose 2 – arms wide
        "left_joint_pos": np.array([0.5, 1.2, 0.8, 0.3, 0.0, 0.0]),
        "right_joint_pos": np.array([-0.5, 1.2, 0.8, -0.3, 0.0, 0.0]),
        "left_gripper_pos": np.array([1.0]),
        "right_gripper_pos": np.array([1.0]),
    },
    {  # Pose 3 – arms tucked, grippers half-closed
        "left_joint_pos": np.array([0.0, 1.4, 1.4, 0.0, -0.3, 0.0]),
        "right_joint_pos": np.array([0.0, 1.4, 1.4, 0.0, 0.3, 0.0]),
        "left_gripper_pos": np.array([0.5]),
        "right_gripper_pos": np.array([0.5]),
    },
    {  # Pose 4 – arms extended, grippers open
        "left_joint_pos": np.array([0.2, 0.6, 0.6, 0.2, 0.2, 0.2]),
        "right_joint_pos": np.array([-0.2, 0.6, 0.6, -0.2, -0.2, -0.2]),
        "left_gripper_pos": np.array([1.0]),
        "right_gripper_pos": np.array([1.0]),
    },
]

ACTION_KEYS = [
    "left_joint_pos",
    "left_gripper_pos",
    "right_joint_pos",
    "right_gripper_pos",
]


class PredefinedPosePolicy(Policy):
    """Cycle through :data:`PREDEFINED_POSES` with cosine interpolation.

    Parameters
    ----------
    num_interpolation_steps:
        Number of waypoints to interpolate between consecutive poses.
        This equals the action chunk length ``T``.  Set this equal to
        ``RolloutConfig.action_horizon`` so one chunk = one pose transition.
    """

    def __init__(self, num_interpolation_steps: int = 30) -> None:
        self.num_interpolation_steps = num_interpolation_steps
        self.pose_index: int = 0
        self.step_count: int = 0

    def reset(self) -> Info | None:
        self.pose_index = 0
        self.step_count = 0
        print("[ExamplePolicy] Reset – restarting from pose 0")
        return None

    def get_action(self, observation: Observation) -> tuple[Action, Info]:
        # Read current robot state from observation
        current = {
            "left_joint_pos": np.asarray(
                observation.get("left_joint_pos", np.zeros(6)), dtype=np.float32
            ),
            "right_joint_pos": np.asarray(
                observation.get("right_joint_pos", np.zeros(6)), dtype=np.float32
            ),
            "left_gripper_pos": np.asarray(
                observation.get("left_gripper_pos", np.array([1.0])), dtype=np.float32
            ),
            "right_gripper_pos": np.asarray(
                observation.get("right_gripper_pos", np.array([1.0])), dtype=np.float32
            ),
        }

        target = PREDEFINED_POSES[self.pose_index]

        # Cosine interpolation: smooth acceleration / deceleration
        T = self.num_interpolation_steps
        chunk: dict[str, np.ndarray] = {}
        for key in ACTION_KEYS:
            start = current[key]
            end = target[key]
            waypoints = []
            for i in range(1, T + 1):
                alpha = (1.0 - np.cos(i / T * np.pi)) / 2.0
                waypoints.append(start * (1.0 - alpha) + end * alpha)
            chunk[key] = np.stack(waypoints).astype(np.float32)

        print(
            f"[ExamplePolicy] Heading to pose {self.pose_index} "
            f"(total steps executed: {self.step_count})"
        )

        # Advance counters
        self.step_count += T
        self.pose_index = (self.pose_index + 1) % len(PREDEFINED_POSES)

        first_action = {k: v[0] for k, v in chunk.items()}
        return first_action, {"action_chunk": chunk}
