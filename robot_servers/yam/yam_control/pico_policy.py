"""End-effector control policy driven by Pico XR controller via Portal."""

import threading
import time

import numpy as np
import portal
from scipy.spatial.transform import Rotation as R

from yam_control.constants import PICO_PORT
from yam_control.policy import Action, Info, Observation, Options, Policy

TRIGGER_PRESS_THRESHOLD = 0.5


class PicoClient:
    def __init__(self, host: str = "localhost", port: int = PICO_PORT):
        self.client = portal.Client(f"{host}:{port}")

    def get_info(self):
        return self.client.get_info().result()


class PicoPolicy(Policy):
    """End-effector control policy driven by Pico controller via Portal."""

    def __init__(
        self, host: str = "localhost", port: int = PICO_PORT, env=None, standalone: bool = False
    ):
        self._client = PicoClient(host, port)
        self._lock = threading.Lock()
        self._env = env
        assert self._env is not None, "Environment must be provided"
        self._standalone = standalone
        self._control_mode = self._env.unwrapped.control_mode

        self._latest_left_wrist_matrix = np.eye(4, dtype=np.float64)
        self._latest_right_wrist_matrix = np.eye(4, dtype=np.float64)
        self._latest_left_trigger = 0.0
        self._latest_right_trigger = 0.0
        self._latest_left_gripper = 1.0
        self._latest_right_gripper = 1.0
        self._latest_button_x = False
        self._latest_button_y = False
        self._latest_button_a = False
        self._latest_button_b = False

        self._side_state = {
            side: {
                "teleop_init": np.eye(4, dtype=np.float64),
                "arm_init": np.eye(4, dtype=np.float64),
                "target": np.eye(4, dtype=np.float64),
                "trigger_pressed": False,
            }
            for side in ("left", "right")
        }

        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def _poll_loop(self):
        while self._running:
            try:
                data = self._client.get_info()
                self._update_state(data)
            except Exception:
                time.sleep(0.01)
            time.sleep(0.001)

    def _update_state(self, data):
        with self._lock:
            self._latest_left_wrist_matrix = np.array(data["left_wrist_pose"], dtype=np.float64)
            self._latest_right_wrist_matrix = np.array(data["right_wrist_pose"], dtype=np.float64)
            buttons = data["buttons"]
            self._latest_left_trigger = float(buttons["left_trigger"])
            self._latest_right_trigger = float(buttons["right_trigger"])
            self._latest_left_gripper = float(buttons["left_gripper"])
            self._latest_right_gripper = float(buttons["right_gripper"])
            self._latest_button_x = bool(buttons["X"])
            self._latest_button_y = bool(buttons["Y"])
            self._latest_button_a = bool(buttons["A"])
            self._latest_button_b = bool(buttons["B"])

    def get_action(
        self, observation: Observation, options: Options | None = None
    ) -> tuple[Action, Info]:
        del options
        action: Action = {}
        info: Info = {}

        with self._lock:
            left_obs_matrix, right_obs_matrix = self._observation_to_matrix(observation)

            left_target_matrix = self._update_side_target(
                "left",
                teleop_matrix=self._latest_left_wrist_matrix,
                obs_matrix=left_obs_matrix,
                trigger_value=self._latest_left_trigger,
            )
            right_target_matrix = self._update_side_target(
                "right",
                teleop_matrix=self._latest_right_wrist_matrix,
                obs_matrix=right_obs_matrix,
                trigger_value=self._latest_right_trigger,
            )

            left_target_pose = self._matrix_to_pose_xyzw(left_target_matrix)
            right_target_pose = self._matrix_to_pose_xyzw(right_target_matrix)

            if self._control_mode == "joint_position":
                left_joint_pos, right_joint_pos = (
                    self._env.unwrapped._kinematics.inverse_kinematics(
                        left_target_pose[:3],
                        left_target_pose[3:],
                        right_target_pose[:3],
                        right_target_pose[3:],
                        seeded=True,
                    )
                )
                action["left_joint_pos"] = left_joint_pos
                action["right_joint_pos"] = right_joint_pos
            elif self._control_mode == "cartesian_position":
                action["left_ee_pos"] = np.asarray(left_target_pose[:3], dtype=np.float32)
                action["left_ee_quat_xyzw"] = np.asarray(left_target_pose[3:], dtype=np.float32)
                action["right_ee_pos"] = np.asarray(right_target_pose[:3], dtype=np.float32)
                action["right_ee_quat_xyzw"] = np.asarray(right_target_pose[3:], dtype=np.float32)
            else:
                raise ValueError(f"Invalid control mode: {self._control_mode}")

            action["left_gripper_pos"] = np.array([self._latest_left_gripper], dtype=np.float32)
            action["right_gripper_pos"] = np.array([self._latest_right_gripper], dtype=np.float32)

            info.update(
                {
                    "left_teleop_wrist_pose": self._matrix_to_pose_wxyz(
                        self._latest_left_wrist_matrix
                    ),
                    "right_teleop_wrist_pose": self._matrix_to_pose_wxyz(
                        self._latest_right_wrist_matrix
                    ),
                    "left_trigger": self._latest_left_trigger,
                    "right_trigger": self._latest_right_trigger,
                    "button_x": self._latest_button_x,
                    "button_y": self._latest_button_y,
                    "button_a": self._latest_button_a,
                    "button_b": self._latest_button_b,
                }
            )

        return action, info

    def _update_side_target(
        self,
        side: str,
        teleop_matrix: np.ndarray,
        obs_matrix: np.ndarray | None,
        trigger_value: float,
    ) -> np.ndarray:
        state = self._side_state[side]
        pressed = trigger_value > TRIGGER_PRESS_THRESHOLD

        if pressed and not state["trigger_pressed"]:
            if obs_matrix is not None:
                state["arm_init"] = obs_matrix.copy()
            state["teleop_init"] = teleop_matrix.copy()

        state["trigger_pressed"] = pressed

        if pressed and obs_matrix is not None:
            try:
                teleop_init_pos = state["teleop_init"][:3, 3]
                teleop_curr_pos = teleop_matrix[:3, 3]
                teleop_init_rot = state["teleop_init"][:3, :3]
                teleop_curr_rot = teleop_matrix[:3, :3]

                pos_delta = teleop_curr_pos - teleop_init_pos

                rot_delta = teleop_curr_rot @ teleop_init_rot.T

                arm_init_pos = state["arm_init"][:3, 3]
                arm_init_rot = state["arm_init"][:3, :3]

                target = np.eye(4)
                target[:3, 3] = arm_init_pos + pos_delta
                target[:3, :3] = rot_delta @ arm_init_rot
            except np.linalg.LinAlgError:
                target = obs_matrix.copy()
        else:
            target = obs_matrix.copy() if obs_matrix is not None else state["target"].copy()

        state["target"] = target.copy()
        return target

    def _observation_to_matrix(
        self,
        observation: Observation | None,
    ) -> np.ndarray | None:
        if observation is None:
            return None

        if self._control_mode == "joint_position":
            (left_pos, left_quat_xyzw, right_pos, right_quat_xyzw) = (
                self._env.unwrapped._kinematics.forward_kinematics(
                    observation["left_joint_pos"], observation["right_joint_pos"]
                )
            )
        elif self._control_mode == "cartesian_position":
            left_pos = observation["left_ee_pos"]
            left_quat_xyzw = observation["left_ee_quat_xyzw"]
            right_pos = observation["right_ee_pos"]
            right_quat_xyzw = observation["right_ee_quat_xyzw"]
        else:
            raise ValueError(f"Invalid control mode: {self._control_mode}")

        return (
            self._components_to_matrix(left_pos, left_quat_xyzw),
            self._components_to_matrix(right_pos, right_quat_xyzw),
        )

    def _components_to_matrix(self, pos, quat) -> np.ndarray:
        pos_arr = np.asarray(pos, dtype=np.float64).reshape(3)
        quat_arr = np.asarray(quat, dtype=np.float64).reshape(4)
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = R.from_quat(quat_arr).as_matrix()
        matrix[:3, 3] = pos_arr
        return matrix

    def _matrix_to_pose_xyzw(self, matrix: np.ndarray) -> np.ndarray:
        rot = R.from_matrix(matrix[:3, :3])
        quat_xyzw = rot.as_quat(scalar_first=False)
        return np.concatenate((matrix[:3, 3], quat_xyzw))

    def _matrix_to_pose_wxyz(self, matrix: np.ndarray) -> np.ndarray:
        rot = R.from_matrix(matrix[:3, :3])
        quat_wxyz = rot.as_quat(scalar_first=True)
        return np.concatenate((matrix[:3, 3], quat_wxyz))
