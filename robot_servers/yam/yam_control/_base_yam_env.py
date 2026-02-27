"""
Base class for YAM bimanual robot station. Contains code shared between simulation and real environments.
"""

from pathlib import Path
from typing import Literal

import gymnasium as gym
import mujoco
import numpy as np

from yam_control.kinematics import YamKinematics


class _BaseYamEnv(gym.Env):
    CAMERA_HEIGHT, CAMERA_WIDTH = 480, 640

    def __init__(
        self,
        control_mode: Literal["joint_position", "cartesian_position"] = "joint_position",
    ):
        self.control_mode = control_mode
        model_path = Path(__file__).parent / "models" / "station.xml"
        self._spec = mujoco.MjSpec.from_file(str(model_path))
        self._spec.copy_during_attach = True
        self._spec = self._build_task_spec(self._spec)

        for side in ["top", "left", "right"]:
            camera = next(x for x in self._spec.cameras if side in x.name)
            camera.resolution = (self.CAMERA_WIDTH, self.CAMERA_HEIGHT)
            camera.sensor_size = (0.003148, 0.002364)  # FOV from rs.rs2_fov()

        self._model = self._spec.compile()

        left_actuator_names = [x.name for x in self._spec.actuators if x.name.startswith("left_")]
        right_actuator_names = [x.name for x in self._spec.actuators if x.name.startswith("right_")]
        self.left_actuator_ids = [self._model.actuator(name).id for name in left_actuator_names]
        self.right_actuator_ids = [self._model.actuator(name).id for name in right_actuator_names]

        self._kinematics = YamKinematics()

        self.observation_space = self._build_observation_space()
        self.action_space = self._build_action_space()

    def _build_task_spec(self, station_spec: mujoco.MjSpec) -> mujoco.MjSpec:
        """Override this method to add task-specific objects to the scene."""
        return station_spec

    def _build_observation_space(self) -> gym.spaces.Dict:
        space = {}

        for side in ["left", "right"]:
            if self.control_mode == "joint_position":
                space[f"{side}_joint_pos"] = gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32,
                )
            elif self.control_mode == "cartesian_position":
                space.update(
                    {
                        f"{side}_ee_pos": gym.spaces.Box(
                            low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32,
                        ),
                        f"{side}_ee_quat_xyzw": gym.spaces.Box(
                            low=-1, high=1, shape=(4,), dtype=np.float32,
                        ),
                    }
                )

            space[f"{side}_gripper_pos"] = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32,
            )

        for camera in ["top", "left", "right"]:
            space[f"{camera}_camera_image"] = gym.spaces.Box(
                low=0, high=255,
                shape=(self.CAMERA_HEIGHT, self.CAMERA_WIDTH, 3), dtype=np.uint8,
            )

        return gym.spaces.Dict(space)

    def _build_action_space(self) -> gym.spaces.Dict:
        space = {}

        for side, actuator_ids in [
            ("left", self.left_actuator_ids),
            ("right", self.right_actuator_ids),
        ]:
            if self.control_mode == "joint_position":
                ctrl_ranges = self._model.actuator_ctrlrange[actuator_ids]
                low = ctrl_ranges[:, 0].astype(np.float32)
                high = ctrl_ranges[:, 1].astype(np.float32)

                space[f"{side}_joint_pos"] = gym.spaces.Box(
                    low=low[:6], high=high[:6], dtype=np.float32,
                )
            elif self.control_mode == "cartesian_position":
                space.update(
                    {
                        f"{side}_ee_pos": gym.spaces.Box(
                            low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32,
                        ),
                        f"{side}_ee_quat_xyzw": gym.spaces.Box(
                            low=-1, high=1, shape=(4,), dtype=np.float32,
                        ),
                    }
                )

            space[f"{side}_gripper_pos"] = gym.spaces.Box(
                low=np.array([0.0], dtype=np.float32),
                high=np.array([1.0], dtype=np.float32),
                dtype=np.float32,
            )

        return gym.spaces.Dict(space)

    def _convert_from_joint_to_cartesian(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        left_joint_pos = obs["left_joint_pos"]
        right_joint_pos = obs["right_joint_pos"]
        del obs["left_joint_pos"]
        del obs["right_joint_pos"]

        left_ee_pos, left_ee_quat_xyzw, right_ee_pos, right_ee_quat_xyzw = (
            self._kinematics.forward_kinematics(left_joint_pos, right_joint_pos)
        )

        obs["left_ee_pos"] = left_ee_pos
        obs["left_ee_quat_xyzw"] = left_ee_quat_xyzw
        obs["right_ee_pos"] = right_ee_pos
        obs["right_ee_quat_xyzw"] = right_ee_quat_xyzw

        return obs

    def _convert_from_cartesian_to_joint(self, action: dict[str, np.ndarray]) -> np.ndarray:
        left_ee_pos = action["left_ee_pos"]
        left_ee_quat_xyzw = action["left_ee_quat_xyzw"]
        right_ee_pos = action["right_ee_pos"]
        right_ee_quat_xyzw = action["right_ee_quat_xyzw"]
        del action["left_ee_pos"]
        del action["left_ee_quat_xyzw"]
        del action["right_ee_pos"]
        del action["right_ee_quat_xyzw"]

        left_joint_pos, right_joint_pos = self._kinematics.inverse_kinematics(
            left_ee_pos, left_ee_quat_xyzw, right_ee_pos, right_ee_quat_xyzw, seeded=True,
        )

        action["left_joint_pos"] = left_joint_pos
        action["right_joint_pos"] = right_joint_pos

        return action
