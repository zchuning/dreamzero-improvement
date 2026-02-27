"""MuJoCo simulation environment for YAM bimanual robot.

Requires ``mujoco`` (install with ``uv sync --extra sim``).
Only supports joint_position control mode, matching :class:`YamRealEnv`.
"""

import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

CAMERA_HEIGHT, CAMERA_WIDTH = 480, 640
MODEL_PATH = Path(__file__).parent / "models" / "station.xml"


class YamSimEnv(gym.Env):
    """Simulated YAM bimanual robot environment using MuJoCo."""

    GRIPPER_CTRL_SCALE = 0.041
    GRIPPER_QPOS_SCALE = 0.0376

    def __init__(self, policy_control_freq: float = 30.0):
        import mujoco

        super().__init__()
        self.policy_control_freq = policy_control_freq
        self.control_period = 1.0 / policy_control_freq
        self.last_step_time = time.time()

        # Load model
        self._spec = mujoco.MjSpec.from_file(str(MODEL_PATH))
        self._spec.copy_during_attach = True

        # Camera resolution
        for side in ["top", "left", "right"]:
            camera = next(x for x in self._spec.cameras if side in x.name)
            camera.resolution = (CAMERA_WIDTH, CAMERA_HEIGHT)
            camera.sensor_size = (0.003148, 0.002364)

        self._model = self._spec.compile()
        self._data = mujoco.MjData(self._model)
        self._n_substeps = int(self.control_period // self._model.opt.timestep)
        self._renderer = None

        # Joint mappings
        self.left_joint_names = [
            x.name for x in self._spec.joints if x.name.startswith("left_")
        ]
        self.right_joint_names = [
            x.name for x in self._spec.joints if x.name.startswith("right_")
        ]
        self.left_joint_ids = np.array(
            [self._model.joint(n).id for n in self.left_joint_names]
        )
        self.right_joint_ids = np.array(
            [self._model.joint(n).id for n in self.right_joint_names]
        )

        # Actuator mappings
        left_act = [x.name for x in self._spec.actuators if x.name.startswith("left_")]
        right_act = [x.name for x in self._spec.actuators if x.name.startswith("right_")]
        self.left_actuator_ids = [self._model.actuator(n).id for n in left_act]
        self.right_actuator_ids = [self._model.actuator(n).id for n in right_act]
        self.actuator_ids = np.concatenate([self.left_actuator_ids, self.right_actuator_ids])

        # Camera name -> id
        self.camera_ids = {}
        for side in ["top", "left", "right"]:
            cam = next(x for x in self._spec.cameras if side in x.name)
            self.camera_ids[side] = cam.name

        # Spaces
        self.observation_space = self._build_observation_space()
        self.action_space = self._build_action_space()

    # -- spaces --

    def _build_observation_space(self) -> gym.spaces.Dict:
        space: dict[str, gym.spaces.Box] = {}
        for side in ["left", "right"]:
            space[f"{side}_joint_pos"] = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
            )
            space[f"{side}_gripper_pos"] = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
            )
        for cam in ["top", "left", "right"]:
            space[f"{cam}_camera_image"] = gym.spaces.Box(
                low=0, high=255, shape=(CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8
            )
        return gym.spaces.Dict(space)

    def _build_action_space(self) -> gym.spaces.Dict:
        space: dict[str, gym.spaces.Box] = {}
        for side in ["left", "right"]:
            space[f"{side}_joint_pos"] = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
            )
            space[f"{side}_gripper_pos"] = gym.spaces.Box(
                low=np.array([0.0], dtype=np.float32),
                high=np.array([1.0], dtype=np.float32),
                dtype=np.float32,
            )
        return gym.spaces.Dict(space)

    # -- step / reset --

    def step(
        self, action: dict[str, np.ndarray]
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        import mujoco

        ctrl = np.concatenate([
            action["left_joint_pos"],
            action["left_gripper_pos"] * self.GRIPPER_CTRL_SCALE,
            action["right_joint_pos"],
            action["right_gripper_pos"] * self.GRIPPER_CTRL_SCALE,
        ])
        self._data.ctrl[self.actuator_ids] = ctrl

        for _ in range(self._n_substeps):
            mujoco.mj_step(self._model, self._data)

        # Maintain control frequency
        sleep_end = self.last_step_time + self.control_period
        if time.time() > sleep_end:
            overrun = (time.time() - self.last_step_time) * 1000
            print(f"Warning: Control loop overrun ({overrun:.1f}ms > {self.control_period*1000:.1f}ms)")
        while time.time() < sleep_end:
            time.sleep(0.0001)
        self.last_step_time = time.time()

        return self._get_obs(), 0.0, False, False, {}

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        import mujoco

        super().reset(seed=seed)
        mujoco.mj_resetData(self._model, self._data)

        for side, joint_ids, act_ids in [
            ("left", self.left_joint_ids, self.left_actuator_ids),
            ("right", self.right_joint_ids, self.right_actuator_ids),
        ]:
            if options and "initial_state" in options:
                s = options["initial_state"]
                self._data.qpos[joint_ids[:6]] = s[f"{side}_joint_pos"]
                self._data.qpos[joint_ids[6:8]] = s[f"{side}_gripper_pos"] * np.array(
                    [self.GRIPPER_QPOS_SCALE, -self.GRIPPER_QPOS_SCALE]
                )
                self._data.ctrl[act_ids[:6]] = s[f"{side}_joint_pos"]
                self._data.ctrl[act_ids[6:7]] = s[f"{side}_gripper_pos"] * self.GRIPPER_CTRL_SCALE
            else:
                self._data.qpos[joint_ids[6:8]] = [
                    self.GRIPPER_QPOS_SCALE, -self.GRIPPER_QPOS_SCALE
                ]
                self._data.ctrl[act_ids[6:7]] = self.GRIPPER_CTRL_SCALE

        for _ in range(self._n_substeps):
            mujoco.mj_step(self._model, self._data)

        self.last_step_time = time.time()
        return self._get_obs(), {}

    # -- observation / rendering --

    def _get_obs(self) -> dict[str, np.ndarray]:
        obs: dict[str, Any] = {}
        for side, joint_ids in [
            ("left", self.left_joint_ids),
            ("right", self.right_joint_ids),
        ]:
            qpos = self._data.qpos[joint_ids]
            obs[f"{side}_joint_pos"] = qpos[:6]
            obs[f"{side}_gripper_pos"] = (
                np.abs(qpos[6:]).mean(keepdims=True) / self.GRIPPER_QPOS_SCALE
            )

        for cam_name in ["top", "left", "right"]:
            obs[f"{cam_name}_camera_image"] = self._render(cam_name)

        return {
            k: v.astype(np.float32) if v.dtype == np.float64 else v
            for k, v in obs.items()
        }

    def _render(self, camera_name: str) -> np.ndarray:
        import mujoco

        if self._renderer is None:
            self._renderer = mujoco.Renderer(self._model, CAMERA_HEIGHT, CAMERA_WIDTH)
            self._renderer.disable_depth_rendering()
            self._renderer.disable_segmentation_rendering()
        self._renderer.update_scene(self._data, camera=self.camera_ids[camera_name])
        return self._renderer.render()


class MujocoViewerWrapper(gym.Wrapper):
    """Wraps a MuJoCo env to show a live interactive viewer window."""

    def __init__(self, env: gym.Env):
        import mujoco.viewer

        super().__init__(env)
        self.viewer = mujoco.viewer.launch_passive(
            env.unwrapped._model, env.unwrapped._data
        )

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        if not self.viewer.is_running():
            raise RuntimeError("MuJoCo viewer was closed")
        self.viewer.sync()
        return obs, reward, terminated, truncated, info

    def close(self):
        self.viewer.close()
