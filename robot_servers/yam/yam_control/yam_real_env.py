"""Real environment for YAM bimanual robot station.

Only supports joint_position control mode.
"""

import os
import threading
import time
from typing import Any

import gymnasium as gym
import numpy as np
import portal
import pyrealsense2 as rs

from yam_control.constants import LEFT_FOLLOWER_PORT, RIGHT_FOLLOWER_PORT
from yam_control.realsense import RealSenseCamera


CAMERA_HEIGHT, CAMERA_WIDTH = 480, 640


class FollowerRobotClient:
    def __init__(self, host: str = "127.0.0.1", port: int = LEFT_FOLLOWER_PORT):
        self._client = portal.Client(f"{host}:{port}")

    def get_joint_pos(self) -> np.ndarray:
        return self._client.get_joint_pos().result()

    def command_joint_pos(self, joint_pos: np.ndarray) -> None:
        self._client.command_joint_pos(joint_pos)

    def command_joint_state(self, joint_state: dict[str, np.ndarray]) -> None:
        self._client.command_joint_state(joint_state)

    def get_observations(self) -> dict[str, np.ndarray]:
        return self._client.get_observations().result()


class NonBlockingRealSense:
    def __init__(self, camera_name: str):
        serial = self._get_device_id(camera_name)
        self.camera = RealSenseCamera(
            device_id=serial,
            resolution=(640, 480),
            fps=60,
            auto_exposure=True,
            brightness=10,
        )
        self.image = None

        # Camera worker
        self.running = True
        self.worker_thread = threading.Thread(target=self._camera_worker, daemon=True)
        self.worker_thread.start()

        # Wait for worker to start
        while self.image is None:
            time.sleep(0.01)

    def _get_device_id(self, camera_name: str) -> str:
        # Check for device symlink
        symlink_path = f"/dev/video_{camera_name}"
        if not os.path.exists(symlink_path):
            raise FileNotFoundError(f"No symlink found: {symlink_path}")

        # Get USB port identifier from symlink
        video = os.path.basename(os.path.realpath(symlink_path))
        device_path = f"/sys/class/video4linux/{video}/device"
        usb_id = os.path.basename(os.path.realpath(device_path))

        # Get camera serial number from USB port identifier
        context = rs.context()
        for dev in context.query_devices():
            if usb_id in dev.get_info(rs.camera_info.physical_port):
                return dev.get_info(rs.camera_info.serial_number)

        raise ValueError(f"No RealSense device found for symlink: {symlink_path}")

    def _camera_worker(self):
        while self.running:
            data = self.camera.read()  # Blocks for ~33 ms
            if data is not None and data.images["rgb"] is not None:
                self.image = data.images["rgb"]

    def get_image(self):
        return self.image.copy()

    def close(self):
        self.running = False
        self.worker_thread.join()
        self.camera.stop()


class YamRealEnv(gym.Env):
    """Real YAM bimanual robot environment (joint_position control only).

    Observation and action spaces are defined directly without loading a model.
    """

    def __init__(self, policy_control_freq: float = 30.0):
        super().__init__()
        self.policy_control_freq = policy_control_freq
        self.control_period = 1.0 / policy_control_freq
        self.last_step_time = time.time()
        self._last_overrun_warn = 0.0

        # Observation and action spaces
        self.observation_space = self._build_observation_space()
        self.action_space = self._build_action_space()

        # Follower arms
        self.follower_arms = {}
        for side, port in [
            ("left", LEFT_FOLLOWER_PORT),
            ("right", RIGHT_FOLLOWER_PORT),
        ]:
            follower_arm = FollowerRobotClient(host="localhost", port=port)

            # Set arm to gravcomp mode
            follower_arm.command_joint_state(
                {
                    "pos": np.zeros(7),
                    "vel": np.zeros(7),
                    "kp": np.zeros(7),
                    "kd": np.zeros(7),
                }
            )

            self.follower_arms[side] = follower_arm

        # Cameras
        self.cameras = {
            camera_name: NonBlockingRealSense(camera_name)
            for camera_name in ["top", "left", "right"]
        }

    def _build_observation_space(self) -> gym.spaces.Dict:
        space = {}
        for side in ["left", "right"]:
            space[f"{side}_joint_pos"] = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
            )
            space[f"{side}_gripper_pos"] = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
            )
        for camera in ["top", "left", "right"]:
            space[f"{camera}_camera_image"] = gym.spaces.Box(
                low=0, high=255, shape=(CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8
            )
        return gym.spaces.Dict(space)

    def _build_action_space(self) -> gym.spaces.Dict:
        space = {}
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

    def step(
        self, action: dict[str, np.ndarray]
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        # Command follower arms to desired joint positions
        for side, follower_arm in self.follower_arms.items():
            follower_arm.command_joint_pos(
                np.concatenate([action[f"{side}_joint_pos"], action[f"{side}_gripper_pos"]])
            )

        # Maintain desired control freq
        sleep_end_time = self.last_step_time + self.control_period
        now = time.time()
        if now > sleep_end_time and now - self._last_overrun_warn >= 5.0:
            print(
                f"Warning: Control loop timing overrun "
                f"(budget {self.control_period * 1000:.1f} ms, "
                f"actual {1000 * (now - self.last_step_time):.1f} ms)"
            )
            self._last_overrun_warn = now
        while time.time() < sleep_end_time:
            time.sleep(0.0001)
        self.last_step_time = time.time()

        obs = self._get_obs()
        reward = 0.0
        info = {}

        return obs, reward, False, False, info

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        options = options if options is not None else {}

        # Interpolate to initial state
        print("Resetting the environment...")
        if "initial_state" in options:
            initial_state = options["initial_state"]
        else:
            initial_state = {
                "left_joint_pos": np.zeros(6),
                "left_gripper_pos": np.ones(1),
                "right_joint_pos": np.zeros(6),
                "right_gripper_pos": np.ones(1),
            }
        self._interpolate_to_target(initial_state)
        print("Done resetting the environment")

        self.last_step_time = time.time()

        obs = self._get_obs()
        info = {}

        return obs, info

    def _interpolate_to_target(
        self,
        joint_state: dict[str, np.ndarray],
        duration: float = 2.0,
    ) -> None:
        """Interpolate to target joint state over given duration."""
        initial_pos = np.concatenate(
            [
                self.follower_arms["left"].get_joint_pos(),
                self.follower_arms["right"].get_joint_pos(),
            ]
        )
        target_pos = np.concatenate(
            [
                joint_state["left_joint_pos"],
                joint_state["left_gripper_pos"],
                joint_state["right_joint_pos"],
                joint_state["right_gripper_pos"],
            ]
        )
        assert initial_pos.shape == (14,) and target_pos.shape == (14,)

        # Use lower gains during interpolation
        kp = np.array([40, 40, 40, 20, 5, 5, 10])  # Default: [80, 80, 80, 40, 10, 10, 20]
        kd = np.array([5, 5, 5, 1.5, 1.5, 1.5, 0.5])  # Default: [5, 5, 5, 1.5, 1.5, 1.5, 0.5]

        start_time = time.time()
        while time.time() - start_time < duration:
            # Linearly interpolate between initial and target joint state
            alpha = (time.time() - start_time) / duration
            interp_pos = (1 - alpha) * initial_pos + alpha * target_pos

            # Command follower arms to interpolated joint state
            for side, joint_pos in [
                ("left", interp_pos[:7]),
                ("right", interp_pos[7:]),
            ]:
                self.follower_arms[side].command_joint_state(
                    {
                        "pos": joint_pos,
                        "vel": np.zeros(7),
                        "kp": kp,
                        "kd": kd,
                    }
                )

            time.sleep(0.02)

    def _get_obs(self) -> dict[str, np.ndarray]:
        obs = {}

        # Joint and gripper states
        for side, follower_arm in self.follower_arms.items():
            follower_obs = follower_arm.get_observations()
            obs[f"{side}_joint_pos"] = follower_obs["joint_pos"]
            obs[f"{side}_gripper_pos"] = follower_obs["gripper_pos"]

        # Camera images
        for camera_name in ["top", "left", "right"]:
            obs[f"{camera_name}_camera_image"] = self.cameras[camera_name].get_image()

        # Convert float64 to float32
        obs = {k: v.astype(np.float32) if v.dtype == np.float64 else v for k, v in obs.items()}

        return obs

    def close(self):
        for camera in self.cameras.values():
            camera.close()
