import threading
import time

import gymnasium as gym
import numpy as np
import portal

from yam_control.constants import LEFT_LEADER_PORT, RIGHT_LEADER_PORT
from yam_control.policy import Action, Info, Observation, Options, Policy


class LeaderRobotClient:
    """
    Client for commanding leader arms with built-in safety limits.

    Safety features:
    - Maximum gain limits to prevent excessive forces
    - Maximum velocity limits to prevent unsafe speeds
    - Input validation and clamping
    """

    # Safety limits.
    # Should all have length 6, since gripper is not actuated on leader
    MAX_KP = np.array([80, 80, 80, 40, 15, 15])  # Max proportional gains
    MAX_KD = np.array([10, 10, 10, 5, 5, 5])  # Max derivative gains
    MAX_VEL = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0])  # Max velocity (rad/s)

    def __init__(self, host: str = "localhost", port: int = LEFT_LEADER_PORT):
        self._client = portal.Client(f"{host}:{port}")

    def get_info(self) -> np.ndarray:
        return self._client.get_info().result()

    def command_joint_state(self, joint_state: dict[str, np.ndarray]) -> None:
        """
        Command full joint state (pos, vel, kp, kd) to the leader arm with safety limits.

        Safety features:
        - Clamps Kp to MAX_KP to limit maximum forces
        - Clamps Kd to MAX_KD to limit damping
        - Clamps velocities to MAX_VEL to prevent unsafe speeds
        - Validates required keys are present

        Args:
            joint_state: Dict with keys "pos", "vel", "kp", "kd" (all np.ndarray)
        """
        # Validate required keys
        required_keys = {"pos", "vel", "kp", "kd"}
        if not required_keys.issubset(joint_state.keys()):
            missing = required_keys - joint_state.keys()
            raise ValueError(f"Missing required keys in joint_state: {missing}")

        # Create safe copy
        safe_state = joint_state.copy()

        # Clamp gains to safety limits
        original_kp = safe_state["kp"].copy()
        original_kd = safe_state["kd"].copy()
        safe_state["kp"] = np.minimum(safe_state["kp"], self.MAX_KP)
        safe_state["kd"] = np.minimum(safe_state["kd"], self.MAX_KD)

        # Clamp velocities to safety limits
        original_vel = safe_state["vel"].copy()
        safe_state["vel"] = np.clip(safe_state["vel"], -self.MAX_VEL, self.MAX_VEL)

        # Log warnings if limits were enforced
        if not np.allclose(original_kp, safe_state["kp"]):
            print(f"[Safety] Kp clamped: {original_kp} -> {safe_state['kp']}")
        if not np.allclose(original_kd, safe_state["kd"]):
            print(f"[Safety] Kd clamped: {original_kd} -> {safe_state['kd']}")
        if not np.allclose(original_vel, safe_state["vel"]):
            print(f"[Safety] Velocity clamped: {original_vel} -> {safe_state['vel']}")

        # Send safe command
        self._client.command_joint_state(safe_state)


class TeleopPolicy(Policy):
    """Simple teleoperation policy that mirrors leader arm positions to follower arms."""

    def __init__(self, host: str = "localhost"):
        self.leader_arms = {
            "left": LeaderRobotClient(host=host, port=LEFT_LEADER_PORT),
            "right": LeaderRobotClient(host=host, port=RIGHT_LEADER_PORT),
        }
        self.prev_pressed = [[False, False], [False, False]]
        self.button_map = {
            (0, 0): "reset",
            (0, 1): "discard",
            (1, 0): "start",
            (1, 1): "save",
        }
        self._first_call = True

        # Discard confirmation state
        self._discard_confirmation: dict | None = None

    def _start_discard_confirmation(self):
        """Spawn background thread to get terminal confirmation."""
        self._discard_confirmation = {"value": None, "ready": False}

        def _prompt():
            result = input("Are you sure you want to discard this recording? (y/n/s): ")
            self._discard_confirmation["value"] = result.strip().lower()
            self._discard_confirmation["ready"] = True

        threading.Thread(target=_prompt, daemon=True).start()

    def _poll_discard_confirmation(self) -> str | None:
        """Check if confirmation is ready. Returns response or None."""
        if self._discard_confirmation and self._discard_confirmation["ready"]:
            value = self._discard_confirmation["value"]
            self._discard_confirmation = None
            return value
        return None

    def get_action(
        self, observation: Observation, options: Options | None = None
    ) -> tuple[Action, Info]:
        # These arguments are not used
        del observation
        del options
        action = {}
        info = {}
        for side_idx, (side, leader) in enumerate(self.leader_arms.items()):
            joint_pos, button = leader.get_info()

            # Clip joint 6 encoder readings to stay within joint limit
            joint_pos[5] = np.clip(joint_pos[5], -np.pi / 2, np.pi / 2)

            # Joint positions
            action[f"{side}_joint_pos"] = joint_pos[:6]
            action[f"{side}_gripper_pos"] = joint_pos[6:]

            # Hardware bug detection: buttons sometimes report 1.0 on first read
            # If any button is pressed on the very first call, this indicates
            # a hardware fault and we cannot trust the button state
            if self._first_call:
                for btn_idx in (0, 1):
                    if button[btn_idx] > 0.5:
                        raise RuntimeError(
                            f"Hardware fault detected: {side} arm button {btn_idx} "
                            f"reports pressed (value={button[btn_idx]:.2f}) on first read. "
                            "This is a known hardware bug. Please restart the system."
                        )

            # Button edge detection
            for btn_idx in (0, 1):
                pressed = button[btn_idx] > 0.5
                if pressed and not self.prev_pressed[side_idx][btn_idx]:
                    info[self.button_map[(side_idx, btn_idx)]] = True
                self.prev_pressed[side_idx][btn_idx] = pressed

        self._first_call = False
        action["source"] = "human"

        # Check for pending discard confirmation result
        confirmation = self._poll_discard_confirmation()
        if confirmation == "y":
            return action, {"discard_confirmed": True}
        elif confirmation == "s":
            return action, {"save": True}
        elif confirmation is not None:  # "n" or invalid
            return action, {"discard_cancelled": True}

        # While waiting for confirmation, swallow all button events
        if self._discard_confirmation is not None:
            return action, {}

        # Start discard confirmation if discard button pressed
        if "discard" in info:
            self._start_discard_confirmation()
            return action, {}

        return action, info

    def reset(self) -> Info:
        while True:
            action, info = self.get_action(None)
            if "reset" in info:
                return {"initial_state": action}

            time.sleep(0.01)


def main():
    from gymnasium.envs.registration import register

    # Environment
    register(id="YamReal-v0", entry_point="yam_control.yam_real_env:YamRealEnv")
    env = gym.make("YamReal-v0")

    # Policy
    policy = TeleopPolicy()
    policy_info = policy.reset()

    # Main loop
    obs, info = env.reset(options={"initial_state": policy_info["initial_state"]})
    try:
        while True:
            action, _ = policy.get_action(obs)
            obs, _, _, _, _ = env.step(action)
    finally:
        env.close()


if __name__ == "__main__":
    main()
