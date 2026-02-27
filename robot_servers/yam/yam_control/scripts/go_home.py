"""Move both follower arms to the all-zeros home pose.

Requires the follower arm servers to be running.

Usage:
    python -m yam_control.scripts.go_home
    python -m yam_control.scripts.go_home --duration 3.0
"""

import time

import numpy as np
import portal
import tyro

from yam_control.constants import LEFT_FOLLOWER_PORT, RIGHT_FOLLOWER_PORT


def go_home(duration: float = 2.0) -> None:
    """Smoothly interpolate both arms to [0,0,0,0,0,0] with gripper open."""
    arms: dict[str, portal.Client] = {
        "left": portal.Client(f"localhost:{LEFT_FOLLOWER_PORT}"),
        "right": portal.Client(f"localhost:{RIGHT_FOLLOWER_PORT}"),
    }

    initial = np.concatenate([
        arms["left"].get_joint_pos().result(),
        arms["right"].get_joint_pos().result(),
    ])
    target = np.zeros(14)
    target[6] = 1.0   # left gripper open
    target[13] = 1.0  # right gripper open

    assert initial.shape == (14,), f"Expected 14-DOF, got {initial.shape}"

    kp = np.array([40, 40, 40, 20, 5, 5, 10])
    kd = np.array([5, 5, 5, 1.5, 1.5, 1.5, 0.5])

    print(f"[GoHome] Moving to home over {duration}s ...")
    start = time.time()
    while time.time() - start < duration:
        alpha = (time.time() - start) / duration
        interp = (1 - alpha) * initial + alpha * target

        for side, pos in [("left", interp[:7]), ("right", interp[7:])]:
            arms[side].command_joint_state({
                "pos": pos,
                "vel": np.zeros(7),
                "kp": kp,
                "kd": kd,
            })
        time.sleep(0.02)

    print("[GoHome] Done.")


if __name__ == "__main__":
    tyro.cli(go_home)
