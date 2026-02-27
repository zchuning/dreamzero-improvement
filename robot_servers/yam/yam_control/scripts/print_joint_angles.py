"""
Script to print joint angles of a YAM arm.

Usage:
    python -m yam_control.scripts.print_joint_angles --channel <channel>
"""

import math
from typing import Literal

from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.utils import GripperType
import tyro


def main(
    channel: Literal["can_leader_l", "can_follow_l", "can_leader_r", "can_follow_r"],
) -> None:
    gripper_type = (
        GripperType.YAM_TEACHING_HANDLE if "leader" in channel else GripperType.LINEAR_4310
    )
    robot = get_yam_robot(
        channel=channel,
        gripper_type=gripper_type,
        zero_gravity_mode=False,
    )
    motor_states = robot.motor_chain.read_states()

    joint_angles = []
    for motor_state in motor_states:
        angle_deg = math.degrees(motor_state.pos)
        joint_angles.append((motor_state.id, angle_deg))

    robot.close()

    print("\nJoint Angles:")
    print("-" * 60)
    for motor_id, angle_deg in joint_angles:
        if abs(angle_deg) > 0.1:
            color = "\033[91m"
            reset = "\033[0m"
            print(f"Joint {motor_id:2d}: {color}{angle_deg:7.2f}°{reset}")
        else:
            print(f"Joint {motor_id:2d}: {angle_deg:7.2f}°")


if __name__ == "__main__":
    tyro.cli(main)
