"""
Script to print motor temperatures of a YAM arm.

Usage:
    python -m yam_control.scripts.print_motor_temps --channel <channel>
"""

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

    motor_temps = []
    for motor_state in motor_states:
        motor_temps.append((motor_state.id, motor_state.temp_mos, motor_state.temp_rotor))

    robot.close()

    print("\nMotor Temperatures:")
    print("-" * 60)
    for motor_id, temp_mos, temp_rotor in motor_temps:
        print(
            f"Motor {motor_id:2d}: " f"MOSFET = {temp_mos:6.1f}°C  " f"Rotor = {temp_rotor:6.1f}°C"
        )


if __name__ == "__main__":
    tyro.cli(main)
