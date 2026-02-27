"""
Script to set zero position for motor 6 on YAM leader arm.

Usage:
    python -m yam_control.scripts.calibrate_leader_motor_6 --channel can_leader_l
"""

from typing import Literal

from i2rt.motor_drivers.dm_driver import (
    ControlMode,
    DMSingleMotorCanInterface,
    MotorType,
)
import tyro


def main(
    channel: Literal["can_leader_l", "can_leader_r"],
) -> None:
    motor_id = 6
    print(f"Calibrating {channel} motor {motor_id}")

    motor_control_interface = DMSingleMotorCanInterface(
        channel=channel, bustype="socketcan", control_mode=ControlMode.MIT
    )

    motor_control_interface.motor_on(motor_id, MotorType.DM4310)

    while True:
        try:
            before_pos = motor_control_interface.set_control(
                motor_id, MotorType.DM4310, 0, 0, 0, 0, 0
            ).position
            motor_control_interface.save_zero_position(motor_id)
            after_pos = motor_control_interface.set_control(
                motor_id, MotorType.DM4310, 0, 0, 0, 0, 0
            ).position
            print(f"  Motor {motor_id}: {before_pos:7.3f} → {after_pos:7.3f}")
            break
        except Exception:
            pass

    motor_control_interface.motor_off(motor_id)

    motor_control_interface.close()
    print("Calibration complete!")


if __name__ == "__main__":
    tyro.cli(main)
