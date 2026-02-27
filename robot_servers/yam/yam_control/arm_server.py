"""
Server for controlling YAM robot arm from client process.

Adapted from `minimum_gello.py` in the `i2rt` repo:
https://github.com/i2rt-robotics/i2rt/blob/main/scripts/minimum_gello.py

Requires the `arm-server` optional dependency:
    pip install -e ".[arm-server]"
"""

from dataclasses import dataclass
from typing import Any, Literal

from yam_control.constants import (
    LEFT_FOLLOWER_CAN_INTERFACE,
    LEFT_FOLLOWER_PORT,
    LEFT_LEADER_CAN_INTERFACE,
    LEFT_LEADER_PORT,
    RIGHT_FOLLOWER_CAN_INTERFACE,
    RIGHT_FOLLOWER_PORT,
    RIGHT_LEADER_CAN_INTERFACE,
    RIGHT_LEADER_PORT,
)
from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.motor_chain_robot import MotorChainRobot
from i2rt.robots.utils import GripperType
import numpy as np
import portal
import tyro


class FollowerRobotServer:
    def __init__(self, robot: MotorChainRobot, port: int):
        self._robot = robot
        self._server = portal.Server(port)
        self._server.bind("get_joint_pos", self._robot.get_joint_pos)
        self._server.bind("command_joint_pos", self._robot.command_joint_pos)
        self._server.bind("command_joint_state", self._robot.command_joint_state)
        self._server.bind("get_observations", self._robot.get_observations)

    def serve(self) -> None:
        self._server.start()


class LeaderRobot:
    def __init__(self, robot: MotorChainRobot):
        self._robot = robot
        self._motor_chain = robot.motor_chain

    def get_info(self) -> np.ndarray:
        qpos = self._robot.get_observations()["joint_pos"]
        encoder_obs = self._motor_chain.get_same_bus_device_states()
        gripper_cmd = 1 - encoder_obs[0].position
        qpos_with_gripper = np.concatenate([qpos, [gripper_cmd]])
        return qpos_with_gripper, encoder_obs[0].io_inputs

    def command_joint_state(self, joint_state: dict[str, np.ndarray]) -> None:
        """
        Command full joint state (pos, vel, kp, kd) to the leader arm.

        This is the only safe way to command leader arms as it requires
        explicit PD gains, preventing uncontrolled motion.
        """
        self._robot.command_joint_state(joint_state)


class LeaderRobotServer:
    def __init__(self, robot: LeaderRobot, port: int):
        self._robot = robot
        self._server = portal.Server(port)
        self._server.bind("get_info", self._robot.get_info)
        # Only expose command_joint_state for safety (requires explicit gains)
        self._server.bind("command_joint_state", self._robot.command_joint_state)

    def serve(self) -> None:
        self._server.start()


@dataclass
class Args:
    mode: Literal["follower", "leader"] = "follower"
    side: Literal["left", "right"] = "left"


def main(args: Args) -> None:
    CAN_INTERFACE_MAP = {
        "follower": {
            "left": LEFT_FOLLOWER_CAN_INTERFACE,
            "right": RIGHT_FOLLOWER_CAN_INTERFACE,
        },
        "leader": {
            "left": LEFT_LEADER_CAN_INTERFACE,
            "right": RIGHT_LEADER_CAN_INTERFACE,
        },
    }

    GRIPPER_TYPE_MAP = {
        "follower": GripperType.from_string_name("linear_4310"),    # Note: Use linear_4310 for new gripper motors; Use crank_4310 for old gripper motors.
        "leader": GripperType.from_string_name("yam_teaching_handle"),
    }

    PORT_MAP = {
        "follower": {
            "left": LEFT_FOLLOWER_PORT,
            "right": RIGHT_FOLLOWER_PORT,
        },
        "leader": {
            "left": LEFT_LEADER_PORT,
            "right": RIGHT_LEADER_PORT,
        },
    }

    SERVER_MAP: dict[str, type[Any]] = {
        "follower": FollowerRobotServer,
        "leader": LeaderRobotServer,
    }

    # Instantiate robot
    can_interface = CAN_INTERFACE_MAP[args.mode][args.side]
    gripper_type = GRIPPER_TYPE_MAP[args.mode]
    robot = get_yam_robot(channel=can_interface, gripper_type=gripper_type)

    # Wrap leader arm to support reading gripper state
    if args.mode == "leader":
        robot = LeaderRobot(robot)

    # Start server
    port = PORT_MAP[args.mode][args.side]
    print(f"Starting server for {args.side} {args.mode} arm at localhost:{port}")
    server_robot = SERVER_MAP[args.mode](robot, port)
    server_robot.serve()


if __name__ == "__main__":
    main(tyro.cli(Args))
