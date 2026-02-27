from typing import Any

from yam_control.policy import Action, Info


class HILPolicyWrapper:
    def __init__(self, base_policy: Any, hil_policy: Any):
        self.base_policy = base_policy
        self.hil_policy = hil_policy

        assert (
            self.hil_policy._control_mode == "joint_position"
        ), "HIL policy must be in joint position mode"

        self._took_control_of_left = False
        self._last_left_joint_pos = None
        self._last_left_gripper_pos = None
        self._took_control_of_right = False
        self._last_right_joint_pos = None
        self._last_right_gripper_pos = None

    def get_action(self, observation) -> tuple[Action, Info]:
        hil_action, hil_info = self.hil_policy.get_action(observation)

        if (not hil_info["left_trigger"] and not hil_info["right_trigger"]) and (
            self._took_control_of_left or self._took_control_of_right
        ):
            print("self.base_policy.reset()")
            self.base_policy.reset()
            self._took_control_of_left = False
            self._took_control_of_right = False

        base_action, policy_info = self.base_policy.get_action(observation)
        base_action = base_action.copy()

        base_action["source"] = "policy"

        if hil_info["left_trigger"]:
            base_action["left_joint_pos"] = hil_action["left_joint_pos"]
            base_action["left_gripper_pos"] = hil_action["left_gripper_pos"]
            self._took_control_of_left = True
            self._last_left_joint_pos = hil_action["left_joint_pos"]
            self._last_left_gripper_pos = hil_action["left_gripper_pos"]

            base_action["source"] = "human"
        elif self._took_control_of_left:
            base_action["left_joint_pos"] = self._last_left_joint_pos
            base_action["left_gripper_pos"] = self._last_left_gripper_pos

        if hil_info["right_trigger"]:
            base_action["right_joint_pos"] = hil_action["right_joint_pos"]
            base_action["right_gripper_pos"] = hil_action["right_gripper_pos"]
            self._took_control_of_right = True
            self._last_right_joint_pos = hil_action["right_joint_pos"]
            self._last_right_gripper_pos = hil_action["right_gripper_pos"]

            base_action["source"] = "human"
        elif self._took_control_of_right:
            base_action["right_joint_pos"] = self._last_right_joint_pos
            base_action["right_gripper_pos"] = self._last_right_gripper_pos

        return base_action, policy_info

    def __getattr__(self, name):
        return getattr(self.base_policy, name)
