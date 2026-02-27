"""
Policy wrapper that adds operator intervention support to neural network policies.

This wrapper allows operators to take manual control during evaluation by
pressing a button on the leader arms. The neural network policy is paused
during intervention, and operator commands via leader arms are sent to the
follower arms instead.

State machine:
- policy_control: Neural network policy controls the robot
- reverse_sync: Leader arms sync to follower positions (button held, waiting for proximity)
- leader_control: Operator controls via leader arms (after proximity reached)
"""

import numpy as np

from yam_control.action_utils import hold_action_from_proprio
from yam_control.intervention_state_machine import InterventionStateMachine
from yam_control.policy import Action, Info, Observation, Policy
from yam_control.teleop_policy import TeleopPolicy


class InterventionPolicyWrapper(Policy):
    """
    Policy wrapper that adds operator intervention capabilities.

    Wraps a neural network policy and allows operators to take manual control
    during evaluation. Manages three states:
    1. policy_control: Neural policy controls the robot
    2. reverse_sync: Syncing leader arms to follower positions (button held)
    3. leader_control: Operator controls robot via leader arms
    """

    MAX_POSITION_STEP = 0.10  # rad per iteration

    SYNC_BUTTON = (0, 0)  # Left arm, top button
    PAUSE_BUTTON = (0, 1)  # Left arm, bottom button

    LOW_KP = np.array([30, 30, 30, 20, 15, 15])
    LOW_KD = np.array([2, 2, 1.5, 1, 1, 1])

    def __init__(
        self,
        policy: Policy,
        proximity_threshold_deg: float,
        host: str = "localhost",
    ):
        self.policy: Policy = policy

        self.teleop = TeleopPolicy(host=host)

        self.teleop.button_map[self.SYNC_BUTTON] = "sync"
        self.teleop.button_map[self.PAUSE_BUTTON] = "pause_toggle"

        self.state_machine = InterventionStateMachine(
            proximity_threshold_deg=proximity_threshold_deg
        )
        self._hold_action = None
        self._total_interventions = 0

    def poll_button_events(self, observation: Observation) -> Info:
        """Poll button events without executing full get_action logic."""
        _, teleop_info = self.teleop.get_action(observation)
        return teleop_info

    def _reset_leader_arms_to_gravcomp(self) -> None:
        for side, leader in self.teleop.leader_arms.items():
            leader.command_joint_state(
                {
                    "pos": np.zeros(6),
                    "vel": np.zeros(6),
                    "kp": np.zeros(6),
                    "kd": np.zeros(6),
                }
            )

    def _command_reverse_sync(
        self,
        leader_pos: dict[str, np.ndarray],
        follower_pos: dict[str, np.ndarray],
        observation: Observation,
    ) -> None:
        """Command leader arms to sync with follower positions during reverse sync."""
        for side, leader in self.teleop.leader_arms.items():
            current_pos = leader_pos[f"{side}_joint_pos"]
            target_pos = follower_pos[f"{side}_joint_pos"]

            pos_diff = target_pos - current_pos

            step = np.clip(pos_diff, -self.MAX_POSITION_STEP, self.MAX_POSITION_STEP)

            step_in_hz = step / 30.0

            commanded_pos = current_pos + step

            leader.command_joint_state(
                {
                    "pos": commanded_pos,
                    "vel": 0.5 * step_in_hz,
                    "kp": self.LOW_KP,
                    "kd": self.LOW_KD,
                }
            )

    def get_action(self, observation: Observation) -> tuple[Action, Info]:
        """Get action with intervention support."""
        follower_pos = {
            "left_joint_pos": observation["left_joint_pos"],
            "right_joint_pos": observation["right_joint_pos"],
        }

        leader_pos, teleop_info = self.teleop.get_action(observation)

        side_idx, btn_idx = self.SYNC_BUTTON

        sync_button_held = self.teleop.prev_pressed[side_idx][btn_idx] > 0.5

        self.state_machine.update(sync_button_held, leader_pos, follower_pos)

        if self.state_machine.is_entering_state("reverse_sync"):
            self._hold_action = hold_action_from_proprio(observation)
            self._hold_action["source"] = "reverse_sync"
        if self.state_machine.is_leaving_state("reverse_sync"):
            print("[Intervention] Exiting reverse_sync, resetting leader arms to gravity comp")
            self._reset_leader_arms_to_gravcomp()
            self._hold_action = None
        if self.state_machine.is_entering_state("policy_control"):
            print("[Intervention] Returning to policy control, resetting policy")
            self.policy.reset()
        if self.state_machine.is_entering_state("leader_control"):
            self._total_interventions += 1
            print(f"[Intervention] Starting intervention number {self._total_interventions}")

        self.state_machine.clear_transition()

        info = {
            **teleop_info,
            "sync_held": sync_button_held,
            "intervention_state": self.state_machine.state,
            "intervention_active": self.state_machine.is_in_intervention(),
        }

        intervention_state = self.state_machine.state

        if intervention_state == "policy_control":
            policy_action, policy_info = self.policy.get_action(observation)
            policy_action["source"] = "policy"
            return policy_action, {
                **policy_info,
                **info,
            }
        elif intervention_state == "reverse_sync":
            self._command_reverse_sync(leader_pos, follower_pos, observation)
            return self._hold_action, info
        elif intervention_state == "leader_control":
            leader_action = leader_pos.copy()
            leader_action["source"] = "human"
            return leader_action, info
        else:
            raise ValueError(f"Invalid intervention state: {intervention_state}")

    def reset(self) -> Info:
        self.state_machine.enter_state("policy_control")
        return self.policy.reset()
