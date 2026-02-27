"""
Intervention State Machine for YAM Robot Leader-Follower Takeover.

This module implements the state machine for allowing operators to take control
during evaluation by syncing leader arms to follower positions.
"""

from typing import Literal

import numpy as np

from yam_control.filter_utils import PeriodicAverageAccumulator

InterventionState = Literal["policy_control", "reverse_sync", "leader_control"]


class InterventionStateMachine:
    """State machine for the intervention/takeover procedure for YAM robot."""

    def __init__(
        self,
        proximity_threshold_deg: float,
    ):
        self._state: InterventionState = "policy_control"
        self._prev_state: InterventionState = "policy_control"
        self.proximity_threshold_rad = np.deg2rad(proximity_threshold_deg)
        self._proximity_debug = PeriodicAverageAccumulator(window_seconds=1.0)

    @property
    def state(self) -> InterventionState:
        return self._state

    def enter_state(self, state: InterventionState) -> None:
        if state == self._state:
            return

        self._prev_state = self._state

        if state == "policy_control":
            print("[Intervention] Returning to policy control mode")
        elif state == "reverse_sync":
            print("[Intervention] Starting reverse sync (follower -> leader)")
        elif state == "leader_control":
            print("[Intervention] Arms synced, switching to leader control")

        self._state = state

    def check_proximity(
        self,
        leader_pos: dict[str, np.ndarray],
        follower_pos: dict[str, np.ndarray],
    ) -> bool:
        """Check if leader and follower arms are close enough (all joints within threshold)."""
        for side in ["left", "right"]:
            leader_joint = leader_pos[f"{side}_joint_pos"][:6]
            follower_joint = follower_pos[f"{side}_joint_pos"][:6]
            diff = np.abs(leader_joint - follower_joint)
            if np.any(diff > self.proximity_threshold_rad):
                return False
        return True

    def update(
        self,
        sync_button_held: bool,
        leader_pos: dict[str, np.ndarray],
        follower_pos: dict[str, np.ndarray],
    ) -> None:
        """Update intervention state based on current conditions."""
        if sync_button_held:
            if self.state == "policy_control":
                self.enter_state("reverse_sync")
            elif self.state == "reverse_sync":
                if self.check_proximity(leader_pos, follower_pos):
                    self.enter_state("leader_control")
                else:
                    all_diffs = np.concatenate(
                        [
                            np.abs(
                                leader_pos[f"{side}_joint_pos"][:6]
                                - follower_pos[f"{side}_joint_pos"][:6]
                            )
                            for side in ["left", "right"]
                        ]
                    )
                    max_diff_deg = np.rad2deg(all_diffs.max())
                    avg = self._proximity_debug.add(max_diff_deg)
                    if avg is not None:
                        threshold_deg = np.rad2deg(self.proximity_threshold_rad)
                        print(
                            f"[Intervention] reverse_sync: max diff {avg:.1f} deg, threshold {threshold_deg:.1f} deg"
                        )
        else:
            if self.state != "policy_control":
                self.enter_state("policy_control")

    def is_in_intervention(self) -> bool:
        return self.state != "policy_control"

    def is_leaving_state(self, state: InterventionState) -> bool:
        return self._prev_state == state and self._state != state

    def is_entering_state(self, state: InterventionState) -> bool:
        return self._prev_state != state and self._state == state

    def clear_transition(self) -> None:
        """Clear transition flags so is_entering/leaving only fires once per transition."""
        self._prev_state = self._state
