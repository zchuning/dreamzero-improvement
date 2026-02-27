import copy

import numpy as np

from yam_control.policy import Action, Info, Observation, Policy


class ClippedActionPolicy(Policy):
    """Wrapper for policies that clips the delta between consecutive actions."""

    def __init__(self, policy: Policy, clip_value: float):
        """
        Args:
            policy: The underlying policy
            clip_value: Maximum allowed change per dimension between consecutive actions.
                       Delta is clipped to [-clip_value, clip_value].
        """
        self.policy = policy
        self.clip_value = clip_value
        self.previous_action = None

    def reset(self) -> Info | None:
        self.previous_action = None
        return None

    def get_action(self, observation: Observation) -> tuple[Action, Info]:
        action_dict, policy_infos = self.policy.get_action(observation)

        if self.previous_action is None:
            clipped_action_dict = action_dict
        else:
            clipped_action_dict = {}
            for key, action_array in action_dict.items():
                if key in self.previous_action:
                    delta = action_array - self.previous_action[key]
                    clipped_delta = np.clip(delta, -self.clip_value, self.clip_value)
                    clipped_action_dict[key] = self.previous_action[key] + clipped_delta
                else:
                    clipped_action_dict[key] = action_array

        policy_infos = policy_infos.copy()

        del policy_infos["action_chunk"]

        self.previous_action = copy.deepcopy(clipped_action_dict)

        return clipped_action_dict, policy_infos
