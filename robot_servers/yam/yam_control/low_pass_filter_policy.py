"""Low-pass filter policy wrapper for smoothing policy actions."""

from yam_control.filter_utils import FIRFilter
from yam_control.policy import Action, Info, Observation


class LowPassFilterPolicyWrapper:
    """Policy wrapper that applies a low-pass FIR filter to smooth actions.

    Uses exponentially-weighted averaging over a rolling window of past actions.
    """

    def __init__(self, policy, *, k: int, alpha: float):
        """
        Args:
            policy: The underlying policy to wrap.
            k: Buffer size (number of samples in rolling buffer).
            alpha: Smoothing factor (0 < alpha <= 1). Closer to 1 = more recent weight.
        """
        self.policy = policy
        self.filter = FIRFilter(k=k, alpha=alpha)

    def get_action(self, observation: Observation) -> tuple[Action, Info]:
        action, info = self.policy.get_action(observation)
        filtered_action = self.filter.step(action)
        return filtered_action, info

    def reset(self) -> Info | None:
        k = self.filter.k
        alpha = self.filter.alpha
        self.filter = FIRFilter(k=k, alpha=alpha)
        return self.policy.reset()
