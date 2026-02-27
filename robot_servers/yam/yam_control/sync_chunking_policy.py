"""Synchronous action chunking policy wrapper.

Takes a policy that outputs action chunks (sequences of future actions)
and executes them one step at a time.
"""

from collections import deque
from typing import Any

import numpy as np

from yam_control.policy import Action, Info, Observation


def chunk_to_action_list(chunk: dict[str, Any]) -> list[dict[str, np.ndarray]]:
    """Convert an action chunk dict (with arrays of shape (H, D) or (B, H, D))
    into a list of per-step action dicts."""
    actions = []
    for key, value in chunk.items():
        seq = np.asarray(value)
        # Handle both (H, D) and (B, H, D) shapes
        if seq.ndim == 3:
            assert seq.shape[0] == 1, "Should only have batch size 1"
            seq = seq[0]
        for t in range(seq.shape[0]):
            while t >= len(actions):
                actions.append(dict())
            actions[t][key] = seq[t]
    return actions


class SyncChunkingPolicy:
    """Wraps a chunk-producing policy and feeds actions one at a time.

    The inner policy's get_action() should return (action, info) where
    info["action_chunk"] is a dict of arrays with shape (horizon, dim).
    This wrapper queues those actions and pops one per get_action() call.
    """

    def __init__(self, policy: Any, action_exec_horizon: int):
        """
        Args:
            policy: Inner policy whose get_action returns action chunks in info.
            action_exec_horizon: Max number of steps to execute from each chunk.
        """
        self.policy = policy
        self.action_queue: deque[dict[str, np.ndarray]] = deque()
        self.last_info: dict[str, Any] = {}
        self.action_exec_horizon = action_exec_horizon

    def reset(self) -> Info | None:
        self.action_queue = deque()
        self.last_info = {}
        return self.policy.reset()

    def _truncate_chunk(self, action_chunk: dict[str, Any]) -> dict[str, np.ndarray]:
        truncated_chunk = {}
        for key, value in action_chunk.items():
            seq = np.asarray(value)
            if seq.ndim == 3:
                truncated_chunk[key] = seq[:, : self.action_exec_horizon, :]
            elif seq.ndim == 2:
                truncated_chunk[key] = seq[: self.action_exec_horizon, :]
            else:
                truncated_chunk[key] = value
        return truncated_chunk

    def get_action(self, observation: Observation) -> tuple[Action, Info]:
        if not self.action_queue:
            _action, info = self.policy.get_action(observation)
            del _action
            truncated_chunk = self._truncate_chunk(info["action_chunk"])
            truncated_info = info.copy()
            truncated_info["action_chunk"] = truncated_chunk
            self.last_info = truncated_info
            self.action_queue.extend(chunk_to_action_list(truncated_chunk))
        action = self.action_queue.popleft()
        self.last_info["remaining_num_action_in_chunk"] = len(self.action_queue)
        return action, self.last_info
