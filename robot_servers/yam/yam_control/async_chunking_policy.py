"""Asynchronous action chunking policy wrapper.

A background thread continuously calls ``policy.get_action()`` to predict
the next action chunk while the main thread pops and executes actions from
a queue.  This eliminates the idle time between chunks that occurs with
:class:`SyncChunkingPolicy`.
"""

import threading
import time
import warnings
import weakref
from collections import deque
from typing import Any

import numpy as np

from yam_control.policy import Action, Info, Observation
from yam_control.sync_chunking_policy import chunk_to_action_list


def _hold_action_from_obs(observation: Observation) -> Action:
    """Compute a hold-position action from the current observation."""
    return {
        k: np.asarray(
            observation.get(k, np.zeros(6 if "joint" in k else 1)),
            dtype=np.float32,
        )
        for k in [
            "left_joint_pos", "left_gripper_pos",
            "right_joint_pos", "right_gripper_pos",
        ]
    }


class AsyncChunkingPolicy:
    """Wraps a chunk-producing policy and feeds actions one at a time,
    while a background thread runs inference ahead of time.

    Args:
        policy: Inner policy whose ``get_action`` returns action chunks in
            ``info["action_chunk"]``.
        action_exec_horizon: Max steps to execute from each chunk.
        policy_latency_steps: How many queued actions to keep before
            starting the next inference.  Increase if you see "waiting"
            warnings.
        max_get_action_seconds: Max time to wait for an action before
            falling back to a hold-position action.
    """

    def __init__(
        self,
        policy: Any,
        action_exec_horizon: int,
        policy_latency_steps: int = 4,
        max_get_action_seconds: float = 0.5,
    ):
        self.policy = policy
        self.action_queue: deque[dict[str, np.ndarray]] = deque()
        self.last_info: Info = {}
        self.action_exec_horizon = action_exec_horizon
        self.policy_latency_steps = policy_latency_steps
        self.max_get_action_seconds = max_get_action_seconds

        self.lock = threading.Lock()
        self.should_exit = False
        self.is_resetting = False
        self.reset_info: Info | None = None
        self.last_obs: Observation | None = None

        self._policy_thread = threading.Thread(
            daemon=True, target=self._policy_loop
        )
        self._policy_thread.start()

    def get_action(self, observation: Observation) -> tuple[Action, Info]:
        first_chunk = self.last_obs is None
        with self.lock:
            self.last_obs = observation

        start = time.monotonic()
        while not self.action_queue:
            if not first_chunk:
                warnings.warn(
                    "AsyncChunkingPolicy waiting for policy.get_action(). "
                    "Consider increasing policy_latency_steps."
                )
            if time.monotonic() - start > self.max_get_action_seconds:
                return _hold_action_from_obs(observation), {}
            time.sleep(0.01)
            if not self._policy_thread.is_alive():
                raise RuntimeError("AsyncChunkingPolicy thread died")
            if self.last_obs is None:
                raise RuntimeError(
                    "Observation reset to None while waiting for actions"
                )

        with self.lock:
            action = self.action_queue.popleft()
            return action, self.last_info

    def reset(self) -> Info | None:
        self.is_resetting = True
        start = time.monotonic()
        while self.is_resetting:
            time.sleep(0.01)
            if not self._policy_thread.is_alive():
                raise RuntimeError("AsyncChunkingPolicy thread died")
            if time.monotonic() - start > 5.0:
                print("[AsyncChunking] Waiting for reset...")
                start = time.monotonic()
        assert len(self.action_queue) == 0
        return self.reset_info

    def _policy_loop(self):
        weak_self = weakref.ref(self)
        first_chunk = True

        while True:
            self_ref = weak_self()
            if self_ref is None or self_ref.should_exit:
                break

            # Handle reset
            if self_ref.is_resetting:
                reset_info = self_ref.policy.reset()
                first_chunk = True
                with self_ref.lock:
                    self_ref.last_obs = None
                    self_ref.action_queue = deque()
                    self_ref.last_info = {}
                    self_ref.reset_info = reset_info
                    self_ref.is_resetting = False

            # Wait if queue is full enough
            if len(self_ref.action_queue) > self_ref.policy_latency_steps:
                del self_ref
                time.sleep(0.001)
                continue

            last_obs = self_ref.last_obs
            if not last_obs:
                del self_ref
                time.sleep(0.1)
                continue

            # Run inference
            forward_start = time.perf_counter()
            _action, info = self_ref.policy.get_action(last_obs)
            forward_ms = (time.perf_counter() - forward_start) * 1000
            del _action

            if first_chunk:
                prefix_steps = 0
                first_chunk = False
            else:
                prefix_steps = self_ref.policy_latency_steps + 1

            truncated_chunk = self_ref._truncate_chunk(
                info["action_chunk"], prefix_steps=prefix_steps
            )
            truncated_info = info.copy()
            truncated_info["action_chunk"] = truncated_chunk
            truncated_info["forward_time_ms"] = forward_ms

            action_list = chunk_to_action_list(truncated_chunk)
            with self_ref.lock:
                self_ref.last_info = truncated_info
                self_ref.action_queue.extend(action_list)

            del self_ref

    def _truncate_chunk(
        self, action_chunk: dict[str, Any], prefix_steps: int
    ) -> dict[str, np.ndarray]:
        truncated: dict[str, np.ndarray] = {}
        for key, value in action_chunk.items():
            seq = np.asarray(value)
            if seq.ndim == 3:
                truncated[key] = seq[:, prefix_steps:self.action_exec_horizon, :]
            elif seq.ndim == 2:
                truncated[key] = seq[prefix_steps:self.action_exec_horizon, :]
            else:
                truncated[key] = value
        return truncated

    def shutdown(self):
        self.should_exit = True

    def __del__(self):
        self.shutdown()
