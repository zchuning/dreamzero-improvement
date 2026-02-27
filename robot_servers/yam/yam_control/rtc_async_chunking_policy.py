"""Real-Time Chunking (RTC) async chunking policy.

Based on LeRobot's RTC implementation. Provides RTCAsyncChunkingPolicy which
implements asynchronous action chunking with RTC support:

1. Background thread continuously runs inference when queue is low
2. Main thread consumes actions from queue at control frequency
3. Leftover actions from previous chunk are passed for RTC inpainting
4. Dynamic latency tracking determines inference delay
"""

import logging
import math
import threading
import time
from typing import Any
import warnings
import weakref

from yam_control.action_utils import hold_action_from_proprio
from yam_control.key_remapping_utils import pack_actions_for_groot
from yam_control.latency_tracker import LatencyTracker
from yam_control.policy import Action, Info, Observation
from yam_control.rtc_action_queue import RTCActionQueue
from yam_control.rtc_config import RTCConfig
from yam_control.sync_chunking_policy import chunk_to_action_list

logger = logging.getLogger(__name__)


class RTCAsyncChunkingPolicy:
    """Asynchronous chunking policy with Real-Time Chunking (RTC) support.

    Two threads: actor (main) consumes actions, inference (background)
    generates chunks.  Action queue tracking for RTC inpainting with
    dynamic latency tracking.

    Args:
        policy: The underlying policy that generates action chunks.
        rtc_config: Configuration for RTC behavior.
        control_freq: Control frequency in Hz.
        action_horizon: Maximum action horizon per chunk.
        max_get_action_seconds: Max time to wait before returning hold action.
        debug: Enable debug plotting of action stitching.
        model_action_horizon: Model's internal action horizon (default 16).
    """

    def __init__(
        self,
        policy,
        rtc_config: RTCConfig,
        control_freq: float,
        action_horizon: int,
        max_get_action_seconds: float = 0.5,
        debug: bool = False,
        model_action_horizon: int = 16,
    ):
        self.policy = policy
        self.rtc_config = rtc_config
        self.control_freq = control_freq
        self.action_horizon = action_horizon
        self.max_get_action_seconds = max_get_action_seconds
        self.debug = debug

        self.time_per_action = 1.0 / control_freq
        self.action_queue = RTCActionQueue(rtc_config, debug=debug)
        self.latency_tracker = LatencyTracker(maxlen=5)

        self.lock = threading.Lock()
        self.should_exit = False
        self.is_resetting = False

        self.last_obs: Observation | None = None
        self.last_info: dict[str, Any] = {}
        self.reset_info: dict[str, Any] | None = None

        self._model_action_horizon: int = model_action_horizon

        self.steps_since_last_inference = 0
        self.inference_requested = threading.Event()
        self.inference_requested.set()

        self._inference_thread = threading.Thread(
            target=self._inference_loop,
            daemon=True,
            name="RTCInference",
        )
        self._inference_thread.start()
        logger.info("[RTC] Started inference thread")

    def get_action(self, observation: Observation) -> tuple[Action, Info]:
        first_chunk = self.last_obs is None

        with self.lock:
            self.last_obs = observation

        if hasattr(self.policy, "observe"):
            try:
                self.policy.observe(observation)
            except Exception:
                pass

        start = time.monotonic()

        while self.action_queue.empty():
            if not first_chunk:
                warnings.warn(
                    "[RTC] Waiting for inference to complete. "
                    "Consider decreasing inference_interval."
                )

            now = time.monotonic()
            if now - start > self.max_get_action_seconds:
                logger.warning("[RTC] Timeout waiting for actions, returning hold action")
                return hold_action_from_proprio(observation), {}

            time.sleep(0.01)

            if not self._inference_thread.is_alive():
                raise RuntimeError("[RTC] Inference thread died unexpectedly")

            if self.last_obs is None:
                raise RuntimeError(
                    "[RTC] Observation reset to None while waiting. "
                    "Was reset() called from another thread?"
                )

        action = self.action_queue.get()
        if action is None:
            return hold_action_from_proprio(observation), {}

        with self.lock:
            self.steps_since_last_inference += 1
            if self.steps_since_last_inference >= self.rtc_config.inference_interval:
                self.inference_requested.set()
            return action, self.last_info.copy()

    def reset(self) -> Info | None:
        logger.info("[RTC] Reset requested")
        self.is_resetting = True
        start_time = time.monotonic()

        while True:
            time.sleep(0.01)

            if not self.is_resetting:
                elapsed = time.monotonic() - start_time
                logger.info(f"[RTC] Reset complete in {elapsed:.2f}s")
                assert self.action_queue.qsize() == 0, "Reset should empty the queue"
                return self.reset_info

            if not self._inference_thread.is_alive():
                raise RuntimeError("[RTC] Inference thread died during reset")

            elapsed = time.monotonic() - start_time
            if elapsed > 5.0:
                logger.warning(f"[RTC] Waiting for reset to complete... ({elapsed:.1f}s)")
                time.sleep(1.0)

    def _inference_loop(self):
        weak_self = weakref.ref(self)
        first_chunk = True

        while True:
            self = weak_self()
            if self is None or self.should_exit:
                logger.info("[RTC] Inference thread exiting")
                break

            if self.is_resetting:
                logger.info("[RTC] Processing reset request")
                reset_info = self.policy.reset()
                first_chunk = True

                with self.lock:
                    self.last_obs = None
                    self.action_queue.clear()
                    self.last_info = {}
                    self.reset_info = reset_info
                    self.is_resetting = False

                continue

            if not self.inference_requested.is_set() and not self.action_queue.empty():
                time.sleep(0.001)
                self = None
                continue

            self.inference_requested.clear()
            with self.lock:
                self.steps_since_last_inference = 0

            with self.lock:
                last_obs = self.last_obs

            if last_obs is None:
                logger.debug("[RTC] Waiting for first observation")
                self = None
                time.sleep(0.1)
                continue

            action_index_before_inference = self.action_queue.get_action_index()
            leftover_actions = self.action_queue.get_left_over_actions()
            inference_delay = int(self.latency_tracker.mean() / self.time_per_action)

            rtc_options = None
            previous_action = None
            if self.rtc_config.enabled:
                if (
                    leftover_actions is not None
                    and len(leftover_actions) > 0
                    and self._model_action_horizon is not None
                ):
                    previous_action = pack_actions_for_groot(
                        leftover_actions, action_horizon=self._model_action_horizon
                    )
                    remaining_actions = len(leftover_actions)
                    effective_delay = min(inference_delay, remaining_actions)
                    rtc_options = {
                        "action_horizon": self.action_horizon,
                        "inference_delay": effective_delay,
                    }
                else:
                    previous_action = pack_actions_for_groot(
                        [], action_horizon=self._model_action_horizon
                    )
                    rtc_options = {
                        "action_horizon": self.action_horizon,
                        "inference_delay": 0,
                    }

            current_time = time.perf_counter()
            try:
                _action, info = self.policy.get_action(
                    last_obs,
                    options=rtc_options,
                    previous_action=previous_action,
                )
            except Exception as e:
                logger.error(f"[RTC] Policy inference failed: {e}")
                time.sleep(0.1)
                continue

            actual_latency = time.perf_counter() - current_time
            actual_delay = math.ceil(actual_latency / self.time_per_action)

            max_reasonable_latency = 20 * self.time_per_action
            if actual_latency < max_reasonable_latency:
                self.latency_tracker.add(actual_latency)
            else:
                logger.warning(
                    f"[RTC] Ignoring latency spike of {actual_latency:.2f}s "
                    f"({actual_delay} steps, max reasonable: 20 steps) "
                    f"- likely warmup/recompilation"
                )

            if rtc_options:
                model_delay = rtc_options.get("inference_delay", 0)
                if actual_delay != model_delay:
                    print(
                        f"[RTC] DELAY MISMATCH: "
                        f"model_delay={model_delay}, actual_delay={actual_delay}, "
                        f"skipping {actual_delay} actions from new chunk"
                    )

            action_chunk = info.get("action_chunk", {})

            if not action_chunk:
                logger.warning("[RTC] No action_chunk in policy info")
                continue

            first_key = next(iter(action_chunk.keys()))
            detected_horizon = action_chunk[first_key].shape[0]
            if detected_horizon != self._model_action_horizon:
                logger.warning(
                    f"[RTC] Model action horizon mismatch: expected {self._model_action_horizon}, "
                    f"got {detected_horizon}. Updating to {detected_horizon}."
                )
                self._model_action_horizon = detected_horizon

            actions = chunk_to_action_list(action_chunk)
            actions = actions[: self.action_horizon]

            if rtc_options:
                prefix_steps_for_debug = rtc_options.get("inference_delay", 0)
                use_explicit_prefix = True
            else:
                prefix_steps_for_debug = 0
                use_explicit_prefix = False

            self.action_queue.merge(
                actions,
                actual_delay if first_chunk is False else 0,
                action_index_before_inference,
                frozen_steps=prefix_steps_for_debug,
                leftover_actions_for_debug=leftover_actions,
                prefix_label="prefix",
                use_explicit_prefix=use_explicit_prefix,
            )

            first_chunk = False

            forward_time_ms = actual_latency * 1000
            with self.lock:
                self.last_info = {
                    "action_chunk": action_chunk,
                    "forward_time_ms": forward_time_ms,
                    "inference_delay": actual_delay,
                    "queue_size": self.action_queue.qsize(),
                    "leftover_count": len(leftover_actions) if leftover_actions else 0,
                }

            logger.debug(
                f"[RTC] Inference complete: {forward_time_ms:.1f}ms, "
                f"delay={actual_delay}, queue={self.action_queue.qsize()}"
            )

    def flush_debug_plots(self):
        """Flush buffered debug plots to disk."""
        self.action_queue.flush_debug_plots()

    def shutdown(self):
        """Shutdown the inference thread."""
        self.should_exit = True
        logger.info("[RTC] Shutdown requested")

    def __del__(self):
        self.shutdown()
