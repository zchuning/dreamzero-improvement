"""Action queue management for Real-Time Chunking (RTC).

Based on LeRobot's implementation:
https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/rtc/action_queue.py

This module provides RTCActionQueue, a thread-safe queue for managing action chunks
in real-time control scenarios. It supports both RTC-enabled and non-RTC modes,
handling action merging and leftover tracking.

Key concepts:
- Action queue: List of action dicts ready for robot execution
- Delay compensation: Automatically skips actions consumed during inference
- Leftover tracking: Returns unconsumed actions for next chunk's inpainting
"""

import logging
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np

from yam_control.rtc_config import RTCConfig

logger = logging.getLogger(__name__)


class RTCActionQueue:
    """Thread-safe queue for managing action chunks in real-time control.

    The queue operates in two modes:
    1. RTC-enabled: Replaces the entire queue with new actions, accounting for inference delay
    2. RTC-disabled: Appends new actions to the queue, maintaining continuity

    Args:
        cfg: Configuration for Real-Time Chunking behavior.
        debug: Whether to enable debug plotting.
    """

    def __init__(self, cfg: RTCConfig, debug: bool = False):
        self.queue: list[dict[str, Any]] | None = None
        self.lock = Lock()
        self.last_index = 0
        self.cfg = cfg
        self.debug = debug
        self._merge_count = 0

        self._debug_buffer: list[dict[str, Any]] = []

        if self.debug:
            self._debug_dir = Path("rtc_debug")
            if self._debug_dir.exists():
                for f in self._debug_dir.glob("*.png"):
                    f.unlink()
            self._debug_dir.mkdir(exist_ok=True)
            logger.info(f"[RTC Debug] Saving plots to {self._debug_dir.absolute()}")

    def get(self) -> dict[str, Any] | None:
        """Get the next action from the queue. Returns a copy or None if empty."""
        with self.lock:
            if self.queue is None or self.last_index >= len(self.queue):
                return None

            action = self.queue[self.last_index]
            self.last_index += 1
            return {k: np.copy(v) if isinstance(v, np.ndarray) else v for k, v in action.items()}

    def qsize(self) -> int:
        """Get the number of remaining actions in the queue."""
        with self.lock:
            if self.queue is None:
                return 0
            return len(self.queue) - self.last_index

    def empty(self) -> bool:
        """Check if the queue is empty."""
        return self.qsize() <= 0

    def get_action_index(self) -> int:
        """Get the current action consumption index."""
        with self.lock:
            return self.last_index

    def get_left_over(self) -> np.ndarray | None:
        """Get leftover actions as numpy array [remaining_steps, action_dim], or None."""
        with self.lock:
            if self.queue is None:
                return None
            remaining = self.queue[self.last_index :]
            if len(remaining) == 0:
                return None
            return self._actions_to_array(remaining)

    def get_left_over_actions(self) -> list[dict[str, Any]] | None:
        """Get leftover actions as list of action dicts, or None."""
        with self.lock:
            if self.queue is None:
                return None
            remaining = self.queue[self.last_index :]
            if len(remaining) == 0:
                return None
            return [
                {k: np.copy(v) if isinstance(v, np.ndarray) else v for k, v in action.items()}
                for action in remaining
            ]

    def _actions_to_array(self, actions: list[dict[str, Any]]) -> np.ndarray:
        """Convert list of action dicts to numpy array [T, D]."""
        if len(actions) == 0:
            return np.array([])

        keys = sorted(actions[0].keys())
        keys = [k for k in keys if k != "source"]

        arrays = []
        for action in actions:
            row = []
            for key in keys:
                val = np.asarray(action[key], dtype=np.float32).flatten()
                row.append(val)
            arrays.append(np.concatenate(row))

        return np.stack(arrays, axis=0)

    def merge(
        self,
        actions: list[dict[str, Any]],
        real_delay: int,
        action_index_before_inference: int | None = 0,
        frozen_steps: int | None = None,
        leftover_actions_for_debug: list[dict[str, Any]] | None = None,
        prefix_label: str = "frozen",
        use_explicit_prefix: bool = False,
    ):
        """Merge new actions into the queue.

        In RTC mode, replaces the queue accounting for inference delay.
        In non-RTC mode, appends to the queue maintaining continuity.
        """
        with self.lock:
            prev_leftover = None
            if self.debug:
                if leftover_actions_for_debug is not None:
                    prev_leftover = [
                        {k: np.copy(v) if isinstance(v, np.ndarray) else v for k, v in a.items()}
                        for a in leftover_actions_for_debug
                    ]
                elif self.queue is not None:
                    prev_leftover = [
                        {k: np.copy(v) if isinstance(v, np.ndarray) else v for k, v in a.items()}
                        for a in self.queue[self.last_index :]
                    ]

            actual_consumed = self._check_delays(real_delay, action_index_before_inference)

            if self.cfg.enabled:
                self._replace_actions_queue(
                    actions,
                    actual_consumed,
                    leftover_actions=leftover_actions_for_debug,
                    use_explicit_prefix=use_explicit_prefix,
                )
            else:
                self._append_actions_queue(actions)

            if self.debug:
                self._debug_buffer.append(
                    {
                        "prev_leftover": prev_leftover,
                        "new_actions": [
                            {
                                k: np.copy(v) if isinstance(v, np.ndarray) else v
                                for k, v in a.items()
                            }
                            for a in actions
                        ],
                        "real_delay": actual_consumed,
                        "frozen_steps": (
                            frozen_steps if frozen_steps is not None else actual_consumed
                        ),
                        "merge_count": self._merge_count,
                        "prefix_label": prefix_label,
                    }
                )
                self._merge_count += 1

    def _replace_actions_queue(
        self,
        actions: list[dict[str, Any]],
        real_delay: int,
        leftover_actions: list[dict[str, Any]] | None = None,
        use_explicit_prefix: bool = False,
    ):
        """Replace the queue with new actions (RTC mode)."""
        real_delay = max(0, min(real_delay, len(actions)))
        self.queue = actions[real_delay:]

        logger.debug(
            f"[RTC] Replaced queue: len={len(self.queue)}, delay={real_delay}, "
            f"explicit_prefix={use_explicit_prefix}"
        )

        self.last_index = 0

    def _append_actions_queue(
        self,
        actions: list[dict[str, Any]],
    ):
        """Append new actions to the queue (non-RTC mode)."""
        if self.queue is None:
            self.queue = actions
            self.last_index = 0
            return

        self.queue = self.queue[self.last_index :] + actions
        self.last_index = 0

    def _check_delays(
        self, real_delay: int, action_index_before_inference: int | None = None
    ) -> int:
        """Validate computed delays and return the actual consumed count."""
        if action_index_before_inference is None:
            return real_delay

        indexes_diff = self.last_index - action_index_before_inference
        if indexes_diff != real_delay:
            logger.warning(
                f"[RTC] Delay mismatch: consumed={indexes_diff} actions during inference, "
                f"but computed delay={real_delay} from latency"
            )
            return indexes_diff

        return real_delay

    def flush_debug_plots(self):
        """Flush buffered debug data and generate plots."""
        if not self.debug or not self._debug_buffer:
            return

        logger.info(f"[RTC Debug] Flushing {len(self._debug_buffer)} buffered plots...")

        for data in self._debug_buffer:
            self._plot_merge(
                data["prev_leftover"],
                data["new_actions"],
                data["real_delay"],
                data["frozen_steps"],
                data["merge_count"],
                data.get("prefix_label", "frozen"),
            )

        self._debug_buffer.clear()
        logger.info("[RTC Debug] Flush complete")

    def _plot_merge(
        self,
        prev_leftover: list[dict[str, Any]] | None,
        new_actions: list[dict[str, Any]],
        real_delay: int,
        frozen_steps: int,
        merge_count: int,
        prefix_label: str = "frozen",
    ):
        """Plot the action stitching for debugging."""
        try:
            import matplotlib.cm as cm
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("[RTC Debug] matplotlib not available, skipping plot")
            return

        if len(new_actions) == 0:
            return
        keys = sorted([k for k in new_actions[0].keys() if k != "source"])

        new_array = self._actions_to_array(new_actions)

        if prev_leftover is not None and len(prev_leftover) > 0:
            prev_array = self._actions_to_array(prev_leftover)
        else:
            prev_array = None

        frozen_diff = None
        if (
            prev_array is not None
            and len(prev_array) >= frozen_steps
            and len(new_array) >= frozen_steps
            and frozen_steps > 0
        ):
            frozen_diff = np.abs(prev_array[:frozen_steps] - new_array[:frozen_steps]).max(axis=0)

        dims_per_key = {}
        offset = 0
        for key in keys:
            dim = np.asarray(new_actions[0][key]).flatten().shape[0]
            dims_per_key[key] = (offset, offset + dim)
            offset += dim

        fig, axes = plt.subplots(len(keys), 1, figsize=(14, 3.5 * len(keys)), squeeze=False)

        title = f"RTC Action Stitching - Merge #{merge_count}\n"
        title += f"real_delay={real_delay}, {prefix_label}_steps={frozen_steps}, chunk_size={len(new_actions)}\n"
        title += "Solid=OLD leftover, Dashed=NEW chunk"
        if frozen_diff is not None:
            max_frozen_diff = frozen_diff.max()
            title += f"\n{prefix_label.capitalize()} region max diff: {max_frozen_diff:.6f}"
        fig.suptitle(title, fontsize=11)

        for idx, key in enumerate(keys):
            ax = axes[idx, 0]
            start, end = dims_per_key[key]
            num_dims = end - start

            colors = cm.tab10(np.linspace(0, 1, max(num_dims, 10)))[:num_dims]

            if prev_array is not None and len(prev_array) > 0:
                t_prev = np.arange(len(prev_array))
                for d_idx, d in enumerate(range(start, end)):
                    ax.plot(
                        t_prev, prev_array[:, d], "-", color=colors[d_idx],
                        linewidth=2, alpha=0.9, label=f"dim {d_idx} (old)",
                    )

            if len(new_array) > 0:
                t_new = np.arange(len(new_array))
                for d_idx, d in enumerate(range(start, end)):
                    ax.plot(
                        t_new, new_array[:, d], "--", color=colors[d_idx],
                        linewidth=2, alpha=0.9, label=f"dim {d_idx} (new)",
                    )

            if frozen_steps > 0:
                ax.axvspan(0, frozen_steps, alpha=0.15, color="blue",
                           label=f"{prefix_label} ({frozen_steps})")
                ax.axvline(x=frozen_steps, color="blue", linestyle=":", linewidth=2,
                           alpha=0.7, label=f"{prefix_label} end")

            if real_delay > 0 and real_delay < len(new_array):
                ax.axvline(x=real_delay, color="red", linestyle="--", linewidth=2,
                           alpha=0.7, label=f"queue start @ t={real_delay}")

            ax.set_ylabel(key, fontsize=10)
            ax.set_xlabel("Time step", fontsize=10)
            ax.legend(loc="upper right", fontsize="x-small", ncol=2)
            ax.grid(True, alpha=0.3)

            if frozen_diff is not None:
                key_max_diff = frozen_diff[start:end].max()
                ax.set_title(f"{key} ({num_dims} dims) - {prefix_label} diff: {key_max_diff:.6f}", fontsize=10)
            else:
                ax.set_title(f"{key} ({num_dims} dims)", fontsize=10)

        plt.tight_layout()

        plot_path = self._debug_dir / f"merge_{merge_count:04d}.png"
        plt.savefig(plot_path, dpi=120)
        plt.close(fig)
        logger.info(f"[RTC Debug] Saved plot to {plot_path}")

    def clear(self):
        """Clear the queue and reset state."""
        with self.lock:
            self.queue = None
            self.last_index = 0

    def set_initial(
        self,
        actions: list[dict[str, Any]],
    ):
        """Set initial actions without delay compensation."""
        with self.lock:
            self.queue = actions
            self.last_index = 0
