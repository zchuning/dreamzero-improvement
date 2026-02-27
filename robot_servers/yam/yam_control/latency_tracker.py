"""Latency tracking utilities for Real-Time Chunking (RTC).

Based on LeRobot's implementation:
https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/rtc/latency_tracker.py

This module provides LatencyTracker, a utility for tracking inference latencies
over a sliding window and computing statistics like max and percentiles.
"""

from collections import deque

import numpy as np


class LatencyTracker:
    """Tracks recent latencies and provides max/percentile queries.

    Used to estimate inference delay for RTC by tracking how long each
    policy inference takes. The max latency is used to conservatively
    estimate how many action steps will be consumed during inference.

    Args:
        maxlen: Sliding window size. Only the most recent `maxlen` latencies are kept.
        skip_first: Number of initial samples to skip (warmup period).

    Example:
        >>> tracker = LatencyTracker(maxlen=5, skip_first=1)
        >>> tracker.add(2.0)   # Skipped (first inference includes warmup)
        >>> tracker.add(0.05)  # 50ms inference - counted
        >>> tracker.add(0.06)  # 60ms inference - counted
        >>> tracker.max()  # Returns 0.06 (not 2.0)
    """

    def __init__(self, maxlen: int = 5, skip_first: int = 1):
        self._values: deque[float] = deque(maxlen=maxlen)
        self._skip_first = skip_first
        self._skipped_count = 0

    def reset(self) -> None:
        """Clear all recorded latencies and reset skip counter."""
        self._values.clear()
        self._skipped_count = 0

    def add(self, latency: float) -> None:
        """Add a latency sample (in seconds). Negative values are ignored."""
        val = float(latency)
        if val < 0:
            return

        if self._skipped_count < self._skip_first:
            self._skipped_count += 1
            return

        self._values.append(val)

    def __len__(self) -> int:
        return len(self._values)

    def max(self) -> float:
        """Return the maximum latency in the current window, or 0.0 if empty."""
        if not self._values:
            return 0.0
        return float(max(self._values))

    def percentile(self, q: float) -> float:
        """Return the q-quantile (q in [0,1]) of recorded latencies, or 0.0 if empty."""
        if not self._values:
            return 0.0
        q = float(q)
        if q <= 0.0:
            return float(min(self._values))
        if q >= 1.0:
            return float(max(self._values))
        vals = np.array(list(self._values), dtype=np.float32)
        return float(np.quantile(vals, q))

    def p95(self) -> float:
        """Return the 95th percentile latency, or 0.0 if empty."""
        return self.percentile(0.95)

    def mean(self) -> float:
        """Return the mean latency, or 0.0 if empty."""
        if not self._values:
            return 0.0
        return float(np.mean(list(self._values)))
