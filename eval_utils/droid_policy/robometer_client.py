"""Lightweight ZMQ client for querying a RobometerServer.

Copied from robometer.RoboMeter_Interface to avoid cross-project imports.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RobometerResult:
    score: float = 0.0
    scores: list[float] = field(default_factory=list)


@dataclass
class PreferenceResult:
    preference_pred: float = 0.0   # 1.0 = chosen preferred, 0.0 = rejected preferred
    preference_prob: float = 0.5   # sigmoid probability that chosen is preferred


class RobometerClient:
    """ZMQ REQ client for querying a RobometerServer."""

    def __init__(self, host: str = "localhost", port: int = 5555, timeout_ms: int = 60000):
        import zmq

        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.REQ)
        self._sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self._sock.setsockopt(zmq.SNDTIMEO, timeout_ms)
        self._addr = f"tcp://{host}:{port}"
        self._sock.connect(self._addr)
        logger.info(f"RobometerClient connected to {self._addr}")

    def score(self, obs: np.ndarray, task: str = "") -> RobometerResult:
        self._sock.send(pickle.dumps({"action": "score", "obs": obs, "task": task}))
        data = self._sock.recv()
        d = pickle.loads(data)
        if "error" in d:
            raise RuntimeError(f"RobometerServer error: {d['error']}")
        if "score" not in d or "scores" not in d:
            raise KeyError("RobometerServer response missing required keys: score/scores")
        return RobometerResult(score=d["score"], scores=d["scores"])

    def score_preference(
        self,
        chosen_obs: np.ndarray,
        rejected_obs: np.ndarray,
        task: str = "",
    ) -> PreferenceResult:
        self._sock.send(pickle.dumps({
            "action": "preference",
            "chosen_obs": chosen_obs,
            "rejected_obs": rejected_obs,
            "task": task,
        }))
        data = self._sock.recv()
        d = pickle.loads(data)
        if "error" in d:
            raise RuntimeError(f"RobometerServer error: {d['error']}")
        if "preference_pred" not in d or "preference_prob" not in d:
            raise KeyError("RobometerServer response missing required keys: preference_pred/preference_prob")
        return PreferenceResult(
            preference_pred=d["preference_pred"],
            preference_prob=d["preference_prob"],
        )

    def close(self):
        self._sock.close()
        self._ctx.term()
