"""Standalone ZMQ client for connecting to a remote GR00T policy server.

This module provides a lightweight ``Gr00tServerPolicy`` that connects to
``run_gr00t_server.py`` (from gr00t_main) over ZMQ without importing any
``groot.*`` packages.  All observation/action key mapping is handled locally.

Usage::

    from yam_control.gr00t_server_policy import Gr00tServerPolicy

    policy = Gr00tServerPolicy(host="gpu-machine", port=5555)
    assert policy.ping(), "Cannot reach server"

    # Use with the rollout pipeline
    from yam_control.run_policy_rollout import run_rollout, RolloutConfig
    run_rollout(policy, RolloutConfig(task_description="pick up the cup"))

Architecture::

    yam-control (robot machine)          gr00t_main (GPU machine)
    ┌──────────────────────┐             ┌──────────────────────┐
    │  Gr00tServerPolicy   │── ZMQ ─────►│  run_gr00t_server.py │
    │  (this file)         │   REQ/REP   │  PolicyServer        │
    │                      │◄────────────│  (zmq.REP :5555)     │
    └──────────────────────┘             └──────────────────────┘
"""

from __future__ import annotations

import io
from enum import Enum
from typing import Any

import msgpack
import numpy as np
import zmq

from yam_control.embodiment import EmbodimentTag
from yam_control.key_remapping_utils import (
    ACTION_KEY_MAP_xdof,
    Gr00tPolicyClientWrapper,
    Resolution,
)
from yam_control.policy import Action, Info, Observation, Policy

DEFAULT_MODEL_SERVER_PORT = 5555


# ---------------------------------------------------------------------------
# Msgpack serialization (mirrors groot PolicyServer protocol)
# ---------------------------------------------------------------------------


class _MsgSerializer:
    """Serialize/deserialize numpy arrays over msgpack (ZMQ wire format)."""

    @staticmethod
    def to_bytes(data: Any) -> bytes:
        return msgpack.packb(data, default=_MsgSerializer._encode)

    @staticmethod
    def from_bytes(data: bytes) -> Any:
        return msgpack.unpackb(data, object_hook=_MsgSerializer._decode)

    @staticmethod
    def _decode(obj: Any) -> Any:
        if not isinstance(obj, dict):
            return obj
        if "__ndarray_class__" in obj:
            return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
        return obj

    @staticmethod
    def _encode(obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            buf = io.BytesIO()
            np.save(buf, obj, allow_pickle=False)
            return {"__ndarray_class__": True, "as_npy": buf.getvalue()}
        if isinstance(obj, Enum):
            return obj.value
        return obj


# ---------------------------------------------------------------------------
# ZMQ PolicyClient (standalone, no groot.* imports)
# ---------------------------------------------------------------------------


class PolicyClient:
    """Lightweight ZMQ REQ client that talks to a ``PolicyServer``.

    Wire protocol:
    - Request:  ``{"endpoint": "<name>", "data": {...}, "api_token": "..."}``
    - Response: msgpack-serialized return value (numpy arrays auto-decoded)
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = DEFAULT_MODEL_SERVER_PORT,
        timeout_ms: int = 15000,
        api_token: str | None = None,
    ):
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self.api_token = api_token
        self._ctx = zmq.Context()
        self._init_socket()

    def _init_socket(self) -> None:
        self._socket = self._ctx.socket(zmq.REQ)
        self._socket.connect(f"tcp://{self.host}:{self.port}")

    def call_endpoint(
        self,
        endpoint: str,
        data: dict | None = None,
        requires_input: bool = True,
    ) -> Any:
        request: dict[str, Any] = {"endpoint": endpoint}
        if requires_input:
            request["data"] = data
        if self.api_token:
            request["api_token"] = self.api_token

        self._socket.send(_MsgSerializer.to_bytes(request))
        raw = self._socket.recv()
        if raw == b"ERROR":
            raise RuntimeError("Server error (raw ERROR response)")
        response = _MsgSerializer.from_bytes(raw)

        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"Server error: {response['error']}")
        return response

    def ping(self) -> bool:
        try:
            self.call_endpoint("ping", requires_input=False)
            return True
        except zmq.error.ZMQError:
            self._init_socket()
            return False

    def get_action(
        self,
        observation: dict[str, Any],
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        response = self.call_endpoint(
            "get_action", {"observation": observation, "options": options}
        )
        return tuple(response)  # type: ignore[return-value]

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.call_endpoint("reset", {"options": options})

    def __del__(self) -> None:
        self._socket.close()
        self._ctx.term()


# ---------------------------------------------------------------------------
# Gr00tServerPolicy — Policy interface for the rollout pipeline
# ---------------------------------------------------------------------------


class Gr00tServerPolicy(Policy):
    """Policy that delegates inference to a remote GR00T server over ZMQ.

    Wraps :class:`PolicyClient` + :class:`Gr00tPolicyClientWrapper` so
    observations are transparently mapped to the server's expected format
    and actions are mapped back to env format.

    This class implements the :class:`~yam_control.policy.Policy` ABC so
    it plugs directly into ``run_rollout()``.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = DEFAULT_MODEL_SERVER_PORT,
        timeout_ms: int = 15000,
        api_token: str | None = None,
        embodiment_tag: EmbodimentTag = EmbodimentTag.XDOF,
        resolutions: list[Resolution] | None = None,
    ):
        self._client = PolicyClient(
            host=host, port=port, timeout_ms=timeout_ms, api_token=api_token,
        )
        self._wrapper = Gr00tPolicyClientWrapper(
            policy_client=self._client,
            embodiment_tag=embodiment_tag,
            resolutions=resolutions,
        )
        self._host = host
        self._port = port

    def ping(self) -> bool:
        return self._client.ping()

    def get_action(self, observation: Observation) -> tuple[Action, Info]:
        return self._wrapper.get_action(observation)

    def reset(self) -> Info | None:
        try:
            return self._wrapper.reset()
        except Exception:
            return None
