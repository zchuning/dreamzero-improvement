"""PortalPolicy: Run any Policy in a subprocess using Portal shared-memory IPC.

Model inference happens in a separate subprocess on the same machine,
communicating via Portal's shared-memory transport for zero-copy numpy
array transfer.  This keeps CUDA / model weights out of the main process.

Usage::

    from yam_control.portal_policy import PortalPolicy

    # Wrap any Policy subclass -- the factory is called *inside* the subprocess
    policy = PortalPolicy(
        policy_factory=lambda: MyPolicy(model_path="/path/to/ckpt"),
        port=8011,
    )

    # Now use it like a normal Policy
    action, info = policy.get_action(observation)

The factory callable is serialized (via cloudpickle) and executed in the
subprocess, so heavy imports (CUDA, torch, model weights) only happen
there -- the main process stays lightweight.
"""

import os
import time
from typing import Any, Callable

import numpy as np
import portal

from yam_control.policy import Action, Info, Observation, Policy


# ---------------------------------------------------------------------------
# Serialization helper
# ---------------------------------------------------------------------------


def _make_arrays_contiguous(obj: Any) -> Any:
    """Recursively ensure all numpy arrays are C-contiguous for Portal."""
    if isinstance(obj, np.ndarray):
        return np.ascontiguousarray(obj)
    if isinstance(obj, dict):
        return {k: _make_arrays_contiguous(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_make_arrays_contiguous(item) for item in obj)
    return obj


# ---------------------------------------------------------------------------
# Subprocess server
# ---------------------------------------------------------------------------


class PortalPolicyServer:
    """Runs inside the subprocess.  Creates the Policy and exposes
    ``get_action`` / ``reset`` via Portal RPC.
    """

    def __init__(self, policy_factory: Callable[[], Policy], port: int) -> None:
        self.port = port

        print("[PortalPolicyServer] Creating policy...")
        self.policy = policy_factory()
        print("[PortalPolicyServer] Policy created successfully")

        self._server = portal.Server(port)
        self._server.bind("get_action", self._handle_get_action)
        self._server.bind("reset", self._handle_reset)
        self._server.bind("health_check", lambda: True)

    def _handle_get_action(self, observation: dict) -> tuple[dict, dict]:
        action, info = self.policy.get_action(observation)
        return _make_arrays_contiguous(action), _make_arrays_contiguous(info)

    def _handle_reset(self) -> dict | None:
        result = self.policy.reset()
        return _make_arrays_contiguous(result) if result else None

    def serve(self) -> None:
        print(f"[PortalPolicyServer] Starting server on port {self.port}")
        self._server.start()


def _run_server(policy_factory: Callable[[], Policy], port: int) -> None:
    """Entry point executed inside the subprocess."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    server = PortalPolicyServer(policy_factory, port)
    server.serve()


# ---------------------------------------------------------------------------
# Main-process client
# ---------------------------------------------------------------------------


class PortalPolicy(Policy):
    """Wraps any Policy in a subprocess using Portal shared-memory IPC.

    The *policy_factory* callable is executed inside the subprocess -- heavy
    imports (CUDA, model weights) only happen there.  Communication with the
    main process uses Portal's shared-memory transport for zero-copy numpy
    array transfer.

    Parameters
    ----------
    policy_factory:
        A callable ``() -> Policy`` that creates the policy.  Must be
        picklable (portal uses cloudpickle).  Typically a lambda or a
        small factory function.
    port:
        Portal server port (localhost only).
    startup_timeout:
        Seconds to wait for the subprocess to become ready.  Model loading
        can take 60-120 s for large checkpoints.
    """

    def __init__(
        self,
        policy_factory: Callable[[], Policy],
        port: int = 8011,
        startup_timeout: float = 120.0,
    ) -> None:
        self.port = port
        self._client: portal.Client | None = None

        # Capture for the closure that portal.Process will serialize
        factory = policy_factory
        p = port

        def _run() -> None:
            _run_server(factory, p)

        self._process = portal.Process(_run, start=True)
        print(f"[PortalPolicy] Started subprocess on port {port}")

        self._wait_for_ready(startup_timeout)

    def _wait_for_ready(self, timeout: float) -> None:
        start = time.time()
        while time.time() - start < timeout:
            try:
                self._client = portal.Client(f"localhost:{self.port}")
                if self._client.health_check().result(timeout=5.0):
                    elapsed = time.time() - start
                    print(f"[PortalPolicy] Subprocess ready after {elapsed:.1f}s")
                    return
            except Exception:
                time.sleep(1.0)

        raise TimeoutError(
            f"[PortalPolicy] Subprocess not ready after {timeout}s. "
            "Model loading may have failed -- check subprocess logs."
        )

    # -- Policy interface ---------------------------------------------------

    def get_action(self, observation: Observation) -> tuple[Action, Info]:
        obs_clean = _make_arrays_contiguous(observation)
        future = self._client.get_action(obs_clean)  # type: ignore[union-attr]
        action, info = future.result()
        return action, info

    def reset(self) -> Info | None:
        future = self._client.reset()  # type: ignore[union-attr]
        return future.result()
