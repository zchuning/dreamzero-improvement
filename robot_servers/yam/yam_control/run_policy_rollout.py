"""Generic rollout runner for YAM robot.

Provides :func:`run_rollout` which takes any :class:`Policy` subclass and
handles environment setup, action chunking (sync or async), safety
wrappers, ViserUI, recording, and the main control loop.

Usage from your own script::

    from yam_control.policy import Policy
    from yam_control.run_policy_rollout import run_rollout, RolloutConfig

    class MyPolicy(Policy):
        def get_action(self, observation):
            ...  # your model inference
            return first_action, {"action_chunk": chunk}

    policy = MyPolicy(...)
    run_rollout(policy, RolloutConfig(task_description="pick up the cup"))
"""

import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import gymnasium as gym
from gymnasium.envs.registration import register
import numpy as np
import portal

from yam_control.filter_utils import PeriodicAverageAccumulator
from yam_control.low_pass_filter_policy import LowPassFilterPolicyWrapper
from yam_control.policy import Policy
from yam_control.record_episode_wrapper import RecordEpisodeWrapper
from yam_control.speed_limit_wrapper import SpeedLimitWrapper
from yam_control.sync_chunking_policy import SyncChunkingPolicy

HOME_POSE = {
    "left_joint_pos": np.zeros(6),
    "right_joint_pos": np.zeros(6),
    "left_gripper_pos": np.ones(1),
    "right_gripper_pos": np.ones(1),
}


# ---------------------------------------------------------------------------
# Home pose helpers
# ---------------------------------------------------------------------------


def interpolate_trajectory(
    start: dict[str, np.ndarray],
    target: dict[str, np.ndarray],
    num_steps: int,
) -> list[dict[str, np.ndarray]]:
    trajectory = []
    for i in range(num_steps + 1):
        t = (1 - np.cos(i / num_steps * np.pi)) / 2
        waypoint = {}
        for key in start:
            if key in target:
                waypoint[key] = start[key] * (1 - t) + target[key] * t
            else:
                waypoint[key] = start[key]
        trajectory.append(waypoint)
    return trajectory


def move_to_home_pose(env, obs: dict, target=None, duration=3.0, freq=30) -> dict:
    if target is None:
        target = HOME_POSE
    current = {
        "left_joint_pos": obs.get("left_joint_pos", np.zeros(6)),
        "right_joint_pos": obs.get("right_joint_pos", np.zeros(6)),
        "left_gripper_pos": obs.get("left_gripper_pos", np.array([1.0])),
        "right_gripper_pos": obs.get("right_gripper_pos", np.array([1.0])),
    }
    num_steps = int(duration * freq)
    trajectory = interpolate_trajectory(current, target, num_steps)
    print(f"[Home] Moving to home over {duration}s ({num_steps} steps)")
    for i, waypoint in enumerate(trajectory[1:], 1):
        obs, _, terminated, truncated, _ = env.step(waypoint)
        if i % (freq // 2) == 0:
            print(f"[Home] {i / num_steps * 100:.0f}%")
        if terminated or truncated:
            print("[Home] WARNING: terminated during movement")
            break
    print("[Home] Done")
    return obs


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class RolloutConfig:
    """Configuration for :func:`run_rollout`.

    Model-agnostic -- does not contain any model-specific fields.
    """

    # Task
    task_description: str = "Do something useful"
    """Task description for the policy."""

    # Control
    policy_control_freq: int = 30
    """Control loop frequency (Hz)."""
    action_horizon: int = 30
    """Steps to execute from each action chunk."""

    # Async rollout
    use_async: bool = False
    """Use async action chunking (background inference thread)."""
    policy_latency_steps: int = 4
    """Queue depth before starting next inference (async mode only)."""

    # Environment
    use_sim: bool = False
    """Use MuJoCo simulation instead of real robot."""

    # Recording
    record_episode: bool = True
    """Record episodes to disk."""
    station: int = 0
    """Station number for output directory."""
    operator: str | None = None
    """Operator name (prompted if None)."""

    # Home pose
    move_to_home_pose: bool = True
    """Move to home pose before starting."""
    home_pose_duration: float = 3.0
    """Home pose movement duration (seconds)."""

    # Low-pass filter
    use_low_pass_filter: bool = True
    """Apply low-pass filter to smooth actions."""
    low_pass_filter_k: int = 3
    """Filter buffer size."""
    low_pass_filter_alpha: float = 0.5
    """Filter smoothing factor (0 < alpha <= 1)."""

    # Intervention / reverse-sync
    use_reverse_sync: bool = False
    """Enable operator intervention via leader arms (real robot only)."""
    proximity_threshold_deg: float = 6.0
    """Proximity threshold (degrees) for reverse-sync handoff."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _reset_env(env, home_state, task_name, cfg, discard_episode=False):
    observation, _ = env.reset(
        options=dict(
            initial_state=deepcopy(home_state),
            task_name=task_name,
            force_reset=True,
            discard_episode=discard_episode,
        )
    )
    if cfg.move_to_home_pose:
        observation = move_to_home_pose(
            env, observation, HOME_POSE,
            cfg.home_pose_duration * 0.67, cfg.policy_control_freq,
        )
    return observation


def _get_recording_elapsed(env) -> str:
    """Walk the wrapper chain to find RecordEpisodeWrapper and return elapsed time."""
    e = env
    while e is not None:
        if isinstance(e, RecordEpisodeWrapper) and e.is_recording:
            secs = e.recording_elapsed_seconds
            return f"{int(secs) // 60:02d}:{int(secs) % 60:02d}"
        e = getattr(e, "env", None)
    return "--:--"


def _run_main_loop(cfg, env, active_policy, home_state, observation):
    from yam_control.viser_ui import StartStopPlayPolicyWrapper, run_viser_subprocess

    wrapped = StartStopPlayPolicyWrapper(
        policy=active_policy, task_description=cfg.task_description,
    )

    def _start_viser():
        run_viser_subprocess(task_description=cfg.task_description)

    portal.Process(_start_viser, start=True)
    print("[Main] ViserUI launched at http://localhost:8080")

    fwd_acc = PeriodicAverageAccumulator(window_seconds=1.0)

    while True:
        t0 = time.monotonic()
        action, info = wrapped.get_action(observation)
        dt_ms = (time.monotonic() - t0) * 1000

        if info:
            avg = fwd_acc.add(dt_ms)
            if avg is not None:
                rec = _get_recording_elapsed(env)
                status = f"\r[Main] rec {rec} | fwd {avg:.1f}ms"
                sys.stdout.write(f"{status:<60}")
                sys.stdout.flush()

        if info.get("event") == "home":
            sys.stdout.write("\n")
            pi = wrapped.reset()
            observation = _reset_env(
                env, home_state, pi.get("task_name", cfg.task_description), cfg,
                discard_episode=pi.get("discard_episode", False),
            )
            continue

        observation, _, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            sys.stdout.write("\n")
            pi = wrapped.reset()
            observation = _reset_env(
                env, home_state, pi.get("task_name", cfg.task_description), cfg,
                discard_episode=pi.get("discard_episode", False),
            )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_rollout(policy: Policy, cfg: RolloutConfig | None = None) -> None:
    """Run the full rollout loop with the given policy.

    This is the main entry point for users.  It handles:
    - Environment creation (real or sim)
    - Wrapper stacking (recording, speed limit)
    - Action chunking (sync or async)
    - Low-pass filtering
    - ViserUI control (http://localhost:8080)
    - Episode management (home pose, reset)

    Args:
        policy: Your :class:`Policy` subclass instance.
        cfg: Rollout configuration.  Defaults to :class:`RolloutConfig()`.
    """
    if cfg is None:
        cfg = RolloutConfig()

    if cfg.operator is None:
        cfg.operator = input("Enter operator username: ")

    # -- Environment --
    if cfg.use_sim:
        register(id="YamSim-v0", entry_point="yam_control.yam_sim_env:YamSimEnv")
        env = gym.make("YamSim-v0", policy_control_freq=cfg.policy_control_freq)
    else:
        register(id="YamReal-v0", entry_point="yam_control.yam_real_env:YamRealEnv")
        env = gym.make("YamReal-v0", policy_control_freq=cfg.policy_control_freq)

    if cfg.record_episode:
        ts = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        suffix = "sim" if cfg.use_sim else "eval"
        output_dir = Path("data") / suffix / f"{ts}-YAM-{cfg.station:02d}-{suffix}"
        env = RecordEpisodeWrapper(
            env, output_dir=str(output_dir), operator=cfg.operator,
            show_inline_timer=False,
        )

    env = SpeedLimitWrapper(env, control_freq=cfg.policy_control_freq)

    # -- Initial reset --
    home_state = {k: v.copy() for k, v in HOME_POSE.items()}
    observation, _ = env.reset(
        options=dict(initial_state=deepcopy(home_state), skip_next_episode=True)
    )

    if cfg.move_to_home_pose:
        observation = move_to_home_pose(
            env, observation, HOME_POSE, cfg.home_pose_duration, cfg.policy_control_freq
        )

    # -- Wrap policy: chunking -> low-pass filter --
    if cfg.use_async:
        from yam_control.async_chunking_policy import AsyncChunkingPolicy

        active_policy: Any = AsyncChunkingPolicy(
            policy=policy,
            action_exec_horizon=cfg.action_horizon,
            policy_latency_steps=cfg.policy_latency_steps,
            max_get_action_seconds=0.5 / cfg.policy_control_freq,
        )
    else:
        active_policy = SyncChunkingPolicy(
            policy=policy, action_exec_horizon=cfg.action_horizon,
        )

    if cfg.use_low_pass_filter:
        active_policy = LowPassFilterPolicyWrapper(
            policy=active_policy,
            k=cfg.low_pass_filter_k,
            alpha=cfg.low_pass_filter_alpha,
        )

    if cfg.use_reverse_sync and not cfg.use_sim:
        from yam_control.intervention_policy_wrapper import InterventionPolicyWrapper

        active_policy = InterventionPolicyWrapper(
            policy=active_policy,
            proximity_threshold_deg=cfg.proximity_threshold_deg,
        )
        print("[Main] Intervention support enabled (left button 0 = sync)")

    # -- Run main loop with ViserUI --
    try:
        _run_main_loop(cfg, env, active_policy, home_state, observation)
    except KeyboardInterrupt:
        print("\n[Main] KeyboardInterrupt received, saving in-progress episode...")
    finally:
        env.close()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from typing import Literal as _Lit

    import tyro

    @dataclass
    class _CliConfig(RolloutConfig):
        """CLI for run_policy_rollout.

        Select a policy with ``--policy``.
        """

        policy: _Lit["example", "gr00t", "gr00t-server"] = "example"
        """Which policy to use."""

        # Example policy settings
        num_interpolation_steps: int = 30
        """Waypoints between consecutive poses (example policy)."""

        # GR00T local settings (used when --policy gr00t)
        model_path: str | None = None
        """Path to GR00T model checkpoint."""
        embodiment_tag: str = "new_embodiment"
        """Embodiment tag for the model."""
        device: str = "cuda:0"
        """CUDA device for model inference."""
        portal_port: int = 8011
        """Portal IPC port for subprocess inference."""

        # GR00T server settings (used when --policy gr00t-server)
        gr00t_host: str = "localhost"
        """Host of the remote GR00T policy server."""
        gr00t_port: int = 5555
        """Port of the remote GR00T policy server."""
        gr00t_timeout_ms: int = 15000
        """ZMQ timeout for server requests (ms)."""

    cli_cfg = tyro.cli(_CliConfig)

    if cli_cfg.policy == "example":
        from yam_control.example_policy import PredefinedPosePolicy
        _policy: Policy = PredefinedPosePolicy(
            num_interpolation_steps=cli_cfg.num_interpolation_steps,
        )
    elif cli_cfg.policy == "gr00t-server":
        from yam_control.gr00t_server_policy import Gr00tServerPolicy

        _policy = Gr00tServerPolicy(
            host=cli_cfg.gr00t_host,
            port=cli_cfg.gr00t_port,
            timeout_ms=cli_cfg.gr00t_timeout_ms,
        )
        if not _policy.ping():
            raise ConnectionError(
                f"Cannot reach GR00T server at {cli_cfg.gr00t_host}:{cli_cfg.gr00t_port}. "
                f"Start the server first with run_gr00t_server.py from gr00t_main."
            )
        print(f"[Main] Connected to GR00T server at {cli_cfg.gr00t_host}:{cli_cfg.gr00t_port}")
    else:
        if not cli_cfg.model_path:
            raise ValueError("--model-path is required for --policy gr00t")

        from yam_control.gr00t_policy import make_gr00t_factory
        from yam_control.portal_policy import PortalPolicy

        _policy = PortalPolicy(
            policy_factory=make_gr00t_factory(
                model_path=cli_cfg.model_path,
                embodiment_tag=cli_cfg.embodiment_tag,
                device=cli_cfg.device,
            ),
            port=cli_cfg.portal_port,
        )

    run_rollout(_policy, cli_cfg)
