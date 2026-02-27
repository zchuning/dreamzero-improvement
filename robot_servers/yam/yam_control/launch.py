"""Tmux launcher for YAM arm servers + main loop."""

from dataclasses import dataclass
import subprocess
from typing import Literal

import tyro


class TmuxSession:
    def __init__(self, session: str):
        self.session = session
        self._created = False

        subprocess.run(
            ["tmux", "kill-session", "-t", session],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

    def new_window(self, name=None, command=None):
        if not self._created:
            args = ["tmux", "new-session", "-d", "-s", self.session]
            if name:
                args += ["-n", name]
            subprocess.run(args, check=True)
            subprocess.run(["tmux", "bind-key", "-n", "C-\\", "kill-server"], check=True)
            subprocess.run([
                "tmux", "set-option", "-t", self.session,
                "status-right", "Press Ctrl+\\ to exit",
            ], check=True)
            self._created = True
        else:
            args = ["tmux", "new-window", "-t", self.session]
            if name:
                args += ["-n", name]
            subprocess.run(args, check=True)

        if command:
            subprocess.run([
                "tmux", "send-keys", "-t",
                f"{self.session}:{name if name else ''}",
                command, "C-m",
            ])

    def attach(self):
        subprocess.run(["tmux", "attach", "-t", self.session])


@dataclass
class Args:
    mode: Literal["data_collection", "evaluation", "server-eval"] = "data_collection"
    """Mode: data_collection, evaluation (local model), or server-eval (remote GR00T server)."""

    task_list_path: str | None = None
    """Optional path to .txt task list file (data collection mode)."""

    # Evaluation settings (evaluation and server-eval modes)
    policy: Literal["example", "gr00t", "gr00t-server"] = "example"
    """Which policy to use (evaluation mode)."""
    model_path: str | None = None
    """Path to model checkpoint (required for --policy gr00t)."""
    embodiment_tag: str = "new_embodiment"
    """Embodiment tag (gr00t policy)."""
    device: str = "cuda:0"
    """CUDA device for model inference."""
    portal_port: int = 8011
    """Portal IPC port for subprocess inference."""
    task_description: str = "Do something useful"
    """Task description for evaluation."""
    action_horizon: int = 30
    """Action horizon for evaluation."""
    policy_control_freq: int = 30
    """Control frequency (Hz)."""
    no_record: bool = False
    """Disable episode recording."""
    use_async: bool = False
    """Use async action chunking."""
    use_sim: bool = False
    """Use MuJoCo simulation instead of real robot."""

    # Remote GR00T server settings (server-eval mode)
    gr00t_host: str = "localhost"
    """Host of the remote GR00T policy server."""
    gr00t_port: int = 5555
    """Port of the remote GR00T policy server."""
    gr00t_timeout_ms: int = 15000
    """ZMQ timeout for server requests (ms)."""

    # Intervention / reverse-sync (server-eval mode)
    use_reverse_sync: bool = False
    """Start leader arm servers for operator intervention."""

    attach: bool = True
    """Attach to the main tmux session."""


def _build_rollout_cmd(args: Args) -> str:
    """Build the run_policy_rollout.py CLI command from args."""
    if args.mode == "server-eval":
        cmd = (
            f"python -m yam_control.run_policy_rollout"
            f" --policy gr00t-server"
            f" --gr00t-host {args.gr00t_host}"
            f" --gr00t-port {args.gr00t_port}"
            f" --gr00t-timeout-ms {args.gr00t_timeout_ms}"
        )
    else:
        cmd = (
            f"python -m yam_control.run_policy_rollout"
            f" --policy {args.policy}"
            f" --portal-port {args.portal_port}"
        )
        if args.model_path:
            cmd += f" --model-path {args.model_path}"
        if args.policy == "gr00t":
            cmd += f" --embodiment-tag {args.embodiment_tag}"
            cmd += f" --device {args.device}"

    cmd += (
        f" --task-description '{args.task_description}'"
        f" --action-horizon {args.action_horizon}"
        f" --policy-control-freq {args.policy_control_freq}"
    )
    if args.no_record:
        cmd += " --no-record-episode"
    if args.use_reverse_sync:
        cmd += " --use-reverse-sync"
    if args.use_async:
        cmd += " --use-async"
    if args.use_sim:
        cmd += " --use-sim"
    return cmd


def main(args: Args):
    session = TmuxSession("robot_server")

    if args.mode == "data_collection":
        cmd = "python -m yam_control.run_data_collection"
        if args.task_list_path:
            cmd += f" --task-list-path {args.task_list_path}"
        session.new_window("main", cmd)
    else:
        session.new_window("main", _build_rollout_cmd(args))

    if not args.use_sim:
        session.new_window(
            "follow_l", "python -m yam_control.arm_server --mode follower --side left"
        )
        session.new_window(
            "follow_r", "python -m yam_control.arm_server --mode follower --side right"
        )
        if args.mode == "data_collection" or args.use_reverse_sync:
            session.new_window(
                "leader_l", "python -m yam_control.arm_server --mode leader --side left"
            )
            session.new_window(
                "leader_r", "python -m yam_control.arm_server --mode leader --side right"
            )

    if args.attach:
        session.attach()


if __name__ == "__main__":
    main(tyro.cli(Args))
