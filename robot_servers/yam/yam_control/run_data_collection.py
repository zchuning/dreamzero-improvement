from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import random
import select
import sys

import gymnasium as gym
from gymnasium.envs.registration import register
from prompt_toolkit.shortcuts import radiolist_dialog
import tyro

from yam_control.record_episode_wrapper import RecordEpisodeWrapper
from yam_control.teleop_policy import TeleopPolicy


@dataclass
class DataCollectionConfig:
    operator: str | None = None
    """Username of operator."""

    task_list_path: str | None = None
    """Path to .txt task list file containing one task per line."""

    station: int = 0
    """Station number (used in output directory name)."""

    output_dir: str | None = None
    """Custom output directory.  If not set, defaults to data/<timestamp>-YAM-<station>."""


def prompt_missing_fields(cfg: DataCollectionConfig) -> DataCollectionConfig:
    if cfg.operator is None:
        cfg.operator = input("Enter operator username: ")
    return cfg


def get_task_list(cfg: DataCollectionConfig) -> list[str]:
    if cfg.task_list_path is not None:
        print(f"[INFO] Loading task list from {cfg.task_list_path}")
        with open(cfg.task_list_path, "r") as f:
            task_list = [line.strip() for line in f if len(line.strip()) > 0]
        assert len(task_list) > 0, "Task list should not be empty"
        for i, task in enumerate(task_list, start=1):
            print(f"{i}. {task}")
    else:
        task_name = input("Enter task name: ").strip()
        print(f"[INFO] Using task name: {task_name}")
        task_list = [task_name]

    return task_list


def prompt_task_name(task_list: list[str]) -> str:
    if len(task_list) == 1:
        return task_list[0]

    task_name = None
    while task_name is None:
        shuffled_task_list = task_list.copy()
        random.shuffle(shuffled_task_list)

        task_name = radiolist_dialog(
            title="Task selection",
            text="Please select the first valid task:",
            values=[(task, task) for task in shuffled_task_list],
        ).run()

    print(f"[INFO] Using task name: {task_name}")
    return task_name


def check_terminal_input() -> str | None:
    """Check if there is any input available from stdin (non-blocking).

    Returns the input command if available, None otherwise.
    """
    if select.select([sys.stdin], [], [], 0)[0]:
        return sys.stdin.readline().strip()
    return None


def run_data_collection_loop(
    env,
    policy,
    get_task_name,
    *,
    initial_state=None,
    max_steps=None,
    check_terminal=True,
):
    """Run the data collection control loop.

    Args:
        env: Gym environment (wrapped with recording wrappers)
        policy: Policy with get_action(obs) -> (action, info) method
        get_task_name: Callable that returns a task name string
        initial_state: Optional initial state for the environment
        max_steps: Maximum steps to run (None for infinite)
        check_terminal: Whether to check for terminal input commands

    Returns:
        Number of steps executed
    """
    is_recording = False
    task_name = get_task_name()

    reset_options = {"skip_next_episode": True}
    if initial_state is not None:
        reset_options["initial_state"] = initial_state

    obs, _ = env.reset(options=reset_options)

    step_count = 0
    while max_steps is None or step_count < max_steps:
        action, policy_info = policy.get_action(obs)
        obs, _, _, _, _ = env.step(action)
        step_count += 1

        # Check for terminal input (keyboard commands)
        if check_terminal:
            terminal_cmd = check_terminal_input()
            if terminal_cmd:
                if terminal_cmd == "save":
                    if is_recording:
                        print("[INFO] Saving current recording (via terminal command)...")
                        obs, _ = env.reset(options={"skip_next_episode": True})
                        is_recording = False
                        print(
                            "[INFO] Recording saved. Press the top right button to start new recording."
                        )
                    else:
                        print("[INFO] No recording in progress.")

                elif terminal_cmd == "delete":
                    if is_recording:
                        print("[INFO] Discarding current recording (via terminal command)...")
                        obs, _ = env.reset(
                            options={"discard_episode": True, "skip_next_episode": True}
                        )
                        is_recording = False
                        print(
                            "[INFO] Recording discarded. Press the top right button to start new recording."
                        )
                    else:
                        print("[INFO] No recording in progress.")

        # Prompt for new task name
        if "reset" in policy_info:
            task_name = get_task_name()

        # Start new recording
        elif "start" in policy_info:
            if not is_recording:
                obs, _ = env.reset(options={"task_name": task_name})
                is_recording = True
                print(f"[INFO] Recording started for task: {task_name}")
                print(
                    "[INFO] Press the bottom right button to save recording. "
                    "Press the bottom left button to discard recording."
                )
            else:
                print("[INFO] Recording already in progress. Did not start new recording.")

        # Discard current recording
        elif "discard_confirmed" in policy_info:
            if is_recording:
                obs, _ = env.reset(options={"discard_episode": True, "skip_next_episode": True})
                is_recording = False
                print(
                    "[INFO] Recording discarded. Press the top right button to start new recording."
                )
            else:
                print("[INFO] No recording in progress.")

        elif "discard_cancelled" in policy_info:
            print("[INFO] Discard cancelled. Recording continues.")

        # Save current recording
        elif "save" in policy_info:
            if is_recording:
                print("[INFO] Saving current recording...")
                obs, _ = env.reset(options={"skip_next_episode": True})
                is_recording = False
                print("[INFO] Recording saved. Press the top right button to start new recording.")
            else:
                print("[INFO] No recording in progress.")

    return step_count


def main(cfg: DataCollectionConfig):
    cfg = prompt_missing_fields(cfg)

    task_list = get_task_list(cfg)

    if cfg.output_dir is not None:
        output_dir = Path(cfg.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        output_dir = Path("data") / f"{timestamp}-YAM-{cfg.station:02d}"
    print(f"[INFO] Starting data collection for operator: {cfg.operator}")
    print(f"[INFO] Local output directory: {output_dir}")

    register(id="YamReal-v0", entry_point="yam_control.yam_real_env:YamRealEnv")
    env = gym.make("YamReal-v0")

    env = RecordEpisodeWrapper(
        env,
        output_dir=str(output_dir),
        operator=cfg.operator,
    )

    policy = TeleopPolicy()
    print("[INFO] Leader arms are ready. Press the top left button to sync the follower arms.")
    policy_info = policy.reset()

    print(
        "[INFO] Follower arms are now in sync. Press the top right button to start new recording."
    )
    print("[INFO] Terminal commands: type 'save' + Enter to save, 'delete' + Enter to discard")

    try:
        run_data_collection_loop(
            env,
            policy,
            get_task_name=lambda: prompt_task_name(task_list),
            initial_state=policy_info.get("initial_state"),
        )
    finally:
        env.close()


if __name__ == "__main__":
    main(tyro.cli(DataCollectionConfig))
