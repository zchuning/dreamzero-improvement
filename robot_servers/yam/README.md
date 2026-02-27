# yam-control

Control framework for the YAM bimanual robot.  Handles **data collection** (teleoperation) and **policy rollout** (sync and async).

---

## 1. Installation

This project uses **[uv](https://docs.astral.sh/uv/)** for dependency management — no Conda or virtualenv setup needed.  `uv` automatically creates and manages a `.venv` for you.

```bash
git clone <repo-url> yam-control
cd yam-control
uv python pin 3.12
uv sync --all-extras         # install everything
```

Or install only the extras you need:

```bash
uv sync                      # core only (numpy, gymnasium, opencv, portal, …)
uv sync --extra arm-server   # + i2rt (robot hardware)
uv sync --extra eval         # + viser, trimesh, scipy (rollout UI & 3D viz)
uv sync --extra convert      # + av, pandas, pyarrow (LeRobot conversion)
uv sync --extra sim          # + mujoco (simulation, no hardware needed)
uv sync --extra kinematics   # + mink, mujoco (FK/IK for cartesian control)
uv sync --extra dataset-replay  # + pandas, pyarrow (LeRobot dataset replay)
uv sync --extra server-eval    # + pyzmq, msgpack (remote GR00T server eval)
```

### [Already Done For You] Camera setup (one-time)

Each RealSense D405 needs a `/dev/video_*` symlink.  Find serial numbers:

```bash
rs-enumerate-devices | grep Serial
```

Create `/etc/udev/rules.d/99-realsense.rules`:

```
SUBSYSTEM=="video4linux", ATTRS{serial}=="SERIAL_TOP",   ATTR{index}=="0", SYMLINK+="video_top"
SUBSYSTEM=="video4linux", ATTRS{serial}=="SERIAL_LEFT",  ATTR{index}=="0", SYMLINK+="video_left"
SUBSYSTEM=="video4linux", ATTRS{serial}=="SERIAL_RIGHT", ATTR{index}=="0", SYMLINK+="video_right"
```

```bash
sudo udevadm control --reload-rules && sudo udevadm trigger
```

---

## 2. Teleoperation

### 2.1  Collect data

```bash
uv sync --extra arm-server
uv run python -m yam_control.launch --mode data_collection
```

This opens a `tmux` session that starts the arm servers and the data collection script.  You will be prompted for your **operator name** and **task name** (both saved in episode metadata).

**Button controls on the leader arms:**

| Button | Location | Action |
|--------|----------|--------|
| Top-left | Left leader arm | Initialize: Sync leader-follower |
| Top-right | Right leader arm | Start recording |
| Bottom-left | Left leader arm | Discard current recording |
| Bottom-right | Right leader arm | Save current recording |

By default episodes are saved to `data/<timestamp>-YAM-<station>/`.  Override with `--output-dir`:

```bash
uv sync --extra arm-server
uv run python -m yam_control.launch --mode data_collection --output-dir /path/to/my_data
```

Press `Ctrl+\` to stop data collection, and `tmux kill-session -t robots` to kill robot tmux sessions when done.

### 2.2  Convert collected data to LeRobot format

```bash
uv sync --extra convert
uv run python -m yam_control.convert_to_lerobot \
    --input data/2025-01-15-10-30-00-YAM-00 \
    --output lerobot_dataset
```

This reads the raw episodes (`.npy` + `.mp4`) and produces a LeRobot v2.1 dataset directory with Parquet metadata and re-encoded video.

---

## 3. Policy Evaluation

### 3.1  Quick test — PredefinedPosePolicy (no model needed)

The built-in `PredefinedPosePolicy` cycles through 5 joint poses with smooth cosine interpolation.  Use it to verify your hardware and software setup before plugging in a real model.

**Simulation (no hardware required):**

```bash
uv sync --extra eval
uv run python -m yam_control.run_policy_rollout \
    --policy example --use-sim --no-record-episode
```

**Real robot:**

```bash
uv sync --extra eval

# Terminal 1 — start arm servers
uv run python -m yam_control.launch --mode evaluation --no-attach

# Terminal 2 — run the example policy
uv run python -m yam_control.run_policy_rollout --policy example
```

Open **http://localhost:8080** in your browser.  You will see the 3D robot model, predicted trajectory, camera feeds, and control buttons.  Press **Start** to begin execution.

### 3.2  Plugging in your own model

The rollout pipeline is model-agnostic.  You implement a `Policy` subclass and the framework handles everything else: action chunking (sync or async), low-pass filtering, speed limiting, ViserUI, and recording.

**Step 1 — Implement the Policy interface:**

```python
# my_policy.py
from yam_control.policy import Policy

class MyPolicy(Policy):
    def __init__(self, model_path: str):
        self.model = load_my_model(model_path)   # your model

    def get_action(self, observation):
        # --- Read observation ---
        images = {
            "top":   observation["top_camera_image"],     # (480,640,3) uint8
            "left":  observation["left_camera_image"],
            "right": observation["right_camera_image"],
        }
        state = {
            "left_arm":     observation["left_joint_pos"],     # (6,) float32
            "left_gripper": observation["left_gripper_pos"],   # (1,) float32
            "right_arm":    observation["right_joint_pos"],
            "right_gripper":observation["right_gripper_pos"],
        }
        task = observation.get("annotation.task", "")

        # --- Your model inference ---
        # Must return a dict of (T, D) arrays (the action chunk)
        action_chunk = self.model.predict(images, state, task)
        # action_chunk = {
        #     "left_joint_pos":    np.ndarray (T, 6),
        #     "left_gripper_pos":  np.ndarray (T, 1),
        #     "right_joint_pos":   np.ndarray (T, 6),
        #     "right_gripper_pos": np.ndarray (T, 1),
        # }

        first_action = {k: v[0] for k, v in action_chunk.items()}
        return first_action, {"action_chunk": action_chunk}

    def reset(self):
        self.model.reset()
        return None
```

**Step 2 — Write a rollout script:**

*Option A — In-process (simple, good for lightweight models):*

```python
# my_rollout.py
from yam_control.run_policy_rollout import run_rollout, RolloutConfig
from my_policy import MyPolicy

policy = MyPolicy("path/to/checkpoint")
run_rollout(policy, RolloutConfig(
    task_description="pick up the red cup",
    use_async=True,       # async chunking (no wait between chunks)
))
```

*Option B — Subprocess via PortalPolicy (recommended for GPU models):*

For heavy models (GPU inference), wrap your policy in `PortalPolicy`.  This runs model inference in a separate subprocess using Portal shared-memory IPC for zero-copy numpy array transfer.  CUDA and model weights stay out of the main control process.

```python
# my_rollout.py
from yam_control.portal_policy import PortalPolicy
from yam_control.run_policy_rollout import run_rollout, RolloutConfig
from my_policy import MyPolicy

policy = PortalPolicy(
    policy_factory=lambda: MyPolicy("path/to/checkpoint"),
    port=8011,
)
run_rollout(policy, RolloutConfig(task_description="pick up the red cup"))
```

**Step 3 — Run:**

```bash
# Start arm servers first (skip if using use_sim=True)
uv run python -m yam_control.launch --mode evaluation --no-attach

# Run your script
uv run python my_rollout.py
```

ViserUI opens at **http://localhost:8080** — press **Start** to begin execution.

### 3.3  GR00T example

The repo includes a `Gr00tPolicy` adapter for [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T).  Model inference runs in a subprocess via `PortalPolicy`.

**Prerequisites:** Install Isaac-GR00T in the same environment.

```bash
# Terminal 1 — start arm servers
uv run python -m yam_control.launch --mode evaluation --no-attach

# Terminal 2 — run rollout
uv run python -m yam_control.run_policy_rollout \
    --policy gr00t \
    --model-path /path/to/checkpoint \
    --task-description "pick up the cup"
```

Or via Python:

```python
from yam_control.portal_policy import PortalPolicy
from yam_control.gr00t_policy import make_gr00t_factory
from yam_control.run_policy_rollout import run_rollout, RolloutConfig

policy = PortalPolicy(
    policy_factory=make_gr00t_factory(
        model_path="/path/to/checkpoint",
        embodiment_tag="new_embodiment",
        device="cuda:0",
    ),
)
run_rollout(policy, RolloutConfig(task_description="pick up the cup"))
```

### 3.4  Remote GR00T Server Evaluation

For production evaluation, VLA inference runs on a GPU machine while robot control runs locally on the robot machine.  The two communicate over ZMQ (port 5555 by default).  No `groot.*` imports are needed on the robot side.

```
┌────────────────────────────────┐          ┌────────────────────────────┐
│  GPU machine (gr00t_main)      │          │  Robot machine (yam-control)│
│                                │          │                            │
│  SPEEDUP_inference.py          │          │  launch.py --mode server-eval│
│    └─ run_gr00t_server.py      │◄─ ZMQ ──│    └─ Gr00tServerPolicy     │
│       (VLA model on GPU)       │  :5555   │       (ZMQ client)          │
└────────────────────────────────┘          └────────────────────────────┘
```

**Step 1 — Start the inference server (GPU machine):**

```bash
cd gr00t_main/groot/control/envs/yam
uv run --extra experimental experimental/SPEEDUP_inference.py \
    --station 1 \
    path/to/checkpoint/
```

The server starts on `tcp://0.0.0.0:5555` and prints connection instructions.

**Step 2 — Start robot control (robot machine):**

```bash
cd yam-control
uv sync --extra server-eval --extra eval

# Option A: Full tmux launcher (starts arm servers + rollout automatically)
uv run python -m yam_control.launch \
    --mode server-eval \
    --gr00t-host <gpu-machine-ip> \
    --gr00t-port 5555 \
    --task-description "pick up the cup"

# Option B: Manual (two terminals)
# Terminal 1 — start arm servers
uv run python -m yam_control.launch --mode evaluation --no-attach

# Terminal 2 — run rollout
uv run python -m yam_control.run_policy_rollout \
    --policy gr00t-server \
    --gr00t-host <gpu-machine-ip> \
    --gr00t-port 5555 \
    --task-description "pick up the cup"
```

ViserUI opens at **http://localhost:8080** — press **Start** to begin.

**With operator intervention (reverse sync):**

```bash
uv run python -m yam_control.launch \
    --mode server-eval \
    --gr00t-host <gpu-machine-ip> \
    --use-reverse-sync
```

Or from Python:

```python
from yam_control.gr00t_server_policy import Gr00tServerPolicy
from yam_control.run_policy_rollout import run_rollout, RolloutConfig

policy = Gr00tServerPolicy(host="gpu-machine", port=5555)
assert policy.ping(), "Cannot reach server"
run_rollout(policy, RolloutConfig(task_description="pick up the cup"))
```

### 3.5  Simulation

MuJoCo simulation uses the same environment interface as the real robot — no hardware needed.

```bash
uv run python -m yam_control.run_policy_rollout \
    --policy example --use-sim --no-record-episode
```

Or from Python:

```python
from yam_control.run_policy_rollout import run_rollout, RolloutConfig
from my_policy import MyPolicy

policy = MyPolicy(...)
run_rollout(policy, RolloutConfig(use_sim=True))
```

### 3.6  ViserUI and recording

**ViserUI** (`yam_control/viser_ui.py`) is a web interface served at **http://localhost:8080** showing:

- 3D robot model (URDF) updated from live joint positions
- Predicted action trajectory (colored spheres or coordinate frames on the end-effectors)
- **Trajectory timer** (`yam_control/viser_timer.py`) — tracks execution time with auto-pause on Step Once
- **Start** / **Pause** / **Home** / **Step Once** buttons
- **Task** text field (passed to your policy as `observation["annotation.task"]`)
- Live camera feeds (top, left, right)

ViserUI launches automatically when you call `run_rollout()` — no manual setup needed.

**Architecture** — two processes communicate via Portal shared-memory IPC:

```
┌──────────────────────────────┐    Portal IPC     ┌──────────────────────┐
│  Main process                │◄─────────────────►│  ViserUI subprocess  │
│  StartStopPlayPolicyWrapper  │  commands (start/  │  Web server :8080    │
│  wraps your Policy           │  pause/home) +     │  3D robot, cameras,  │
│                              │  state updates     │  timer, trajectories │
└──────────────────────────────┘                    └──────────────────────┘
```

**Using ViserUI components in a custom loop** (advanced):

```python
from yam_control.viser_ui import StartStopPlayPolicyWrapper, ViserUI
import portal, time

# Main process: wrap your policy with start/stop control
wrapped = StartStopPlayPolicyWrapper(policy=my_policy)

# Subprocess: launch the web UI (portal.Process handles IPC)
def start_ui():
    ui = ViserUI(task_description="pick up the cup")
    while True:
        ui.update_ui()
        time.sleep(1/20)   # 20 Hz refresh

portal.Process(start_ui, start=True)

# Your control loop
while True:
    action, info = wrapped.get_action(observation)
    observation, _, _, _, _ = env.step(action)
```

**Using TrajectoryTimer standalone** (e.g. in a non-Viser script):

```python
from yam_control.viser_timer import TrajectoryTimer

timer = TrajectoryTimer()
timer.start()                         # begin timing
# ... run trajectory ...
print(timer.get_display_string())     # "0:12"
timer.pause()                         # freeze at current time
timer.start()                         # resume from paused time
timer.home()                          # stop; next start() resets to 0:00
```

**Recording** is enabled by default.  Episodes are saved to `data/eval/` (real) or `data/sim/` (simulation).  Disable with `--no-record-episode` or `record_episode=False` in `RolloutConfig`.

### 3.7  Pipeline and configuration

```
Your Policy.get_action()               <- runs in subprocess if using PortalPolicy
    |  returns action chunk (T, D)        (shared-memory IPC, zero-copy numpy)
    v
SyncChunkingPolicy / AsyncChunkingPolicy
    |  executes one step at a time
    v
LowPassFilterPolicyWrapper
    |  smooths actions
    v
StartStopPlayPolicyWrapper (ViserUI)
    |  start / pause / home control
    v
SpeedLimitWrapper
    |  clips velocity for safety
    v
RecordEpisodeWrapper
    |  saves episodes to disk
    v
YamRealEnv / YamSimEnv
    |  sends commands to robot / MuJoCo
```

**Sync vs async rollout:**

- **Sync** (`use_async=False`, default): Predict -> execute full chunk -> predict again.  Brief pause between chunks while waiting for the next prediction.
- **Async** (`use_async=True`): A background thread continuously predicts the next chunk while the current one executes.  No pauses.  Set `policy_latency_steps` to tune queue depth.

**RolloutConfig options:**

| Field | Default | Description |
|-------|---------|-------------|
| `task_description` | `"Do something useful"` | Task for the policy |
| `policy_control_freq` | `30` | Control frequency (Hz) |
| `action_horizon` | `30` | Steps per action chunk |
| `use_async` | `False` | Async action chunking |
| `policy_latency_steps` | `4` | Queue depth (async only) |
| `use_sim` | `False` | MuJoCo simulation |
| `use_low_pass_filter` | `True` | Action smoothing |
| `record_episode` | `True` | Record to disk |
| `move_to_home_pose` | `True` | Home pose at start |

---

## 4. Diagnostics & Calibration Scripts

Hardware diagnostic scripts are available in `yam_control/scripts/`.

```bash
# Daily calibration check — reads all arm joint angles and flags drift
python -m yam_control.scripts.daily_calibration_check

# Check specific arm(s) only
python -m yam_control.scripts.daily_calibration_check --channel can_leader_l can_follow_r

# Print joint angles for a single arm
python -m yam_control.scripts.print_joint_angles --channel can_follow_l

# Print motor temperatures
python -m yam_control.scripts.print_motor_temps --channel can_follow_l

# Calibrate leader arm motor 6 (zero position)
python -m yam_control.scripts.calibrate_leader_motor_6 --channel can_leader_l
```

The calibration check is also available as a CLI command (after `pip install`):

```bash
yam-calibration-check
yam-calibration-check --channel can_leader_l
```

---

## 5. Package Structure

```
yam_control/
├── policy.py                       # Policy ABC — subclass this
├── run_policy_rollout.py           # run_rollout() entry point + RolloutConfig
├── run_data_collection.py          # Data collection entry point
├── portal_policy.py                # PortalPolicy: subprocess inference via shared-memory IPC
├── example_policy.py               # PredefinedPosePolicy (5-pose cycling, no model needed)
├── gr00t_policy.py                 # GR00T adapter: Gr00tPolicy + make_gr00t_factory
├── gr00t_server_policy.py          # ZMQ client for remote GR00T server (no groot.* imports)
│
├── sync_chunking_policy.py         # Execute action chunks one step at a time
├── async_chunking_policy.py        # Async: background thread inference
├── rtc_async_chunking_policy.py    # RTC async chunking (Real-Time Chunking)
├── low_pass_filter_policy.py       # FIR low-pass filter for action smoothing
├── speed_limit_wrapper.py          # Safety wrapper (joint/gripper velocity limits)
├── clip_action_policy.py           # Clips delta between consecutive actions
├── replay_policy.py                # Replay recorded action sequences
├── dataset_replay_policy.py        # Replay from LeRobot dataset
├── hil_policy.py                   # Human-in-the-loop policy wrapper
├── intervention_policy_wrapper.py  # Operator intervention (takeover) support
├── intervention_state_machine.py   # State machine for intervention procedure
├── pico_policy.py                  # Pico XR headset teleoperation
│
├── rtc_config.py                   # Real-Time Chunking configuration
├── rtc_action_queue.py             # Thread-safe action queue for RTC
├── latency_tracker.py              # Inference latency tracking
│
├── filter_utils.py                 # Signal processing utilities (FIR filter)
├── action_utils.py                 # Action manipulation utilities
├── key_remapping_utils.py          # Observation/action key remapping for Gr00t
├── embodiment.py                   # EmbodimentTag enum (standalone)
│
├── viser_ui.py                     # ViserUI + 3D URDF robot + trajectory viz
├── viser_timer.py                  # Trajectory timer for Viser UI
├── record_episode_wrapper.py       # Gym wrapper for recording episodes
│
├── yam_real_env.py                 # Real robot environment (portal + RealSense)
├── yam_sim_env.py                  # MuJoCo simulation environment
├── _base_yam_env.py                # Base environment (shared sim/real logic)
├── kinematics.py                   # FK/IK via mink + mujoco
├── models/                         # URDF, MuJoCo XML, and mesh assets
│
├── launch.py                       # Tmux launcher for arm servers + main loop
├── arm_server.py                   # Arm server (requires i2rt)
├── realsense.py                    # RealSense camera driver
├── teleop_policy.py                # Teleoperation policy (leader arm mirroring)
├── convert_to_lerobot.py           # Raw data -> LeRobot v2.1 converter
├── constants.py                    # Port numbers and CAN interfaces
│
└── scripts/                        # Hardware diagnostics & calibration
    ├── daily_calibration_check.py  # Full calibration check
    ├── print_joint_angles.py       # Read joint angles from CAN bus
    ├── print_motor_temps.py        # Read motor temperatures
    └── calibrate_leader_motor_6.py # Motor 6 zero-position calibration
```
