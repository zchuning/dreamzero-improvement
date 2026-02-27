"""ViserUI: web-based control interface for policy evaluation.

Provides a Viser web UI with start/pause/home buttons, trajectory timer,
live camera feeds, a 3D URDF robot model updated from proprioception,
and predicted action trajectory visualization via forward kinematics.
Communicates with the policy wrapper via Portal IPC.

**Automatic usage** — ViserUI is launched automatically when you call
:func:`~yam_control.run_policy_rollout.run_rollout`::

    from yam_control.run_policy_rollout import run_rollout, RolloutConfig
    from yam_control.example_policy import PredefinedPosePolicy

    policy = PredefinedPosePolicy()
    run_rollout(policy, RolloutConfig(use_sim=True))
    # ViserUI opens at http://localhost:8080

**Manual usage** — use ``StartStopPlayPolicyWrapper`` + ``ViserUI`` directly
if you need custom control over the rollout loop::

    from yam_control.viser_ui import StartStopPlayPolicyWrapper, ViserUI
    import portal, time

    # In the main process: wrap your policy
    wrapped = StartStopPlayPolicyWrapper(policy=my_policy)

    # In a subprocess: launch the web UI
    def start_ui():
        ui = ViserUI(task_description="pick up the cup")
        while True:
            ui.update_ui()
            time.sleep(1/20)

    portal.Process(start_ui, start=True)

    # Main control loop
    while True:
        action, info = wrapped.get_action(observation)
        observation, _, _, _, _ = env.step(action)

**Architecture** — two processes communicate via Portal shared-memory IPC::

    ┌─────────────────────────────┐     Portal IPC     ┌──────────────────────┐
    │  Main process               │◄──────────────────►│  ViserUI subprocess  │
    │  StartStopPlayPolicyWrapper │  commands (start/   │  Web server :8080    │
    │  wraps your Policy          │  pause/home) +      │  3D robot, cameras,  │
    │                             │  state updates      │  timer, trajectories │
    └─────────────────────────────┘                     └──────────────────────┘
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Literal

import numpy as np
import portal

from yam_control.policy import Action, Info, Observation, Policy
from yam_control.viser_timer import TrajectoryTimer


def _make_arrays_contiguous(obj: Any, memo: dict | None = None) -> Any:
    """Recursively ensure numpy arrays are contiguous for Portal serialization."""
    if memo is None:
        memo = {}
    if id(obj) in memo:
        return memo[id(obj)]
    if isinstance(obj, np.ndarray):
        return np.ascontiguousarray(obj)
    if isinstance(obj, dict):
        new: dict = {}
        memo[id(obj)] = new
        for k, v in obj.items():
            new[k] = _make_arrays_contiguous(v, memo)
        return new
    if isinstance(obj, (list, tuple)):
        new_list: list = []
        memo[id(obj)] = new_list
        for item in obj:
            new_list.append(_make_arrays_contiguous(item, memo))
        return new_list
    return obj


State = Literal["pause", "start", "step_once", "home", "reset_policy"]

MIN_REAL_ACTIONS_TO_RECORD = 3

CAMERA_KEYS = ["top_camera_image", "left_camera_image", "right_camera_image"]

_URDF_PATH = Path(__file__).resolve().parent / "models" / "station.urdf"

# Offset (metres) along the TCP +Z axis to the actual grasp site.
_GRASP_SITE_OFFSET_M = 0.1347

# TCP link names in the URDF (right first, then left — same order as the
# private repo default ``ik_tcp_link_names``).
_TCP_LINK_NAMES = ("right_tcp", "left_tcp")


# ---------------------------------------------------------------------------
# Policy wrapper (runs in main process)
# ---------------------------------------------------------------------------


class StartStopPlayPolicyWrapper:
    """Wraps a policy with start/stop/home control, driven by ViserUI via Portal IPC.

    The ViserUI subprocess sends commands (enter_state, set_task_command) to
    this wrapper, and this wrapper sends observation/action state back for
    display.
    """

    def __init__(
        self,
        policy: Policy,
        task_description: str = "Do something useful",
        policy_port: int = 8009,
        viser_host: str = "localhost",
        viser_port: int = 8010,
    ):
        self.policy = policy
        self._state: State = "pause"
        self._task_command: str = task_description
        self._hold_action: dict[str, Any] | None = None
        self._current_obs_action: dict[str, Any] | None = None
        self._reset_count: int = 0
        self._real_actions_this_episode: int = 0
        self._last_viser_update: float = 0.0
        self._viser_update_period: float = 1.0 / 10  # 10 Hz to ViserUI
        self._pending_future: Any = None
        self.lock = threading.Lock()

        # Portal server: receive commands from ViserUI
        self._server = portal.Server(policy_port)
        self._server.bind("enter_state", self._rpc_enter_state)
        self._server.bind("set_task_command", self._rpc_set_task)
        threading.Thread(target=self._server.start, daemon=True).start()
        time.sleep(0.1)

        # Portal client: send state updates to ViserUI
        self._client = portal.Client(f"{viser_host}:{viser_port}")
        print(
            f"[Policy] Portal IPC ready "
            f"(commands <-:{policy_port}, updates ->{viser_host}:{viser_port})"
        )

    # --- Portal RPC handlers ---

    def _rpc_enter_state(self, state: str) -> bool:
        with self.lock:
            print(f"[Policy] State -> {state}")
            self._state = state  # type: ignore[assignment]
            if state == "pause":
                self._hold_action = None
        return True

    def _rpc_set_task(self, task_command: str) -> bool:
        with self.lock:
            self._task_command = task_command
            # Changing the task resets the policy
            self._state = "reset_policy"
            print(f"[Policy] Task -> {task_command}")
        return True

    # --- Viser update ---

    def _send_update(self, observation: Observation):
        now = time.monotonic()
        if now - self._last_viser_update < self._viser_update_period:
            return
        self._last_viser_update = now

        # Drain the previous future (non-blocking check) to avoid queue buildup
        if self._pending_future is not None:
            try:
                self._pending_future.result()
            except Exception:
                pass
            self._pending_future = None

        try:
            msg = _make_arrays_contiguous({
                "observation": observation,
                "action": self._current_obs_action.get("action") if self._current_obs_action else None,
                "info": self._current_obs_action.get("info") if self._current_obs_action else None,
                "reset_count": self._reset_count,
            })
            self._pending_future = self._client.update_state(msg)
        except Exception:
            pass  # ViserUI not ready yet

    # --- Public API (matches Policy protocol) ---

    def get_action(self, observation: Observation) -> tuple[Action, Info]:
        with self.lock:
            # Internal reset (e.g. task changed)
            if self._state == "reset_policy":
                self.policy.reset()
                self._state = "pause"
                self._hold_action = None
                self._current_obs_action = None

            if self._state == "pause":
                # Check for physical button press (pause_toggle) to unpause
                if hasattr(self.policy, "poll_button_events"):
                    button_info = self.policy.poll_button_events(observation)
                    if button_info.get("pause_toggle"):
                        print("[Policy] Pause toggle detected (button), entering start state")
                        self._state = "start"

                self._send_update(observation)
                if self._hold_action is None:
                    self._hold_action = {
                        k: np.asarray(
                            observation.get(k, np.zeros(6 if "joint" in k else 1)),
                            dtype=np.float32,
                        )
                        for k in [
                            "left_joint_pos", "left_gripper_pos",
                            "right_joint_pos", "right_gripper_pos",
                        ]
                    }
                    self._hold_action["source"] = None
                return self._hold_action, {}

            if self._state in ("start", "step_once"):
                self._real_actions_this_episode += 1
                observation["annotation.task"] = self._task_command
                action, info = self.policy.get_action(observation)
                self._current_obs_action = {"action": action, "info": info}
                self._send_update(observation)
                if "source" not in action:
                    action["source"] = "policy"
                if self._state == "step_once":
                    self._state = "pause"
                elif info.get("pause_toggle"):
                    print("[Policy] Pause toggle detected (button), entering pause state")
                    self._state = "pause"
                    self._hold_action = None
                return action, info

            if self._state == "home":
                # Still call get_action so ViserUI gets a last update
                observation["annotation.task"] = self._task_command
                action, info = self.policy.get_action(observation)
                self._current_obs_action = {"action": action, "info": info}
                self._send_update(observation)
                self._reset_count += 1
                self.policy.reset()
                self._state = "pause"
                self._hold_action = None
                self._current_obs_action = None
                info["event"] = "home"
                if "source" not in action:
                    action["source"] = "policy"
                return action, info

            raise ValueError(f"Unknown state: {self._state}")

    def reset(self) -> Info:
        self._reset_count += 1
        self.policy.reset()
        self._state = "pause"
        self._hold_action = None
        self._current_obs_action = None
        discard = self._real_actions_this_episode < MIN_REAL_ACTIONS_TO_RECORD
        if discard:
            print(
                f"[Policy] Discarding episode ({self._real_actions_this_episode} actions "
                f"< {MIN_REAL_ACTIONS_TO_RECORD} minimum)"
            )
        else:
            print(f"[Policy] Episode had {self._real_actions_this_episode} actions")
        self._real_actions_this_episode = 0
        return {"task_name": self._task_command, "discard_episode": discard}


# ---------------------------------------------------------------------------
# ViserUI (runs in subprocess via portal.Process)
# ---------------------------------------------------------------------------


class ViserUI:
    """Web-based control UI with 3D robot, trajectory preview, and camera feeds.

    Opens a Viser web server on port 8080.  Open http://localhost:8080 in a
    browser to see the 3D robot model, predicted action trajectories, live
    camera feeds, and start/pause/home buttons.
    """

    def __init__(
        self,
        task_description: str = "Do something useful",
        camera_keys: list[str] | None = None,
        policy_host: str = "localhost",
        policy_port: int = 8009,
        viser_port: int = 8010,
        web_port: int = 8080,
    ):
        import pyroki as pk
        import trimesh
        import viser
        from scipy.spatial.transform import Rotation as R
        from viser.extras import ViserUrdf
        from yourdfpy import URDF

        self._R = R  # stash for use in update methods
        self._trimesh = trimesh

        self._camera_keys = camera_keys or list(CAMERA_KEYS)
        self._latest_state: dict[str, Any] | None = None
        self._state_lock = threading.Lock()
        self._timer = TrajectoryTimer()
        self._step_once_active = False

        # Portal server: receive state updates from policy
        self._portal_server = portal.Server(viser_port)
        self._portal_server.bind("update_state", self._rpc_update_state)
        threading.Thread(target=self._portal_server.start, daemon=True).start()
        time.sleep(0.1)

        # Portal client: send commands to policy
        self._portal_client = portal.Client(f"{policy_host}:{policy_port}")

        # Viser web server
        self.server = viser.ViserServer(host="0.0.0.0", port=web_port)

        # --- Control buttons ---
        with self.server.gui.add_folder("Policy Control"):
            start_btn = self.server.gui.add_button("Start")
            pause_btn = self.server.gui.add_button("Pause")
            home_btn = self.server.gui.add_button("Home")
            step_btn = self.server.gui.add_button("Step Once")

        @start_btn.on_click
        def _on_start(_e: Any) -> None:
            self._timer.start()
            self._send_cmd("start")

        @pause_btn.on_click
        def _on_pause(_e: Any) -> None:
            self._timer.pause()
            self._send_cmd("pause")

        @home_btn.on_click
        def _on_home(_e: Any) -> None:
            self._timer.home()
            self._send_cmd("home")

        @step_btn.on_click
        def _on_step(_e: Any) -> None:
            self._timer.step_once()
            self._step_once_active = True
            self._send_cmd("step_once")

        # --- Trajectory timer display ---
        with self.server.gui.add_folder("Trajectory Timer"):
            self._timer_display = self.server.gui.add_text(
                label="Time", initial_value="0:00 (stopped)"
            )

        # --- Task input ---
        self._task_input = self.server.gui.add_text(
            label="Task", initial_value=task_description
        )

        @self._task_input.on_update  # type: ignore[attr-defined]
        def _on_task_update(_e: Any) -> None:
            try:
                self._portal_client.set_task_command(self._task_input.value).result()
            except Exception:
                pass

        # Send initial task
        try:
            self._portal_client.set_task_command(task_description).result()
        except Exception:
            pass

        # --- Camera images ---
        self._gui_images: dict[str, Any] = {}
        with self.server.gui.add_folder("Cameras"):
            for key in self._camera_keys:
                label = key.replace("_camera_image", "").replace("_image", "")
                self._gui_images[key] = self.server.gui.add_image(
                    label=label,
                    image=np.zeros((240, 320, 3), dtype=np.uint8),
                )

        # --- Visualization toggles ---
        with self.server.gui.add_folder("Visualization"):
            self._ee_as_poses_cb = self.server.gui.add_checkbox(
                "EE as poses (frames)", False
            )

        # --- 3D Robot (URDF) ---
        urdf_path = _URDF_PATH

        # ViserUrdf for 3D display
        self._urdf_vis = ViserUrdf(
            self.server, urdf_or_path=urdf_path, load_meshes=True
        )
        self._urdf_joint_names: list[str] = list(
            self._urdf_vis.get_actuated_joint_limits().keys()
        )

        # PyRoki robot for forward kinematics
        urdf_model = URDF.load(str(urdf_path))
        self._ik_robot = pk.Robot.from_urdf(urdf_model)
        self._link_names: list[str] = list(self._ik_robot.links.names)
        self._ik_actuated_names: list[str] = list(self._ik_robot.joints.actuated_names)
        self._ik_idx_right: list[int] = [
            i for i, n in enumerate(self._ik_actuated_names) if n.startswith("right_joint")
        ][:6]
        self._ik_idx_left: list[int] = [
            i for i, n in enumerate(self._ik_actuated_names) if n.startswith("left_joint")
        ][:6]

        # Cache TCP link indices
        self._tcp_link_idx: dict[str, int] = {}
        for name in _TCP_LINK_NAMES:
            if name in self._link_names:
                self._tcp_link_idx[name] = self._link_names.index(name)

        # EE visualization handles (lazily populated)
        self._ee_points_left: list[Any] = []
        self._ee_points_right: list[Any] = []
        self._ee_frames_left: list[Any] = []
        self._ee_frames_right: list[Any] = []

        # Add a ground grid
        self.server.scene.add_grid(
            "/grid", width=2, height=2, position=(0.0, 0.0, 0.0)
        )

        print(f"[ViserUI] Web UI at http://localhost:{web_port}")

    # --- Portal RPC ---

    def _rpc_update_state(self, state: dict[str, Any]) -> bool:
        with self._state_lock:
            self._latest_state = state
        return True

    def _send_cmd(self, state: str):
        try:
            self._portal_client.enter_state(state).result()
        except Exception as e:
            print(f"[ViserUI] Command failed: {e}")

    def _update_timer_display(self):
        time_str = self._timer.get_display_string()
        state = self._timer.state
        if state == "running":
            self._timer_display.value = f"{time_str} (running)"
        elif state == "paused":
            self._timer_display.value = f"{time_str} (paused)"
        else:
            self._timer_display.value = f"{time_str} (stopped)"

    # --- UI update (called in a loop) ---

    def update_ui(self):
        with self._state_lock:
            state = self._latest_state
            self._latest_state = None

        if state is None:
            self._update_timer_display()
            return

        obs = state.get("observation", {})
        info = state.get("info") or {}

        # 1. Update camera images
        self._update_images(obs)

        # 2. Update URDF robot model from proprioception
        self._update_urdf_from_proprio(obs)

        # 3. Visualize predicted trajectory from action_chunk
        action_chunk = info.get("action_chunk")
        if action_chunk:
            try:
                self._update_prediction(action_chunk)
            except Exception as e:
                print(f"[ViserUI] Trajectory visualization error: {e}")

        # 4. Auto-pause timer after step_once completes
        if self._step_once_active:
            self._timer.pause()
            self._step_once_active = False

        self._update_timer_display()

    # --- Camera image update ---

    def _update_images(self, obs: dict[str, Any]):
        for key in self._camera_keys:
            if key in obs and key in self._gui_images:
                img = np.asarray(obs[key], dtype=np.uint8)
                if img.ndim == 2:
                    img = np.stack([img] * 3, axis=-1)
                self._gui_images[key].image = img

    # --- URDF robot update from proprioception ---

    def _update_urdf_from_proprio(self, obs: dict[str, Any]):
        """Set URDF joint positions from the current observation."""
        left_joints = obs.get("left_joint_pos")
        right_joints = obs.get("right_joint_pos")
        if left_joints is None or right_joints is None:
            return
        left_joints = np.asarray(left_joints, dtype=float)
        right_joints = np.asarray(right_joints, dtype=float)

        name_to_val: dict[str, float] = {}
        for i in range(min(6, len(left_joints))):
            name_to_val[f"left_joint{i + 1}"] = float(left_joints[i])
        for i in range(min(6, len(right_joints))):
            name_to_val[f"right_joint{i + 1}"] = float(right_joints[i])
        cfg = np.array(
            [name_to_val.get(n, 0.0) for n in self._ik_actuated_names], dtype=float
        )
        self._urdf_vis.update_cfg(cfg)

    # --- Trajectory prediction visualization ---

    @staticmethod
    def _point_color(t_norm: float) -> tuple[float, float, float, float]:
        """Red-to-blue gradient based on time index."""
        return (float(t_norm), 0.2, float(1.0 - t_norm), 1.0)

    def _update_prediction(self, action_chunk: dict[str, Any]) -> None:
        """Visualize predicted EE trajectory from action_chunk joint sequences.

        Uses pyroki forward kinematics to compute TCP poses, then renders
        them as colored spheres or coordinate frames in the 3D scene.
        """
        if self._ik_robot is None:
            return

        # Extract sequences (support multiple key naming conventions)
        def _get_seq(key_candidates: list[str]) -> np.ndarray | None:
            for k in key_candidates:
                if k in action_chunk:
                    seq = np.asarray(action_chunk[k], dtype=float)
                    if seq.ndim == 3:
                        seq = seq[0]
                    return seq
            return None

        left_seq = _get_seq(["left_joint_pos", "joint_pos_action_left", "left_arm_joints"])
        right_seq = _get_seq(["right_joint_pos", "joint_pos_action_right", "right_arm_joints"])
        steps = max(
            (left_seq.shape[0] if left_seq is not None else 0),
            (right_seq.shape[0] if right_seq is not None else 0),
        )
        if steps <= 0:
            return
        idxs = np.linspace(0, steps - 1, num=int(steps), dtype=int)

        # Compute TCP poses via pyroki FK
        left_positions: list[np.ndarray] = []
        right_positions: list[np.ndarray] = []
        left_wxyzs: list[np.ndarray] = []
        right_wxyzs: list[np.ndarray] = []

        for s in idxs:
            # Build actuated joint configuration vector
            cfg = np.zeros((len(self._ik_actuated_names),), dtype=float)
            if right_seq is not None and len(self._ik_idx_right) >= 6:
                cfg[np.array(self._ik_idx_right[: right_seq.shape[-1]])] = right_seq[s]
            if left_seq is not None and len(self._ik_idx_left) >= 6:
                cfg[np.array(self._ik_idx_left[: left_seq.shape[-1]])] = left_seq[s]

            # pyroki FK: returns (1, num_links, 7) as [wxyz, xyz]
            Ts_world_links = np.asarray(
                self._ik_robot.forward_kinematics(cfg[None, ...])
            )

            for side, acc in (
                ("right", (right_positions, right_wxyzs)),
                ("left", (left_positions, left_wxyzs)),
            ):
                name = _TCP_LINK_NAMES[0] if side == "right" else _TCP_LINK_NAMES[1]
                link_idx = self._tcp_link_idx.get(name)
                if link_idx is None:
                    continue
                wxyz_xyz = Ts_world_links[0, link_idx]
                p_tcp = np.asarray(wxyz_xyz[4:7], dtype=float)
                q_tcp = np.asarray(wxyz_xyz[0:4], dtype=float)
                # Apply grasp-site offset along local +Z
                R_tcp = self._R.from_quat(q_tcp, scalar_first=True).as_matrix()
                p_grasp = p_tcp + R_tcp[:, 2] * _GRASP_SITE_OFFSET_M
                acc[0].append(p_grasp)
                acc[1].append(q_tcp)

        npts = len(idxs)
        use_frames = bool(self._ee_as_poses_cb.value)

        if use_frames:
            self._ensure_ee_frames(npts)
            # Hide points
            for h in self._ee_points_left:
                h.visible = False
            for h in self._ee_points_right:
                h.visible = False
            # Update frames
            for i in range(npts):
                if i < len(left_positions):
                    p, q = left_positions[i], left_wxyzs[i]
                    hl = self._ee_frames_left[i]
                    hl.position = (float(p[0]), float(p[1]), float(p[2]))
                    hl.wxyz = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
                    hl.visible = True
                if i < len(right_positions):
                    p, q = right_positions[i], right_wxyzs[i]
                    hr = self._ee_frames_right[i]
                    hr.position = (float(p[0]), float(p[1]), float(p[2]))
                    hr.wxyz = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
                    hr.visible = True
            # Hide extras
            for i in range(npts, len(self._ee_frames_left)):
                self._ee_frames_left[i].visible = False
            for i in range(npts, len(self._ee_frames_right)):
                self._ee_frames_right[i].visible = False
        else:
            self._ensure_ee_points(npts)
            # Hide frames
            for h in self._ee_frames_left:
                h.visible = False
            for h in self._ee_frames_right:
                h.visible = False
            # Update points
            for i in range(npts):
                if i < len(left_positions):
                    p = left_positions[i]
                    hl = self._ee_points_left[i]
                    hl.position = (float(p[0]), float(p[1]), float(p[2]))
                    hl.visible = True
                if i < len(right_positions):
                    p = right_positions[i]
                    hr = self._ee_points_right[i]
                    hr.position = (float(p[0]), float(p[1]), float(p[2]))
                    hr.visible = True

        # Hide extras beyond npts
        for i in range(npts, len(self._ee_points_left)):
            self._ee_points_left[i].visible = False
        for i in range(npts, len(self._ee_points_right)):
            self._ee_points_right[i].visible = False

    # --- Lazy creation of EE visualization handles ---

    def _ensure_ee_frames(self, count: int) -> None:
        while len(self._ee_frames_left) < count:
            i = len(self._ee_frames_left)
            t_norm = i / max(count - 1, 1)
            rgb = self._point_color(t_norm)[:3]
            rgb255 = (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))
            h = self.server.scene.add_frame(
                f"/pred/left_pose_{i}",
                show_axes=True,
                axes_length=0.02,
                axes_radius=0.002,
                origin_radius=0.005,
                origin_color=rgb255,
                visible=False,
            )
            self._ee_frames_left.append(h)
        while len(self._ee_frames_right) < count:
            i = len(self._ee_frames_right)
            t_norm = i / max(count - 1, 1)
            rgb = self._point_color(t_norm)[:3]
            rgb255 = (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))
            h = self.server.scene.add_frame(
                f"/pred/right_pose_{i}",
                show_axes=True,
                axes_length=0.02,
                axes_radius=0.002,
                origin_radius=0.005,
                origin_color=rgb255,
                visible=False,
            )
            self._ee_frames_right.append(h)

    def _ensure_ee_points(self, count: int) -> None:
        radius = 0.008
        while len(self._ee_points_left) < count:
            i = len(self._ee_points_left)
            t_norm = i / max(count - 1, 1)
            rgba = self._point_color(t_norm)
            sphere = self._trimesh.creation.icosphere(subdivisions=2, radius=radius)
            vc = (np.array(rgba) * 255).astype(np.uint8)
            vis = getattr(sphere, "visual", None)
            if vis is not None and hasattr(vis, "vertex_colors"):
                vis.vertex_colors = np.tile(vc, (sphere.vertices.shape[0], 1))
            h = self.server.scene.add_mesh_trimesh(
                f"/pred/left_pt_{i}", sphere, position=(0.0, 0.0, 0.0)
            )
            self._ee_points_left.append(h)
        while len(self._ee_points_right) < count:
            i = len(self._ee_points_right)
            t_norm = i / max(count - 1, 1)
            rgba = self._point_color(t_norm)
            sphere = self._trimesh.creation.icosphere(subdivisions=2, radius=radius)
            vc = (np.array(rgba) * 255).astype(np.uint8)
            vis = getattr(sphere, "visual", None)
            if vis is not None and hasattr(vis, "vertex_colors"):
                vis.vertex_colors = np.tile(vc, (sphere.vertices.shape[0], 1))
            h = self.server.scene.add_mesh_trimesh(
                f"/pred/right_pt_{i}", sphere, position=(0.0, 0.0, 0.0)
            )
            self._ee_points_right.append(h)


# ---------------------------------------------------------------------------
# Subprocess entry point
# ---------------------------------------------------------------------------


def run_viser_subprocess(
    task_description: str = "Do something useful",
    camera_keys: list[str] | None = None,
    policy_host: str = "localhost",
    policy_port: int = 8009,
    viser_port: int = 8010,
    web_port: int = 8080,
):
    """Entry point for ViserUI subprocess (launched via portal.Process)."""
    ui = ViserUI(
        task_description=task_description,
        camera_keys=camera_keys,
        policy_host=policy_host,
        policy_port=policy_port,
        viser_port=viser_port,
        web_port=web_port,
    )
    while True:
        ui.update_ui()
        time.sleep(1 / 20)  # 20 Hz
