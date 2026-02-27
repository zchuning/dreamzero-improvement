"""Key remapping utilities for observation/action mapping between environment and policy formats.

Provides two mapping modes:

1. **In-process Gr00tPolicy** -- ``map_observation()`` / ``map_action()`` convert between
   env format and the ``(images, proprio)`` tuple expected by a locally-loaded model.

2. **Remote Gr00t server** -- ``map_observation_for_gr00t_server()`` converts env
   observations into the flat ``video.*`` / ``state.*`` / ``annotation.task`` format
   expected by ``run_gr00t_server.py`` over ZMQ.
"""

import time
from typing import Any, Dict, List, Literal

import numpy as np
from PIL import Image

from yam_control.action_utils import hold_action_from_proprio  # noqa: F401
from yam_control.embodiment import EmbodimentTag

Resolution = Literal["240p", "480p"]

# =============================================================================
# Key maps for observation/action mapping
# =============================================================================

PROPRIO_KEY_MAP_xdof_oss = {
    "left_joint_pos": "left_joint_pos",
    "left_gripper_pos": "left_gripper_pos",
    "right_joint_pos": "right_joint_pos",
    "right_gripper_pos": "right_gripper_pos",
}

ACTION_KEY_MAP_xdof_oss = {
    "left_joint_pos": "left_joint_pos",
    "left_gripper_pos": "left_gripper_pos",
    "right_joint_pos": "right_joint_pos",
    "right_gripper_pos": "right_gripper_pos",
}

CAMERA_KEY_MAP_xdof_oss = {
    "top_camera_image": "top",
    "left_camera_image": "left",
    "right_camera_image": "right",
}

PROPRIO_KEY_MAP_xdof = {
    "left_joint_pos": "joint_pos_obs_left",
    "left_gripper_pos": "gripper_pos_obs_left",
    "right_joint_pos": "joint_pos_obs_right",
    "right_gripper_pos": "gripper_pos_obs_right",
}

ACTION_KEY_MAP_xdof = {
    "joint_pos_action_left": "left_joint_pos",
    "gripper_pos_action_left": "left_gripper_pos",
    "joint_pos_action_right": "right_joint_pos",
    "gripper_pos_action_right": "right_gripper_pos",
}

CAMERA_KEY_MAP_xdof_240 = {
    "top_camera_image": "top_camera-images-rgb_320_240",
    "left_camera_image": "left_camera-images-rgb_320_240",
    "right_camera_image": "right_camera-images-rgb_320_240",
}

CAMERA_KEY_MAP_xdof_480 = {
    "top_camera_image": "top_camera-images-rgb",
    "left_camera_image": "left_camera-images-rgb",
    "right_camera_image": "right_camera-images-rgb",
}


# =============================================================================
# Observation/action mapping functions
# =============================================================================


def map_observation(
    observation: Dict[str, Any],
    embodiment_tag: EmbodimentTag,
    resolutions: List[Resolution] | Resolution,
):
    """Map environment observation to (images, proprio) format for policy."""
    if isinstance(resolutions, str):
        resolutions = [resolutions]

    proprio = {}
    if embodiment_tag == EmbodimentTag.XDOF_OSS_DATA:
        PROPRIO_KEY_MAP = PROPRIO_KEY_MAP_xdof_oss
        CAMERA_KEY_MAP = CAMERA_KEY_MAP_xdof_oss
    elif embodiment_tag == EmbodimentTag.XDOF:
        PROPRIO_KEY_MAP = PROPRIO_KEY_MAP_xdof
        CAMERA_KEY_MAP = CAMERA_KEY_MAP_xdof_240
    else:
        raise ValueError(f"Embodiment tag {embodiment_tag} not supported")

    for key, value in PROPRIO_KEY_MAP.items():
        proprio[value] = observation[key]
    images = {}
    for key, value in CAMERA_KEY_MAP.items():
        images[value] = Image.fromarray(observation[key]).resize((320, 240))
    if "480p" in resolutions and embodiment_tag == EmbodimentTag.XDOF:
        for key, value in CAMERA_KEY_MAP_xdof_480.items():
            images[value] = Image.fromarray(observation[key])
    return images, proprio


def map_action(action: Dict[str, Any], embodiment_tag: EmbodimentTag):
    """Map policy action to environment action format."""
    action_dict = {}
    if embodiment_tag == EmbodimentTag.XDOF_OSS_DATA:
        ACTION_KEY_MAP = ACTION_KEY_MAP_xdof_oss
    elif embodiment_tag == EmbodimentTag.XDOF:
        ACTION_KEY_MAP = ACTION_KEY_MAP_xdof
    else:
        raise ValueError(f"Embodiment tag {embodiment_tag} not supported")
    for key, value in ACTION_KEY_MAP.items():
        action_dict[value] = action[key]
    return action_dict


# =============================================================================
# Packing/unpacking utilities for Gr00t policy format
# =============================================================================

ENV_TO_GROOT_ACTION_KEY_MAP_xdof = {v: k for k, v in ACTION_KEY_MAP_xdof.items()}


def unpack_action_from_groot(
    action: Dict[str, Any],
    embodiment_tag: EmbodimentTag = EmbodimentTag.XDOF,
) -> Dict[str, np.ndarray]:
    """Convert Gr00tPolicy action output to env action format.

    Gr00tPolicy returns actions with shape (B, T, D) and keys like
    'joint_pos_action_left'. This function converts to env format
    with shape (D,) and keys like 'left_joint_pos'.
    """
    if embodiment_tag != EmbodimentTag.XDOF:
        raise ValueError(f"Embodiment tag {embodiment_tag} not supported for unpacking")

    env_action = {}
    for groot_key, env_key in ACTION_KEY_MAP_xdof.items():
        if groot_key in action:
            val = action[groot_key]
            if isinstance(val, np.ndarray):
                if val.ndim == 3:  # (B, T, D)
                    val = val[0, 0]
                elif val.ndim == 2:  # (T, D)
                    val = val[0]
            env_action[env_key] = val

    return env_action


def unpack_action_chunk_from_groot(
    action: Dict[str, Any],
    embodiment_tag: EmbodimentTag = EmbodimentTag.XDOF,
) -> Dict[str, np.ndarray]:
    """Convert Gr00tPolicy action chunk to env format, keeping time dimension."""
    if embodiment_tag != EmbodimentTag.XDOF:
        raise ValueError(f"Embodiment tag {embodiment_tag} not supported for unpacking")

    env_action = {}
    for groot_key, env_key in ACTION_KEY_MAP_xdof.items():
        if groot_key in action:
            val = action[groot_key]
            if isinstance(val, np.ndarray):
                if val.ndim == 3:  # (B, T, D)
                    val = val[0]
            env_action[env_key] = val

    return env_action


def pack_actions_for_groot(
    actions: list[Dict[str, Any]],
    embodiment_tag: EmbodimentTag = EmbodimentTag.XDOF,
    action_horizon: int | None = None,
) -> Dict[str, np.ndarray]:
    """Convert list of env-format action dicts to Gr00t format for RTC inpainting."""
    if embodiment_tag != EmbodimentTag.XDOF:
        raise ValueError(f"Embodiment tag {embodiment_tag} not supported for packing")

    if len(actions) == 0:
        if action_horizon is not None and action_horizon > 0:
            groot_action = {}
            action_dims = {
                "gripper_pos_action_left": 1,
                "gripper_pos_action_right": 1,
                "joint_pos_action_left": 6,
                "joint_pos_action_right": 6,
            }
            for groot_key, dim in action_dims.items():
                groot_action[groot_key] = np.zeros((1, action_horizon, dim), dtype=np.float32)
            return groot_action
        return {}

    groot_action = {}
    for env_key, groot_key in ENV_TO_GROOT_ACTION_KEY_MAP_xdof.items():
        values = []
        for action in actions:
            if env_key in action:
                val = np.asarray(action[env_key], dtype=np.float32)
                if val.ndim == 0:
                    val = val.reshape(1)
                values.append(val)

        if values:
            stacked = np.stack(values, axis=0)

            if action_horizon is not None and stacked.shape[0] < action_horizon:
                pad_count = action_horizon - stacked.shape[0]
                last_action = stacked[-1:, :]
                padding = np.repeat(last_action, pad_count, axis=0)
                stacked = np.concatenate([stacked, padding], axis=0)

            groot_action[groot_key] = stacked[None]

    return groot_action


# =============================================================================
# Gr00t Policy Server observation/action mapping (ZMQ server format)
# =============================================================================

GR00T_CAMERA_KEY_MAP_xdof_240 = {
    "top_camera_image": "video.top_camera-images-rgb_320_240",
    "left_camera_image": "video.left_camera-images-rgb_320_240",
    "right_camera_image": "video.right_camera-images-rgb_320_240",
}

GR00T_STATE_KEY_MAP_xdof = {
    "left_joint_pos": "state.joint_pos_obs_left",
    "left_gripper_pos": "state.gripper_pos_obs_left",
    "right_joint_pos": "state.joint_pos_obs_right",
    "right_gripper_pos": "state.gripper_pos_obs_right",
}


def map_observation_for_gr00t_server(
    observation: Dict[str, Any],
    embodiment_tag: EmbodimentTag,
    resolutions: List[Resolution] | Resolution,
) -> Dict[str, Any]:
    """Map environment observation to the format expected by run_gr00t_server.py.

    The server expects:
    - Video keys: ``video.{camera_name}`` with list of JPEG-encoded bytes
    - State keys: ``state.{state_name}`` with shape ``(T, D)`` where ``T=1``
    - Language: ``annotation.task`` with tuple of strings
    """
    import cv2

    if isinstance(resolutions, str):
        resolutions = [resolutions]

    if embodiment_tag != EmbodimentTag.XDOF:
        raise ValueError(f"Embodiment tag {embodiment_tag} not supported for gr00t server mapping")

    if "240p" not in resolutions:
        raise ValueError(f"Resolution {resolutions} not supported for gr00t server mapping yet")

    gr00t_obs: Dict[str, Any] = {}

    for env_key, gr00t_key in GR00T_CAMERA_KEY_MAP_xdof_240.items():
        if env_key in observation:
            img = observation[env_key]
            if img.shape[:2] != (240, 320):
                img = np.array(Image.fromarray(img).resize((320, 240)))
            _, encoded = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            gr00t_obs[gr00t_key] = [encoded.tobytes()]

    for env_key, gr00t_key in GR00T_STATE_KEY_MAP_xdof.items():
        if env_key in observation:
            state = np.asarray(observation[env_key], dtype=np.float32)
            if state.ndim == 1:
                state = state.reshape(1, -1)
            gr00t_obs[gr00t_key] = state

    if "annotation.task" in observation:
        gr00t_obs["annotation.task"] = (observation["annotation.task"],)
    else:
        gr00t_obs["annotation.task"] = ("",)

    return gr00t_obs


class Gr00tPolicyClientWrapper:
    """Wraps a ZMQ PolicyClient with env-to-server observation/action mapping.

    Transforms YAM environment observations into the flat ``video.*`` / ``state.*``
    format expected by ``run_gr00t_server.py``, and maps the returned
    ``action.*`` keys back to environment format.
    """

    def __init__(
        self,
        policy_client,
        embodiment_tag: EmbodimentTag,
        resolutions: List[Resolution] | None = None,
    ):
        self.policy_client = policy_client
        self.embodiment_tag = embodiment_tag
        self.resolutions = resolutions or ["240p"]

    def get_action(
        self, observation: Dict[str, Any], options: Dict[str, Any] | None = None
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        gr00t_obs = map_observation_for_gr00t_server(
            observation, self.embodiment_tag, self.resolutions
        )
        t0 = time.perf_counter()
        action, info = self.policy_client.get_action(gr00t_obs, options)
        if "forward_time_ms" not in info:
            info["forward_time_ms"] = (time.perf_counter() - t0) * 1000

        env_action: Dict[str, Any] = {}
        for key, value in action.items():
            if key.startswith("action."):
                action_key = key.replace("action.", "")
                if action_key in ACTION_KEY_MAP_xdof:
                    env_action[ACTION_KEY_MAP_xdof[action_key]] = value
                else:
                    env_action[action_key] = value
            else:
                env_action[key] = value

        if "action_chunk" not in info:
            info["action_chunk"] = env_action

        return env_action, info

    def reset(self, options: Dict[str, Any] | None = None) -> Dict[str, Any]:
        return self.policy_client.reset(options)

    def ping(self) -> bool:
        return self.policy_client.ping()
