"""GR00T policy adapter for yam-control.

Provides :class:`Gr00tPolicy` which loads an Isaac-GR00T model and
implements the :class:`Policy` interface.  Use it directly (in-process)
or wrap it in :class:`PortalPolicy` for subprocess inference with
shared-memory IPC.

Subprocess usage (recommended)::

    from yam_control.portal_policy import PortalPolicy
    from yam_control.gr00t_policy import make_gr00t_factory

    policy = PortalPolicy(
        policy_factory=make_gr00t_factory(
            model_path="/path/to/checkpoint",
            embodiment_tag="new_embodiment",
        ),
    )

In-process usage::

    from yam_control.gr00t_policy import Gr00tPolicy

    policy = Gr00tPolicy(model_path="/path/to/checkpoint")

Requires Isaac-GR00T to be installed (``pip install -e .`` in the
Isaac-GR00T repo).
"""

from typing import Any, Callable

import numpy as np

from yam_control.policy import Action, Info, Observation, Policy

# Default key mappings between YAM env keys and GR00T model modality keys.
# Must match the modality_keys in the modality config used during finetuning.

DEFAULT_VIDEO_KEY_MAP = {
    "top_camera_image": "top_camera",
    "left_camera_image": "left_camera",
    "right_camera_image": "right_camera",
}

DEFAULT_STATE_KEY_MAP = {
    "left_joint_pos": "left_arm",
    "left_gripper_pos": "left_gripper",
    "right_joint_pos": "right_arm",
    "right_gripper_pos": "right_gripper",
}

DEFAULT_ACTION_KEY_MAP = {
    "left_arm": "left_joint_pos",
    "left_gripper": "left_gripper_pos",
    "right_arm": "right_joint_pos",
    "right_gripper": "right_gripper_pos",
}

DEFAULT_LANGUAGE_KEY = "annotation.human.action.task_description"


class Gr00tPolicy(Policy):
    """Loads an Isaac-GR00T model and runs inference.

    This class handles the observation/action format conversion between
    the yam-control flat dict format and the nested
    ``{video: {}, state: {}, language: {}}`` format with ``(B, T, ...)``
    dimensions that the GR00T policy API expects.

    Parameters
    ----------
    model_path:
        Path to the GR00T model checkpoint directory.
    embodiment_tag:
        Embodiment tag string (converted to ``EmbodimentTag`` enum).
    device:
        CUDA device for inference (e.g. ``"cuda:0"``).
    video_key_map / state_key_map / action_key_map / language_key:
        Override the default key mappings if your finetuned model uses
        different modality keys.
    """

    def __init__(
        self,
        model_path: str,
        embodiment_tag: str = "new_embodiment",
        device: str = "cuda:0",
        video_key_map: dict[str, str] | None = None,
        state_key_map: dict[str, str] | None = None,
        action_key_map: dict[str, str] | None = None,
        language_key: str | None = None,
    ) -> None:
        from gr00t.data.embodiment_tags import EmbodimentTag
        from gr00t.policy.gr00t_policy import Gr00tPolicy as _Gr00tPolicy

        self.video_key_map = video_key_map or DEFAULT_VIDEO_KEY_MAP
        self.state_key_map = state_key_map or DEFAULT_STATE_KEY_MAP
        self.action_key_map = action_key_map or DEFAULT_ACTION_KEY_MAP
        self.language_key = language_key or DEFAULT_LANGUAGE_KEY

        print(f"[Gr00tPolicy] Loading model from {model_path}...")
        self.gr00t_policy = _Gr00tPolicy(
            embodiment_tag=EmbodimentTag(embodiment_tag),
            model_path=model_path,
            device=device,
            strict=False,
        )
        print("[Gr00tPolicy] Model loaded")

    def _build_observation(
        self, env_obs: dict[str, Any], task_description: str
    ) -> dict[str, Any]:
        obs: dict[str, Any] = {"video": {}, "state": {}, "language": {}}
        for env_key, model_key in self.video_key_map.items():
            if env_key in env_obs:
                obs["video"][model_key] = np.asarray(
                    env_obs[env_key], dtype=np.uint8
                )[np.newaxis, np.newaxis, ...]
        for env_key, model_key in self.state_key_map.items():
            if env_key in env_obs:
                obs["state"][model_key] = np.asarray(
                    env_obs[env_key], dtype=np.float32
                )[np.newaxis, np.newaxis, ...]
        obs["language"][self.language_key] = [[task_description]]
        return obs

    def _parse_action(self, model_action: dict[str, Any]) -> dict[str, np.ndarray]:
        chunk: dict[str, np.ndarray] = {}
        for model_key, env_key in self.action_key_map.items():
            if model_key in model_action:
                arr = np.asarray(model_action[model_key], dtype=np.float32)
                if arr.ndim == 3:
                    arr = arr[0]
                chunk[env_key] = arr
        return chunk

    def get_action(self, observation: Observation) -> tuple[Action, Info]:
        task = observation.get("annotation.task", "Do something useful")
        model_action, model_info = self.gr00t_policy.get_action(
            self._build_observation(observation, task)
        )
        action_chunk = self._parse_action(model_action)
        first_action = {k: v[0] for k, v in action_chunk.items()}
        info: Info = {"action_chunk": action_chunk}
        if model_info:
            info.update(model_info)
        return first_action, info

    def reset(self) -> Info | None:
        self.gr00t_policy.reset()
        return None


# ---------------------------------------------------------------------------
# Factory for PortalPolicy
# ---------------------------------------------------------------------------


def make_gr00t_factory(
    model_path: str,
    embodiment_tag: str = "new_embodiment",
    device: str = "cuda:0",
    video_key_map: dict[str, str] | None = None,
    state_key_map: dict[str, str] | None = None,
    action_key_map: dict[str, str] | None = None,
    language_key: str | None = None,
) -> Callable[[], Policy]:
    """Create a factory callable for use with :class:`PortalPolicy`.

    The returned callable is serialized and executed *inside* the
    subprocess, so CUDA / model loading only happens there.

    Example::

        from yam_control.portal_policy import PortalPolicy
        from yam_control.gr00t_policy import make_gr00t_factory

        policy = PortalPolicy(
            policy_factory=make_gr00t_factory(
                model_path="/path/to/checkpoint",
                embodiment_tag="new_embodiment",
                device="cuda:0",
            ),
        )
    """

    def _factory() -> Policy:
        return Gr00tPolicy(
            model_path=model_path,
            embodiment_tag=embodiment_tag,
            device=device,
            video_key_map=video_key_map,
            state_key_map=state_key_map,
            action_key_map=action_key_map,
            language_key=language_key,
        )

    return _factory
