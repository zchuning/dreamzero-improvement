import logging
import pickle
import traceback


import numpy as np
import torch
import torch.distributed as dist
from tianshou.data import Batch

from groot.vla.model.n1_5.sim_policy import GrootSimPolicy
from .utils import FrameBuffer, VideoSaver

logger = logging.getLogger(__name__)

# Camera key mapping: roboarena (0-indexed) -> AR_droid (1-indexed)
_IMAGE_KEY_MAPPING = {
    "observation/exterior_image_0_left": "video.exterior_image_1_left",
    "observation/exterior_image_1_left": "video.exterior_image_2_left",
    "observation/wrist_image_left": "video.wrist_image_left",
}

_DROID_CAMERA_KEYS = list(_IMAGE_KEY_MAPPING.values())

class ARDroidRoboarenaPolicy:
    """Wrapper policy that implements roboarena.policy.BasePolicy interface for AR_droid.

    Handles:
    - Observation format conversion (roboarena -> AR_droid format)
    - Frame accumulation across calls (roboarena sends single frames, AR_droid expects multi-frame video)
    - Action format conversion (AR_droid dict -> roboarena array format)
    - Distributed inference coordination
    """

    FRAMES_PER_CHUNK = 4

    def __init__(
        self,
        groot_policy: GrootSimPolicy,
        signal_group: dist.ProcessGroup,
        output_dir: str | None = None,
    ) -> None:
        self._policy = groot_policy
        self._signal_group = signal_group

        self._frame_buffer = FrameBuffer(_DROID_CAMERA_KEYS, self.FRAMES_PER_CHUNK)
        self._call_count = 0
        self._is_first_call = True

        # Session tracking
        self._current_session_id: str | None = None

        # Video saving
        self._output_dir = output_dir
        self._video_saver: VideoSaver | None = None
        if output_dir:
            self._video_saver = VideoSaver(
                output_dir, groot_policy.trained_model.action_head
            )

    def _convert_observation(self, obs: dict) -> dict:
        """Convert roboarena observation format to AR_droid format.

        Roboarena format:
            - observation/exterior_image_0_left: (H, W, 3) single frame
            - observation/exterior_image_1_left: (H, W, 3) single frame
            - observation/wrist_image_left: (H, W, 3) single frame
            - observation/joint_position: (7,)
            - observation/gripper_position: (1,)
            - prompt: str

        AR_droid format:
            - video.exterior_image_1_left: (T, H, W, 3) multi-frame
            - video.exterior_image_2_left: (T, H, W, 3) multi-frame
            - video.wrist_image_left: (T, H, W, 3) multi-frame
            - state.joint_position: (1, 7)
            - state.gripper_position: (1, 1)
            - annotation.language.action_text: str
        """
        converted = {}

        # Accumulate frames
        for roboarena_key, droid_key in _IMAGE_KEY_MAPPING.items():
            if roboarena_key in obs:
                data = obs[roboarena_key]
                if isinstance(data, np.ndarray):
                    self._frame_buffer.append(droid_key, data)

        # First call: 1 frame; subsequent: FRAMES_PER_CHUNK frames
        num_frames = 1 if self._is_first_call else self.FRAMES_PER_CHUNK

        for droid_key in _DROID_CAMERA_KEYS:
            if self._frame_buffer.has_frames(droid_key):
                converted[droid_key] = self._frame_buffer.get_frames(droid_key, num_frames)

        # State
        if "observation/joint_position" in obs:
            joint_pos = obs["observation/joint_position"]
            if joint_pos.ndim == 1:
                joint_pos = joint_pos.reshape(1, -1)
            converted["state.joint_position"] = joint_pos.astype(np.float64)
        else:
            converted["state.joint_position"] = np.zeros((1, 7), dtype=np.float64)

        if "observation/gripper_position" in obs:
            gripper_pos = obs["observation/gripper_position"]
            if gripper_pos.ndim == 1:
                gripper_pos = gripper_pos.reshape(1, -1)
            converted["state.gripper_position"] = gripper_pos.astype(np.float64)
        else:
            converted["state.gripper_position"] = np.zeros((1, 1), dtype=np.float64)

        # Prompt
        converted["annotation.language.action_text"] = obs.get("prompt", "")

        return converted

    def _convert_action(self, action_dict: dict) -> np.ndarray:
        """Convert AR_droid action dict to roboarena (N, 8) array."""
        joint_action = None
        gripper_action = None

        for key, value in action_dict.items():
            if "joint_position" in key:
                joint_action = value
            elif "gripper_position" in key or "gripper" in key:
                gripper_action = value

        if joint_action is None:
            return np.zeros((1, 8), dtype=np.float32)

        if isinstance(joint_action, torch.Tensor):
            joint_action = joint_action.cpu().numpy()
        if joint_action.ndim == 1:
            joint_action = joint_action.reshape(1, -1)

        N = joint_action.shape[0]

        if gripper_action is not None:
            if isinstance(gripper_action, torch.Tensor):
                gripper_action = gripper_action.cpu().numpy()
            if gripper_action.ndim == 1:
                gripper_action = gripper_action.reshape(-1, 1)
            elif gripper_action.ndim == 0:
                gripper_action = gripper_action.reshape(1, 1)
        else:
            gripper_action = np.zeros((N, 1), dtype=np.float32)

        return np.concatenate([joint_action, gripper_action], axis=-1).astype(np.float32)

    def infer(self, obs: dict) -> np.ndarray:
        """Infer actions from observations (roboarena format in, (N,8) array out)."""
        # Session change detection
        session_id = obs.get("session_id", None)
        if session_id is not None and session_id != self._current_session_id:
            if self._current_session_id is not None:
                logger.info(f"Session changed from '{self._current_session_id}' to '{session_id}', resetting state")
                self._reset_state()
            else:
                logger.info(f"New session started: '{session_id}'")
            self._current_session_id = session_id

        self._call_count += 1

        converted_obs = self._convert_observation(obs)

        # Signal workers to continue (0 = continue)
        signal_tensor = torch.zeros(1, dtype=torch.int32, device="cpu")
        dist.broadcast(signal_tensor, src=0, group=self._signal_group)

        # Broadcast obs to workers
        broadcast_to_workers(converted_obs)
        batch = Batch(obs=converted_obs)

        # Distributed forward pass
        dist.barrier()
        with torch.no_grad():
            result_batch, video_pred = self._policy.lazy_joint_forward_causal(batch)
        dist.barrier()

        # Accumulate video latents
        if self._video_saver is not None:
            self._video_saver.append(video_pred)

        # Extract and convert action
        action_chunk_dict = result_batch.act
        action_dict = {}
        for k in dir(action_chunk_dict):
            if k.startswith("action."):
                action_dict[k] = getattr(action_chunk_dict, k)

        action = self._convert_action(action_dict)

        if self._is_first_call:
            self._is_first_call = False

        return action

    def _reset_state(self, save_video: bool = True) -> None:
        if save_video and self._video_saver is not None:
            self._video_saver.save_and_clear(reason="reset")
        elif self._video_saver is not None:
            self._video_saver.clear()

        self._frame_buffer.clear()
        self._call_count = 0
        self._is_first_call = True

    def reset(self) -> None:
        """Reset the policy state for a new episode."""
        self._reset_state(save_video=True)


class DistributedWorkerLoop:
    """Worker loop that participates in distributed forward passes driven by rank 0.

    Parameters
    ----------
    policy
        The ``GrootSimPolicy`` (or compatible) instance — must expose
        ``lazy_joint_forward_causal(batch)``.
    signal_group : dist.ProcessGroup
        A gloo process group used for signal broadcasting between ranks.
    """

    def __init__(self, policy, signal_group: dist.ProcessGroup) -> None:
        self._policy = policy
        self._signal_group = signal_group

    async def run(self) -> None:
        """Blocking worker loop — listens for signals from rank 0 and participates in inference."""
        rank = dist.get_rank()
        logger.info(f"Worker loop started for rank {rank}")
        signal_tensor = torch.zeros(1, dtype=torch.int32, device="cpu")

        while True:
            try:
                dist.broadcast(signal_tensor, src=0, group=self._signal_group)
                signal = signal_tensor.item()

                if signal == 1:
                    logger.info(f"Rank {rank} received shutdown signal")
                    break

                if signal == 2:
                    logger.info(f"Rank {rank} received idle signal. Waiting for next client.")
                    continue

                # signal == 0 → continue with inference
                obs = receive_from_rank0()
                batch = Batch(obs=obs)

                dist.barrier()
                with torch.no_grad():
                    self._policy.lazy_joint_forward_causal(batch)
                dist.barrier()

            except Exception as e:
                logger.error(f"Worker loop error on rank {rank}: {e}")
                traceback.print_exc()
                break


# ------------------------------------------------------------------
# Static broadcast / receive helpers (used by rank 0 and workers)
# ------------------------------------------------------------------

def broadcast_to_workers(obs: dict) -> None:
    """Serialize *obs* and broadcast from rank 0 to all other ranks."""
    serialized = pickle.dumps(obs)
    data_size = len(serialized)

    size_tensor = torch.tensor([data_size], dtype=torch.int64, device="cuda")
    dist.broadcast(size_tensor, src=0)

    data_tensor = torch.frombuffer(serialized, dtype=torch.uint8).cuda()
    dist.broadcast(data_tensor, src=0)


def receive_from_rank0() -> dict:
    """Receive broadcast data from rank 0 and return as a ``Batch``."""
    size_tensor = torch.zeros(1, dtype=torch.int64, device="cuda")
    dist.broadcast(size_tensor, src=0)
    data_size = size_tensor.item()

    data_tensor = torch.zeros(data_size, dtype=torch.uint8, device="cuda")
    dist.broadcast(data_tensor, src=0)

    obs = pickle.loads(data_tensor.cpu().numpy().tobytes())
    return obs