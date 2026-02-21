"""Video saver for decoding and saving VAE video latents to MP4.
"""

import datetime
import logging
import os

import imageio
import numpy as np
import torch
from einops import rearrange

logger = logging.getLogger(__name__)


class FrameBuffer:
    """Accumulates camera frames and returns fixed-size windows for policy input.

    The policy expects multi-frame video tensors (T, H, W, C) but observations
    arrive one frame at a time. This buffer stores incoming frames per camera key
    and returns the most recent ``num_frames`` when requested, padding by repeating
    the first frame if not enough frames have been accumulated yet.
    """

    def __init__(self, camera_keys: list[str], frames_per_chunk: int = 4) -> None:
        self._buffers: dict[str, list[np.ndarray]] = {k: [] for k in camera_keys}
        self.frames_per_chunk = frames_per_chunk

    def append(self, key: str, frame: np.ndarray) -> None:
        """Append a single (H, W, 3) or batch (T, H, W, 3) frame to the buffer."""
        if key not in self._buffers:
            raise KeyError(f"Unknown camera key: {key}")
        if frame.ndim == 4:
            self._buffers[key].extend(list(frame))
        else:
            self._buffers[key].append(frame)

    def get_frames(self, key: str, num_frames: int) -> np.ndarray:
        """Return the last ``num_frames`` frames as (T, H, W, C), padding if needed."""
        buf = self._buffers[key]
        if len(buf) == 0:
            raise ValueError(f"No frames in buffer for key: {key}")
        if len(buf) >= num_frames:
            frames = buf[-num_frames:]
        else:
            frames = list(buf)
            while len(frames) < num_frames:
                frames.insert(0, buf[0])
        return np.stack(frames, axis=0)

    @property
    def camera_keys(self) -> list[str]:
        return list(self._buffers.keys())

    def has_frames(self, key: str) -> bool:
        return len(self._buffers[key]) > 0

    def clear(self) -> None:
        """Clear all frame buffers."""
        for key in self._buffers:
            self._buffers[key] = []



class VideoSaver:
    """Accumulates video latent tensors and decodes/saves them as MP4 files.

    Parameters
    ----------
    output_dir : str
        Directory where MP4 files are written.
    action_head
        Reference to ``policy.trained_model.action_head`` — used to access the
        VAE and its tiling parameters (``tiled``, ``tile_size_*``, ``tile_stride_*``).
    """

    def __init__(self, output_dir: str, action_head) -> None:
        self._output_dir = output_dir
        self._action_head = action_head
        self._latents: list[torch.Tensor] = []
        os.makedirs(output_dir, exist_ok=True)

    def append(self, video_latent: torch.Tensor) -> None:
        """Store a video latent tensor for later decoding."""
        self._latents.append(video_latent)

    @property
    def count(self) -> int:
        return len(self._latents)

    def should_save(self) -> bool:
        """Heuristic: save when we have accumulated more than 10 latents."""
        return len(self._latents) > 10

    def should_save_on_frame_reset(self) -> bool:
        """Save when the action head has reset its frame counter and we have >1 latent."""
        ah = self._action_head
        return (
            ah.current_start_frame == 1 + ah.num_frame_per_block
            and len(self._latents) > 1
        )

    def save_and_clear(self, reason: str = "") -> str | None:
        """Decode accumulated latents, write MP4, clear buffer.

        Returns the path to the saved file, or ``None`` on failure / empty buffer.
        """
        if not self._latents:
            return None
        return self._decode_and_save(self._latents, reason)

    def save_all_but_last_and_rotate(self, reason: str = "") -> str | None:
        """Decode all but the last latent, save, keep the last one in the buffer.

        Used for the frame-reset heuristic where the last latent belongs to the
        next segment.
        """
        if len(self._latents) <= 1:
            return None
        to_save = self._latents[:-1]
        keep = self._latents[-1]
        path = self._decode_and_save(to_save, reason)
        self._latents = [keep]
        return path

    def clear(self) -> None:
        self._latents = []

    def _decode_and_save(self, latents: list[torch.Tensor], reason: str) -> str | None:
        try:
            video_cat = torch.cat(latents, dim=2)
            ah = self._action_head
            frames = ah.vae.decode(
                video_cat,
                tiled=ah.tiled,
                tile_size=(ah.tile_size_height, ah.tile_size_width),
                tile_stride=(ah.tile_stride_height, ah.tile_stride_width),
            )
            frames = rearrange(frames, "B C T H W -> B T H W C")
            frames = frames[0]
            frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)

            frame_list = list(frames)
            if not frame_list:
                return None

            sample = frame_list[0]
            if not (sample.ndim == 3 and sample.shape[2] in (1, 3, 4)):
                logger.warning(f"Invalid frame shape {sample.shape}, skipping save.")
                return None

            all_mp4 = [f for f in os.listdir(self._output_dir) if f.endswith(".mp4")]
            ts = datetime.datetime.now().strftime("%m_%d_%H_%M_%S")
            n = (len(frame_list) - 1) // 8
            path = os.path.join(
                self._output_dir,
                f"{len(all_mp4):06}_{ts}_n{n}.mp4",
            )
            imageio.mimsave(path, frame_list, fps=5, codec="libx264")
            logger.info(f"Saved video{f' ({reason})' if reason else ''}: {path}")
            return path
        except Exception as e:
            logger.warning(f"Failed to save video: {e}")
            return None
