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
