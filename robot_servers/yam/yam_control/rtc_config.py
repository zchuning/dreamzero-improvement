"""Configuration for Real-Time Chunking (RTC) inference.

Based on LeRobot's RTC implementation:
https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/rtc/configuration_rtc.py

RTC improves real-time inference by treating chunk generation as an inpainting problem,
using the model's training-time RTC capability to handle inference delays gracefully.
"""

from dataclasses import dataclass


@dataclass
class RTCConfig:
    """Configuration for Real Time Chunking (RTC) inference.

    RTC improves real-time inference by treating chunk generation as an inpainting problem.
    The model is trained with simulated inference delays, so at inference time we simply
    pass the actual delay and the model handles the rest.

    Attributes:
        enabled: Toggle RTC on/off.
        inference_interval: Trigger new inference every N action steps.
    """

    enabled: bool = True

    inference_interval: int = 10

    def __post_init__(self):
        if self.inference_interval <= 0:
            raise ValueError(f"inference_interval must be positive, got {self.inference_interval}")
