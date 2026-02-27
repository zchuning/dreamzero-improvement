"""Embodiment tag definitions for YAM robot configurations.

This module provides a standalone EmbodimentTag enum for identifying robot
embodiment configurations, decoupled from the gr00t VLA training infrastructure.
"""

from enum import Enum


class EmbodimentTag(str, Enum):
    """Identifies the robot embodiment and data format."""

    XDOF = "xdof"
    """Standard XDOF robot configuration."""

    XDOF_H16 = "xdof_h16"
    """XDOF robot with action horizon 16."""

    XDOF_OSS_DATA = "xdof_oss_data"
    """XDOF data with open-source processing pipeline."""
