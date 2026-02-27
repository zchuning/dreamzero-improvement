"""Policy interface for YAM robot control.

To plug in your own model, subclass :class:`Policy` and implement :meth:`get_action`.  
The rollout infrastructure (action chunking, safety wrappers, ViserUI, recording) is handled by ``run_rollout()``.

Observation dict
----------------
The environment passes a flat dict with the following keys:

    Proprioception (float32):
        left_joint_pos   -- (6,)   left arm joint angles (rad)
        left_gripper_pos -- (1,)   left gripper position [0=closed, 1=open]
        right_joint_pos  -- (6,)   right arm joint angles (rad)
        right_gripper_pos-- (1,)   right gripper position [0=closed, 1=open]

    Camera images (uint8):
        top_camera_image   -- (480, 640, 3) RGB
        left_camera_image  -- (480, 640, 3) RGB
        right_camera_image -- (480, 640, 3) RGB

    Task description (str, set by ViserUI):
        annotation.task  -- e.g. "pick up the red cup"

Action dict
-----------
``get_action`` must return ``(first_action, info)`` where:

    first_action: dict with keys
        left_joint_pos    -- (6,)  float32
        left_gripper_pos  -- (1,)  float32
        right_joint_pos   -- (6,)  float32
        right_gripper_pos -- (1,)  float32

    info: dict that **must** contain
        "action_chunk" -- dict of arrays with shape (T, D) representing
                          the full predicted action sequence.  T is the
                          temporal horizon of your model.  The chunking
                          wrappers (SyncChunkingPolicy / AsyncChunkingPolicy)
                          use this to execute one step at a time.

Minimal example
---------------
::

    from yam_control.policy import Policy

    class MyPolicy(Policy):
        def __init__(self, model):
            self.model = model

        def get_action(self, observation):
            images = {k: observation[k] for k in
                      ["top_camera_image", "left_camera_image", "right_camera_image"]}
            state = {k: observation[k] for k in
                     ["left_joint_pos", "left_gripper_pos",
                      "right_joint_pos", "right_gripper_pos"]}
            task = observation.get("annotation.task", "")

            # Your model returns action chunk: dict of (T, D) arrays
            action_chunk = self.model.predict(images, state, task)

            first_action = {k: v[0] for k, v in action_chunk.items()}
            return first_action, {"action_chunk": action_chunk}

        def reset(self):
            self.model.reset()
            return None
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

Observation = dict[str, Any]
Action = dict[str, np.ndarray]
Options = dict[str, Any]
Info = dict[str, Any]


class Policy(ABC):
    """Base class for all policies.  Subclass this and implement ``get_action``."""

    def reset(self) -> Info | None:
        """Called between episodes.  Override to clear model state."""
        return None

    @abstractmethod
    def get_action(self, observation: Observation) -> tuple[Action, Info]:
        """Return ``(first_action, info)`` given an observation.

        ``info`` **must** contain ``"action_chunk"`` -- a dict of numpy
        arrays with shape ``(T, D)`` representing the predicted action
        sequence, so the chunking wrappers can execute it step by step.
        """
        ...
