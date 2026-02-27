"""Trajectory timer for Viser UI.

Tracks execution time across Start/Pause/Home/StepOnce transitions.
Automatically integrated into :class:`~yam_control.viser_ui.ViserUI` —
the timer display appears in the web interface at http://localhost:8080.

Standalone usage (e.g. in a custom control loop)::

    from yam_control.viser_timer import TrajectoryTimer

    timer = TrajectoryTimer()
    timer.start()           # begin timing
    # ... execute trajectory ...
    print(timer.get_display_string())   # "0:12"
    timer.pause()           # freeze — resumes from here on next start()
    timer.start()           # resume from paused time
    timer.home()            # stop and mark for reset on next start()
    timer.start()           # resets to 0:00 and starts fresh
"""

import time
from typing import Literal

TimerState = Literal["stopped", "running", "paused"]


class TrajectoryTimer:
    """Timer for tracking trajectory execution time.

    Behavior:
    - Start after Home: Resets to 0:00 (new trajectory)
    - Start after Pause: Resumes from paused time
    - Home: Stops timer, displays final time (does NOT clear)
    - Multiple Home presses: No effect, time persists
    """

    def __init__(self):
        self._start_time: float | None = None
        self._elapsed_time: float = 0.0
        self._state: TimerState = "stopped"
        self._should_reset_on_start: bool = True

    def start(self) -> None:
        """Start or resume timer based on whether reset is needed."""
        if self._should_reset_on_start:
            self._elapsed_time = 0.0
            self._start_time = time.time()
            self._should_reset_on_start = False
        else:
            self._start_time = time.time()

        self._state = "running"

    def pause(self) -> None:
        """Pause timer and save current elapsed time."""
        if self._state == "running":
            self._elapsed_time += time.time() - self._start_time
            self._start_time = None

        self._state = "paused"

    def home(self) -> None:
        """Stop timer and display final time (does NOT clear)."""
        if self._state == "running":
            self._elapsed_time += time.time() - self._start_time
            self._start_time = None

        self._state = "stopped"
        self._should_reset_on_start = True

    def step_once(self) -> None:
        """Start timer for a single step, will auto-pause after step completes.

        Behavior:
        - From Home (stopped): Reset to 0 and start timing
        - From Pause (after previous steps): Resume timing (accumulate)
        - Preserves reset flag so Start after Step Once will still reset
        """
        if self._state == "stopped":
            self._elapsed_time = 0.0

        if self._state != "running":
            self._start_time = time.time()
            self._state = "running"

    def get_elapsed_time(self) -> float:
        """Get current elapsed time in seconds."""
        if self._state == "running":
            return self._elapsed_time + (time.time() - self._start_time)
        return self._elapsed_time

    def get_display_string(self) -> str:
        """Get formatted time string (MM:SS)."""
        total_seconds = int(self.get_elapsed_time())
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        return f"{minutes}:{seconds:02d}"

    @property
    def state(self) -> TimerState:
        """Current timer state (stopped/running/paused)."""
        return self._state
