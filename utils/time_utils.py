# =============================================================================
# utils/time_utils.py  –  FPS counter and timestamp helpers
# =============================================================================

import time
from collections import deque
from datetime import datetime


class FPSCounter:
    """Sliding-window FPS counter (thread-safe-ish for single writer)."""

    def __init__(self, window: int = 30):
        self._times: deque = deque(maxlen=window)

    def tick(self) -> float:
        """Record a frame and return current FPS."""
        self._times.append(time.monotonic())
        if len(self._times) < 2:
            return 0.0
        elapsed = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / elapsed if elapsed > 0 else 0.0

    @property
    def fps(self) -> float:
        if len(self._times) < 2:
            return 0.0
        elapsed = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / elapsed if elapsed > 0 else 0.0


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def now_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def uptime_str(start_time: float) -> str:
    """Human-readable uptime from a start timestamp."""
    secs = int(time.time() - start_time)
    h, rem = divmod(secs, 3600)
    m, s   = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"
