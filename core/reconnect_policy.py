# =============================================================================
# core/reconnect_policy.py  –  Reconnection backoff policies  (v4)
# =============================================================================
# FIX #5: Added SteppedReconnectPolicy with explicit delay ladder:
#           attempt 1 → 1 s
#           attempt 2 → 3 s
#           attempt 3 → 5 s
#           attempt 4+ → 10 s
#         Used by the RTSP reader thread to avoid log spam.
# =============================================================================

import time
from utils.logger import get_logger

logger = get_logger("ReconnectPolicy")


class ReconnectPolicy:
    """Exponential back-off – original behaviour, unchanged."""

    def __init__(
        self,
        initial_delay:  float = 1.0,
        max_delay:       float = 60.0,
        backoff_factor:  float = 2.0,
        max_attempts:    int   = 0,
    ):
        self.initial_delay  = initial_delay
        self.max_delay       = max_delay
        self.backoff_factor  = backoff_factor
        self.max_attempts    = max_attempts
        self._current_delay  = initial_delay
        self._attempt        = 0

    def wait(self) -> None:
        if self.max_attempts and self._attempt >= self.max_attempts:
            logger.error(f"Reconnect exhausted after {self.max_attempts} attempts")
            time.sleep(self.max_delay)
            return
        logger.info(
            f"Reconnect attempt {self._attempt + 1}"
            + (f"/{self.max_attempts}" if self.max_attempts else "")
            + f" – waiting {self._current_delay:.1f}s"
        )
        time.sleep(self._current_delay)
        self._attempt      += 1
        self._current_delay = min(
            self._current_delay * self.backoff_factor, self.max_delay
        )

    def reset(self) -> None:
        self._current_delay = self.initial_delay
        self._attempt       = 0

    @property
    def attempt(self) -> int:
        return self._attempt


class SteppedReconnectPolicy:
    """
    FIX #5 – explicit stepped delay ladder for RTSP streams.

    attempt 1  →  1 s
    attempt 2  →  3 s
    attempt 3  →  5 s
    attempt 4+ → 10 s

    Does NOT grow exponentially; instead provides a stable, predictable
    reconnect cadence suitable for industrial RTSP cameras.
    """

    _STEPS = [1.0, 3.0, 5.0, 10.0]

    def __init__(self, max_attempts: int = 0):
        self.max_attempts = max_attempts   # 0 = unlimited
        self._attempt     = 0

    def wait(self) -> None:
        if self.max_attempts and self._attempt >= self.max_attempts:
            logger.error(f"RTSP reconnect exhausted ({self.max_attempts} attempts)")
            time.sleep(self._STEPS[-1])
            return

        delay = self._STEPS[min(self._attempt, len(self._STEPS) - 1)]
        logger.info(
            f"RTSP reconnect attempt {self._attempt + 1} – waiting {delay:.0f}s"
        )
        time.sleep(delay)
        self._attempt += 1

    def reset(self) -> None:
        self._attempt = 0

    @property
    def attempt(self) -> int:
        return self._attempt

    @property
    def current_delay(self) -> float:
        return self._STEPS[min(self._attempt, len(self._STEPS) - 1)]
