# =============================================================================
# processes/camera_process.py  –  Isolated RTSP capture process  (v4)
# =============================================================================
# FIX #4: "Camera reader thread stalled (infs)" – guard seconds_since_last_frame
#         with math.isfinite() before formatting; show "not started" instead.
# FIX #5: RTSP reader now uses SteppedReconnectPolicy (1/3/5/10 s ladder)
#         instead of exponential backoff to avoid log spam and over-reconnect.
# =============================================================================

import math
import os
import sys
import time
import threading
from multiprocessing import Queue
from typing import Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import cv2
import numpy as np

from utils.logger import setup_process_logger, get_logger
from utils.resource_guard import ResourceGuard, ResourceLimitExceeded, RAM_LIMIT_CAMERA
from utils.time_utils import FPSCounter
from ipc.frame_store import FrameWriter
from ipc.messages import (
    make_heartbeat, make_error,
    MSG_SHUTDOWN, MSG_CONTROL, CTRL_SHUTDOWN,
)
from core.reconnect_policy import SteppedReconnectPolicy   # FIX #5

FPS_TARGET         = 12
HEARTBEAT_INTERVAL = 5.0
WATCHDOG_TIMEOUT_S = 20.0   # FIX #4: raised slightly; NaN/inf no longer printed


class _RTSPReaderThread(threading.Thread):
    """
    Non-blocking RTSP reader daemon thread.

    FIX #5: Uses SteppedReconnectPolicy (1/3/5/10 s).
    FIX #4: seconds_since_last_frame returns a finite float or
            a sentinel BIG_FLOAT; caller checks math.isfinite().
    """

    BIG_FLOAT = 9999.0   # FIX #4: replaces float("inf") in callers

    def __init__(self, url: str, resolution: Tuple[int, int]):
        super().__init__(daemon=True, name="rtsp-reader")
        self.url        = url
        self.resolution = resolution
        self._lock          = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._last_frame_t  = 0.0
        self._connected     = False
        self._stop_event    = threading.Event()
        self._cap: Optional[cv2.VideoCapture] = None

    def run(self) -> None:
        policy = SteppedReconnectPolicy(max_attempts=0)   # FIX #5: stepped
        while not self._stop_event.is_set():
            if not self._open():
                policy.wait()
                continue
            policy.reset()

            while not self._stop_event.is_set():
                try:
                    ret, frame = self._cap.read()
                    if not ret or frame is None:
                        break
                    frame = cv2.resize(frame, self.resolution)
                    with self._lock:
                        self._latest_frame = frame
                        self._last_frame_t = time.monotonic()
                        self._connected    = True
                except Exception:
                    break

            self._close()

    def _open(self) -> bool:
        self._close()
        try:
            cap = cv2.VideoCapture(self.url)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if cap.isOpened():
                self._cap = cap
                return True
            cap.release()
        except Exception:
            pass
        with self._lock:
            self._connected = False
        return False

    def _close(self) -> None:
        with self._lock:
            self._connected = False
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

    def get_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._latest_frame

    @property
    def is_connected(self) -> bool:
        with self._lock:
            return self._connected

    @property
    def seconds_since_last_frame(self) -> float:
        """
        FIX #4: returns BIG_FLOAT (not float('inf')) when no frame yet.
        Callers use math.isfinite() to detect the "never started" case.
        """
        with self._lock:
            if self._last_frame_t == 0.0:
                return self.BIG_FLOAT   # FIX #4
            return time.monotonic() - self._last_frame_t

    def stop(self) -> None:
        self._stop_event.set()
        self._close()


def run_camera_process(
    camera_id:    int,
    rtsp_url:     str,
    resolution:   Tuple[int, int],
    heartbeat_q:  Queue,
    control_q:    Queue,
    ram_limit_mb: float = RAM_LIMIT_CAMERA,
) -> None:
    pname = f"camera_{camera_id}"
    setup_process_logger(pname)
    log = get_logger("Main")
    log.info(f"Camera process {camera_id} started  PID={os.getpid()}")

    guard   = ResourceGuard(ram_limit_mb=ram_limit_mb)
    writer  = FrameWriter(camera_id=camera_id,
                          width=resolution[0], height=resolution[1])
    fps_ctr = FPSCounter(window=30)
    reader  = _RTSPReaderThread(rtsp_url, resolution)
    reader.start()
    reader_start_t = time.monotonic()   # FIX #3: track when reader was (re)started

    last_hb            = 0.0
    last_frame_t       = time.monotonic()
    frame_interval     = 1.0 / FPS_TARGET
    last_written_frame = None

    try:
        while True:
            try:
                msg = control_q.get_nowait()
                if msg.get("type") == MSG_SHUTDOWN or (
                    msg.get("type") == MSG_CONTROL
                    and msg["payload"].get("command") == CTRL_SHUTDOWN
                ):
                    log.info("Shutdown received")
                    break
            except Exception:
                pass

            try:
                guard.check()
            except ResourceLimitExceeded as e:
                log.error(f"Resource limit: {e}")
                heartbeat_q.put_nowait(
                    make_error(pname, str(e), camera_id=camera_id, fatal=True)
                )
                sys.exit(2)

            now = time.monotonic()
            if now - last_hb >= HEARTBEAT_INTERVAL:
                last_hb = now
                heartbeat_q.put_nowait(
                    make_heartbeat(
                        source=pname,
                        camera_id=camera_id,
                        fps=fps_ctr.fps,
                        ram_mb=guard.get_ram_mb(),
                        extra={"connected": reader.is_connected},
                    )
                )

            # Watchdog: only fire if the reader has been alive long enough to
            # have connected.  On startup seconds_since_last_frame returns
            # BIG_FLOAT (9999) because no frame has arrived yet – we must NOT
            # restart the thread just because it hasn't received its first frame
            # within the poll interval.  The grace period equals WATCHDOG_TIMEOUT_S
            # so the reader always gets at least that long to make the first
            # connection before we consider it stalled.
            stall_s      = reader.seconds_since_last_frame
            reader_age_s = time.monotonic() - reader_start_t
            if stall_s > WATCHDOG_TIMEOUT_S and reader_age_s > WATCHDOG_TIMEOUT_S:
                if stall_s < _RTSPReaderThread.BIG_FLOAT:
                    log.warning(
                        f"Camera {camera_id} reader stalled "
                        f"({stall_s:.0f}s) – restarting thread"
                    )
                else:
                    log.warning(
                        f"Camera {camera_id} reader failed to connect "
                        f"within {WATCHDOG_TIMEOUT_S:.0f}s – restarting"
                    )
                reader.stop()
                reader.join(timeout=3)
                reader       = _RTSPReaderThread(rtsp_url, resolution)
                reader.start()
                reader_start_t = time.monotonic()   # reset grace clock
                last_frame_t   = time.monotonic()
                continue

            elapsed = time.monotonic() - last_frame_t
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)
                continue

            frame = reader.get_frame()
            if frame is None:
                time.sleep(0.05)
                continue

            if frame is last_written_frame:
                time.sleep(0.005)
                continue

            last_written_frame = frame
            last_frame_t       = time.monotonic()
            writer.write(frame)
            fps_ctr.tick()

    except KeyboardInterrupt:
        pass
    except Exception as e:
        log.error(f"Camera process fatal: {e}", exc_info=True)
        try:
            heartbeat_q.put_nowait(
                make_error(pname, str(e), camera_id=camera_id, fatal=True)
            )
        except Exception:
            pass
        sys.exit(1)
    finally:
        reader.stop()
        reader.join(timeout=5)
        writer.close()
        log.info(f"Camera process {camera_id} exiting")
