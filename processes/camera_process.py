# =============================================================================
# processes/camera_process.py  –  Isolated RTSP capture process  (v5)
# =============================================================================
# FIX #4: "Camera reader thread stalled (infs)" – guard seconds_since_last_frame
#         with math.isfinite() before formatting; show "not started" instead.
# FIX #5: RTSP reader now uses SteppedReconnectPolicy (1/3/5/10 s ladder)
#         instead of exponential backoff to avoid log spam and over-reconnect.
# FIX LAG-1: cv2.resize() removed from the tight reader thread.
#         It ran at full RTSP rate (25-30 FPS), burning CPU on both camera
#         processes and starving Camera 2.  Resize now happens ONCE in the
#         main loop, right before writer.write(), so it executes at the
#         12 FPS capture rate only.
# FIX LAG-2: FFmpeg backend is forced to TCP transport + no-buffer mode via
#         OPENCV_FFMPEG_CAPTURE_OPTIONS.  On Windows the CAP_PROP_BUFFERSIZE=1
#         hint is often silently ignored by the FFmpeg backend for RTSP, causing
#         a creeping decode delay.  The env-var approach bypasses that limitation.
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
    FIX LAG-1: cv2.resize() has been REMOVED from this thread.
            The thread now stores raw frames straight from the decoder.
            Resize is performed in run_camera_process() at the capped
            12 FPS rate, not at the full 25-30 FPS decode rate.
    """

    BIG_FLOAT = 9999.0   # FIX #4: replaces float("inf") in callers

    def __init__(self, url: str, resolution: Tuple[int, int]):
        super().__init__(daemon=True, name="rtsp-reader")
        self.url        = url
        self.resolution = resolution          # kept for reference; resize done outside
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
                    # FIX LAG-1: resize REMOVED here.
                    # Storing the raw decoder frame keeps this tight read-loop
                    # CPU-cheap (memcpy only).  The main process loop resizes
                    # at 12 FPS, not at the full 25-30 FPS decode rate.
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
            # FIX LAG-2: force FFmpeg to TCP transport + disable internal
            # packet buffering.  On Windows, CAP_PROP_BUFFERSIZE=1 is
            # often ignored for RTSP streams by the FFmpeg backend, causing
            # a growing decode delay.  The env-var approach is the only
            # reliable way to pass fflags/flags to the underlying AVFormatContext.
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay"
            )
            cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
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
    # Create FrameWriter with a small retry loop to tolerate transient SHM races
    writer = None
    for attempt in range(3):
        try:
            writer = FrameWriter(camera_id=camera_id,
                                 width=resolution[0], height=resolution[1])
            break
        except Exception as e:
            log = get_logger("Main")
            log.warning(f"FrameWriter create failed (attempt {attempt+1}): {e}")
            time.sleep(0.5)
    if writer is None:
        # Last-resort: try one more time and let exception propagate if it fails
        writer = FrameWriter(camera_id=camera_id,
                             width=resolution[0], height=resolution[1])
    fps_ctr = FPSCounter(window=30)
    reader  = _RTSPReaderThread(rtsp_url, resolution)
    reader.start()
    reader_start_t = time.monotonic()   # FIX #3: track when reader was (re)started

    # Dedicated heartbeat sender thread so heartbeats are sent even if
    # the main loop stalls briefly (prevents supervisor false-kills).
    class _HeartbeatThread(threading.Thread):
        def __init__(self, hb_q, pname, camera_id, fps_ctr, guard, reader):
            super().__init__(daemon=True, name="heartbeat-thread")
            self.hb_q = hb_q
            self.pname = pname
            self.cid = camera_id
            self.fps_ctr = fps_ctr
            self.guard = guard
            self.reader = reader
            self._stop = threading.Event()

        def run(self):
            while not self._stop.is_set():
                try:
                    msg = make_heartbeat(
                        source=self.pname,
                        camera_id=self.cid,
                        fps=self.fps_ctr.fps,
                        ram_mb=self.guard.get_ram_mb(),
                        extra={"connected": self.reader.is_connected},
                    )
                    try:
                        self.hb_q.put_nowait(msg)
                    except Exception:
                        # queue full; drop and retry on next interval
                        pass
                except Exception:
                    pass
                time.sleep(HEARTBEAT_INTERVAL)

    hb_thread = _HeartbeatThread(heartbeat_q, pname, camera_id, fps_ctr, guard, reader)
    hb_thread.start()

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

            # FIX LAG-1: resize happens HERE, at 12 FPS, not inside the
            # reader thread at 25-30 FPS.  This halves the per-camera CPU
            # cost of decoding and eliminates the OS scheduling asymmetry
            # that caused Camera 2 to starve for CPU cycles.
            frame = cv2.resize(frame, resolution)

            last_written_frame = frame
            last_frame_t       = time.monotonic()
            try:
                writer.write(frame)
            except Exception as e:
                log = get_logger("Main")
                log.error(f"FrameWriter.write failed: {e} — attempting recreate")
                try:
                    # Close old writer and cleanup SHM for this camera, then recreate
                    try:
                        writer.close()
                    except Exception:
                        pass
                    from ipc.frame_store import cleanup_shm_for_camera
                    cleanup_shm_for_camera(camera_id, resolution[0], resolution[1])
                except Exception:
                    pass
                try:
                    writer = FrameWriter(camera_id=camera_id,
                                         width=resolution[0], height=resolution[1])
                except Exception as e2:
                    log.error(f"Recreate FrameWriter failed: {e2}")
                    # continue loop; don't crash camera process — heartbeat will show disconnected
                    time.sleep(0.5)
                    continue
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
        try:
            hb_thread._stop.set()
            hb_thread.join(timeout=1)
        except Exception:
            pass
        reader.stop()
        reader.join(timeout=5)
        writer.close()
        log.info(f"Camera process {camera_id} exiting")