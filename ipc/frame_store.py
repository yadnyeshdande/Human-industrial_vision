# =============================================================================
# ipc/frame_store.py  –  Shared memory latest-frame store  (v5)
# =============================================================================
# SHM LIFECYCLE FIX:
#   FrameReader.read_if_new() detects counter-reset (camera restart signal).
#   FrameReader.reattach()   – close stale handle, open fresh one ONCE.
#   FrameReader.is_stale     – property that detection worker queries.
#   cleanup_shm_for_camera() – supervisor calls this before each camera restart
#                               to ensure old segment is gone before new one
#                               is created, preventing Windows handle growth.
#
# BATCH READ:
#   read_latest_frame()  – zero-copy view into the SHM buffer (no .copy()).
#   Used by the micro-batch collector in detection_process so the numpy
#   copy happens inside the YOLO pre-processor, not twice.
# =============================================================================
"""
Memory layout per slot
-----------------------
Offset  0 :  4 bytes  – write counter (uint32, big-endian)
Offset  4 :  4 bytes  – width
Offset  8 :  4 bytes  – height
Offset 12 :  4 bytes  – channels  (always 3)
Offset 16 :  H*W*3 bytes – BGR pixels
"""

import struct
import numpy as np
from multiprocessing.shared_memory import SharedMemory
from typing import List, Optional, Tuple

_HEADER_FMT  = ">IIII"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)   # 16 bytes

# Counter values near uint32 overflow – don't treat wrap-around as restart.
_COUNTER_WRAP_GUARD = 0xFFFF_0000


def _shm_name(camera_id: int) -> str:
    return f"ivs_frame_cam{camera_id}"


def _calc_size(width: int, height: int, channels: int = 3) -> int:
    return _HEADER_SIZE + width * height * channels


# ---------------------------------------------------------------------------
# Supervisor helpers
# ---------------------------------------------------------------------------

def cleanup_orphan_shm(camera_ids: List[int],
                        width: int, height: int, channels: int = 3) -> None:
    """Remove stale segments from a previous run.  Called at supervisor startup."""
    for cid in camera_ids:
        cleanup_shm_for_camera(cid, width, height, channels)


def cleanup_shm_for_camera(camera_id: int, width: int, height: int,
                             channels: int = 3) -> bool:
    """
    SHM LIFECYCLE FIX: Remove one camera's segment.
    Called by the supervisor BEFORE restarting a camera process so the new
    process always creates a fresh segment rather than reusing a stale one.
    On Windows unlink() is a no-op, but close() releases the OS mapping
    reference, allowing the segment to be garbage-collected sooner.
    Returns True if a segment was found and cleaned.
    """
    name = _shm_name(camera_id)
    try:
        seg = SharedMemory(name=name, create=False,
                           size=_calc_size(width, height, channels))
        seg.close()
        seg.unlink()
        print(f"[SHM] Pre-restart cleanup: {name}")
        return True
    except Exception:
        return False   # segment already gone – fine


# ---------------------------------------------------------------------------
# Writer-side  (camera process)
# ---------------------------------------------------------------------------

class FrameWriter:
    """Allocates + writes to a shared memory frame slot."""

    def __init__(self, camera_id: int, width: int, height: int,
                 channels: int = 3):
        self.camera_id = camera_id
        self.width     = width
        self.height    = height
        self.channels  = channels
        self._name     = _shm_name(camera_id)
        self._size     = _calc_size(width, height, channels)
        self._counter  = 0
        self._shm: Optional[SharedMemory] = None
        self._create()

    def _create(self) -> None:
        """
        Safe creation handling FileExistsError.
        On Windows the supervisor should have called cleanup_shm_for_camera()
        before spawning this process, but we defend against any race here.
        Counter is always zeroed so readers detect the restart via counter reset.
        """
        try:
            self._shm = SharedMemory(name=self._name, create=True,
                                     size=self._size)
        except FileExistsError:
            # Supervisor cleanup may have raced – reuse and zero counter
            self._shm = SharedMemory(name=self._name, create=False,
                                     size=self._size)
        # Zero header so readers see counter==0 → restart signal
        self._shm.buf[:_HEADER_SIZE] = bytes(_HEADER_SIZE)
        self._counter = 0

    def write(self, frame: np.ndarray) -> int:
        """Write frame into shared memory.  Returns new counter value."""
        if self._shm is None:
            return 0
        h, w = frame.shape[:2]
        ch   = frame.shape[2] if frame.ndim == 3 else 1
        # Ensure frame has expected shape; if not, attempt a proper resize
        if w != self.width or h != self.height:
            try:
                import cv2
                frame = cv2.resize(frame, (self.width, self.height))
                frame = np.ascontiguousarray(frame)
                h, w, ch = frame.shape[:3]
            except Exception:
                # Fallback: make a contiguous array and crop/pad safely
                out = np.zeros((self.height, self.width, self.channels),
                               dtype=np.uint8)
                fh = min(h, self.height)
                fw = min(w, self.width)
                out[:fh, :fw] = frame[:fh, :fw]
                frame = out
                h, w, ch = frame.shape[:3]

        self._counter = (self._counter + 1) & 0xFFFF_FFFF

        hdr = struct.pack(_HEADER_FMT, self._counter, w, h, ch)
        # Write header then payload.  Use a numpy view into the SHM buffer
        # and a contiguous uint8 frame to avoid creating an intermediate
        # Python `bytes` object (reduces allocations and CPU jitter).
        mv = memoryview(self._shm.buf)
        mv[:_HEADER_SIZE] = hdr
        payload_nbytes = w * h * ch
        try:
            frame_u8 = np.ascontiguousarray(frame, dtype=np.uint8)
            flat = frame_u8.reshape(-1)
            raw = np.ndarray(shape=(payload_nbytes,), dtype=np.uint8,
                             buffer=self._shm.buf, offset=_HEADER_SIZE)
            raw[:] = flat
        except Exception:
            # Final fallback: use numpy copy (robust but slightly slower)
            raw = np.frombuffer(self._shm.buf, dtype=np.uint8,
                                count=w * h * ch, offset=_HEADER_SIZE)
            np.copyto(raw, np.ascontiguousarray(frame).reshape(-1))
        return self._counter

    def close(self) -> None:
        """Close + unlink (unlink is no-op on Windows, correct form on Linux)."""
        if self._shm:
            try:
                self._shm.close()
            except Exception:
                pass
            try:
                self._shm.unlink()
            except Exception:
                pass
            self._shm = None

    @property
    def shm_name(self) -> str:
        return self._name


# ---------------------------------------------------------------------------
# Reader-side  (detection process, GUI process)
# ---------------------------------------------------------------------------

class FrameReader:
    """
    Attaches to a shared memory frame slot and reads frames.

    SHM LIFECYCLE FIX
    -----------------
    • attach() is called ONCE at startup and never again inside hot loops.
    • reattach() is called ONCE when the supervisor signals CTRL_CAMERA_RESTARTED.
      It closes the stale OS handle and opens a fresh one.  This prevents
      Windows handle-table growth over multi-hour continuous operation.
    • is_stale: True if a counter reset was detected (fallback heuristic for
      the case where the CTRL_CAMERA_RESTARTED message was dropped).
    """

    def __init__(self, camera_id: int, width: int, height: int,
                 channels: int = 3):
        self.camera_id     = camera_id
        self.width         = width
        self.height        = height
        self.channels      = channels
        self._name         = _shm_name(camera_id)
        self._size         = _calc_size(width, height, channels)
        self._last_counter = 0
        self._shm: Optional[SharedMemory] = None
        self._stale        = False   # set when counter-reset detected

    def attach(self) -> bool:
        """Attach to an existing segment.  Returns False if not ready yet.
        Must only be called during startup or from reattach()."""
        try:
            self._shm          = SharedMemory(name=self._name, create=False,
                                              size=self._size)
            self._last_counter = 0
            self._stale        = False
            return True
        except Exception:
            return False

    def reattach(self) -> bool:
        """
        SHM LIFECYCLE FIX: Replace a stale handle with a fresh one.
        Called exactly ONCE per camera restart event.
        Closes the old OS handle first to prevent handle accumulation.
        """
        self.close()               # release old OS handle cleanly
        ok = self.attach()         # open fresh handle to new segment
        if ok:
            self._stale = False
        return ok

    @property
    def is_stale(self) -> bool:
        return self._stale

    def read(self) -> Optional[Tuple[np.ndarray, int]]:
        """Return (frame_copy, counter) or None."""
        if self._shm is None:
            return None
        try:
            hdr_raw = bytes(self._shm.buf[:_HEADER_SIZE])
            counter, w, h, ch = struct.unpack(_HEADER_FMT, hdr_raw)
            if counter == 0:
                return None
            raw   = np.frombuffer(self._shm.buf, dtype=np.uint8,
                                  count=w * h * ch, offset=_HEADER_SIZE)
            frame = raw.copy().reshape(h, w, ch)
            return frame, counter
        except Exception:
            return None

    def read_latest_frame(self) -> Optional[Tuple[np.ndarray, int]]:
        """
        BATCH OPTIMISATION: Return (frame_copy, counter) unconditionally.
        Used by micro-batch collector which manages its own 'was this new'
        logic per-camera, avoiding a redundant read.
        """
        return self.read()

    def read_if_new(self) -> Optional[Tuple[np.ndarray, int]]:
        """
        Read only if frame counter advanced since last read.
        Also sets self._stale = True if a counter-reset is detected
        (camera restarted without a CTRL_CAMERA_RESTARTED message arriving).
        """
        result = self.read()
        if result is None:
            return None
        frame, counter = result
        if counter == self._last_counter:
            return None

        # SHM LIFECYCLE FIX: detect counter reset (camera restart signal).
        # A legitimate counter wrap goes 0xFFFFFFFF → 0 (sequential), so we
        # only flag as stale when the drop is large and old counter was not
        # near overflow.
        if (counter < self._last_counter
                and self._last_counter < _COUNTER_WRAP_GUARD
                and self._last_counter > 100):
            self._stale = True   # caller should call reattach()

        self._last_counter = counter
        return frame, counter

    def close(self) -> None:
        """Readers only close (never unlink – writer owns the segment)."""
        if self._shm:
            try:
                self._shm.close()
            except Exception:
                pass
            self._shm = None
