# =============================================================================
# core/relay_hardware.py  –  Relay hardware abstraction  (v5)
# =============================================================================
#
# ROOT-CAUSE FIX  –  pyhid_usb_relay API was wrong in TWO places:
#
# BUG 1 – find() return type
#   OLD (broken):
#       devices = find()          # assumed list
#       self._device = devices[0] # TypeError: Controller not subscriptable
#   CORRECT:
#       self._device = find()     # find() returns ONE Controller directly
#       self._device = find(find_all=True)[0]  # only when filtering serials
#
# BUG 2 – activation / deactivation method names
#   OLD (broken):
#       device.turn_on(relay_id)  # AttributeError: no such method
#       device.turn_off(relay_id) # AttributeError: no such method
#   CORRECT:
#       device.set_state(relay_id, True)   # activate
#       device.set_state(relay_id, False)  # deactivate
#
# The Controller class in pyhid_usb_relay exposes only:
#   set_state(relay, state)   – relay=int 1-indexed, state=bool
#   get_state(relay)          – returns bool
#   toggle_state(relay)       – flip current state
#   set_serial(new_serial)    – rename device
# There is NO turn_on() or turn_off() method.
# =============================================================================

import time
import threading
from abc import ABC, abstractmethod
from typing import Dict, Optional, Set
from utils.logger import get_logger

logger = get_logger("Relay")


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------

class RelayInterface(ABC):
    @abstractmethod
    def connect(self) -> bool: ...
    @abstractmethod
    def activate(self, relay_id: int) -> bool: ...
    @abstractmethod
    def deactivate(self, relay_id: int) -> bool: ...
    @abstractmethod
    def deactivate_all(self) -> bool: ...
    @property
    @abstractmethod
    def is_connected(self) -> bool: ...


# ---------------------------------------------------------------------------
# Simulator  (used when use_usb_relay=False or hardware not found)
# ---------------------------------------------------------------------------

class RelaySimulator(RelayInterface):
    def __init__(self, num_channels: int = 8):
        self._channels  = num_channels
        self._states:   Dict[int, bool] = {}
        self._connected = False

    def connect(self) -> bool:
        self._connected = True
        logger.info(f"RelaySimulator connected ({self._channels} ch)")
        return True

    def activate(self, relay_id: int) -> bool:
        self._states[relay_id] = True
        logger.info(f"[SIM] Relay {relay_id} → ON")
        return True

    def deactivate(self, relay_id: int) -> bool:
        self._states[relay_id] = False
        logger.info(f"[SIM] Relay {relay_id} → OFF")
        return True

    def deactivate_all(self) -> bool:
        for rid in list(self._states):
            self._states[rid] = False
        logger.info("[SIM] All relays OFF")
        return True

    @property
    def is_connected(self) -> bool:
        return self._connected

    def get_state(self, relay_id: int) -> bool:
        return self._states.get(relay_id, False)


# ---------------------------------------------------------------------------
# USB HID relay  –  pyhid_usb_relay  (FIXED API)
# ---------------------------------------------------------------------------

class RelayUSBHID(RelayInterface):
    """
    USB HID relay via pyhid_usb_relay.

    CORRECT API (verified against library source):
        from pyhid_usb_relay import find

        # Single device (no serial filter):
        device = find()                      # returns Controller directly

        # Specific device by serial:
        device = find(serial="AAAAA")        # returns matching Controller

        # Multiple devices:
        devices = find(find_all=True)        # returns list of Controllers

        # Activate / deactivate  (relay_id is 1-indexed int):
        device.set_state(relay_id, True)     # ON
        device.set_state(relay_id, False)    # OFF
        device.get_state(relay_id)           # → bool
        device.toggle_state(relay_id)        # flip
    """

    def __init__(self, num_channels: int = 8, serial: Optional[str] = None):
        self._channels  = num_channels
        self._serial    = serial
        self._device    = None
        self._connected = False
        self._lock      = threading.Lock()

    def connect(self) -> bool:
        with self._lock:
            return self._connect_unlocked()

    def _connect_unlocked(self) -> bool:
        """Open/reopen device handle. Lock must be held by caller."""
        try:
            from pyhid_usb_relay import find, DeviceNotFoundError

            if self._serial:
                # FIX BUG 1: use serial= kwarg, still returns single Controller
                self._device = find(serial=self._serial)
            else:
                # FIX BUG 1: find() returns ONE Controller, not a list
                self._device = find()

            self._connected = True
            logger.info(
                f"USB relay connected – serial={getattr(self._device,'serial','N/A')} "
                f"channels={getattr(self._device,'num_relays', self._channels)}"
            )
            return True

        except ImportError:
            logger.error(
                "pyhid_usb_relay not installed. Run: pip install pyhid-usb-relay"
            )
            self._connected = False
            return False
        except Exception as e:
            logger.error(f"USB relay connect failed: {e}")
            self._connected = False
            self._device = None
            return False

    def activate(self, relay_id: int) -> bool:
        with self._lock:
            try:
                if not self._connected or self._device is None:
                    if not self._connect_unlocked():
                        return False
                # FIX BUG 2: set_state(relay_id, True), NOT turn_on(relay_id)
                self._device.set_state(relay_id, True)
                logger.info(f"Relay {relay_id} → ON")
                return True
            except Exception as e:
                logger.error(f"Relay {relay_id} activate failed: {e}")
                self._connected = False
                return False

    def deactivate(self, relay_id: int) -> bool:
        with self._lock:
            try:
                if not self._connected or self._device is None:
                    return False
                # FIX BUG 2: set_state(relay_id, False), NOT turn_off(relay_id)
                self._device.set_state(relay_id, False)
                logger.info(f"Relay {relay_id} → OFF")
                return True
            except Exception as e:
                logger.error(f"Relay {relay_id} deactivate failed: {e}")
                return False

    def deactivate_all(self) -> bool:
        with self._lock:
            try:
                if self._device:
                    for i in range(1, self._channels + 1):
                        try:
                            self._device.set_state(i, False)
                        except Exception:
                            pass
                return True
            except Exception as e:
                logger.error(f"deactivate_all failed: {e}")
                return False

    def get_channel_state(self, relay_id: int) -> bool:
        """Read current hardware state of one channel."""
        with self._lock:
            if not self._connected or self._device is None:
                return False
            try:
                return bool(self._device.get_state(relay_id))
            except Exception:
                return False

    @property
    def is_connected(self) -> bool:
        return self._connected


# ---------------------------------------------------------------------------
# RelayManager  –  cooldown + auto-reset + retry
# ---------------------------------------------------------------------------

class RelayManager:
    def __init__(
        self,
        interface:           Optional[RelayInterface] = None,
        cooldown:             float = 5.0,
        activation_duration:  float = 1.0,
        max_retries:          int   = 3,
    ):
        self.interface           = interface or RelaySimulator()
        self.cooldown            = cooldown
        self.activation_duration = activation_duration
        self.max_retries         = max_retries
        self._active_relays:     Set[int] = set()
        self._last_trigger:      Dict[int, float] = {}
        self._lock               = threading.Lock()
        self._reset_timers:      Dict[int, threading.Timer] = {}
        self.interface.connect()

    def trigger(self, relay_id: int) -> bool:
        with self._lock:
            now  = time.time()
            last = self._last_trigger.get(relay_id, 0.0)
            if now - last < self.cooldown:
                logger.debug(
                    f"Relay {relay_id} in cooldown "
                    f"({self.cooldown - (now - last):.1f}s remaining)"
                )
                return False
            self._last_trigger[relay_id] = now

        ok = self._activate_with_retry(relay_id)
        if ok:
            with self._lock:
                self._active_relays.add(relay_id)
            t = threading.Timer(self.activation_duration,
                                self._auto_reset, args=(relay_id,))
            t.daemon = True
            with self._lock:
                old = self._reset_timers.pop(relay_id, None)
                if old:
                    old.cancel()
                self._reset_timers[relay_id] = t
            t.start()
        return ok

    def _activate_with_retry(self, relay_id: int) -> bool:
        for attempt in range(self.max_retries):
            try:
                if self.interface.activate(relay_id):
                    return True
            except Exception as e:
                logger.warning(
                    f"Relay {relay_id} attempt {attempt + 1} failed: {e}"
                )
                if attempt < self.max_retries - 1:
                    time.sleep(0.2 * (attempt + 1))
                    self.interface.connect()
        logger.error(f"Relay {relay_id} failed after {self.max_retries} attempts")
        return False

    def _auto_reset(self, relay_id: int) -> None:
        try:
            self.interface.deactivate(relay_id)
            with self._lock:
                self._active_relays.discard(relay_id)
                self._reset_timers.pop(relay_id, None)
            logger.info(f"Relay {relay_id} auto-reset (OFF)")
        except Exception as e:
            logger.error(f"Relay {relay_id} auto-reset failed: {e}")

    def reset_all(self) -> None:
        try:
            self.interface.deactivate_all()
            with self._lock:
                self._active_relays.clear()
        except Exception as e:
            logger.error(f"reset_all failed: {e}")

    def is_active(self, relay_id: int) -> bool:
        with self._lock:
            return relay_id in self._active_relays

    def get_active_relays(self) -> Set[int]:
        with self._lock:
            return self._active_relays.copy()

    def reinitialize(self) -> bool:
        try:
            self.interface.connect()
            logger.info("Relay reinitialized")
            return True
        except Exception as e:
            logger.error(f"Relay reinitialize failed: {e}")
            return False
