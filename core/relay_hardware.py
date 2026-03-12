# =============================================================================
# core/relay_hardware.py  –  Relay hardware abstraction  (v4)
# =============================================================================
# FIX #7: RelayUSBHID now uses pyhid_usb_relay correctly:
#           from pyhid_usb_relay import find
#           relays = find()
#           relay  = relays[0]           (or filter by serial)
#           relay.turn_on(channel)
#           relay.turn_off(channel)
#         Removed incorrect HIDRelay import that caused:
#           ImportError: cannot import name 'HIDRelay'
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
# Simulator
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
# USB HID relay  (pyhid_usb_relay)  –  FIX #7
# ---------------------------------------------------------------------------

class RelayUSBHID(RelayInterface):
    """
    USB HID relay via the pyhid_usb_relay library.

    FIX #7 – correct API:
        from pyhid_usb_relay import find
        relays = find()           # returns list of Relay objects
        device = relays[0]        # first device (or filter by serial)
        device.turn_on(channel)   # 1-indexed channel number
        device.turn_off(channel)

    The old HIDRelay class does not exist in pyhid_usb_relay.
    """

    def __init__(self, num_channels: int = 8, serial: Optional[str] = None):
        self._channels  = num_channels
        self._serial    = serial
        self._device    = None
        self._connected = False
        self._lock      = threading.Lock()

    def connect(self) -> bool:
        with self._lock:
            try:
                from pyhid_usb_relay import find   # FIX #7: correct import
                devices = find()
                if not devices:
                    logger.error("No USB relay devices found")
                    self._connected = False
                    return False

                if self._serial:
                    # Filter by serial if specified
                    matched = [d for d in devices
                               if hasattr(d, "serial") and d.serial == self._serial]
                    if not matched:
                        logger.error(
                            f"USB relay with serial '{self._serial}' not found. "
                            f"Available: {[getattr(d,'serial','?') for d in devices]}"
                        )
                        self._connected = False
                        return False
                    self._device = matched[0]
                else:
                    self._device = devices[0]   # use first available device

                self._connected = True
                logger.info(
                    f"USB relay connected "
                    f"(serial={getattr(self._device,'serial','N/A')}, "
                    f"channels={self._channels})"
                )
                return True

            except ImportError:
                logger.error(
                    "pyhid_usb_relay not installed. "
                    "Run: pip install pyhid-usb-relay"
                )
                self._connected = False
                return False
            except Exception as e:
                logger.error(f"USB relay connect failed: {e}")
                self._connected = False
                return False

    def activate(self, relay_id: int) -> bool:
        with self._lock:
            try:
                if not self._connected or self._device is None:
                    if not self._connect_unlocked():
                        return False
                self._device.turn_on(relay_id)   # FIX #7: turn_on(channel)
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
                self._device.turn_off(relay_id)   # FIX #7: turn_off(channel)
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
                            self._device.turn_off(i)   # FIX #7
                        except Exception:
                            pass
                return True
            except Exception as e:
                logger.error(f"deactivate_all failed: {e}")
                return False

    def _connect_unlocked(self) -> bool:
        """Connect without acquiring lock (already held by caller)."""
        try:
            from pyhid_usb_relay import find
            devices = find()
            if not devices:
                return False
            if self._serial:
                matched = [d for d in devices
                           if hasattr(d, "serial") and d.serial == self._serial]
                self._device = matched[0] if matched else devices[0]
            else:
                self._device = devices[0]
            self._connected = True
            return True
        except Exception as e:
            logger.error(f"Relay reconnect failed: {e}")
            self._connected = False
            return False

    @property
    def is_connected(self) -> bool:
        return self._connected


# ---------------------------------------------------------------------------
# RelayManager  –  cooldown + auto-reset + retry (unchanged)
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
            logger.info(f"Relay {relay_id} auto-reset")
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
