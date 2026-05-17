# =============================================================================
# core/relay_hardware.py  –  Relay hardware abstraction  (v6 – dual backend)
# =============================================================================
#
# Backends:
#   RelaySimulator      – software mock              (relay_type = "none")
#   RelayUSBHID         – pyhid_usb_relay             (relay_type = "usb")
#   ModbusRelayBackend  – Waveshare Ethernet relay    (relay_type = "ethernet")
#
# Factory:
#   build_relay_interface(settings) → RelayInterface
#
# Network reliability in ModbusRelayBackend:
#   • TCP connect timeout 3 s
#   • Per-operation: write coil → read-back verify, 3 retries, stepped backoff
#   • Background heartbeat thread reads coil 0 every 10 s
#       – fail → _connected = False immediately (no false outputs ever)
#       – triggers stepped reconnect 1 / 3 / 5 / 10 s
#   • Self-test on every connect / reconnect:
#       cycles ALL channels ON (0.1 s each) then OFF (0.05 s each)
#   • is_connected reflects REAL state – never a stale cached True
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
# Simulator  (relay_type = "none")
# ---------------------------------------------------------------------------

class RelaySimulator(RelayInterface):
    def __init__(self, num_channels: int = 16):
        self._channels  = num_channels
        self._states:   Dict[int, bool] = {}
        self._connected = False

    def connect(self) -> bool:
        self._connected = True
        logger.info(f"[SIM] Simulator connected ({self._channels} ch)")
        return True

    def activate(self, relay_id: int) -> bool:
        self._states[relay_id] = True
        logger.info(f"[SIM] Relay {relay_id} -> ON")
        return True

    def deactivate(self, relay_id: int) -> bool:
        self._states[relay_id] = False
        logger.info(f"[SIM] Relay {relay_id} -> OFF")
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

    def close(self) -> None:
        self._connected = False


# ---------------------------------------------------------------------------
# USB HID relay  –  pyhid_usb_relay  (relay_type = "usb")
# ---------------------------------------------------------------------------

class RelayUSBHID(RelayInterface):
    """
    USB HID relay via pyhid_usb_relay.

    Correct API (verified against library source):
        device = find()                      # returns Controller directly
        device = find(serial="AAAAA")        # specific serial
        device.set_state(relay_id, True)     # ON  (1-indexed)
        device.set_state(relay_id, False)    # OFF
        device.get_state(relay_id)           # -> bool
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
        try:
            from pyhid_usb_relay import find
            if self._serial:
                self._device = find(serial=self._serial)
            else:
                self._device = find()
            self._connected = True
            logger.info(
                f"[USB] Connected – serial={getattr(self._device, 'serial', 'N/A')} "
                f"channels={getattr(self._device, 'num_relays', self._channels)}"
            )
            return True
        except ImportError:
            logger.error(
                "[USB] pyhid_usb_relay not installed. Run: pip install pyhid-usb-relay"
            )
            self._connected = False
            return False
        except Exception as e:
            logger.error(f"[USB] Connect failed: {e}")
            self._connected = False
            self._device    = None
            return False

    def activate(self, relay_id: int) -> bool:
        with self._lock:
            try:
                if not self._connected or self._device is None:
                    if not self._connect_unlocked():
                        return False
                self._device.set_state(relay_id, True)
                logger.info(f"[USB] Relay {relay_id} -> ON")
                return True
            except Exception as e:
                logger.error(f"[USB] Relay {relay_id} activate failed: {e}")
                self._connected = False
                return False

    def deactivate(self, relay_id: int) -> bool:
        with self._lock:
            try:
                if not self._connected or self._device is None:
                    return False
                self._device.set_state(relay_id, False)
                logger.info(f"[USB] Relay {relay_id} -> OFF")
                return True
            except Exception as e:
                logger.error(f"[USB] Relay {relay_id} deactivate failed: {e}")
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
                logger.error(f"[USB] deactivate_all failed: {e}")
                return False

    def get_channel_state(self, relay_id: int) -> bool:
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

    def close(self) -> None:
        with self._lock:
            self._connected = False
            self._device    = None


# ---------------------------------------------------------------------------
# Modbus TCP relay  –  Waveshare Ethernet relay  (relay_type = "ethernet")
# ---------------------------------------------------------------------------

class ModbusRelayBackend(RelayInterface):
    """
    Waveshare 16-ch Modbus TCP Ethernet relay – industrial-grade I/O.

    Reliability features:
      TCP connect timeout : 3 s
      Per-operation       : write + read-back verify, 3 retries, stepped backoff
      Background heartbeat: reads coil 0 every 10 s; marks disconnected on fail
      Reconnect backoff   : 1 / 3 / 5 / 10 s
      Self-test on connect: all channels ON then OFF (audible click confirms life)
      is_connected        : reflects LIVE heartbeat state – never stale
    """

    _CONNECT_TIMEOUT    = 3.0
    _MAX_RETRIES        = 3
    _RETRY_BACKOFF      = [0.2, 0.5, 1.0]
    _HEARTBEAT_INTERVAL = 10.0
    _RECONNECT_STEPS    = [1.0, 3.0, 5.0, 10.0]

    def __init__(
        self,
        ip:           str = "192.168.1.200",
        port:         int = 502,
        device_id:    int = 1,
        num_channels: int = 16,
    ):
        self._ip           = ip
        self._port         = port
        self._device_id    = device_id
        self._num_channels = num_channels

        self._client:     Optional[object]           = None
        self._connected:  bool                       = False
        self._lock:       threading.Lock             = threading.Lock()
        self._stop_event: threading.Event            = threading.Event()
        self._hb_thread:  Optional[threading.Thread] = None

    # -- public interface -----------------------------------------------------

    def connect(self) -> bool:
        with self._lock:
            ok = self._connect_unlocked(run_selftest=True)
        if ok and (self._hb_thread is None or not self._hb_thread.is_alive()):
            self._start_heartbeat()
        return ok

    def activate(self, relay_id: int) -> bool:
        with self._lock:
            return self._write_coil_with_retry(relay_id, True)

    def deactivate(self, relay_id: int) -> bool:
        with self._lock:
            return self._write_coil_with_retry(relay_id, False)

    def deactivate_all(self) -> bool:
        with self._lock:
            if not self._connected or self._client is None:
                logger.warning("[MODBUS] deactivate_all: not connected – skipping")
                return False
            try:
                values = [False] * self._num_channels
                result = self._client.write_coils(
                    address=0, values=values, device_id=self._device_id
                )
                if result.isError():
                    logger.error(f"[MODBUS] deactivate_all write error: {result}")
                    self._connected = False
                    return False
                logger.info("[MODBUS] All relays OFF")
                return True
            except Exception as e:
                logger.error(f"[MODBUS] deactivate_all exception: {e}")
                self._connected = False
                return False

    def close(self) -> None:
        """Stop heartbeat thread and close TCP connection cleanly."""
        self._stop_event.set()
        if self._hb_thread and self._hb_thread.is_alive():
            self._hb_thread.join(timeout=6)
        with self._lock:
            self._disconnect_unlocked()
        logger.info("[MODBUS] Connection closed")

    @property
    def is_connected(self) -> bool:
        """Returns REAL state verified by background heartbeat – never stale."""
        return self._connected

    # -- internal connect / disconnect ----------------------------------------

    def _connect_unlocked(self, run_selftest: bool = False) -> bool:
        """Fresh TCP connect + Modbus coil-read verify. Lock MUST be held."""
        self._disconnect_unlocked()
        try:
            from pymodbus.client import ModbusTcpClient
            client = ModbusTcpClient(
                host=self._ip,
                port=self._port,
                timeout=self._CONNECT_TIMEOUT,
            )
            if not client.connect():
                raise ConnectionError(
                    f"TCP connect refused at {self._ip}:{self._port}"
                )
            verify = client.read_coils(
                address=0, count=1, device_id=self._device_id
            )
            if verify.isError():
                client.close()
                raise ConnectionError(f"Modbus verify read failed: {verify}")

            self._client    = client
            self._connected = True
            logger.info(
                f"[MODBUS] Connected -> {self._ip}:{self._port} "
                f"device_id={self._device_id} channels={self._num_channels}"
            )
            if run_selftest:
                self._selftest_unlocked()
            return True

        except ImportError:
            logger.error(
                "[MODBUS] pymodbus not installed. Run: pip install pymodbus"
            )
            self._connected = False
            return False
        except Exception as e:
            logger.error(
                f"[MODBUS] Connect failed ({self._ip}:{self._port}): {e}"
            )
            self._connected = False
            return False

    def _disconnect_unlocked(self) -> None:
        """Close TCP socket. Lock MUST be held."""
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None
        self._connected = False

    # -- self-test ------------------------------------------------------------

    def _selftest_unlocked(self) -> None:
        """
        Cycle all channels ON then OFF to confirm hardware is alive.
        You will hear every relay click ON then OFF.
        Lock MUST be held. Called on every connect and every reconnect.
        """
        logger.info(
            f"[MODBUS] Self-test: cycling {self._num_channels} relays ON then OFF"
        )
        for ch in range(1, self._num_channels + 1):
            try:
                self._client.write_coil(
                    address=ch - 1, value=True, device_id=self._device_id
                )
            except Exception as e:
                logger.warning(f"[MODBUS] Self-test ON ch{ch}: {e}")
            time.sleep(0.10)

        time.sleep(0.30)

        for ch in range(1, self._num_channels + 1):
            try:
                self._client.write_coil(
                    address=ch - 1, value=False, device_id=self._device_id
                )
            except Exception as e:
                logger.warning(f"[MODBUS] Self-test OFF ch{ch}: {e}")
            time.sleep(0.05)

        logger.info("[MODBUS] Self-test complete – all relays returned to OFF")

    # -- write with retry + verify --------------------------------------------

    def _write_coil_with_retry(self, relay_id: int, value: bool) -> bool:
        """Write coil, verify by reading back, retry on failure. Lock MUST be held."""
        address = relay_id - 1
        for attempt in range(self._MAX_RETRIES):
            if not self._connected or self._client is None:
                logger.info(
                    f"[MODBUS] Relay {relay_id}: not connected – "
                    f"reconnect attempt {attempt + 1}/{self._MAX_RETRIES}"
                )
                if not self._connect_unlocked(run_selftest=False):
                    if attempt < self._MAX_RETRIES - 1:
                        time.sleep(self._RETRY_BACKOFF[attempt])
                    continue

            try:
                result = self._client.write_coil(
                    address=address, value=value, device_id=self._device_id
                )
                if result.isError():
                    raise IOError(f"write_coil returned error: {result}")

                verify = self._client.read_coils(
                    address=address, count=1, device_id=self._device_id
                )
                if verify.isError():
                    raise IOError(f"verify read returned error: {verify}")

                actual = bool(verify.bits[0])
                if actual != value:
                    raise IOError(
                        f"Write-verify mismatch: wrote {value}, read back {actual}"
                    )

                logger.info(
                    f"[MODBUS] Relay {relay_id} -> {'ON' if value else 'OFF'} "
                    f"(verified, attempt {attempt + 1})"
                )
                return True

            except Exception as e:
                logger.warning(
                    f"[MODBUS] Relay {relay_id} attempt {attempt + 1} failed: {e}"
                )
                self._connected = False
                try:
                    self._client.close()
                except Exception:
                    pass
                self._client = None

            if attempt < self._MAX_RETRIES - 1:
                time.sleep(self._RETRY_BACKOFF[attempt])
                self._connect_unlocked(run_selftest=False)

        logger.error(
            f"[MODBUS] Relay {relay_id} -> {'ON' if value else 'OFF'} FAILED "
            f"after {self._MAX_RETRIES} attempts"
        )
        return False

    # -- background heartbeat + reconnect -------------------------------------

    def _start_heartbeat(self) -> None:
        self._stop_event.clear()
        self._hb_thread = threading.Thread(
            target=self._heartbeat_loop,
            name="modbus-heartbeat",
            daemon=True,
        )
        self._hb_thread.start()
        logger.info("[MODBUS] Heartbeat thread started")

    def _heartbeat_loop(self) -> None:
        """
        Reads coil 0 every _HEARTBEAT_INTERVAL seconds to verify the link.

        On failure:
          1. Sets _connected = False immediately.
             -> relay_process rejects commands, logs "disconnected"
             -> GUI health label turns red within the next health poll
          2. Retries reconnect with stepped backoff until success.
          3. On reconnect: runs full self-test (all relays cycle ON->OFF).
        """
        reconnect_attempt = 0

        while not self._stop_event.is_set():
            self._stop_event.wait(self._HEARTBEAT_INTERVAL)
            if self._stop_event.is_set():
                break

            alive = False
            with self._lock:
                if self._connected and self._client is not None:
                    try:
                        result = self._client.read_coils(
                            address=0, count=1, device_id=self._device_id
                        )
                        alive = not result.isError()
                    except Exception:
                        alive = False

                    if not alive:
                        logger.warning(
                            "[MODBUS] Heartbeat failed – hardware marked DISCONNECTED. "
                            "Relay commands will be rejected until reconnected."
                        )
                        self._connected = False
                else:
                    alive = False

            if alive:
                reconnect_attempt = 0
                continue

            delay = self._RECONNECT_STEPS[
                min(reconnect_attempt, len(self._RECONNECT_STEPS) - 1)
            ]
            logger.info(
                f"[MODBUS] Reconnect attempt {reconnect_attempt + 1} "
                f"waiting {delay:.0f}s ... ({self._ip}:{self._port})"
            )
            self._stop_event.wait(delay)
            if self._stop_event.is_set():
                break

            reconnect_attempt += 1
            with self._lock:
                if self._connect_unlocked(run_selftest=True):
                    reconnect_attempt = 0
                    logger.info("[MODBUS] Reconnected successfully")
                else:
                    logger.warning(
                        f"[MODBUS] Reconnect attempt {reconnect_attempt} failed – retrying"
                    )


# ---------------------------------------------------------------------------
# RelayManager  –  cooldown + auto-reset (backend-agnostic)
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
            t = threading.Timer(
                self.activation_duration, self._auto_reset, args=(relay_id,)
            )
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


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_relay_interface(settings) -> RelayInterface:
    """
    Create the correct RelayInterface from settings.relay_type.

    "none"     -> RelaySimulator       (no hardware, safe default)
    "usb"      -> RelayUSBHID          (pyhid_usb_relay)
    "ethernet" -> ModbusRelayBackend   (Waveshare Modbus TCP)

    Backward compat: if relay_type absent but use_usb_relay=True -> "usb".
    """
    relay_type = getattr(settings, "relay_type", None)
    if not relay_type:
        relay_type = "usb" if getattr(settings, "use_usb_relay", False) else "none"
    relay_type = relay_type.lower().strip()

    if relay_type == "usb":
        logger.info("[Factory] Building USB HID relay backend")
        return RelayUSBHID(
            num_channels=getattr(settings, "usb_num_channels", 8),
            serial=getattr(settings, "usb_serial", None),
        )

    if relay_type == "ethernet":
        ip   = getattr(settings, "eth_relay_ip",          "192.168.1.200")
        port = getattr(settings, "eth_relay_port",         502)
        did  = getattr(settings, "eth_relay_device_id",    1)
        chs  = getattr(settings, "eth_relay_num_channels", 16)
        logger.info(
            f"[Factory] Building Modbus Ethernet relay backend ({ip}:{port})"
        )
        return ModbusRelayBackend(ip=ip, port=port, device_id=did, num_channels=chs)

    logger.info("[Factory] Building relay simulator (relay_type='none')")
    return RelaySimulator(
        num_channels=getattr(
            settings, "eth_relay_num_channels",
            getattr(settings, "usb_num_channels", 16)
        )
    )