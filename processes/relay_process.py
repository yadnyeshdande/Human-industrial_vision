# =============================================================================
# processes/relay_process.py  –  Relay control process  (v3 – dual backend)
# =============================================================================
#
# Changes from v2:
#   • Reads SETTINGS.relay_type to select backend (none / usb / ethernet)
#   • CTRL_RELOAD_SETTINGS: hot-swaps backend if relay_type changed
#   • ModbusRelayBackend heartbeat tracks REAL connection state
#   • Accurate health reporting: MSG_RELAY_HEALTH -> status_q -> GUI
#     (never reports hw_connected=True when cable is unplugged)
#   • relay_process is the SOLE authority for all relay outputs
#   • GUI never touches relay hardware directly
#
# FUNCTION SIGNATURE CHANGED FROM v2:
#   OLD: run_relay_process(hb_q, ctrl_q, relay_q, status_q,
#                          use_usb_relay, usb_num_channels, usb_serial,
#                          activation_duration, cooldown, ram_limit_mb)
#   NEW: run_relay_process(hb_q, ctrl_q, relay_q, status_q, ram_limit_mb)
#        All relay config is read from SETTINGS directly at startup
#        and on every CTRL_RELOAD_SETTINGS.
#        supervisor.py MUST be updated to match (see change #7).
# =============================================================================

import os
import sys
import time
from multiprocessing import Queue

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.logger import setup_process_logger, get_logger
from utils.resource_guard import ResourceGuard, ResourceLimitExceeded, RAM_LIMIT_RELAY
from ipc.messages import (
    make_heartbeat, make_error, make_relay_status, make_relay_health,
    MSG_RELAY_COMMAND, MSG_RELAY_HEALTH,
    MSG_SHUTDOWN, MSG_CONTROL,
    CTRL_SHUTDOWN, CTRL_RELOAD_SETTINGS,
)

HEARTBEAT_INTERVAL = 5.0    # s between supervisor heartbeats
HEALTH_INTERVAL    = 10.0   # s between GUI health pushes
QUEUE_TIMEOUT      = 1.0    # relay_q.get() timeout


def run_relay_process(
    heartbeat_q:  Queue,
    control_q:    Queue,
    relay_q:      Queue,
    status_q:     Queue,
    ram_limit_mb: float = RAM_LIMIT_RELAY,
) -> None:
    """
    Single-process relay authority.

    Reads SETTINGS at startup to build the correct backend.
    Reacts to CTRL_RELOAD_SETTINGS to hot-swap backends at runtime.
    Reports accurate health (MSG_RELAY_HEALTH) to status_q.
    """
    pname = "relay"
    setup_process_logger(pname)
    log = get_logger("Main")
    log.info(f"Relay process started  PID={os.getpid()}")

    guard = ResourceGuard(ram_limit_mb=ram_limit_mb)

    # Initial backend construction
    from config.loader import SETTINGS
    SETTINGS.load()

    interface, hw_type = _build_interface(SETTINGS, log)

    from core.relay_hardware import RelayManager
    manager = RelayManager(
        interface=interface,
        cooldown=SETTINGS.relay_cooldown,
        activation_duration=SETTINGS.relay_duration,
    )

    last_hb     = 0.0
    last_health = 0.0
    active_type = hw_type

    try:
        while True:
            # Control queue
            try:
                ctrl  = control_q.get_nowait()
                mtype = ctrl.get("type", "")
                cmd   = ctrl.get("payload", {}).get("command", "")

                if mtype == MSG_SHUTDOWN or cmd == CTRL_SHUTDOWN:
                    log.info("Shutdown – resetting relays and exiting")
                    manager.reset_all()
                    _close_interface(interface, hw_type, log)
                    break

                if cmd == CTRL_RELOAD_SETTINGS:
                    log.info("CTRL_RELOAD_SETTINGS received")
                    SETTINGS.load()
                    manager.cooldown            = SETTINGS.relay_cooldown
                    manager.activation_duration = SETTINGS.relay_duration

                    new_type = _resolve_type(SETTINGS)
                    if new_type != active_type:
                        log.info(
                            f"Relay type changed: {active_type} -> {new_type} "
                            "– hot-swapping backend"
                        )
                        manager.reset_all()
                        _close_interface(interface, active_type, log)
                        interface, hw_type = _build_interface(SETTINGS, log)
                        manager.interface  = interface
                        active_type        = hw_type
                        _publish_health(status_q, interface, hw_type, pname)

                    log.info(
                        f"Settings reloaded: relay_type={SETTINGS.relay_type} "
                        f"cooldown={SETTINGS.relay_cooldown}s "
                        f"duration={SETTINGS.relay_duration}s"
                    )

            except Exception:
                pass

            # Resource guard
            try:
                guard.check()
            except ResourceLimitExceeded as e:
                log.error(f"Resource limit: {e}")
                heartbeat_q.put_nowait(make_error(pname, str(e), fatal=True))
                sys.exit(2)

            # Supervisor heartbeat
            now = time.monotonic()
            if now - last_hb >= HEARTBEAT_INTERVAL:
                last_hb = now
                try:
                    heartbeat_q.put_nowait(
                        make_heartbeat(
                            source=pname,
                            ram_mb=guard.get_ram_mb(),
                            extra={
                                "active_relays": list(manager.get_active_relays()),
                                "hw_connected":  interface.is_connected,
                                "hw_type":       hw_type,
                            },
                        )
                    )
                except Exception:
                    pass

            # GUI health push
            if now - last_health >= HEALTH_INTERVAL:
                last_health = now
                _publish_health(status_q, interface, hw_type, pname)

            # Relay commands
            try:
                msg = relay_q.get(timeout=QUEUE_TIMEOUT)
            except Exception:
                continue

            if msg.get("type") != MSG_RELAY_COMMAND:
                continue

            payload  = msg.get("payload", {})
            relay_id = payload.get("relay_id", 1)
            action   = payload.get("action", "trigger")
            cam_id   = msg.get("camera_id")
            zone_id  = payload.get("zone_id")

            if action == "trigger":
                # Guard: reject command if hardware is known disconnected
                if not interface.is_connected and hw_type != "none":
                    log.warning(
                        f"Relay {relay_id} SKIPPED – {hw_type} hardware disconnected "
                        f"(cam={cam_id} zone={zone_id})"
                    )
                    _publish_health(
                        status_q, interface, hw_type, pname,
                        error=f"{hw_type} relay disconnected"
                    )
                    continue

                ok = manager.trigger(relay_id)
                log.info(
                    f"Relay {relay_id} {'TRIGGERED' if ok else 'COOLDOWN/FAILED'} "
                    f"cam={cam_id} zone={zone_id}"
                )
                try:
                    status_q.put_nowait(
                        make_relay_status(
                            source=pname,
                            relay_id=relay_id,
                            state=ok,
                            reason="triggered" if ok else "cooldown",
                        )
                    )
                except Exception:
                    pass

            elif action == "reset":
                manager.reset_all()

    except KeyboardInterrupt:
        pass
    except Exception as e:
        log.error(f"Relay process fatal: {e}", exc_info=True)
        try:
            heartbeat_q.put_nowait(make_error(pname, str(e), fatal=True))
        except Exception:
            pass
        sys.exit(1)
    finally:
        try:
            manager.reset_all()
        except Exception:
            pass
        _close_interface(interface, hw_type, log)
        log.info("Relay process exiting")


# =============================================================================
# Helpers
# =============================================================================

def _resolve_type(settings) -> str:
    t = getattr(settings, "relay_type", None)
    if not t:
        t = "usb" if getattr(settings, "use_usb_relay", False) else "none"
    return t.lower().strip()


def _build_interface(settings, log):
    """Build RelayInterface from settings. Returns (interface, hw_type_str)."""
    from core.relay_hardware import build_relay_interface
    hw_type   = _resolve_type(settings)
    interface = build_relay_interface(settings)
    log.info(f"[Relay] Initialising backend: {hw_type}")
    return interface, hw_type


def _close_interface(interface, hw_type: str, log) -> None:
    """Cleanly shut down the backend (stops heartbeat thread for Ethernet)."""
    try:
        if hasattr(interface, "close"):
            interface.close()
        elif hasattr(interface, "deactivate_all"):
            interface.deactivate_all()
        log.info(f"[Relay] Backend '{hw_type}' closed")
    except Exception as e:
        log.warning(f"[Relay] Backend close error: {e}")


def _publish_health(status_q: Queue, interface, hw_type: str,
                    source: str, error: str = "") -> None:
    """Push MSG_RELAY_HEALTH to status_q so GUI shows accurate real-time state."""
    try:
        channels = getattr(interface, "_num_channels",
                           getattr(interface, "_channels", 0))
        status_q.put_nowait(
            make_relay_health(
                source=source,
                hw_type=hw_type,
                connected=interface.is_connected,
                channels=channels,
                error=error,
            )
        )
    except Exception:
        pass