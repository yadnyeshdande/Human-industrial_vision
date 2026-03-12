# =============================================================================
# processes/relay_process.py  –  USB relay control process  (v2 – patched)
# =============================================================================
# FIX #1:  RAM limit raised to 512 MB (Python runtime ~310 MB baseline)
# FIX #8:  Handles CTRL_RELOAD_SETTINGS → re-reads cooldown/duration at runtime
# FIX #13: setup_process_logger("relay") → logs/relay.log
# =============================================================================

import os
import sys
import time
from multiprocessing import Queue
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.logger import setup_process_logger, get_logger
from utils.resource_guard import ResourceGuard, ResourceLimitExceeded, RAM_LIMIT_RELAY
from ipc.messages import (
    make_heartbeat, make_error, make_relay_status,
    MSG_RELAY_COMMAND, MSG_SHUTDOWN, MSG_CONTROL,
    CTRL_SHUTDOWN, CTRL_RELOAD_SETTINGS,
)

HEARTBEAT_INTERVAL = 5.0
QUEUE_TIMEOUT      = 1.0


def run_relay_process(
    heartbeat_q:         Queue,
    control_q:           Queue,
    relay_q:             Queue,
    status_q:            Queue,
    use_usb_relay:       bool  = False,
    usb_num_channels:    int   = 8,
    usb_serial:          Optional[str] = None,
    activation_duration: float = 1.0,
    cooldown:            float = 5.0,
    ram_limit_mb:        float = RAM_LIMIT_RELAY,   # FIX #1: 512 MB
) -> None:
    pname = "relay"
    setup_process_logger(pname)   # FIX #13
    log = get_logger("Main")
    log.info(f"Relay process started  PID={os.getpid()}")

    guard = ResourceGuard(ram_limit_mb=ram_limit_mb)

    interface = None
    if use_usb_relay:
        try:
            from core.relay_hardware import RelayUSBHID
            interface = RelayUSBHID(num_channels=usb_num_channels,
                                    serial=usb_serial)
            log.info("USB HID relay hardware initialised")
        except Exception as e:
            log.error(f"USB relay init failed: {e} – falling back to simulator")

    if interface is None:
        from core.relay_hardware import RelaySimulator
        interface = RelaySimulator()
        log.info("Relay simulator active")

    from core.relay_hardware import RelayManager
    manager = RelayManager(
        interface=interface,
        cooldown=cooldown,
        activation_duration=activation_duration,
    )

    last_hb = 0.0

    try:
        while True:
            # ── control messages ─────────────────────────────────────────────
            try:
                ctrl = control_q.get_nowait()
                mtype = ctrl.get("type", "")
                cmd   = ctrl.get("payload", {}).get("command", "")

                if mtype == MSG_SHUTDOWN or cmd == CTRL_SHUTDOWN:
                    log.info("Shutdown – resetting relays and exiting")
                    manager.reset_all()
                    break

                # FIX #8: hot-reload cooldown/duration from saved settings
                if cmd == CTRL_RELOAD_SETTINGS:
                    log.info("Reloading settings")
                    try:
                        from config.loader import SETTINGS
                        SETTINGS.load()
                        manager.cooldown            = SETTINGS.relay_cooldown
                        manager.activation_duration = SETTINGS.relay_duration
                        log.info(
                            f"Relay updated: cooldown={SETTINGS.relay_cooldown}s "
                            f"duration={SETTINGS.relay_duration}s"
                        )
                    except Exception as e:
                        log.warning(f"Settings reload failed: {e}")

            except Exception:
                pass

            # ── resource guard ───────────────────────────────────────────────
            try:
                guard.check()
            except ResourceLimitExceeded as e:
                log.error(f"Resource limit: {e}")
                heartbeat_q.put_nowait(make_error(pname, str(e), fatal=True))
                sys.exit(2)

            # ── heartbeat ────────────────────────────────────────────────────
            now = time.monotonic()
            if now - last_hb >= HEARTBEAT_INTERVAL:
                last_hb = now
                heartbeat_q.put_nowait(
                    make_heartbeat(
                        source=pname,
                        ram_mb=guard.get_ram_mb(),
                        extra={
                            "active_relays": list(manager.get_active_relays()),
                            "hw_connected":  interface.is_connected,
                        },
                    )
                )

            # ── relay commands ───────────────────────────────────────────────
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
                ok = manager.trigger(relay_id)
                log.info(
                    f"Relay {relay_id} {'OK' if ok else 'COOLDOWN'} "
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
        heartbeat_q.put_nowait(make_error(pname, str(e), fatal=True))
        sys.exit(1)
    finally:
        try:
            manager.reset_all()
        except Exception:
            pass
        log.info("Relay process exiting")
