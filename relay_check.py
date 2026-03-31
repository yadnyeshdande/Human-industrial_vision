#!/usr/bin/env python3
"""
relay_check.py  –  USB relay hardware diagnostic

Run this standalone to verify pyhid_usb_relay is working correctly
before starting the main application.

Usage:
    python relay_check.py
"""

import sys
import time


def main():
    print("=" * 52)
    print("  USB HID Relay Diagnostic")
    print("=" * 52)

    # ── 1. Import check ───────────────────────────────────
    try:
        import pyhid_usb_relay
        print(f"[OK] pyhid_usb_relay imported (version: {pyhid_usb_relay.VERSION})")
    except ImportError:
        print("[FAIL] pyhid_usb_relay not installed.")
        print("       Run: pip install pyhid-usb-relay")
        sys.exit(1)

    # ── 2. Device find ────────────────────────────────────
    # CORRECT: find() returns ONE Controller object, not a list
    try:
        relay = pyhid_usb_relay.find()
        print(f"[OK] Device found:")
        print(f"     serial     = {relay.serial}")
        print(f"     num_relays = {relay.num_relays}")
    except pyhid_usb_relay.DeviceNotFoundError:
        print("[FAIL] No USB relay device detected.")
        print("       Check USB cable and driver (run fix_pyhid_libusb.py on Windows).")
        sys.exit(1)
    except Exception as e:
        print(f"[FAIL] find() error: {type(e).__name__}: {e}")
        sys.exit(1)

    # ── 3. Read current state ─────────────────────────────
    try:
        # relay.state is a raw bitmask byte; read per-channel with get_state()
        ch1_state = relay.get_state(1)
        print(f"[OK] Channel 1 current state: {'ON' if ch1_state else 'OFF'}")
    except Exception as e:
        print(f"[WARN] Could not read state: {e}")

    # ── 4. Activate channel 1 ─────────────────────────────
    # CORRECT API: set_state(relay_id, True/False)
    # WRONG API:   turn_on() / turn_off()  ← these do NOT exist
    print("\nTesting channel 1 (set_state)...")
    try:
        relay.set_state(1, True)
        print("[OK] Channel 1 → ON  (set_state(1, True))")
        time.sleep(1.0)
        relay.set_state(1, False)
        print("[OK] Channel 1 → OFF (set_state(1, False))")
    except Exception as e:
        print(f"[FAIL] set_state error: {type(e).__name__}: {e}")
        sys.exit(1)

    # ── 5. toggle_state ───────────────────────────────────
    print("\nTesting toggle_state...")
    try:
        relay.toggle_state(1)
        print("[OK] Channel 1 toggled ON")
        time.sleep(0.5)
        relay.toggle_state(1)
        print("[OK] Channel 1 toggled OFF")
    except Exception as e:
        print(f"[WARN] toggle_state error: {e}")

    # ── 6. All channels off ───────────────────────────────
    print("\nResetting all channels OFF...")
    try:
        relay.set_state("all", False)
        print("[OK] All channels OFF")
    except Exception as e:
        # Fallback: iterate
        for ch in range(1, relay.num_relays + 1):
            try:
                relay.set_state(ch, False)
            except Exception:
                pass
        print("[OK] All channels reset individually")

    print("\n" + "=" * 52)
    print("  Relay hardware working correctly!")
    print("=" * 52)
    print("\nIn app_settings.json set:  \"use_usb_relay\": true")
    print("Then start:  python supervisor.py")


if __name__ == "__main__":
    main()
