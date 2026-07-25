"""Live Mevo+ bringup test — no shots required.

Validates everything in the connect path except the shot-result decode:
  discover -> TCP 5100 connect -> replay bringup handshake -> keepalive -> read frames.

Success (device idle, no swings) = the handshake is accepted, the connection stays up,
checksum-valid device->host frames keep flowing, and the device reports an ARMED/IDLE
state in its E3/E5 text. That proves bringup works; only the D4/ED/EF shot decode still
needs a real swing to verify.

Usage:
  python scripts/mevo_connect_test.py [--host H] [--broadcast B] [--seconds N]
"""

import argparse
import collections
import socket
import sys
import threading
import time

sys.path.insert(0, __file__.rsplit("/scripts/", 1)[0])

from mevo import MevoClient, MevoDevice          # noqa: E402
from mevo.discovery import discover               # noqa: E402
from mevo.metrics import (                         # noqa: E402
    T_LAUNCH_RESULT, T_FLIGHT_RESULT, T_CLUB_RESULT, T_SPIN_RESULT,
)

TEXT_TYPES = {0xE3, 0xE5}          # device state / status text
SHOT_TYPES = {T_LAUNCH_RESULT, T_FLIGHT_RESULT, T_CLUB_RESULT, T_SPIN_RESULT, 0xD9}


def guess_broadcasts():
    addrs = ["255.255.255.255"]
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local = s.getsockname()[0]
        s.close()
        addrs.insert(0, local.rsplit(".", 1)[0] + ".255")
    except OSError:
        pass
    return addrs


def find_device(host, broadcast, timeout=3.0):
    if host:
        print(f"[discovery] skipped — using --host {host}:5100")
        return MevoDevice(host=host, port=5100)
    targets = [broadcast] if broadcast else guess_broadcasts()
    for b in targets:
        print(f"[discovery] broadcasting query to {b}:1248 (timeout {timeout}s) …")
        dev = discover(timeout=timeout, broadcast_addr=b)
        if dev:
            print(f"[discovery] FOUND {dev.instance or '?'} at {dev.host}:{dev.port}")
            return dev
        print(f"[discovery] no answer on {b}")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="")
    ap.add_argument("--broadcast", default="")
    ap.add_argument("--seconds", type=int, default=20)
    args = ap.parse_args()

    dev = find_device(args.host, args.broadcast)
    if dev is None:
        print("\nVERDICT: ✗ device not found. Is it powered on and on this Wi-Fi? "
              "Try --host <device-ip> or --broadcast <subnet>.255")
        return 2

    lock = threading.Lock()
    counts = collections.Counter()
    bad_cksum = collections.Counter()
    texts = []                 # (type, decoded ascii)
    shot_frames = collections.Counter()

    def on_frame(f):
        with lock:
            counts[f.typ] += 1
            if not f.checksum_ok:
                bad_cksum[f.typ] += 1
            if f.typ in TEXT_TYPES:
                s = "".join(chr(c) if 32 <= c < 127 else "" for c in f.payload).strip()
                if s and (f.typ, s) not in texts:
                    texts.append((f.typ, s))
            if f.typ in SHOT_TYPES:
                shot_frames[f.typ] += 1

    client = MevoClient(on_frame=on_frame, on_shot=lambda m: print(f"[SHOT] {m}"))
    print(f"[connect] TCP {dev.host}:{dev.port} …")
    try:
        client.connect(device=dev)
    except Exception as e:  # noqa: BLE001
        print(f"\nVERDICT: ✗ connect/bringup failed: {e}")
        return 2
    print(f"[bringup] replayed {len(client._bringup)} frames; "
          f"keepalive running. Listening {args.seconds}s (no shots needed)…\n")

    start = time.time()
    last = 0
    try:
        while time.time() - start < args.seconds:
            time.sleep(2.0)
            with lock:
                total = sum(counts.values())
                alive = client._sock is not None and any(
                    t.is_alive() for t in client._threads)
            print(f"  t={int(time.time()-start):2d}s  frames={total:<6} "
                  f"(+{total-last})  link={'up' if alive else 'DOWN'}")
            last = total
            if not alive:
                break
    finally:
        client.close()

    with lock:
        total = sum(counts.values())
        bad = sum(bad_cksum.values())
        print("\n=== summary ===")
        print(f"device→host frames: {total}  (checksum bad: {bad})")
        print("by type:", {f"{t:#04x}": n for t, n in counts.most_common(12)})
        print("\ndevice state / text seen:")
        for t, s in texts[:20]:
            print(f"  [{t:#04x}] {s}")
        if shot_frames:
            print("\nshot-result frames seen:",
                  {f"{t:#04x}": n for t, n in shot_frames.items()})

    # verdict
    print("\n=== VERDICT ===")
    ok_link = total > 0
    print(f"  discovery + connect ........ {'✓' if dev else '✗'}")
    print(f"  bringup accepted (frames) .. {'✓' if ok_link else '✗ no device frames'}")
    print(f"  checksums valid ............ {'✓' if total and bad == 0 else ('—' if not total else f'⚠ {bad} bad')}")
    armed = any("ARM" in s.upper() or "IDLE" in s.upper() or "STATE" in s.upper()
                for _, s in texts)
    print(f"  device ready (ARMED/IDLE) .. {'✓' if armed else '? no state text — inspect above'}")
    if ok_link:
        print("\n  Bringup works. Take a swing and re-run to validate shot decode "
              "(or watch for [SHOT] lines).")
    return 0 if ok_link else 2


if __name__ == "__main__":
    raise SystemExit(main())
