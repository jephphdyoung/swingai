"""MevoClient -- live FlightScope Mevo+ control over TCP 5100.

Lifecycle: discover -> connect -> replay the captured bringup handshake -> run a ~1 Hz
keepalive (STATUS polls) -> read result frames and fire on_shot(ShotMetrics) per swing.

The bringup frames and keepalive set are replayed verbatim from mevo/handshake_capture.json.

CAVEAT (see mevo/STATE.md): the PI 0x90 bringup frames carry a blob that is likely
device-specific calibration/license for serial M2-047295. Verbatim replay is unverified
against a different unit; if bringup fails on other hardware this is the first suspect.
"""

from __future__ import annotations

import json
import os
import socket
import threading
import time
from collections.abc import Callable

from .frames import Deframer, Frame
from .metrics import ShotMetrics, ShotAssembler
from .discovery import MevoDevice, discover

_HANDSHAKE_PATH = os.path.join(os.path.dirname(__file__), "handshake_capture.json")

ShotCallback = Callable[[ShotMetrics], None]
FrameCallback = Callable[[Frame], None]


def _load_handshake() -> tuple[list[bytes], list[bytes]]:
    with open(_HANDSHAKE_PATH) as f:
        d = json.load(f)
    bringup = [bytes.fromhex(h) for h in d["bringup_frames_hex"]]
    keepalive = [bytes.fromhex(h) for h in d["keepalive_frames_hex"]]
    return bringup, keepalive


class MevoClient:
    """Drives a Mevo+ as the sole TCP 5100 client.

    Typical use:
        client = MevoClient(on_shot=lambda m: print(m))
        client.connect()           # discover + handshake; starts keepalive + reader
        ...                         # shots arrive on the callback thread
        client.close()
    """

    def __init__(
        self,
        on_shot: ShotCallback | None = None,
        on_frame: FrameCallback | None = None,
        keepalive_interval: float = 1.0,
    ) -> None:
        self.on_shot = on_shot
        self.on_frame = on_frame
        self.keepalive_interval = keepalive_interval

        self.device: MevoDevice | None = None
        self._sock: socket.socket | None = None
        self._bringup, self._keepalive = _load_handshake()

        self._deframer = Deframer()
        self._assembler = ShotAssembler()

        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

    # -- connection -----------------------------------------------------------

    def connect(self, device: MevoDevice | None = None, discover_timeout: float = 3.0,
                broadcast_addr: str = "255.255.255.255") -> MevoDevice:
        """Discover (unless `device` given), TCP-connect, replay handshake, start threads."""
        if device is None:
            device = discover(timeout=discover_timeout, broadcast_addr=broadcast_addr)
            if device is None:
                raise ConnectionError("no Mevo+ device answered discovery")
        self.device = device

        sock = socket.create_connection((device.host, device.port), timeout=5.0)
        sock.settimeout(1.0)
        self._sock = sock

        self._send_bringup()

        self._stop.clear()
        self._threads = [
            threading.Thread(target=self._reader_loop, name="mevo-reader", daemon=True),
            threading.Thread(target=self._keepalive_loop, name="mevo-keepalive", daemon=True),
        ]
        for t in self._threads:
            t.start()
        return device

    def _send_bringup(self) -> None:
        assert self._sock is not None
        for frame in self._bringup:
            self._sock.sendall(frame)
            time.sleep(0.005)  # pace the burst as the app does

    def close(self) -> None:
        self._stop.set()
        for t in self._threads:
            if t.is_alive():
                t.join(timeout=2.0)
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None

    def __enter__(self) -> "MevoClient":
        self.connect()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- threads --------------------------------------------------------------

    def _keepalive_loop(self) -> None:
        while not self._stop.is_set():
            try:
                if self._sock is not None:
                    for frame in self._keepalive:
                        self._sock.sendall(frame)
            except OSError:
                break
            self._stop.wait(self.keepalive_interval)

    def _reader_loop(self) -> None:
        while not self._stop.is_set():
            try:
                chunk = self._sock.recv(65536) if self._sock else b""
            except socket.timeout:
                continue
            except OSError:
                break
            if not chunk:
                break  # device closed the connection
            for frame in self._deframer.feed(chunk):
                self._handle_frame(frame)

    # -- result assembly ------------------------------------------------------

    def _handle_frame(self, frame: Frame) -> None:
        if self.on_frame is not None:
            self.on_frame(frame)
        shot = self._assembler.feed(frame)
        if shot is not None and self.on_shot is not None:
            self.on_shot(shot)
