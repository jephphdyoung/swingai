"""Mevo+ TCP 5100 frame layer.

Wire frame:  F0 | DEST | SRC | TYPE | payload | CS | F1
  - Everything between F0 (SOF) and F1 (EOF) is byte-stuffed.
  - Bus addresses (DEST/SRC), not a direction flag: APP=0x10, PI=0x12, AVR=0x30, DSP=0x40.
  - CS = 16-bit sum of the *stuffed* wire bytes from DEST through the last payload byte,
    emitted big-endian with a leading zero high byte dropped (1 or 2 bytes), then stuffed.

This module's build/parse was validated to round-trip all 106 captured host->device
frames in mevo/handshake_capture.json byte-for-byte.
"""

from __future__ import annotations

from dataclasses import dataclass

# Bus node addresses.
APP = 0x10
PI = 0x12
AVR = 0x30
DSP = 0x40

SOF = 0xF0
EOF = 0xF1
ESC = 0xFD

# value -> escaped 2-byte sequence
_STUFF = {0xF0: b"\xfd\x01", 0xF1: b"\xfd\x02", 0xFD: b"\xfd\x03", 0xFA: b"\xfd\x04"}
# escape-follow byte -> original value
_UNSTUFF = {0x01: 0xF0, 0x02: 0xF1, 0x03: 0xFD, 0x04: 0xFA}


def stuff(data: bytes) -> bytes:
    """Byte-stuff F0/F1/FD/FA so they can't be confused with SOF/EOF/ESC."""
    out = bytearray()
    for b in data:
        esc = _STUFF.get(b)
        if esc is not None:
            out += esc
        else:
            out.append(b)
    return bytes(out)


def unstuff(data: bytes) -> bytes:
    """Reverse stuff(). Raises ValueError on a truncated/invalid escape sequence."""
    out = bytearray()
    i = 0
    n = len(data)
    while i < n:
        b = data[i]
        if b == ESC:
            if i + 1 >= n:
                raise ValueError("truncated escape sequence")
            orig = _UNSTUFF.get(data[i + 1])
            if orig is None:
                raise ValueError(f"invalid escape byte {data[i + 1]:#04x}")
            out.append(orig)
            i += 2
        else:
            out.append(b)
            i += 1
    return bytes(out)


def _checksum_bytes(stuffed_body: bytes) -> bytes:
    """Checksum field for an already-stuffed DEST..payload body."""
    s = sum(stuffed_body) & 0xFFFF
    raw = bytes([s >> 8, s & 0xFF]) if s > 0xFF else bytes([s & 0xFF])
    return stuff(raw)


def build_frame(dest: int, src: int, typ: int, payload: bytes = b"") -> bytes:
    """Assemble a complete F0..F1 wire frame."""
    body = stuff(bytes([dest, src, typ]) + payload)
    return bytes([SOF]) + body + _checksum_bytes(body) + bytes([EOF])


@dataclass(frozen=True)
class Frame:
    """A parsed frame. `payload` excludes DEST/SRC/TYPE and the checksum."""

    dest: int
    src: int
    typ: int
    payload: bytes
    checksum_ok: bool

    @property
    def type_hex(self) -> str:
        return f"{self.typ:02x}"


def parse_frame(raw: bytes) -> Frame:
    """Parse one raw F0..F1 frame (stuffed, inclusive of SOF/EOF).

    The checksum is the trailing 1 or 2 unstuffed bytes. We try 1 byte first to match the
    builder's canonical minimal-BE encoding (single byte when the sum fits in 0xFF), then
    2 bytes; this resolves the ambiguity where a trailing 0x00 payload byte looks like a
    zero checksum high byte.
    """
    if len(raw) < 4 or raw[0] != SOF or raw[-1] != EOF:
        raise ValueError("not a framed F0..F1 message")
    body = unstuff(raw[1:-1])  # DEST SRC TYPE payload... CS
    if len(body) < 4:
        raise ValueError("frame too short")
    dest, src, typ = body[0], body[1], body[2]

    for cslen in (1, 2):
        if len(body) - cslen < 3:
            continue
        payload = body[3:len(body) - cslen]
        expect = sum(stuff(body[:len(body) - cslen])) & 0xFFFF
        cs = int.from_bytes(body[len(body) - cslen:], "big")
        if expect == cs:
            return Frame(dest, src, typ, payload, checksum_ok=True)

    # Checksum didn't verify at either length; return best-effort payload (assume 2-byte CS).
    payload = body[3:-2] if len(body) > 5 else b""
    return Frame(dest, src, typ, payload, checksum_ok=False)


class Deframer:
    """Streaming deframer: feed arbitrary TCP byte chunks, get back whole Frames.

    Mevo frames span multiple TCP segments, so reassembly happens here. Bytes outside
    an F0..F1 pair (and any leading garbage before the first SOF) are dropped.
    """

    def __init__(self) -> None:
        self._buf = bytearray()
        self._in_frame = False

    def feed(self, chunk: bytes) -> list[Frame]:
        frames: list[Frame] = []
        for b in chunk:
            if not self._in_frame:
                if b == SOF:
                    self._in_frame = True
                    self._buf = bytearray([b])
                # else: drop inter-frame byte
                continue

            self._buf.append(b)
            if b == EOF:
                try:
                    frames.append(parse_frame(bytes(self._buf)))
                except ValueError:
                    pass  # malformed frame; skip
                self._in_frame = False
                self._buf = bytearray()
        return frames
