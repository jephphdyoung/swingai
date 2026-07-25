"""Provisional decode of the Mevo+ club/ball tracking streams (EC / EE on TCP 5100).

These frames carry the *raw per-frame trajectory* the FlightScope app uses to compute the
clubface metrics (face angle, club path, AoA, spin axis, impact location) — none of which
are transmitted as finished scalars (see mevoinfo/mevo_protocol.md S6/S7). This module
extracts the record structure so that geometry work can build on it; the spatial channels
are NOT yet calibrated to real-world units or assigned to club-vs-ball.

Record structure (validated on mevo_p4_fullswing2.pcapng):
  EC: 4-byte frame header + N x 60-byte records.  off0 = monotonic time index (+32/rec).
  EE: 1-byte frame header + N x 76-byte records.  off0 = frame#, off2 = time index (+32).
Columns that vary smoothly over a swing are trajectory channels; the rest are packed
low-bytes / flags / quality and are exposed raw.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

from .frames import Frame

EC_HEADER = 4
EC_RECORD = 60
EE_HEADER = 1
EE_RECORD = 76

# Offsets (into the record) of the columns that trace smooth time-series in the p4 capture.
# Provisional — identities/units unconfirmed. Use as starting points for geometry work.
EC_SMOOTH_OFFSETS = (16, 18, 32, 34, 38, 40, 42, 48)
EE_SMOOTH_OFFSETS = (22, 28, 34, 48, 50, 54, 56, 58)


@dataclass
class TrackRecord:
    """One tracking sample. `time_index` is the device's monotonic counter; `channels`
    maps each smooth-column offset to its int16 BE value (provisional, uncalibrated)."""

    time_index: int
    channels: dict[int, int]
    raw: bytes  # full record bytes, for further decoding


def _i16be(b: bytes, off: int) -> int:
    return struct.unpack_from(">h", b, off)[0]


def _split_records(payload: bytes, header: int, record: int) -> list[bytes]:
    body = payload[header:]
    return [body[i:i + record] for i in range(0, len(body) - record + 1, record)]


def parse_ec(payload: bytes) -> list[TrackRecord]:
    out = []
    for r in _split_records(payload, EC_HEADER, EC_RECORD):
        out.append(TrackRecord(
            time_index=_i16be(r, 0),
            channels={off: _i16be(r, off) for off in EC_SMOOTH_OFFSETS},
            raw=r,
        ))
    return out


def parse_ee(payload: bytes) -> list[TrackRecord]:
    out = []
    for r in _split_records(payload, EE_HEADER, EE_RECORD):
        out.append(TrackRecord(
            time_index=_i16be(r, 2),  # off0 is the frame#, off2 the time index
            channels={off: _i16be(r, off) for off in EE_SMOOTH_OFFSETS},
            raw=r,
        ))
    return out


def parse_tracking(frame: Frame) -> list[TrackRecord]:
    """Parse an EC (0xEC) or EE (0xEE) frame into its tracking records; [] for others."""
    if frame.typ == 0xEC:
        return parse_ec(frame.payload)
    if frame.typ == 0xEE:
        return parse_ee(frame.payload)
    return []
