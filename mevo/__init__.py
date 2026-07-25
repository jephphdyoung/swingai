"""Mevo+ launch-monitor client for SwingAI.

Reverse-engineered FlightScope Mevo+ network protocol (see mevoinfo/mevo_protocol.md
and mevo/STATE.md). All traffic is plaintext over Wi-Fi.

Layers:
  frames.py       -- F0..F1 framing: byte-stuffing, checksum, streaming deframer.
  metrics.py      -- ShotMetrics + D4/ED/EF result-frame decode.
  pcap_source.py  -- offline replay of a pcap's device->host TCP 5100 stream.
  discovery.py    -- UDP 1248 broadcast discovery -> (host, port).
  client.py       -- MevoClient: discover, connect, replay handshake, keepalive, read shots.
"""

from .frames import Frame, build_frame, stuff, unstuff, Deframer, APP, PI, AVR, DSP
from .metrics import ShotMetrics, ShotAssembler, parse_shot_frames
from .discovery import MevoDevice, discover
from .client import MevoClient
from .tracking import TrackRecord, parse_tracking

__all__ = [
    "Frame", "build_frame", "stuff", "unstuff", "Deframer",
    "APP", "PI", "AVR", "DSP",
    "ShotMetrics", "ShotAssembler", "parse_shot_frames",
    "MevoDevice", "discover",
    "MevoClient",
    "TrackRecord", "parse_tracking",
]
