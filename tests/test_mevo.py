"""Mevo+ frame/metric decode tests.

Framing tests run anywhere (they use the committed handshake_capture.json). The pcap
decode tests are skipped when the gitignored *.pcapng captures aren't present.
"""

import json
import os

import pytest

from mevo.frames import (
    Deframer, Frame, build_frame, parse_frame, stuff, unstuff, AVR, APP,
)
from mevo.metrics import parse_shot_frames

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HANDSHAKE = os.path.join(REPO, "mevo", "handshake_capture.json")
MEVOINFO = os.path.join(REPO, "mevoinfo")


def _handshake_frames():
    with open(HANDSHAKE) as f:
        d = json.load(f)
    return d["bringup_frames_hex"] + d["keepalive_frames_hex"]


# --- framing -----------------------------------------------------------------

def test_stuff_unstuff_roundtrip():
    raw = bytes(range(256)) + b"\xf0\xf1\xfd\xfa" * 4
    assert unstuff(stuff(raw)) == raw


def test_builder_roundtrips_real_frames():
    """Every captured host->device frame must rebuild byte-for-byte."""
    for hx in _handshake_frames():
        f = parse_frame(bytes.fromhex(hx))
        assert f.checksum_ok, hx
        assert build_frame(f.dest, f.src, f.typ, f.payload).hex() == hx


def test_checksum_handles_stuffed_checksum():
    # f03010b0010000fd02f1: checksum value 0x00F1 -> low byte F1 -> stuffed to FD 02.
    f = parse_frame(bytes.fromhex("f03010b0010000fd02f1"))
    assert f.checksum_ok
    assert f.dest == AVR and f.src == APP and f.typ == 0xB0


def test_deframer_reassembles_across_chunks():
    f1 = build_frame(AVR, APP, 0xAA, b"\x01\x01\x00")
    f2 = build_frame(APP, AVR, 0xD4, b"\x00\xfd\xf0\xf1")  # payload forces stuffing
    stream = f1 + f2
    # Feed one byte at a time to stress reassembly.
    deframer = Deframer()
    got: list[Frame] = []
    for b in stream:
        got += deframer.feed(bytes([b]))
    assert len(got) == 2
    assert got[0].typ == 0xAA and got[0].payload == b"\x01\x01\x00"
    assert got[1].typ == 0xD4 and got[1].payload == b"\x00\xfd\xf0\xf1"


def test_deframer_drops_interframe_garbage():
    good = build_frame(AVR, APP, 0xAA, b"\x01")
    assert len(Deframer().feed(b"\x00\x99" + good + b"\xab")) == 1


# --- metric decode vs ground truth -------------------------------------------

def _decode(pcap):
    from mevo.pcap_source import iter_frames
    frames = list(iter_frames(os.path.join(MEVOINFO, pcap)))
    return [s for s in parse_shot_frames(frames) if not s.is_empty()]


@pytest.mark.skipif(not os.path.exists(os.path.join(MEVOINFO, "mevo_p3_fullswing.pcapng")),
                    reason="p3 capture not present (gitignored)")
def test_p3_ball_speed_matches_ground_truth():
    shots = _decode("mevo_p3_fullswing.pcapng")
    ball = [round(s.ball_speed_mph, 1) for s in shots]
    assert ball == [91.1, 101.5, 97.4, 97.4, 85.7]


@pytest.mark.skipif(not os.path.exists(os.path.join(MEVOINFO, "mevo_p4_fullswing2.pcapng")),
                    reason="p4 capture not present (gitignored)")
def test_p4_club_and_spin_match_csv():
    shots = _decode("mevo_p4_fullswing2.pcapng")
    assert len(shots) == 8
    club = [round(s.club_speed_mph, 1) for s in shots]
    assert club == [80.7, 77.9, 78.8, 78.7, 76.9, 86.7, 85.5, 88.2]
    # Spin matches the CSV to within a couple rpm.
    spin_gt = [5644, 5594, 5938, 5240, 5286, 2815, 3406, 3566]
    for got, gt in zip([s.spin_rpm for s in shots], spin_gt):
        assert abs(got - gt) <= 3
    # Launch direction (experimental E8 field) is within ~0.3 deg of the CSV on this session.
    ldir_gt = [5.0, 1.9, 3.2, 0.0, 3.0, 1.4, 0.5, 2.2]
    for got, gt in zip([s.launch_direction_deg for s in shots], ldir_gt):
        assert got is not None and abs(got - gt) <= 0.3


@pytest.mark.skipif(not os.path.exists(os.path.join(MEVOINFO, "mevo_p4_fullswing2.pcapng")),
                    reason="p4 capture not present (gitignored)")
def test_tracking_records_have_monotonic_time():
    from mevo.pcap_source import iter_frames
    from mevo.tracking import parse_tracking
    ec_ts, ee_ts = [], []
    for f in iter_frames(os.path.join(MEVOINFO, "mevo_p4_fullswing2.pcapng")):
        if f.typ == 0xEC:
            ec_ts += [r.time_index for r in parse_tracking(f)]
        elif f.typ == 0xEE:
            ee_ts += [r.time_index for r in parse_tracking(f)]
    # EC time index steps by 32 within a frame's records.
    assert ec_ts and ee_ts
    assert ec_ts[1] - ec_ts[0] == 32
