"""Shot-metric decode from Mevo+ TCP 5100 result frames.

For full swings the radar path puts measured metrics on the wire (unlike putts, which
are camera/GVP-derived). Only three scalars are *measured* and stored on the wire; the
rest (launch angle/direction, spin axis, AoA, carry, ...) are computed app-side and are
NOT decoded here.

Validated offsets (r ~= 1.000 vs app CSV export, see mevoinfo/mevo_protocol.md S6):
  ball speed  = D4 payload off 16, INT24 BE, mm/s
  club speed  = D4 payload off 85, INT24 BE, mm/s   (also ED off 6, INT16 BE, cm/s)
  spin rate   = EF payload off 106, INT16 BE, RPM
  shot number = D4 payload off 1

Experimental (r ~= 0.997, single-session calibration -- see mevoinfo/mevo_protocol.md S7):
  launch direction = E8 payload off 25, INT16 LE, linear-calibrated to degrees (+ = right).

NOT on the wire as scalars (computed app-side from the EC/EE club-track point-clouds):
  face angle, club path, AoA, spin axis, face-to-path, dynamic loft, impact location.
"""

from __future__ import annotations

from dataclasses import dataclass

from .frames import Frame

MPS_TO_MPH = 2.2369362920544

# Result-frame TYPE bytes.
T_LAUNCH_RESULT = 0xE8   # "early launch conditions" (precedes D4 in the lifecycle)
T_FLIGHT_RESULT = 0xD4
T_CLUB_RESULT = 0xED
T_SPIN_RESULT = 0xEF

# Launch-direction calibration (E8 off25 INT16 LE -> degrees, + = right).
# Fitted on the 8-shot p4 session; experimental, accurate to ~0.2 deg there.
_LAUNCH_DIR_SLOPE = -9.700980e-04
_LAUNCH_DIR_INTERCEPT = 0.131347


def _int_be(payload: bytes, off: int, size: int, signed: bool = False) -> int | None:
    """Big-endian integer at `off`, or None if the payload is too short."""
    if off < 0 or off + size > len(payload):
        return None
    return int.from_bytes(payload[off:off + size], "big", signed=signed)


@dataclass
class ShotMetrics:
    """Measured launch-monitor metrics for one shot. Speeds in SI; mph helpers provided."""

    shot_number: int | None = None
    ball_speed_mps: float | None = None
    club_speed_mps: float | None = None
    spin_rpm: int | None = None
    launch_direction_deg: float | None = None  # experimental; + = right of target

    def is_empty(self) -> bool:
        return (self.ball_speed_mps is None and self.club_speed_mps is None
                and self.spin_rpm is None and self.launch_direction_deg is None)

    @property
    def ball_speed_mph(self) -> float | None:
        return None if self.ball_speed_mps is None else self.ball_speed_mps * MPS_TO_MPH

    @property
    def club_speed_mph(self) -> float | None:
        return None if self.club_speed_mps is None else self.club_speed_mps * MPS_TO_MPH


def _apply_e8(shot: ShotMetrics, payload: bytes) -> None:
    if 25 + 2 <= len(payload):
        raw = int.from_bytes(payload[25:27], "little", signed=True)
        shot.launch_direction_deg = _LAUNCH_DIR_SLOPE * raw + _LAUNCH_DIR_INTERCEPT


def _apply_d4(shot: ShotMetrics, payload: bytes) -> None:
    shot.shot_number = _int_be(payload, 1, 1)
    ball = _int_be(payload, 16, 3)      # mm/s
    club = _int_be(payload, 85, 3)      # mm/s
    if ball is not None:
        shot.ball_speed_mps = ball / 1000.0
    if club is not None:
        shot.club_speed_mps = club / 1000.0


def _apply_ed(shot: ShotMetrics, payload: bytes) -> None:
    # Club speed also appears here (cm/s); prefer it only if D4 didn't supply one.
    if shot.club_speed_mps is None:
        club = _int_be(payload, 6, 2)   # cm/s
        if club is not None:
            shot.club_speed_mps = club / 100.0


def _apply_ef(shot: ShotMetrics, payload: bytes) -> None:
    spin = _int_be(payload, 106, 2)     # RPM
    if spin is not None:
        shot.spin_rpm = spin


class ShotAssembler:
    """Stateful per-shot assembler shared by offline parsing and the live client.

    Lifecycle order on the wire is E8 (launch) -> D4 (flight) -> ED x2 (club) -> EF (spin).
    A shot begins on the first E8/D4 after the previous shot's spin and is *complete* once
    EF arrives. feed() returns a finished ShotMetrics on the EF that closes a shot, else None.
    """

    def __init__(self) -> None:
        self._cur: ShotMetrics | None = None

    def _ensure(self) -> ShotMetrics:
        # Start a fresh shot if none is open or the open one is already complete.
        if self._cur is None or self._cur.spin_rpm is not None:
            self._cur = ShotMetrics()
        return self._cur

    def feed(self, frame: Frame) -> ShotMetrics | None:
        if frame.typ == T_LAUNCH_RESULT:
            _apply_e8(self._ensure(), frame.payload)
        elif frame.typ == T_FLIGHT_RESULT:
            _apply_d4(self._ensure(), frame.payload)
        elif self._cur is None:
            return None
        elif frame.typ == T_CLUB_RESULT:
            _apply_ed(self._cur, frame.payload)
        elif frame.typ == T_SPIN_RESULT:
            _apply_ef(self._cur, frame.payload)
            shot, self._cur = self._cur, None
            return shot if not shot.is_empty() else None
        return None

    def flush(self) -> ShotMetrics | None:
        """Return any in-progress shot that never received an EF (incomplete tail)."""
        shot, self._cur = self._cur, None
        return shot if shot is not None and not shot.is_empty() else None


def parse_shot_frames(frames: list[Frame]) -> list[ShotMetrics]:
    """Group an in-order sequence of result frames into per-shot metrics."""
    asm = ShotAssembler()
    shots: list[ShotMetrics] = []
    for f in frames:
        shot = asm.feed(f)
        if shot is not None:
            shots.append(shot)
    tail = asm.flush()
    if tail is not None:
        shots.append(tail)
    return shots
