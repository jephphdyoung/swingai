# FlightScope Mevo+ — Network Protocol (reverse-engineered)

Captured via Wireshark/tshark on the controlling PC, June 2026.
Device under test: **InstanceName `M2-047295`** (serial 047295), firmware `0.43 FS Mevo 2`,
internal build `0.38 M2 TR / May 28 2026`. Internally a Raspberry-Pi-class embedded
Linux board with a **Raspberry Pi Camera Module V2** and an Edimax USB Wi-Fi dongle.

> All traffic is **plaintext** — no TLS anywhere. The unit joins the local Wi-Fi
> (DHCP) and the app talks to it directly over IP.

---

## 1. Channel map

| Proto / Port | Direction | Format | Role |
|---|---|---|---|
| **UDP 1248** | app ⇄ unit (bcast) | XML | Service discovery |
| **TCP 5100** | app ⇄ unit | Binary, `F0…F1` framed | Radar control + raw sample telemetry + device info |
| **TCP 1258** | app ⇄ unit | JSON, newline-delimited | Golf Video Processor (GVP): triggers, config, status, logs |
| **HTTP 8080** | unit → app | `multipart/x-mixed-replace` | MJPEG camera stream (mjpg-streamer, boundary `boundarydonotcross`) |
| **UDP 123 (NTP)** | app → unit | NTP | App continuously syncs PC clock to the unit (~1/sec) |

---

## 2. Discovery (UDP 1248, XML)

App broadcasts a query; unit replies with how/where to connect:

```xml
<!-- app -> -->
<ServiceQuery><ServiceName>FlightScope Device</ServiceName></ServiceQuery>

<!-- unit -> -->
<ServiceInfo>
  <ServiceName>FlightScope Device</ServiceName>
  <ApplicationName>Mevo2</ApplicationName>
  <ApplicationVersion>Issue 1</ApplicationVersion>
  <InstanceName>M2-047295</InstanceName>
  <TransportProtocol>TCP</TransportProtocol>
  <Host>192.168.120.169</Host>
  <Port>5100</Port>          <!-- "connect to me on TCP 5100" -->
</ServiceInfo>
```

---

## 3. Control + telemetry (TCP 5100, binary)

### Frame envelope (every message)

```
0xF0 │ dir │ opcode │ … header + payload … │ [cksum] │ 0xF1
```

- `0xF0` = start of frame, `0xF1` = end of frame (no F0/F1 seen mid-payload).
- **dir** (byte 1): `0x40` = host→device (commands), `0x10` = device→host (replies/data).
- **opcode** (byte 2): `0x10` host poll · `0x40` status reply · `0x30` data/stream · `0x12` device info.
- A checksum byte typically precedes the trailing `0xF1`.

### Idle loop (~1 Hz)

```
host -> F0 40 10 AA 01 0100 FC F1                (poll, 9B)
unit -> F0 10 40 ... F1                          (status reply, 136B)
unit -> F0 10 30 ... F1                          (data, 32B)
unit -> F0 10 12 ... F1                          (info, 27B)
```

### Streaming radar record — opcode `0x30`, subtype `0xD3` (~216B)

Field offsets are into the TCP payload (byte 0 = `0xF0`):

| Offset | Field | Notes |
|---|---|---|
| `[0]` | `0xF0` | SOF |
| `[1]` | `0x10` | dir = device→host |
| `[2]` | `0x30` | opcode = data/stream |
| `[3:4]` | `D3 CF` / `D3 CD` | format id; CF↔CD toggles one status bit |
| `[7]` | `0x02` | constant (channel/format selector) |
| `[9:10]` | **cumulative sample counter** (BE) | **+90 between adjacent records** |
| `[12:13]` | **samples in this record** | ~90 (drops to ~70) |
| `[14:15]` | **record sequence #** (BE) | +1 per record |
| `[17]` | status / flags | varies ff / fd / ea |
| `[18 : end-2]` | **raw radar samples (8-bit)** | ~196 bytes; full 0–255 range, mean ≈119 (DC offset) |
| `[end-1]` | checksum | precedes `0xF1` |
| `[end]` | `0xF1` | EOF |

Larger frames (432 / 648 / 1296 B) are simply **N×216 records coalesced** into one
TCP segment — not larger shots.

### Device info — opcode `0x12` (27B)

```
F0 10 12 AA 13 01 0668 04D0 0000 0000 0500 2010 03E8 03E8 0000 0104 4C F1
                                              ^^^^ ^^^^ = 1000, 1000 (rate/scale consts)
```

### Status — opcode `0x40` (136B)

Carries an array of 16-bit little-endian values clustered ~3000–4000
(`0x0c35`, `0x0cf9`, `0x0ca5`, …) — likely **per-bin signal levels / channel
magnitudes** from the radar front-end, plus counters.

---

## 4. Golf Video Processor (TCP 1258, JSON)

Newline-delimited JSON objects, each with a `type` and `version`.

- `CONFIG_REQUEST` / `CONFIG` — camera calibration, ROI, pre/post-trigger buffer sizes,
  `saveVideosEnabled`, live-preview processing config.
- `STATUS` — buffer state: `IDLE → TRIGGERED → CONVERTING → IDLE`.
- `TRIGGER` — fires on shot detection:
  ```json
  {"epochTime":1782009207.36,"guid":"{e9845936-…}","savePath":"{e9845936-…}",
   "skipTracking":false,"type":"TRIGGER","version":6}
  ```
- `MT_GOLF_EXPECTED_CLUB_TRACK` — app sends the unit a predicted club path
  (polynomials in image U/V coords + radius) to guide vision tracking:
  ```json
  {"type":"MT_GOLF_EXPECTED_CLUB_TRACK","duration":0.1,
   "polyU":[396.18,-0.0146,1.60,18.08,-899.99],
   "polyV":[512.53,-0.0036,1.57,2.99,-604.96],
   "polyRadius":[1,6.49,2.13,0,0]}
  ```
- `LOG` — `[GVP]` golf-vision-processor log lines (GVP version 1.25, image proc v22).

---

## 5. Key conclusion for reverse engineering

> **UPDATE (full-swing analysis): this conclusion is PUTT-ONLY.** For **full swings**
> the radar path *does* transmit the metrics on TCP 5100 (message D4/ED/EF), and we have
> decoded ball speed. See Section 6. The text below applies to putts (camera/GVP path).

**Computed metrics (ball speed, spin, launch, distance) are NOT transmitted as labeled
numbers.** The wire carries:

1. **Raw radar samples** (binary, TCP 5100) — framed 8-bit sample blocks,
2. **Trigger / control events** (JSON, TCP 1258),
3. **MJPEG video** (HTTP 8080).

The app/processor derives the final numbers from the raw 5100 stream. To get speeds you
must replicate the DSP: **subtract DC (~119) → FFT over the 8-bit sample stream →
Doppler frequency → velocity.**

The `d3` records split into two interleaved subtypes — `cf` (~710k samples) and
`cd` (~254k samples) — likely two radar channels (I/Q or two antennas). Exported per
channel as `mevo_iq_cf.bin` / `mevo_iq_cd.bin` (raw 8-bit), plus `_u8.csv` (0–255) and
`_centered.csv` (DC-removed, ±127) for direct FFT. Per-record metadata in
`mevo_iq_records.csv`.

### Observed anomaly
During the putting test, exactly **one** GVP `TRIGGER` fired and its video processing
**failed**: `Failed to open the header` / `Failed to parse header.32k` → returned to IDLE.
If shots aren't registering in the app, this camera-processing path is worth investigating.

---

## 6. Full-swing binary protocol (TCP 5100) — VALIDATED

Reference: **github.com/divotmaker/ironsight** `docs/` (WIRE.md, SEQUENCE.md, CAMERA.md),
cross-checked against our own `mevo_p3_fullswing.pcapng` (5 shots, ground-truth carries).

### Frame structure (corrected)

```
0xF0 │ DEST │ SRC │ TYPE │ payload │ CS_HI │ CS_LO │ 0xF1
```

- **DEST/SRC are bus addresses**, not a direction flag: `0x10`=APP, `0x12`=PI,
  `0x30`=AVR (radar I/O), `0x40`=DSP (radar core). So `F0 10 30 …` = AVR→APP.
- The `AA` we saw everywhere = **STATUS** message TYPE (polled ~1/s to each node).
- **Byte-stuffing** (escape = `0xFD`): `F0→FD 01`, `F1→FD 02`, `FD→FD 03`, `FA→FD 04`.
  Must un-stuff before parsing. **Frames span multiple TCP segments → reassemble first.**
- Checksum = 16-bit sum of the *stuffed wire bytes* from DEST through last payload byte.
- Verified on the 9-byte poll `F0 40 10 AA 01 01 00 FC F1`: 0x40+0x10+0xAA+0x01+0x01 = 0xFC ✓.

### Shot lifecycle (per swing, all on 5100)

`E5 "BALL TRIGGER" → E9 TRACKING → E8 FLIGHT_RESULT_V1 → EC PRC_DATA bursts →
E5 "Clubimpact" (text leaks Vc=club m/s, Vb=ball m/s) → D4 FLIGHT_RESULT →
ED CLUB_RESULT (×2) → D9 SPEED_PROFILE → EF SPIN_RESULT → E5 "PROCESSED" → E5 "IDLE"`

Counts seen for 5 shots: 5×D4, 5×E8, 10×ED, 5×EF, 5×D9, 25×E5. **0 GVP triggers on 1258**
(full swings are radar-detected, unlike putts).

### D4 FLIGHT_RESULT decode — `[1 header byte] + 52 × INT24 big-endian`

| Field | Payload offset | Meaning | Confidence |
|---|---|---|---|
| f0 | 1 | shot number (1,2,3,…) | confirmed |
| **f5** | **16** | **ball speed, mm/s** (÷1000 = m/s) | **confirmed** — matches device Vb to 0.1 m/s |
| **f28** | **85** | **club speed, mm/s** | **confirmed** — r=1.000 vs 8 shots (36073 mm/s = 80.7 mph) |
| f8 / f11 | 25 / 34 | distance estimate, mm (tracks ball speed) | likely |

### Other validated result fields (session 2, 8 shots)

| Message | Offset | Encoding | Meaning | Check |
|---|---|---|---|---|
| **EF** SPIN_RESULT | **106** | INT16 BE | **spin rate, RPM (raw)** | 5646 vs GT 5644 ✓ |
| **ED** CLUB_RESULT | **6** | INT16 BE | **club speed, cm/s** (×0.01 m/s) | 3607 = 36.07 m/s = 80.7 mph ✓ |

Validated this session — ball speed 91.1/101.5/97.4/97.4/85.7 mph (8i shots 1–5);
spin rate to ~1 rpm; club speed to exact mph.

### NOT stored as scalar fields (next target)

**Launch angle, launch direction (azimuth), spin axis, AoA** do NOT appear as clean
INT16/24/32 or FLOAT40 fields in D4/E8/EF/ED/D9 (best r only ~0.88–0.99, no consistent
scale). Working hypothesis: the **angles are derived from velocity-vector components**
(DSP frame X=range, Y=vertical, Z=lateral) — find 3 D4 fields whose vector magnitude ≈
ball speed, then launch = atan(v_y/v_x), direction = atan(v_z/v_x). Spin axis likely sits
near the spin-rate field in EF; AoA near club speed in ED.

Displayed **carry is app-computed** (flight model from speed+angle+spin), NOT a stored
D4 field — confirmed again (no field correlates cleanly with carry across 8 shots).

### Method that worked (for next time)

1. tshark filter `tcp.port==5100 && ip.src==<mevo> && frame.time_relative` in a ±0.5 s
   window around each shot; concatenate `tcp.payload` (reassembles across segments).
2. Frame on `F0…F1`; un-stuff; strip DEST/SRC/TYPE + 2 CS bytes → payload.
3. Brute-force every offset × {INT16/24/32 BE/LE, FLOAT40} and correlate against known
   ground truth (Pearson for ranked values; ratio-CoV when exact values known).
4. PowerShell gotcha: cast bytes to `[int]` before `-shl` (byte shift wraps otherwise).

### Session 3 — CSV export cross-check (31 cols × 8 shots)

App CSV export (`8shots_windows-1252.csv`) gave precise ground truth for every metric.
Fit each column against ~3000 candidate fields (all offsets × INT16/24/32 BE/LE + FLOAT40)
across D4/E8/EF/ED/D9. Result:

**Directly measured (locked, r ≈ 1.000):**
- Ball speed = D4 off 16 (INT24 BE, mm/s) — this is the **range / Doppler velocity**.
- Club speed = D4 off 85 (mm/s) & ED off 6 (cm/s).
- Spin rate = EF off 106 (INT16 BE, raw RPM).

**Everything else is computed, NOT stored as a scalar field.** Launch V/H, spin axis, AoA,
club path, dynamic/spin loft, descent, height, carry, total, lateral — *none* lock
(best r only 0.88–0.99, always with spurious huge-slope LE encodings). The app derives
them from the raw measurements + tracking data.

**Velocity-vector probe (partial):** computed expected components from CSV
(vₓ=S·cosV·cosH, v_y=S·sinV, v_z=S·cosV·sinH) and searched:
- vₓ ≈ ball-speed field (off 16 overlaps) → **off-16 "ball speed" is really the range
  velocity component vₓ**, r=0.999.
- v_z (lateral): weak hit at E8 off 19 (r≈−0.99, shot 4≈0 matching launchH=0).
- v_y (vertical): **no clean BE field** → launch elevation is not a simple stored component.

So launch/direction likely come from the **E8 polynomial** or the **EC PRC point-clouds**
(raw tracking), fused client-side — same pattern as the putt face-impact (CAMERA.md:
"computed client-side, NOT in any wire message").

### To finish the decode (next session)
- Parse the **2nd ED frame** (cnt=1) and EC/EF sub-record structures for spin axis & AoA.
- Decode the **E8 polynomial** (doc: E8 = early launch conditions + polynomial) — likely
  source of launch angle/direction.
- PowerShell gotchas hit repeatedly: `$P`/`$p` are the same var (case-insensitive — use
  distinct names, e.g. `$spd` not `$S` next to loop `$s`); `+=` on a hashtable into an
  array flattens it (use ArrayList); unary `,` double-wraps; cast bytes to `[int]` before
  `-shl`; FLOAT40 on garbage offsets yields ∞ — guard before writing to file.

---

## 7. Python re-decode (Jun 2026) — clubface data & tracking streams

Re-ran the field search cleanly in Python (`mevo/` package, scapy `PcapReader`) against
`mevo_p4_fullswing2.pcapng` + the 8-shot CSV. Confirms S6 and adds findings.

### Frame inventory per shot (non-D3/AA), p4
`E8`×1 (94B, launch) · `D4`×1 (157B, flight) · `ED`×2 (167B, club) · `D9`×1 (speed
profile) · `EF`×1 (138B, spin) · `D1`×1 (242B) · `0x84`×1 (67B) · plus streams
`EC`×40/shot (tracking) · `EE`×~22/shot (tracking) · `E3`/`E5` text.

- **`0x84` = ASCII shot timestamp** e.g. `2026.06.21T19.07.01.4` (shot ID, not metrics).
- **`E5` text** (per shot): `BALL TRIGGER: NN ms back, at Epoch <us>`, `Clubimpact at
  Epoch <us>, ClubImpactIndex=<n>`, `RAW SAMPLE save … .fsb`, `PROCESSED`, `IDLE`. Gives
  exact **club-impact epoch (µs)** — usable for shot↔video pairing. (No Vc/Vb leak in p4.)
- **`E3` text**: radar/system state (`Fs=37500`, `MaxVel=120`, `ARMED DetectionMode`, …).

### Re-confirmed measured scalars (r≈1.000)
Ball = D4 off16 (INT24 BE mm/s); club = D4 off85 (mm/s) & ED off6 (INT16 BE cm/s);
spin = EF off106/108 (INT16 BE RPM, slope≈1.0).

### NEW: launch direction IS on the wire (S6 had said it wasn't)
**E8 off25, INT16 LE**, clean **Q8.8 fixed-point** (raw values are exact ×256 multiples).
Linear-calibrated to deg: `dir = -9.701e-4*raw + 0.131`, **r=0.997, max err 0.16°** on
p4 (+ = right). Shipped in `ShotMetrics.launch_direction_deg` (experimental, single-session
calibration). Launch elevation + a loft angle also correlate in E8 (r≈0.99) but with no
clean scale → left out.

### Clubface metrics are NOT transmitted as scalars (confirmed rigorously)
Brute-forced every per-shot frame (D4/E8/ED×2/D9/EF/D1) × all offsets × {INT16/24/32
BE/LE, FLOAT32} vs the CSV. **Face angle (FTT), club path, AoA, spin axis, face-to-path,
dynamic loft, impact location do NOT lock** (best r only 0.79–0.96, no clean scale → the
high-r hits are coincidental on 8 points). FTP = FTT − ClubPath holds exactly in the CSV,
i.e. the app derives them together. These are **computed app-side from the raw tracking**.

### EC / EE = the raw club/ball tracking (source of clubface) — provisional decode
Both are per-frame trajectory streams (`mevo/tracking.py`):
- **EC** = 4B frame header + N×**60-byte** records. off0 = monotonic time index (+32/rec).
  Smooth trajectory channels at off **16,18,32,34,38,40,42,48**; other columns are packed
  low-bytes/flags. ~168 records/shot.
- **EE** = 1B header + N×**76-byte** records. off0 = frame#, off2 = time index (+32).
  Smooth channels at off **22,28,34,48,50,54,56,58**. ~65 records/shot.

**Feasibility of computing clubface ourselves:** the determining data is present (smooth
3D-ish trajectories through impact), but turning it into validated face/path/AoA needs:
(1) full channel decode + units, (2) club-vs-ball + position-vs-velocity assignment,
(3) the device camera/radar **calibration** (cf. the device-specific `0x90` bringup blob),
(4) replicating FlightScope's impact-plane geometry. That's a separate multi-week research
spike, and hard to validate without per-frame ground truth. Recommendation: ship measured
scalars + launch direction now; treat clubface-from-tracking as future research.
