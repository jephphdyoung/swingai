# Mevo+ Integration — Work-in-progress state

_Last updated: 2026-06-21. Resume point for the SwingAI ↔ Mevo+ integration._

## Goal (decided)
- **Active full client**: SwingAI becomes the device client — discover → replay bringup
  handshake → keepalive → read shot results. Replaces the FlightScope app.
- **Tag each swing with metrics**: store ball/club speed + spin alongside each recorded
  swing and show them in the Streamlit UI next to the P-position analysis.

## Network facts (from `mevoinfo/` captures)
- Mevo = `192.168.120.169`, App/PC = `192.168.120.177` (in the captures).
- Channels: UDP 1248 (XML discovery) · **TCP 5100 (binary control + results)** ·
  TCP 1258 (GVP JSON) · HTTP 8080 (MJPEG video) · UDP 123 (NTP).
- Frame = `F0 DEST SRC TYPE payload CS_hi CS_lo F1`; CS = 16-bit sum of DEST..last payload
  byte; byte-stuffed (FD escapes: F0→FD01, F1→FD02, FD→FD03, FA→FD04). Nodes: APP=0x10,
  PI=0x12, AVR=0x30, DSP=0x40. STATUS type = 0xAA (the ~1Hz keepalive poll).

## Python `mevo/` package — BUILT (2026-06-21)
Bottom-up, validated offline against the captures:
- `frames.py` — stuffing + 16-bit checksum + streaming `Deframer`. Checksum algorithm
  derived empirically: 16-bit sum of the *stuffed* DEST..payload bytes, emitted minimal
  big-endian (1 byte if ≤0xFF else 2), then stuffed. **Round-trips all 106 captured frames
  byte-for-byte.** (Parser tries 1-byte CS first to resolve the trailing-0x00 ambiguity.)
- `metrics.py` — `ShotMetrics` + D4/ED/EF decode (ball=D4 off16, club=D4 off85 / ED off6,
  spin=EF off106).
- `pcap_source.py` — streaming `PcapReader` over device→host 5100 (no full-file load).
- `discovery.py` — UDP 1248 broadcast → `MevoDevice(host, port, instance)`.
- `client.py` — `MevoClient`: discover → connect → replay bringup → keepalive thread +
  reader thread → `on_shot(ShotMetrics)`. Streaming assembler fires per shot (anchored on
  D4, finalized on EF).
- `tests/test_mevo.py` — 7 tests, all pass. Framing round-trip + deframer reassembly +
  decode vs ground truth: **p3 ball speeds exact (91.1/101.5/97.4/97.4/85.7), p4 club exact
  & spin within ~2 rpm.**
- `pages/1_Launch_Monitor.py` — standalone Streamlit page (live device OR offline pcap
  replay). Intentionally NOT wired into `app.py` yet.
- `scapy` added to `requirements.txt` (pcap replay; imported lazily).

Known nuance: D4 off16 ("ball speed") is the **range-velocity component vₓ**, ~1 mph off
the app's vector-magnitude ball speed on angled shots (matches the RE notes). Club + spin
are exact.

## Clubface / "what's actually sent" investigation (2026-06-21) — see mevo_protocol.md §7
- **Measured scalars on the wire:** ball, club, spin (r≈1.0) + **NEW launch direction**
  (E8 off25 INT16 LE, Q8.8, r=0.997 — now in `ShotMetrics.launch_direction_deg`, experimental).
- **Clubface metrics (face angle, club path, AoA, spin axis, FTP, dyn loft, impact) are NOT
  transmitted as scalars** — confirmed by exhaustive brute force. The app computes them from
  the **raw club/ball tracking** streamed in `EC`/`EE`.
- `mevo/tracking.py` — provisional EC (60B records) / EE (76B records) extractor: time index
  + smooth trajectory channels identified, **uncalibrated**. Computing clubface ourselves is
  a separate multi-week spike (needs channel units + club/ball assignment + device calibration
  + impact geometry). Deferred.
- Useful byproducts: `E5` text gives **club-impact epoch (µs)** and `0x84` a per-shot
  timestamp → both directly usable for the shot↔video pairing question.

## Earlier DONE (reverse-engineering)
1. `mevoinfo/` copied into repo; `*.pcapng` gitignored, docs/csv committed.
2. **Read path proven** — my own framer+unstuff+decode reproduced documented ground truth
   exactly: D4 off16 INT24-BE mm/s ball speed = 91.1/101.5/97.4/97.4/85.7 mph (p3 shots 1-5).
   Club speed = D4 off85; (still need to re-confirm in code) spin = EF off106 INT16-BE RPM,
   club = ED off6 INT16-BE cm/s.
3. **Bringup handshake extracted** → `mevo/handshake_capture.json` (102 host→device frames,
   time-ordered, from `mevo_app2.pcapng`). Keepalive set = the 4 distinct AA frames:
   `f04010aa010100fcf1`, `f03010aa010100ecf1`, `f01210aa010100cef1`, `f01210aa010300d0f1`.
4. Discovery XML confirmed: query =
   `<?xml version="1.0"?><ServiceQuery><ServiceName>FlightScope Device</ServiceName></ServiceQuery>`
   broadcast to `<subnet>.255:1248`; reply `<ServiceInfo>` gives Host + Port 5100.
5. GVP (1258) handshake seen: app sends `{"type":"CONFIG_REQUEST","version":1}` then a CONFIG.

## LIVE TEST (2026-06-21) — bringup PROVEN on hardware
Ran `scripts/mevo_connect_test.py` against the real unit (still at 192.168.120.169 via DHCP
reservation; same serial M2-047295).
- **Bringup handshake ACCEPTED** — incl. the device-specific `0x90` blob (worked because same
  unit; still unverified on a *different* unit). Link stable 20s, keepalive holding,
  184 device→host frames, **0 checksum errors**. DSP sent `#DSP: CameraParam Read PASS`.
- **UDP 1248 discovery got no reply** (AP likely drops the broadcast reply) → use `--host`.
  Direct TCP 5100 connect works fine.
- **NOT yet armed for shot detection**: only 1 `D3` radar frame + no ARMED/IDLE text in the
  window. The 102-frame bringup connects+responds but full arming probably needs the
  **alignment/radar-config** step (`0x30/A4` + AVR config, ~99× in `mevo_p2_alignment`) and/or
  **GVP TCP 1258**. NEXT: replay alignment to arm, then verify a real shot → `[SHOT]` line.

## CAVEAT to verify
- The PI `0x90` bringup frames carry a base64 blob (`YAAQ...`) that is **likely
  device-specific calibration/license for serial M2-047295**. Replaying verbatim may not work
  on a different unit. NEEDS testing against the real device, or finding which fields are
  device-specific vs constant.

## NEXT STEPS (resume here — 2026-06-22)
Offline package built & tested (8 pass). Live bringup PROVEN (see LIVE TEST above). Resume:

1. **Confirm shot detection works (start here).** Power on device, then:
   `venv/bin/python scripts/mevo_connect_test.py --host 192.168.120.169 --seconds 60`
   and take a swing. If a `[SHOT]` line prints with sane ball/club/spin → whole pipeline
   proven end-to-end. (Discovery's UDP reply doesn't come back on this AP — always use `--host`.)
2. **If no shot registers → arm the device.** Extract the alignment/radar-config frames from
   `mevoinfo/mevo_p2_alignment.pcapng` (host→device: `0x30/A4` ~99×, AVR config types
   be/d2/d0/23/9b) and append to the bringup; possibly also bring up **GVP TCP 1258**
   (`{"type":"CONFIG_REQUEST","version":1}` + CONFIG). Re-test until `D3` streams heavily +
   ARMED/IDLE text appears, then a swing yields `[SHOT]`.
3. **Auto-refresh** the live Streamlit page (manual Refresh button today) — `st_autorefresh`.
4. **Pair shot ↔ video** (open product question below); then optionally surface metrics on the
   Analyze page in `app.py`. Page is standalone for now.

Files added this session: `mevo/{frames,metrics,discovery,client,pcap_source,tracking}.py`,
`mevo/__init__.py`, `tests/test_mevo.py`, `pages/1_Launch_Monitor.py`,
`scripts/mevo_connect_test.py`. Nothing committed yet.
4. Stretch (RE, not needed for MVP): decode E8 polynomial (launch angle/direction) and
   EC PRC point-clouds for the computed/angular metrics.

## How to run analysis env
`cd <repo>; source venv/bin/activate` (scapy installed there). Captures in `mevoinfo/`.
No tshark/tcpdump-only on this box — use scapy `rdpcap`.

## Open product question (not yet decided)
- How a shot is paired to a video clip in time (trigger timestamp ↔ video record start).
  NTP on the device (UDP 123) means clocks are synced ~1/sec — usable for alignment.
