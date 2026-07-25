# Mevo+ Protocol Reverse-Engineering — Info Pack

Reverse-engineering of the FlightScope Mevo+ network protocol (own device, own network),
June 2026. All traffic is plaintext over Wi-Fi.

## Start here
- **`mevo_protocol.md`** — the full protocol reference (channel map, frame format, decoded
  fields). Section 6 is the validated full-swing protocol + CSV cross-check.
- **`project_log.md`** — chronological log of how we got here, with next steps and gotchas.

## Files
| File | What it is |
|---|---|
| `mevo_protocol.md` | Main reference doc |
| `project_log.md` | Running project notes / next steps |
| `8shots_windows-1252.csv` | App export — ground truth for the 8 full swings |
| `mevo_boot.pcapng` | Power-on / boot |
| `mevo_app.pcapng`, `mevo_app2.pcapng` | App connect + discovery handshake |
| `mevo_putting.pcapng` | Putting session (camera path) |
| `mevo_p1_poweron_init.pcapng` | Full-swing day: power-on |
| `mevo_p2_alignment.pcapng` | Alignment + radar settings |
| `mevo_p3_fullswing.pcapng` | 5 full swings (session 1) |
| `mevo_p4_fullswing2.pcapng` | **8 full swings — main analysis set (matches the CSV)** |

## Key findings (summary)
- **Transport:** UDP 1248 (XML discovery) → TCP 5100 (binary control + results),
  TCP 1258 (JSON camera/GVP), HTTP 8080 (MJPEG video), UDP 123 (NTP).
- **Frame:** `F0 | DEST | SRC | TYPE | payload | CS_hi | CS_lo | F1`; addresses
  APP=0x10, PI=0x12, AVR=0x30, DSP=0x40; byte-stuffing with `FD`; 16-bit checksum;
  frames span TCP segments (reassemble first).
- **Decoded & validated (r≈1.0 vs CSV):**
  - Ball speed = D4 off 16, INT24 BE, mm/s (= range velocity component)
  - Club speed = D4 off 85 (mm/s) / ED off 6 (cm/s)
  - Spin rate = EF off 106, INT16 BE, raw RPM
- **Computed by the app (not on the wire):** launch angle/direction, spin axis, AoA,
  club path, loft, carry, total, apex, etc.
- **Next:** decode the E8 polynomial (launch angle/direction) and ED/EF/EC sub-records
  (spin axis, AoA); the EC PRC point-clouds hold the raw tracking data.

## Reference
External decode that matched our captures: github.com/divotmaker/ironsight (docs/ =
WIRE.md, SEQUENCE.md, CAMERA.md).

## Tools
Analyzed with Wireshark/tshark (`C:\Program Files\Wireshark\tshark.exe`) + PowerShell.
