# ADR 0001 — Hybrid Rust/Python runtime

- **Status:** Accepted
- **Date:** 2026-07-31
- **Supersedes:** the "defer Rust until a proven hot path" line in `docs/ROADMAP.md`

## Context

SwingAI today is a Python research prototype: MediaPipe pose extraction, heuristic
P-position detection, DTW experiments, and a Streamlit UI. It reads video files that
somebody else recorded.

The next product step (see `docs/ROADMAP.md`) is a capture booth: two cameras running
continuously at 240fps into a ring buffer, a microphone listening for impact, and a
retroactive clip extracted from the seconds *before* the trigger fired. That is a
soft-real-time job with hard requirements Python is a poor fit for — a continuous
acquisition loop, a second always-listening audio thread, multi-GB ring buffers, and an
audio-timestamp-to-frame mapping that has to be correct to a couple of milliseconds.

None of that changes the fact that the research work — pose models, club detection, rule
discovery, evaluation — is where Python is strongest and where the project's actual
knowledge lives.

## Decision

Split the system along the capture/analysis boundary.

**Rust owns the runtime:** camera capture, frame timing, cross-camera synchronization,
ring buffers, session storage on disk, and production orchestration. Later, ONNX
inference.

**Python stays the research environment:** pose and club models, dataset preparation,
rule discovery, evaluation harnesses, PyTorch training, and the current analysis pipeline
in `analyzer.py` / `utils/`.

**They talk through versioned JSON files and subprocess execution.** Rust writes a
capture manifest; Python reads it, analyses the clip, and writes an analysis result.

This PR lays the foundation only: the ADR, the two JSON contracts, and a Rust workspace
with the corresponding types and a contract-checking CLI. No camera code.

## Rationale

### Why Rust is being introduced

Continuous capture is not a hot path we can optimize into later — it is a different
execution model. A 240fps acquisition loop cannot tolerate a GC-style pause or the GIL
serializing two camera threads against an audio thread, and a 5-second two-camera ring
buffer is multi-gigabyte, so copies matter. Rust gives predictable latency and real
threads without a rewrite risk at the point where the timing has to be right.

The trigger path is the specific thing that forced this: the microphone timestamp has to
resolve to a frame in the video ring buffer, which needs one monotonic clock shared by
the audio and camera threads. Owning both threads in one runtime is the only way to make
that measurable rather than assumed.

### Why Python remains part of the project

The research loop is Python's: MediaPipe, PyTorch, NumPy/SciPy, notebooks, and 207
existing tests encoding what has already been learned about P-position detection. Porting
that to Rust would buy nothing — the analysis runs after the replay has already started,
so it is off the critical path — and would throw away the regression baseline that tells
us whether a detector change helped.

### Why the capture boundary is the first Rust responsibility

It is the part that Python does badly, the part that does not exist yet (so there is
nothing to port), and the part with the cleanest interface: everything downstream needs
only "here are the frames, here is when each one happened."

### Why communication is versioned JSON files and subprocesses

A file is the simplest thing that gives us a durable record of what was captured, and a
subprocess is the simplest thing that gives us process isolation — a MediaPipe crash
cannot take down the capture loop. Both are debuggable by reading them.

The alternatives all cost more than they are worth today: an embedded Python runtime
couples the two lifecycles and the CUDA/MediaPipe dependency trees; a message queue or
local network service adds an operational component to a single-box application; a shared
memory transport is premature before we know the analysis call rate (roughly one per
swing).

### Why the schema version must match exactly, for now

Versioning is explicit because the two sides will be edited independently. A first draft
of this ADR said consumers would accept any compatible minor version and preserve fields
they did not recognize. That was retracted before any of it shipped, because the
implementation could not honour it: unknown fields were preserved only at the document
root, so a field added inside a stream or an event would have been dropped silently. A
compatibility guarantee that holds in some places and not others is worse than none — it
looks like it works.

So both contracts accept **exactly** the version they were built for and reject anything
else with a clear error. Writer and reader move together. The extension points that
remain are the explicit `metadata` and `context` maps, which are open by design and do
round-trip unchanged; unknown keys anywhere else are ignored on read and not preserved.

A later ADR can introduce real minor-version compatibility once there is a concrete need
for two versions to coexist — with per-object preservation to back it up, not a promise
in prose.

### Why timestamps are canonical, not frame indexes

A frame index only means something relative to one stream that never dropped a frame.
Cross-component, it is a lie waiting to happen:

- Two cameras that drop different frames diverge, and every downstream index is off.
- The microphone has no frame index at all — the trigger is an instant.
- Analysis output has to survive a re-encode, a trim, or a decimation to 60Hz for
  detection. The existing `data/ground_truth.json` schema already keys on `timestamp_ms`
  for exactly this reason.

So every cross-component time in these contracts is an **integer count of nanoseconds
since the capture session's monotonic origin**, and frame indexes are stream-local
bookkeeping only.

**The persisted origin is zero.** Every `*_timestamp_ns` in a document is an offset from
the start of its own capture session, which makes values comparable within a session,
meaningless across sessions, and never negative. The representation is unsigned, so a
negative timestamp cannot be constructed or deserialized at all — if one shows up, a
clock-domain conversion went wrong upstream and the seam should say so rather than pass
it along.

Nanoseconds because a 240fps frame period is 4.17ms — millisecond integers would quantize
frame timing to 24% of a frame. `u64` nanoseconds spans 584 years, and a session's
offsets stay under 2^53 for its first 104 days, so values remain exactly representable in
JSON consumers that use doubles.

#### Device clocks are not the session clock

Cameras and audio devices report time on **their own clocks**. They may use a different
epoch, a different tick frequency, or a counter that is not nanoseconds at all — a
machine-vision SDK frame stamp does not inherently share this clock domain, and must not
be assumed to. Converting is the capture runtime's job: measure each device's offset and
rate against the session clock rather than assuming them, and write only converted values.

Raw device stamps may be retained alongside, in a stream's `metadata` map, for diagnosing
drift. They must never appear as a `*_timestamp_ns`. This is precisely the mapping that
STATUS.md flags as the riskiest unknown in the trigger path, so the contract is built to
make an unconverted value obvious rather than plausible.

Wall-clock creation times are separate fields, validated as RFC 3339 on the way in, and
are for humans and filing — never for correlating streams. They are a distinct Rust type
from session timestamps so the two cannot be confused at a call site.

### Why contract paths are forward-slash regardless of host

A contract path is always spelled with `/`, and a backslash is rejected outright on every
platform rather than normalised. Normalising would be guesswork: on Linux
`clips\shot.mkv` is a single legal filename that happens to contain a backslash, so
reinterpreting it as a directory separator invents a path the writer never meant. Since
SwingAI writes manifests on Windows and reads them on Linux, one spelling has to win, and
rejecting at the seam turns a Windows writer emitting native separators into a loud bug in
that writer.

That one rule also disposes of `C:\...`, UNC `\\server\share` and extended-length
`\\?\C:\...` spellings for free, leaving only Unix-absolute, drive-relative `C:rel`, and
`..` traversal to reject explicitly.

### Why Windows 11 is the first production target

The cameras are the constraint. The Fox / Hikrobot MVS SDK is best supported on Windows,
and the capture booth is a dedicated machine, not a fleet. Targeting one OS first means
the camera integration can be written once against the vendor's primary platform.

### Why core crates must stay portable to Linux

Development, CI, and all the GPU/model work happen on the Fedora box described in
`docs/STATUS.md`. If the domain types and contract handling only built on Windows, every
contract change would require the Windows machine, and the Python side — which is
developed on Linux — could not be tested against the Rust types.

So the split is: `swingai-core` and `swingai-contracts` contain no platform-specific code
and must compile and test on both. Windows-specific code, when it arrives, goes in a
separate capture crate behind a trait, and stays there.

### Why SwingAI avoids Registry-backed application state

Configuration lives in files next to the application or under the user's data directory.
The Registry is invisible to `git diff`, cannot be copied to another booth machine by
copying a folder, does not exist on Linux (so it would immediately fork the config path),
and survives uninstall. None of that is a trade we want for a single-box application whose
state is already file-shaped.

### Why the existing Python implementation is retained, not rewritten

It is the baseline. `docs/STATUS.md` is explicit that detection quality is *unmeasured* —
there is no `data/ground_truth.json` yet. Rewriting an unvalidated detector in another
language would replace one unmeasured implementation with a second unmeasured
implementation and destroy the ability to tell them apart. The Python suite stays as the
regression harness until there is a measurement that says a port is an improvement.

## Consequences

- Two toolchains. Contributors touching capture need Rust; contributors touching models
  need the Python container.
- Contract changes are now a two-sided edit. That is the cost of the seam, and the reason
  the schemas and the round-trip tests exist.
- **Exact version matching means a contract change is a lockstep deploy.** Bumping
  `schema_version` breaks every reader until it is rebuilt. Acceptable while both sides
  ship together on one machine; it is the first thing to revisit when that stops being
  true.
- The capture runtime now owes a **measured** conversion from each device clock to the
  session clock before it can write a manifest at all. That is deliberate — it forces the
  riskiest unknown in the trigger path to be confronted rather than assumed — but it means
  the first capture PR carries that work.
- `swingai-core` takes one dependency beyond serde: `time`, for RFC 3339 parsing, with
  default features off so no timezone database or system clock comes with it. The
  portability test enforces both the allow-list and the feature restriction.
- The `dtl` / `face_on` view names used by `utils/swing_pairing.py` and
  `data/annotations.json` do not match the contract's `down_the_line` / `face_on`. The
  contract spells it out; whichever component bridges the two performs the mapping. The
  Rust side rejects `dtl` rather than guessing.
- Path A of the P-position code calls the velocity peak `P6` while path B calls it `P7`
  (see `CLAUDE.md`). The contract's `SwingEvent.name` is a free string and deliberately
  does not adjudicate this. It must be reconciled before any cross-path evaluation —
  tracked as B3 in `docs/STATUS.md`.

## Deferred decisions

None of these are settled by this ADR, and none should be inferred from it.

| Decision | Why deferred |
|---|---|
| **Fox / MVS SDK integration** | Needs the USB3 bandwidth arithmetic for 2×240fps first, and the vendor API shape drives the capture trait. |
| **Desktop UI framework** | The replay surface can stay Streamlit or become native; nothing in the capture boundary forces the choice yet. |
| **Video codec and container** | Depends on what the cameras deliver (likely Bayer/mono) and whether we store raw for research or encode for playback. The manifest carries a pixel format and a path, so either works. |
| **ONNX Runtime integration** | Only worth it once a model is chosen and the Python detector has a measured baseline to beat. |
| **Hardware camera synchronization** | Software trigger with per-frame timestamps may be sufficient; measure the drift before buying a sync cable. |
| **Installer / updater** | Not until there is a second machine to install on. |
