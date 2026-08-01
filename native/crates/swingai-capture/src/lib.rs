//! The deterministic half of SwingAI's capture runtime: frame sources, per-camera
//! ring buffers, timestamp-based extraction, and the shot-directory writer that
//! produces a `capture-manifest.json` the analysis side can read.
//!
//! There are no cameras here. [`SyntheticSource`] stands in for one, and it is
//! enough to prove the model that the Fox/MVS integration will later have to
//! satisfy: two independent streams, a trigger, and a synchronized clip pulled
//! out of the seconds *before* it. See
//! `docs/adr/0001-hybrid-rust-python-runtime.md`.
//!
//! # Everything is on one capture-session clock
//!
//! Every [`Timestamp`](swingai_core::Timestamp) in this crate is nanoseconds
//! since the capture session's monotonic origin, and the persisted origin is
//! zero. The sources in this crate *are already in that domain* — they are
//! synthetic, so there is nothing to convert.
//!
//! **A real camera is not.** A device reports on its own clock, with its own
//! epoch and tick rate, and converting it is a measurement, not an assumption.
//! That conversion is deliberately absent from this crate: a source hands over
//! frames that are already session-clock, so when the MVS binding arrives the
//! conversion has an obvious place to live and an obvious way to be wrong.
//!
//! # Why extraction is by timestamp
//!
//! A trigger is an instant, not a frame. The microphone that will eventually
//! fire it has no frame index at all, and two cameras that drop different frames
//! disagree about what index *n* means within milliseconds of starting. So
//! [`CaptureSession::trigger`] asks each buffer for the frames whose timestamps
//! fall in `[trigger - pre_roll, trigger]`, independently, and never assumes a
//! frame rate to convert a duration into a frame count. Streams may start at
//! different instants and return different frame counts from the same request;
//! that is the correct answer, not a defect.
//!
//! # Why sequence numbers exist anyway
//!
//! Each frame carries the stream-local sequence number its source reported, and
//! it is used for exactly one thing: noticing that frames went missing. A jump
//! from 41 to 44 is three lost frames, which timestamps alone cannot distinguish
//! from a camera that simply paused. Sequence numbers are never compared across
//! cameras and never used to align anything — [`ClipGap::after_frame_index`] is
//! re-derived as an index into the extracted clip, precisely so nothing
//! downstream is tempted to treat a source sequence number as a position.
//!
//! # Why the writer emits image sequences
//!
//! [`write_shot`] stores 8-bit grayscale PGM files. PGM is a header and a block
//! of bytes, so it costs no dependency and no codec decision — and the codec
//! decision is one ADR 0001 explicitly defers, because it depends on what the
//! cameras actually deliver (likely Bayer or mono) and on whether raw frames are
//! worth keeping for research. Choosing H.264 here to make a fixture look
//! production-shaped would be deciding that by accident. The manifest already
//! carries `image_sequence` as a first-class media kind, so switching to a video
//! container later is a writer change and not a contract change.

mod clip;
mod clock;
mod error;
mod frame;
mod ring;
mod session;
mod source;
mod synthetic;
mod writer;

pub use clip::{ClipGap, PreRollWindow, ShotExtraction, StreamClip};
pub use clock::{now_utc, shot_id_for};
pub use error::{CaptureError, WriteError};
pub use frame::CapturedFrame;
pub use ring::{FrameRingBuffer, RingBufferConfig};
pub use session::CaptureSession;
pub use source::{CameraDescriptor, FrameSource};
pub use synthetic::{SyntheticSource, SyntheticSourceConfig};
pub use writer::{FRAME_FILENAME_PATTERN, WrittenShot, write_shot};

/// Re-exported so a caller driving a capture need not depend on `swingai-core`
/// directly just to name a camera or an instant.
pub use swingai_core::{
    CameraId, CameraView, PixelFormat, Rfc3339Timestamp, ShotId, Timestamp, ValidationError,
    ValidationErrors,
};
