//! Platform-neutral domain types for the SwingAI native runtime.
//!
//! Everything here compiles on Windows and Linux and depends on nothing but
//! `std` and `serde`. Platform-specific code (camera SDKs, OS clocks) belongs in
//! crates that build on top of this one, never in here — see
//! `docs/adr/0001-hybrid-rust-python-runtime.md`.
//!
//! The one convention worth internalising: **time is a [`Timestamp`], never a
//! frame index.** A frame index is meaningful only within a single stream that
//! dropped nothing; a timestamp is comparable across cameras, across the
//! microphone, and across a re-encode.

mod error;
mod frame_sequence;
mod ids;
mod pixel_format;
mod timestamp;
mod view;

pub use error::{ValidationError, ValidationErrors};
pub use frame_sequence::FrameSequence;
pub use ids::{CameraId, ShotId};
pub use pixel_format::PixelFormat;
pub use timestamp::Timestamp;
pub use view::CameraView;
