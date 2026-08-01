//! Rust types for the versioned JSON contracts SwingAI's capture runtime and
//! analysis pipeline exchange.
//!
//! The authoritative documents are `schemas/capture-manifest.schema.json` and
//! `schemas/analysis-result.schema.json`; these types mirror them. The reasoning
//! behind the seam is in `docs/adr/0001-hybrid-rust-python-runtime.md`.
//!
//! Four conventions run through everything here:
//!
//! - **Times are [`Timestamp`](swingai_core::Timestamp)s**, integer nanoseconds
//!   since the capture session's monotonic origin, which is zero as persisted.
//!   Frame indexes are stream-local bookkeeping and never a cross-component
//!   reference. Wall-clock times are a separate type,
//!   [`Rfc3339Timestamp`](swingai_core::Rfc3339Timestamp), and are for filing
//!   only.
//! - **Paths are [`RelativePath`]s** — forward-slash, host-independent, resolved
//!   against the directory holding the document. A capture folder can be moved
//!   or copied to another machine without rewriting it.
//! - **The schema version must match exactly.** Both sides move together; a
//!   mismatch is a clear error rather than a silent partial read. See
//!   [`SchemaVersion`].
//! - **`metadata` and `context` maps are the only open extension points.**
//!   Unknown keys inside them round-trip unchanged; unknown keys elsewhere are
//!   ignored on read and not preserved.
//!
//! Parse through [`CaptureManifest::from_json_str`] or
//! [`AnalysisResult::from_json_str`] rather than `serde_json::from_str`: those
//! also run the semantic checks that deserialization cannot express.

mod analysis;
mod capture;
mod error;
mod relative_path;
mod version;

pub use analysis::{
    AnalysisResult, AnalysisStatus, AnalyzerInfo, Artifact, Confidence, Diagnostic, Measurement,
    SwingEvent, TimestampRange,
};
pub use capture::{CameraStreamManifest, CaptureManifest, MediaSource, SequenceGap};
pub use error::ContractError;
pub use relative_path::RelativePath;
pub use version::{AnalysisResultVersion, CaptureManifestVersion, SchemaVersion};

/// Re-exported so downstream crates need not depend on `swingai-core` directly
/// just to name a timestamp.
pub use swingai_core::{
    CameraId, CameraView, FrameSequence, PixelFormat, Rfc3339Timestamp, ShotId, Timestamp,
};
