use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use swingai_core::{
    CameraId, CameraView, FrameSequence, PixelFormat, ShotId, Timestamp, ValidationErrors,
};

use crate::{CaptureManifestVersion, ContractError, RelativePath};

/// What the capture runtime recorded for one shot.
///
/// Mirrors `schemas/capture-manifest.schema.json`. Written by Rust, read by the
/// Python analysis pipeline.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CaptureManifest {
    pub schema_version: CaptureManifestVersion,
    pub shot_id: ShotId,
    /// RFC 3339 wall-clock time the manifest was written. For humans and filing
    /// only — never for correlating streams, which is what
    /// [`Timestamp`] is for. Kept as a string because nothing in the native
    /// runtime computes on wall-clock time yet, and a date-time crate would be a
    /// dependency bought for formatting alone.
    pub created_at: String,
    /// When the shot trigger fired — the microphone hearing impact, today.
    /// Absent for a capture that was not triggered.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trigger_timestamp_ns: Option<Timestamp>,
    pub streams: Vec<CameraStreamManifest>,
    /// Top-level fields written by a newer minor version. Preserved so a
    /// round-trip through this build does not delete them.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

impl CaptureManifest {
    /// Parse and check in one step. Prefer this to `serde_json::from_str`:
    /// deserialization catches shape and schema version, this also catches the
    /// semantic constraints — ordering, uniqueness, arithmetic that cannot hold.
    pub fn from_json_str(json: &str) -> Result<Self, ContractError> {
        let manifest: Self = serde_json::from_str(json)?;
        manifest.validate().into_result()?;
        Ok(manifest)
    }

    pub fn to_json_string_pretty(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// The stream for a given view, if exactly one camera holds it.
    #[must_use]
    pub fn stream_for_view(&self, view: CameraView) -> Option<&CameraStreamManifest> {
        let mut matching = self.streams.iter().filter(|stream| stream.view == view);
        let first = matching.next()?;
        matching.next().is_none().then_some(first)
    }

    /// Everything wrong with this manifest, collected so one run of the checker
    /// reports every problem rather than the first.
    pub fn validate(&self) -> ValidationErrors {
        let mut errors = ValidationErrors::new();

        if self.created_at.trim().is_empty() {
            errors.push("created_at", "must not be empty");
        }

        if self.streams.is_empty() {
            errors.push("streams", "a capture must have at least one camera stream");
        }

        let mut seen = BTreeSet::new();
        for (index, stream) in self.streams.iter().enumerate() {
            let path = format!("streams[{index}]");
            errors.extend_at(&path, stream.validate());
            if !seen.insert(stream.camera_id.as_str()) {
                errors.push(
                    format!("{path}.camera_id"),
                    format!("{} appears on more than one stream", stream.camera_id),
                );
            }
        }

        errors
    }
}

/// One camera's contribution to a shot.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CameraStreamManifest {
    pub camera_id: CameraId,
    /// Hardware serial as the camera reports it, when available. Distinct from
    /// `camera_id`, which names the position in the booth: swapping the hardware
    /// changes this and not that.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub serial_number: Option<String>,
    pub view: CameraView,
    pub media: MediaSource,
    pub width: u32,
    pub height: u32,
    pub pixel_format: PixelFormat,
    /// Flattened into the stream object rather than nested, matching the schema.
    #[serde(flatten)]
    pub frames: FrameSequence,
    /// Where the dropped frames were, when the camera said.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub gaps: Vec<SequenceGap>,
    /// Camera-specific extras — exposure, gain, vendor fields. Opaque here and
    /// round-tripped unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Map<String, Value>>,
}

impl CameraStreamManifest {
    pub fn validate(&self) -> ValidationErrors {
        let mut errors = ValidationErrors::new();

        if self.width == 0 {
            errors.push("width", "must be at least 1");
        }
        if self.height == 0 {
            errors.push("height", "must be at least 1");
        }

        if let MediaSource::ImageSequence { pattern, .. } = &self.media
            && pattern.trim().is_empty()
        {
            errors.push("media.pattern", "must not be empty for an image sequence");
        }

        // The frame-sequence fields are flattened into this object, so their
        // error paths are already relative to it.
        errors.extend(self.frames.validate());

        let first = self.frames.first_timestamp_ns;
        let last = self.frames.last_timestamp_ns;
        let mut accounted = 0u64;

        for (index, gap) in self.gaps.iter().enumerate() {
            let path = format!("gaps[{index}]");
            errors.extend_at(&path, gap.validate());
            accounted += u64::from(gap.missing_frame_count);

            if first <= last && (gap.start_timestamp_ns < first || gap.end_timestamp_ns > last) {
                errors.push(
                    path,
                    format!(
                        "lies outside the stream's own span ({}..{})",
                        first.as_nanos(),
                        last.as_nanos()
                    ),
                );
            }
        }

        if accounted > u64::from(self.frames.dropped_frame_count) {
            errors.push(
                "gaps",
                format!(
                    "account for {accounted} missing frames, but dropped_frame_count is {}",
                    self.frames.dropped_frame_count
                ),
            );
        }

        errors
    }
}

/// Where a stream's frames live, relative to the manifest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum MediaSource {
    /// A single container file.
    Video { path: RelativePath },
    /// A directory of numbered images. Kept as an option because raw
    /// machine-vision output may not be worth encoding for research use — the
    /// codec decision is deferred, see ADR 0001.
    ImageSequence {
        path: RelativePath,
        /// printf-style filename pattern within the directory, e.g.
        /// `frame_%06d.png`.
        pattern: String,
    },
}

impl MediaSource {
    #[must_use]
    pub fn path(&self) -> &RelativePath {
        match self {
            Self::Video { path } | Self::ImageSequence { path, .. } => path,
        }
    }
}

/// A run of frames the camera reported but did not deliver.
///
/// Bounded by timestamps rather than indexes: after a gap, an index means
/// something different in this stream than in the one beside it, which is the
/// whole reason indexes are not the cross-component reference.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SequenceGap {
    /// Timestamp of the last frame delivered before the gap.
    pub start_timestamp_ns: Timestamp,
    /// Timestamp of the first frame delivered after the gap.
    pub end_timestamp_ns: Timestamp,
    pub missing_frame_count: u32,
    /// Index within this stream of the last frame before the gap. A convenience
    /// for stream-local seeking; not a cross-component time reference.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub after_frame_index: Option<u32>,
}

impl SequenceGap {
    pub fn validate(&self) -> ValidationErrors {
        let mut errors = ValidationErrors::new();

        if self.end_timestamp_ns < self.start_timestamp_ns {
            errors.push(
                "end_timestamp_ns",
                format!(
                    "is before start_timestamp_ns ({} vs {})",
                    self.end_timestamp_ns.as_nanos(),
                    self.start_timestamp_ns.as_nanos()
                ),
            );
        }

        if self.missing_frame_count == 0 {
            errors.push(
                "missing_frame_count",
                "must be at least 1; a gap with no missing frames is not a gap",
            );
        }

        errors
    }
}
