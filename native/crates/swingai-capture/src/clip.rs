use std::time::Duration;

use swingai_core::{FrameSequence, Timestamp};

use crate::{CameraDescriptor, CapturedFrame};

/// A run of frames missing from the middle of an extracted clip.
///
/// Bounded by the timestamps of the frames either side of the hole, because that
/// is what survives being copied to another machine or resampled to 60Hz.
/// `after_frame_index` is an index into *this clip*, recomputed at extraction —
/// the source's own sequence numbers are not it, and never leave this crate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClipGap {
    /// Timestamp of the last frame delivered before the hole.
    pub start_timestamp: Timestamp,
    /// Timestamp of the first frame delivered after the hole.
    pub end_timestamp: Timestamp,
    pub missing_frame_count: u32,
    /// Index within the extracted clip of the frame at `start_timestamp`.
    pub after_frame_index: u32,
}

/// One camera's frames for one trigger, already restricted to the requested
/// window.
///
/// Non-empty by construction: [`FrameRingBuffer::extract`](crate::FrameRingBuffer::extract)
/// returns `None` rather than an empty clip, so every accessor here is
/// infallible and the manifest — which requires `frame_count >= 1` — can always
/// be built from one.
#[derive(Debug, Clone)]
pub struct StreamClip {
    descriptor: CameraDescriptor,
    frames: Vec<CapturedFrame>,
    gaps: Vec<ClipGap>,
    buffered_from: Timestamp,
    requested_start: Timestamp,
}

impl StreamClip {
    /// # Panics
    ///
    /// If `frames` is empty. Callers inside this crate check first; the type's
    /// whole contract is that a clip has frames.
    pub(crate) fn new(
        descriptor: CameraDescriptor,
        frames: Vec<CapturedFrame>,
        gaps: Vec<ClipGap>,
        buffered_from: Timestamp,
        requested_start: Timestamp,
    ) -> Self {
        assert!(!frames.is_empty(), "a stream clip must have frames");
        Self {
            descriptor,
            frames,
            gaps,
            buffered_from,
            requested_start,
        }
    }

    #[must_use]
    pub fn descriptor(&self) -> &CameraDescriptor {
        &self.descriptor
    }

    #[must_use]
    pub fn frames(&self) -> &[CapturedFrame] {
        &self.frames
    }

    #[must_use]
    pub fn gaps(&self) -> &[ClipGap] {
        &self.gaps
    }

    #[must_use]
    pub fn first_timestamp(&self) -> Timestamp {
        self.frames
            .first()
            .expect("non-empty by construction")
            .timestamp()
    }

    #[must_use]
    pub fn last_timestamp(&self) -> Timestamp {
        self.frames
            .last()
            .expect("non-empty by construction")
            .timestamp()
    }

    /// The oldest instant the camera's buffer still held when the trigger fired,
    /// which may be earlier than this clip's first frame.
    #[must_use]
    pub const fn buffered_from(&self) -> Timestamp {
        self.buffered_from
    }

    /// Whether the buffer reached back to the start of the requested window.
    ///
    /// `false` means the pre-roll is short — either the camera had not been
    /// running long enough, or retention evicted the frames before the trigger
    /// arrived. Both are the same thing to a caller deciding whether the clip is
    /// usable, and both are reported rather than silently returning whatever was
    /// there.
    #[must_use]
    pub fn full_pre_roll_available(&self) -> bool {
        self.buffered_from <= self.requested_start
    }

    /// Frames the camera reported but that never reached this clip.
    #[must_use]
    pub fn dropped_frame_count(&self) -> u32 {
        self.gaps
            .iter()
            .map(|gap| gap.missing_frame_count)
            .fold(0u32, u32::saturating_add)
    }

    /// The timing spine of this clip, in the shape the manifest wants.
    ///
    /// Describes the extracted window only — never the source's whole lifetime.
    #[must_use]
    pub fn frame_sequence(&self) -> FrameSequence {
        FrameSequence {
            frame_count: u32::try_from(self.frames.len()).unwrap_or(u32::MAX),
            nominal_fps: self.descriptor.nominal_fps,
            first_timestamp_ns: self.first_timestamp(),
            last_timestamp_ns: self.last_timestamp(),
            dropped_frame_count: self.dropped_frame_count(),
        }
    }
}

/// Everything one trigger pulled out of a capture session.
#[derive(Debug, Clone)]
pub struct ShotExtraction {
    trigger_timestamp: Timestamp,
    requested_start: Timestamp,
    pre_roll: Duration,
    streams: Vec<StreamClip>,
}

impl ShotExtraction {
    pub(crate) const fn new(
        trigger_timestamp: Timestamp,
        requested_start: Timestamp,
        pre_roll: Duration,
        streams: Vec<StreamClip>,
    ) -> Self {
        Self {
            trigger_timestamp,
            requested_start,
            pre_roll,
            streams,
        }
    }

    #[must_use]
    pub const fn trigger_timestamp(&self) -> Timestamp {
        self.trigger_timestamp
    }

    /// The start of the requested window: `trigger - pre_roll`, floored at the
    /// session origin, since nothing precedes it.
    #[must_use]
    pub const fn requested_start(&self) -> Timestamp {
        self.requested_start
    }

    #[must_use]
    pub const fn pre_roll(&self) -> Duration {
        self.pre_roll
    }

    #[must_use]
    pub fn streams(&self) -> &[StreamClip] {
        &self.streams
    }

    /// Whether every camera reached back to the start of the requested window.
    #[must_use]
    pub fn full_pre_roll_available(&self) -> bool {
        self.streams.iter().all(StreamClip::full_pre_roll_available)
    }
}
