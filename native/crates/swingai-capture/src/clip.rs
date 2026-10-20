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

/// The interval a trigger asked for, and whether the session clock could express
/// all of it.
///
/// # Three facts, kept apart on purpose
///
/// - [`start`](Self::start) — where extraction actually began. Floored at the
///   session origin, because timestamps are unsigned and nothing precedes zero.
/// - [`pre_roll`](Self::pre_roll) — the duration that was *asked for*.
/// - [`reaches_before_origin`](Self::reaches_before_origin) — whether that
///   duration extended past the origin, so `start` is the floor rather than the
///   instant requested.
///
/// Collapsing the third into the first is a bug this type exists to prevent. A
/// trigger at 4ms asking for 30 seconds of pre-roll floors to `start == 0`, and
/// a buffer holding everything since the session began does reach zero — so a
/// check of "did the buffer reach `start`" alone answers *yes, complete*, for a
/// clip containing 4ms of a requested 30 seconds. Reaching the origin does not
/// satisfy a longer requested duration; there simply was not that much history.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PreRollWindow {
    start: Timestamp,
    end: Timestamp,
    pre_roll: Duration,
    reaches_before_origin: bool,
}

impl PreRollWindow {
    /// The window a trigger at `at` reaching back `pre_roll` asks for.
    #[must_use]
    pub fn new(at: Timestamp, pre_roll: Duration) -> Self {
        // A pre-roll too long for `u64` nanoseconds is tracked as its own fact
        // rather than saturated into one. Saturating alone gets the extreme case
        // backwards: a pre-roll of 584 billion years clamps to `u64::MAX`
        // nanoseconds, which compares equal to a trigger at `u64::MAX` and so
        // looks like it fits — when it is the most obviously impossible request
        // there is.
        let (pre_roll_nanos, longer_than_the_clock) = match u64::try_from(pre_roll.as_nanos()) {
            Ok(nanos) => (nanos, false),
            Err(_) => (u64::MAX, true),
        };

        Self {
            start: Timestamp::from_nanos(at.as_nanos().saturating_sub(pre_roll_nanos)),
            end: at,
            pre_roll,
            reaches_before_origin: longer_than_the_clock || pre_roll_nanos > at.as_nanos(),
        }
    }

    /// A plain inclusive range, for extracting without a trigger behind it.
    ///
    /// Nothing is floored here — the caller named both ends — so the requested
    /// duration is by definition expressible.
    #[must_use]
    pub fn between(start: Timestamp, end: Timestamp) -> Self {
        Self {
            start,
            end,
            pre_roll: end.duration_since(start).unwrap_or_default(),
            reaches_before_origin: false,
        }
    }

    /// Where extraction begins. May be the session origin even when more was
    /// asked for — see [`reaches_before_origin`](Self::reaches_before_origin).
    #[must_use]
    pub const fn start(self) -> Timestamp {
        self.start
    }

    /// The trigger instant, and the inclusive end of the window.
    #[must_use]
    pub const fn end(self) -> Timestamp {
        self.end
    }

    /// The duration that was asked for, which is not always the duration
    /// [`start`](Self::start) and [`end`](Self::end) span.
    #[must_use]
    pub const fn pre_roll(self) -> Duration {
        self.pre_roll
    }

    /// Whether `end - pre_roll` fell before the session origin.
    ///
    /// When true, the requested duration is longer than the session itself and
    /// no buffer can satisfy it, however full.
    #[must_use]
    pub const fn reaches_before_origin(self) -> bool {
        self.reaches_before_origin
    }

    /// The duration extraction could actually cover: `end - start`, which is
    /// shorter than [`pre_roll`](Self::pre_roll) exactly when the window was
    /// floored.
    #[must_use]
    pub fn expressible_span(self) -> Duration {
        self.end.duration_since(self.start).unwrap_or_default()
    }
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
    window: PreRollWindow,
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
        window: PreRollWindow,
    ) -> Self {
        assert!(!frames.is_empty(), "a stream clip must have frames");
        Self {
            descriptor,
            frames,
            gaps,
            buffered_from,
            window,
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

    /// The window this clip was extracted for.
    #[must_use]
    pub const fn window(&self) -> PreRollWindow {
        self.window
    }

    /// Whether the *full requested pre-roll duration* was available.
    ///
    /// Two independent ways for this to be `false`, and both must be checked:
    ///
    /// - the requested duration reaches before the session origin, so that much
    ///   history never existed for any camera; or
    /// - this camera's buffer does not reach the start of the window — either it
    ///   had not been running long enough, or retention evicted the frames
    ///   before the trigger arrived.
    ///
    /// Checking only the second is what made a 30-second pre-roll at a 4ms
    /// trigger report complete: the window floors to zero and a buffer that
    /// starts at zero reaches it. Reaching the origin is not the same as having
    /// the requested duration behind you.
    ///
    /// A `false` here does not mean the clip is empty. The frames that *did*
    /// exist are still returned; this says only that there are fewer of them
    /// than were asked for, which is the caller's decision to make.
    #[must_use]
    pub fn full_pre_roll_available(&self) -> bool {
        !self.window.reaches_before_origin() && self.buffered_from <= self.window.start()
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
    window: PreRollWindow,
    streams: Vec<StreamClip>,
}

impl ShotExtraction {
    pub(crate) const fn new(window: PreRollWindow, streams: Vec<StreamClip>) -> Self {
        Self { window, streams }
    }

    /// The window every stream was extracted for.
    #[must_use]
    pub const fn window(&self) -> PreRollWindow {
        self.window
    }

    #[must_use]
    pub const fn trigger_timestamp(&self) -> Timestamp {
        self.window.end()
    }

    /// Where extraction began: `trigger - pre_roll`, floored at the session
    /// origin.
    ///
    /// This is the *actual* start, not proof that the requested duration
    /// existed — [`window().reaches_before_origin()`](PreRollWindow::reaches_before_origin)
    /// is what says whether the floor was applied.
    #[must_use]
    pub const fn requested_start(&self) -> Timestamp {
        self.window.start()
    }

    /// The duration that was asked for.
    #[must_use]
    pub const fn pre_roll(&self) -> Duration {
        self.window.pre_roll()
    }

    #[must_use]
    pub fn streams(&self) -> &[StreamClip] {
        &self.streams
    }

    /// Whether every camera had the full requested pre-roll duration behind it.
    ///
    /// The origin check is stated here as well as per-stream so the answer does
    /// not depend on the stream list being non-empty: a window reaching before
    /// the session origin is incomplete regardless of how many cameras agree
    /// about it.
    #[must_use]
    pub fn full_pre_roll_available(&self) -> bool {
        !self.window.reaches_before_origin()
            && self.streams.iter().all(StreamClip::full_pre_roll_available)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MS: u64 = 1_000_000;

    #[test]
    fn a_window_inside_the_session_is_not_floored() {
        let window = PreRollWindow::new(Timestamp::from_nanos(30_000 * MS), Duration::from_secs(5));
        assert_eq!(window.start(), Timestamp::from_nanos(25_000 * MS));
        assert_eq!(window.end(), Timestamp::from_nanos(30_000 * MS));
        assert!(!window.reaches_before_origin());
        assert_eq!(window.expressible_span(), Duration::from_secs(5));
    }

    #[test]
    fn a_window_longer_than_the_session_floors_and_says_so() {
        let window = PreRollWindow::new(Timestamp::from_nanos(4 * MS), Duration::from_secs(30));

        assert_eq!(window.start(), Timestamp::ZERO, "floored, as it must be");
        assert!(window.reaches_before_origin());
        assert_eq!(
            window.pre_roll(),
            Duration::from_secs(30),
            "and the requested duration is still remembered verbatim"
        );
        assert_eq!(
            window.expressible_span(),
            Duration::from_millis(4),
            "which is not what could actually be covered"
        );
    }

    #[test]
    fn a_pre_roll_exactly_as_long_as_the_session_is_not_floored() {
        // The boundary: `trigger - pre_roll` lands exactly on the origin, which
        // is expressible, so nothing was lost.
        let window = PreRollWindow::new(Timestamp::from_nanos(30 * MS), Duration::from_millis(30));
        assert_eq!(window.start(), Timestamp::ZERO);
        assert!(!window.reaches_before_origin());
    }

    #[test]
    fn a_pre_roll_too_long_for_the_clock_reaches_before_the_origin() {
        let window = PreRollWindow::new(
            Timestamp::from_nanos(u64::MAX),
            Duration::from_secs(u64::MAX),
        );
        assert!(window.reaches_before_origin());
        assert_eq!(window.start(), Timestamp::ZERO);
    }

    #[test]
    fn a_plain_range_is_never_treated_as_floored() {
        let window = PreRollWindow::between(Timestamp::ZERO, Timestamp::from_nanos(10 * MS));
        assert!(!window.reaches_before_origin());
        assert_eq!(window.pre_roll(), Duration::from_millis(10));
    }
}
