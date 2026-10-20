use std::collections::VecDeque;
use std::time::Duration;

use swingai_core::{Timestamp, ValidationErrors};

use crate::{CameraDescriptor, CaptureError, CapturedFrame, ClipGap, PreRollWindow, StreamClip};

/// How much history one camera's buffer keeps.
///
/// Two limits, because they answer different questions. `retention` is the
/// product decision — how far back a trigger can reach. `max_payload_bytes` is
/// the safety cap: at 240fps a two-camera 5-second window is multiple gigabytes,
/// and a camera that runs faster than configured, or a retention someone typed
/// an extra zero into, must not be able to consume the machine. Whichever binds
/// first wins, and hitting the byte cap means the effective retention is shorter
/// than asked for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RingBufferConfig {
    pub retention: Duration,
    pub max_payload_bytes: u64,
}

impl RingBufferConfig {
    #[must_use]
    pub const fn new(retention: Duration, max_payload_bytes: u64) -> Self {
        Self {
            retention,
            max_payload_bytes,
        }
    }

    pub fn validate(&self) -> ValidationErrors {
        let mut errors = ValidationErrors::new();
        if self.retention.is_zero() {
            errors.push(
                "retention",
                "must be greater than zero, or the buffer holds only the newest frame",
            );
        }
        if self.max_payload_bytes == 0 {
            errors.push("max_payload_bytes", "must be greater than zero");
        }
        errors
    }
}

/// A gap as the buffer recorded it, in the buffer's own terms. Turned into a
/// [`ClipGap`] — with a clip-local index — only at extraction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RecordedGap {
    start: Timestamp,
    end: Timestamp,
    missing: u64,
}

/// One camera's rolling window of recent frames, addressed by time.
///
/// Frames go in as they arrive and fall out of the front once they are older
/// than `retention` relative to the newest frame, or once the byte cap is
/// exceeded. Nothing here assumes a frame rate: a request for `[a, b]` is
/// answered by comparing timestamps, so a camera that stutters returns fewer
/// frames rather than the wrong ones.
///
/// The buffer is strict about what it accepts. A stream whose timestamps or
/// sequence numbers move the wrong way has no usable timeline, and the failure
/// is far cheaper to diagnose at the push than three stages downstream in a
/// manifest that looks plausible.
#[derive(Debug)]
pub struct FrameRingBuffer {
    descriptor: CameraDescriptor,
    config: RingBufferConfig,
    frames: VecDeque<CapturedFrame>,
    gaps: VecDeque<RecordedGap>,
    payload_bytes: u64,
    last_timestamp: Option<Timestamp>,
    last_sequence: Option<u64>,
    dropped_frame_total: u64,
}

impl FrameRingBuffer {
    pub fn new(
        descriptor: CameraDescriptor,
        config: RingBufferConfig,
    ) -> Result<Self, CaptureError> {
        let mut errors = descriptor.validate();
        errors.extend(config.validate());
        errors.into_result()?;

        Ok(Self {
            descriptor,
            config,
            frames: VecDeque::new(),
            gaps: VecDeque::new(),
            payload_bytes: 0,
            last_timestamp: None,
            last_sequence: None,
            dropped_frame_total: 0,
        })
    }

    #[must_use]
    pub fn descriptor(&self) -> &CameraDescriptor {
        &self.descriptor
    }

    #[must_use]
    pub const fn config(&self) -> &RingBufferConfig {
        &self.config
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.frames.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// Bytes of image payload currently retained.
    #[must_use]
    pub const fn payload_bytes(&self) -> u64 {
        self.payload_bytes
    }

    /// Frames this camera has dropped since the buffer was created, including
    /// any whose gap has since been evicted.
    #[must_use]
    pub const fn dropped_frame_total(&self) -> u64 {
        self.dropped_frame_total
    }

    /// The span of time currently retained, oldest first.
    #[must_use]
    pub fn buffered_span(&self) -> Option<(Timestamp, Timestamp)> {
        Some((
            self.frames.front()?.timestamp(),
            self.frames.back()?.timestamp(),
        ))
    }

    /// Accept a frame, then evict whatever no longer fits.
    ///
    /// Every check runs before any state changes, so a rejected frame leaves the
    /// buffer exactly as it was.
    pub fn push(&mut self, frame: CapturedFrame) -> Result<(), CaptureError> {
        self.check_identity(&frame)?;
        self.check_ordering(&frame)?;

        let frame_bytes = frame.payload_len();
        if frame_bytes > self.config.max_payload_bytes {
            return Err(CaptureError::ByteLimitTooSmall {
                camera_id: self.descriptor.camera_id.clone(),
                frame_bytes,
                max_payload_bytes: self.config.max_payload_bytes,
            });
        }

        if let (Some(previous_sequence), Some(previous_timestamp)) =
            (self.last_sequence, self.last_timestamp)
        {
            let missing = frame.sequence() - previous_sequence - 1;
            if missing > 0 {
                self.dropped_frame_total += missing;
                self.gaps.push_back(RecordedGap {
                    start: previous_timestamp,
                    end: frame.timestamp(),
                    missing,
                });
            }
        }

        self.last_sequence = Some(frame.sequence());
        self.last_timestamp = Some(frame.timestamp());
        self.payload_bytes += frame_bytes;
        self.frames.push_back(frame);

        self.evict();
        Ok(())
    }

    fn check_identity(&self, frame: &CapturedFrame) -> Result<(), CaptureError> {
        let descriptor = &self.descriptor;

        if frame.camera_id() != &descriptor.camera_id {
            return Err(CaptureError::CameraMismatch {
                expected: descriptor.camera_id.clone(),
                actual: frame.camera_id().clone(),
            });
        }
        if (frame.width(), frame.height()) != (descriptor.width, descriptor.height) {
            return Err(CaptureError::DimensionsChanged {
                camera_id: descriptor.camera_id.clone(),
                expected: (descriptor.width, descriptor.height),
                actual: (frame.width(), frame.height()),
            });
        }
        if frame.pixel_format() != &descriptor.pixel_format {
            return Err(CaptureError::PixelFormatChanged {
                camera_id: descriptor.camera_id.clone(),
                expected: descriptor.pixel_format.clone(),
                actual: frame.pixel_format().clone(),
            });
        }

        Ok(())
    }

    fn check_ordering(&self, frame: &CapturedFrame) -> Result<(), CaptureError> {
        let camera_id = || self.descriptor.camera_id.clone();

        // Compared against the last frame *pushed*, not the last frame retained:
        // eviction must not make a stale timestamp acceptable again.
        if let Some(previous) = self.last_timestamp {
            if frame.timestamp() == previous {
                return Err(CaptureError::DuplicateTimestamp {
                    camera_id: camera_id(),
                    timestamp: previous,
                });
            }
            if frame.timestamp() < previous {
                return Err(CaptureError::TimestampWentBackward {
                    camera_id: camera_id(),
                    previous,
                    offered: frame.timestamp(),
                });
            }
        }

        if let Some(previous) = self.last_sequence
            && frame.sequence() <= previous
        {
            return Err(CaptureError::SequenceWentBackward {
                camera_id: camera_id(),
                previous,
                offered: frame.sequence(),
            });
        }

        Ok(())
    }

    fn evict(&mut self) {
        let Some(newest) = self.frames.back().map(CapturedFrame::timestamp) else {
            return;
        };

        // Age is measured against the newest frame rather than a wall clock:
        // the buffer has no clock of its own, and the session clock only
        // advances when frames arrive.
        while self.frames.len() > 1 {
            let oldest = self.frames.front().expect("len > 1").timestamp();
            let age = newest.duration_since(oldest).unwrap_or_default();
            if age > self.config.retention {
                self.pop_oldest();
            } else {
                break;
            }
        }

        // A single frame larger than the cap is refused at push, so this cannot
        // empty the buffer.
        while self.payload_bytes > self.config.max_payload_bytes && self.frames.len() > 1 {
            self.pop_oldest();
        }

        self.prune_gaps();
    }

    fn pop_oldest(&mut self) {
        if let Some(frame) = self.frames.pop_front() {
            self.payload_bytes -= frame.payload_len();
        }
    }

    /// Drop gaps whose leading frame is gone.
    ///
    /// A gap is only expressible while both bounding frames are retained —
    /// `after_frame_index` has to point at a frame that is actually in the clip.
    /// Keeping a half-evicted gap would either panic at extraction or produce an
    /// index into nothing.
    fn prune_gaps(&mut self) {
        let Some(oldest) = self.frames.front().map(CapturedFrame::timestamp) else {
            self.gaps.clear();
            return;
        };
        while self.gaps.front().is_some_and(|gap| gap.start < oldest) {
            self.gaps.pop_front();
        }
    }

    /// The frames whose timestamps fall inside `window`, inclusive at both ends,
    /// or `None` if there are none.
    ///
    /// Takes a [`PreRollWindow`] rather than a bare pair of instants because the
    /// clip has to carry more than the range it covers: whether the *requested*
    /// duration was expressible at all is not recoverable from `start` and `end`
    /// once the start has been floored at the session origin. Use
    /// [`PreRollWindow::between`] for a plain range with no trigger behind it.
    ///
    /// Payloads are shared, not copied — the returned frames point at the same
    /// pixels the buffer still holds.
    #[must_use]
    pub fn extract(&self, window: PreRollWindow) -> Option<StreamClip> {
        let (start, end) = (window.start(), window.end());
        if end < start {
            return None;
        }

        let frames: Vec<CapturedFrame> = self
            .frames
            .iter()
            .filter(|frame| frame.timestamp() >= start && frame.timestamp() <= end)
            .cloned()
            .collect();

        if frames.is_empty() {
            return None;
        }

        let gaps = self
            .gaps
            .iter()
            .filter(|gap| gap.start >= start && gap.end <= end)
            .map(|gap| {
                let after_frame_index = frames
                    .iter()
                    .position(|frame| frame.timestamp() == gap.start)
                    .expect("a retained gap's leading frame is retained too — see prune_gaps");
                ClipGap {
                    start_timestamp: gap.start,
                    end_timestamp: gap.end,
                    missing_frame_count: u32::try_from(gap.missing).unwrap_or(u32::MAX),
                    after_frame_index: u32::try_from(after_frame_index).unwrap_or(u32::MAX),
                }
            })
            .collect();

        let buffered_from = self
            .frames
            .front()
            .expect("frames were found, so the buffer is not empty")
            .timestamp();

        Some(StreamClip::new(
            self.descriptor.clone(),
            frames,
            gaps,
            buffered_from,
            window,
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use swingai_core::{CameraId, CameraView, PixelFormat};

    use super::*;

    const FRAME_BYTES: usize = 4;

    fn descriptor() -> CameraDescriptor {
        CameraDescriptor {
            camera_id: CameraId::new("cam").unwrap(),
            view: CameraView::FaceOn,
            width: 2,
            height: 2,
            pixel_format: PixelFormat::new(PixelFormat::MONO8).unwrap(),
            nominal_fps: 100.0,
        }
    }

    fn buffer(retention_ms: u64, max_bytes: u64) -> FrameRingBuffer {
        FrameRingBuffer::new(
            descriptor(),
            RingBufferConfig::new(Duration::from_millis(retention_ms), max_bytes),
        )
        .unwrap()
    }

    fn frame(sequence: u64, nanos: u64) -> CapturedFrame {
        CapturedFrame::new(
            CameraId::new("cam").unwrap(),
            sequence,
            Timestamp::from_nanos(nanos),
            2,
            2,
            PixelFormat::new(PixelFormat::MONO8).unwrap(),
            vec![sequence as u8; FRAME_BYTES],
        )
    }

    #[test]
    fn a_zero_retention_or_zero_cap_is_rejected() {
        assert!(
            FrameRingBuffer::new(descriptor(), RingBufferConfig::new(Duration::ZERO, 1_024))
                .is_err()
        );
        assert!(
            FrameRingBuffer::new(
                descriptor(),
                RingBufferConfig::new(Duration::from_millis(10), 0)
            )
            .is_err()
        );
    }

    #[test]
    fn a_frame_from_another_camera_is_rejected() {
        let mut buffer = buffer(100, 1_024);
        let foreign = CapturedFrame::new(
            CameraId::new("other").unwrap(),
            0,
            Timestamp::ZERO,
            2,
            2,
            PixelFormat::new(PixelFormat::MONO8).unwrap(),
            vec![0; FRAME_BYTES],
        );
        assert!(matches!(
            buffer.push(foreign),
            Err(CaptureError::CameraMismatch { .. })
        ));
    }

    #[test]
    fn a_rejected_frame_leaves_the_buffer_untouched() {
        let mut buffer = buffer(100, 1_024);
        buffer.push(frame(0, 0)).unwrap();
        let before = (buffer.len(), buffer.payload_bytes());

        assert!(buffer.push(frame(1, 0)).is_err(), "duplicate timestamp");
        assert_eq!((buffer.len(), buffer.payload_bytes()), before);
    }

    #[test]
    fn a_frame_larger_than_the_cap_is_refused_rather_than_silently_dropped() {
        let mut buffer = buffer(100, FRAME_BYTES as u64 - 1);
        let error = buffer.push(frame(0, 0)).unwrap_err();
        assert!(matches!(error, CaptureError::ByteLimitTooSmall { .. }));
        assert!(buffer.is_empty());
        assert!(error.to_string().contains("retains nothing"), "{error}");
    }

    #[test]
    fn eviction_never_empties_the_buffer() {
        // Retention shorter than the frame interval: each new frame is already
        // "too old" relative to itself, but the newest must survive.
        let mut buffer = buffer(1, FRAME_BYTES as u64 * 8);
        for i in 0..5u64 {
            buffer.push(frame(i, i * 10_000_000)).unwrap();
        }
        assert_eq!(buffer.len(), 1);
        assert_eq!(buffer.payload_bytes(), FRAME_BYTES as u64);
    }

    #[test]
    fn payload_accounting_survives_repeated_eviction() {
        let mut buffer = buffer(20, FRAME_BYTES as u64 * 4);
        for i in 0..200u64 {
            buffer.push(frame(i, i * 1_000_000)).unwrap();
        }
        assert_eq!(
            buffer.payload_bytes(),
            buffer.len() as u64 * FRAME_BYTES as u64
        );
        assert!(buffer.len() <= 4, "byte cap holds four frames");
    }

    #[test]
    fn an_empty_window_extracts_nothing() {
        let mut buffer = buffer(1_000, 1_024);
        buffer.push(frame(0, 5_000)).unwrap();
        assert!(
            buffer
                .extract(PreRollWindow::between(
                    Timestamp::from_nanos(0),
                    Timestamp::from_nanos(1_000)
                ))
                .is_none()
        );
        assert!(
            buffer
                .extract(PreRollWindow::between(
                    Timestamp::from_nanos(9_000),
                    Timestamp::from_nanos(1_000)
                ))
                .is_none(),
            "a backwards window is empty, not a panic"
        );
    }

    #[test]
    fn extraction_shares_the_pixels_it_returns() {
        let mut buffer = buffer(1_000, 1_024);
        buffer.push(frame(0, 1_000)).unwrap();
        let clip = buffer
            .extract(PreRollWindow::between(
                Timestamp::from_nanos(0),
                Timestamp::from_nanos(2_000),
            ))
            .unwrap();

        let buffered = buffer.frames.front().unwrap().payload_handle();
        assert!(Arc::ptr_eq(buffered, clip.frames()[0].payload_handle()));
    }

    #[test]
    fn a_gap_whose_leading_frame_was_evicted_is_forgotten() {
        let mut buffer = buffer(10, FRAME_BYTES as u64 * 64);
        buffer.push(frame(0, 0)).unwrap();
        // Two frames missing, then a frame far enough ahead to evict frame 0.
        buffer.push(frame(3, 50_000_000)).unwrap();
        assert_eq!(buffer.len(), 1, "the leading frame aged out");
        assert!(
            buffer.gaps.is_empty(),
            "so its gap is no longer expressible"
        );
        // The drop is still counted for the stream's lifetime.
        assert_eq!(buffer.dropped_frame_total(), 2);
    }
}
