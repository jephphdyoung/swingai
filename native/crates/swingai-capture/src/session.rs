use std::time::Duration;

use swingai_core::{CameraId, Timestamp};

use crate::{
    CameraDescriptor, CaptureError, CapturedFrame, FrameRingBuffer, PreRollWindow,
    RingBufferConfig, ShotExtraction,
};

/// A capture session: one ring buffer per camera, one shared clock, and a
/// trigger that pulls the same interval out of all of them.
///
/// Cameras are held in a `Vec` rather than a fixed pair. The booth uses two
/// today, but ADR 0001 leaves an overhead and an impact camera on the table, and
/// a coordinator that hard-codes two would have to be rewritten rather than
/// configured. Lookup is linear over a handful of cameras, which is not worth a
/// map.
///
/// Buffers are fully independent. Nothing here interleaves streams, compares one
/// camera's sequence numbers to another's, or lets a drop in one shift the
/// other: push order across cameras cannot affect the result, because a frame
/// only ever touches its own buffer.
#[derive(Debug, Default)]
pub struct CaptureSession {
    buffers: Vec<FrameRingBuffer>,
}

impl CaptureSession {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Configure a camera. Rejects a duplicate id, because a frame has to route
    /// to exactly one buffer.
    pub fn add_camera(
        &mut self,
        descriptor: CameraDescriptor,
        config: RingBufferConfig,
    ) -> Result<(), CaptureError> {
        if self.buffer_for(&descriptor.camera_id).is_some() {
            return Err(CaptureError::DuplicateCamera {
                camera_id: descriptor.camera_id.clone(),
            });
        }

        self.buffers.push(FrameRingBuffer::new(descriptor, config)?);
        Ok(())
    }

    #[must_use]
    pub fn camera_count(&self) -> usize {
        self.buffers.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.buffers.is_empty()
    }

    #[must_use]
    pub fn buffers(&self) -> &[FrameRingBuffer] {
        &self.buffers
    }

    #[must_use]
    pub fn buffer_for(&self, camera_id: &CameraId) -> Option<&FrameRingBuffer> {
        self.buffers
            .iter()
            .find(|buffer| &buffer.descriptor().camera_id == camera_id)
    }

    /// Route a frame to its camera's buffer.
    pub fn push(&mut self, frame: CapturedFrame) -> Result<(), CaptureError> {
        let buffer = self
            .buffers
            .iter_mut()
            .find(|buffer| buffer.descriptor().camera_id == *frame.camera_id())
            .ok_or_else(|| CaptureError::UnknownCamera {
                camera_id: frame.camera_id().clone(),
            })?;

        buffer.push(frame)
    }

    /// Extract `[at - pre_roll, at]` from every camera.
    ///
    /// The window is computed once, in the session clock, and applied to each
    /// buffer independently. Cameras that started at different instants or
    /// dropped different frames return different frame counts for the same
    /// window — which is the honest answer, and the reason nothing here converts
    /// the pre-roll into a frame count.
    ///
    /// Extraction begins at the session origin when the pre-roll reaches back
    /// past it, since timestamps are unsigned. That floor is recorded rather
    /// than forgotten: [`PreRollWindow::reaches_before_origin`] stays true, and
    /// [`ShotExtraction::full_pre_roll_available`] is consequently `false`. A
    /// buffer reaching zero does not mean 30 seconds of history existed at a
    /// trigger 4ms into the session.
    pub fn trigger(
        &self,
        at: Timestamp,
        pre_roll: Duration,
    ) -> Result<ShotExtraction, CaptureError> {
        if self.buffers.is_empty() {
            return Err(CaptureError::NoCameras);
        }

        let window = PreRollWindow::new(at, pre_roll);

        let mut streams = Vec::with_capacity(self.buffers.len());
        for buffer in &self.buffers {
            let camera_id = buffer.descriptor().camera_id.clone();
            let Some((buffered_from, buffered_to)) = buffer.buffered_span() else {
                return Err(CaptureError::NoFramesBuffered { camera_id });
            };

            let clip = buffer.extract(window).ok_or(CaptureError::WindowIsEmpty {
                camera_id,
                start: window.start(),
                end: window.end(),
                buffered_from,
                buffered_to,
            })?;

            streams.push(clip);
        }

        Ok(ShotExtraction::new(window, streams))
    }
}

#[cfg(test)]
mod tests {
    use swingai_core::{CameraView, PixelFormat};

    use super::*;

    fn descriptor(id: &str) -> CameraDescriptor {
        CameraDescriptor {
            camera_id: CameraId::new(id).unwrap(),
            view: CameraView::FaceOn,
            width: 2,
            height: 2,
            pixel_format: PixelFormat::new(PixelFormat::MONO8).unwrap(),
            nominal_fps: 100.0,
        }
    }

    fn config() -> RingBufferConfig {
        RingBufferConfig::new(Duration::from_secs(1), 4_096)
    }

    fn frame(id: &str, sequence: u64, nanos: u64) -> CapturedFrame {
        CapturedFrame::new(
            CameraId::new(id).unwrap(),
            sequence,
            Timestamp::from_nanos(nanos),
            2,
            2,
            PixelFormat::new(PixelFormat::MONO8).unwrap(),
            vec![sequence as u8; 4],
        )
    }

    #[test]
    fn a_duplicate_camera_id_is_rejected() {
        let mut session = CaptureSession::new();
        session.add_camera(descriptor("cam"), config()).unwrap();

        let error = session.add_camera(descriptor("cam"), config()).unwrap_err();
        assert!(matches!(error, CaptureError::DuplicateCamera { .. }));
        assert_eq!(session.camera_count(), 1);
    }

    #[test]
    fn a_frame_for_an_unconfigured_camera_is_rejected() {
        let mut session = CaptureSession::new();
        session.add_camera(descriptor("cam"), config()).unwrap();

        let error = session.push(frame("ghost", 0, 0)).unwrap_err();
        assert!(matches!(error, CaptureError::UnknownCamera { .. }));
        assert!(error.to_string().contains("ghost"), "{error}");
    }

    #[test]
    fn a_trigger_without_cameras_is_an_error() {
        let session = CaptureSession::new();
        let error = session
            .trigger(Timestamp::from_nanos(1_000), Duration::from_millis(1))
            .unwrap_err();
        assert!(matches!(error, CaptureError::NoCameras));
    }

    /// A session holding every millisecond since its origin, five frames long.
    fn four_millisecond_session() -> CaptureSession {
        let mut session = CaptureSession::new();
        session.add_camera(descriptor("cam"), config()).unwrap();
        for i in 0..5u64 {
            session.push(frame("cam", i, i * 1_000_000)).unwrap();
        }
        session
    }

    #[test]
    fn a_pre_roll_longer_than_the_session_extracts_from_zero_but_reports_incomplete() {
        // The case the first implementation got wrong: the window floors to the
        // origin, the buffer does reach the origin, and yet 4ms is not 30
        // seconds. Reaching zero is not evidence that the requested duration
        // existed.
        let extraction = four_millisecond_session()
            .trigger(Timestamp::from_nanos(4_000_000), Duration::from_secs(30))
            .unwrap();

        assert_eq!(
            extraction.requested_start(),
            Timestamp::ZERO,
            "extraction still begins at the origin — the boundary must stay valid"
        );
        assert_eq!(
            extraction.pre_roll(),
            Duration::from_secs(30),
            "and the requested duration is remembered rather than rewritten"
        );
        assert!(extraction.window().reaches_before_origin());
        assert!(!extraction.full_pre_roll_available());
        assert!(
            !extraction.streams()[0].full_pre_roll_available(),
            "every affected stream must say so too"
        );

        // The frames that did exist are still returned.
        assert_eq!(extraction.streams()[0].frames().len(), 5);
    }

    #[test]
    fn a_pre_roll_the_session_can_cover_reports_complete() {
        let mut session = CaptureSession::new();
        // Retention and byte cap both wide enough to hold the whole window, so
        // the only thing under test is the origin arithmetic.
        let generous = RingBufferConfig::new(Duration::from_secs(30), 1 << 20);
        session.add_camera(descriptor("cam"), generous).unwrap();

        // A trigger 30s into the session with 5s of pre-roll: the window starts
        // at 25s, nowhere near the origin. Frames from 24s so the buffer reaches
        // back past the window start.
        for i in 24_000..30_001u64 {
            session.push(frame("cam", i, i * 1_000_000)).unwrap();
        }

        let extraction = session
            .trigger(
                Timestamp::from_nanos(30_000_000_000),
                Duration::from_secs(5),
            )
            .unwrap();

        assert!(!extraction.window().reaches_before_origin());
        assert!(extraction.full_pre_roll_available());
        assert!(extraction.streams()[0].full_pre_roll_available());
    }

    #[test]
    fn a_trigger_exactly_one_pre_roll_into_the_session_may_be_complete() {
        // The boundary: `trigger - pre_roll` lands precisely on the origin, so
        // nothing was floored away and a buffer reaching zero really does have
        // the whole requested duration.
        let extraction = four_millisecond_session()
            .trigger(Timestamp::from_nanos(4_000_000), Duration::from_millis(4))
            .unwrap();

        assert_eq!(extraction.requested_start(), Timestamp::ZERO);
        assert!(!extraction.window().reaches_before_origin());
        assert!(extraction.full_pre_roll_available());
    }
}
