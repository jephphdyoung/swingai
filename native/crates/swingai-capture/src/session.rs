use std::time::Duration;

use swingai_core::{CameraId, Timestamp};

use crate::{
    CameraDescriptor, CaptureError, CapturedFrame, FrameRingBuffer, RingBufferConfig,
    ShotExtraction,
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
    /// The requested start is floored at the session origin: the origin is zero
    /// and nothing precedes it, so a pre-roll longer than the trigger's own
    /// timestamp reaches back to the start of the session and is reported as
    /// incomplete rather than wrapping into a huge number.
    pub fn trigger(
        &self,
        at: Timestamp,
        pre_roll: Duration,
    ) -> Result<ShotExtraction, CaptureError> {
        if self.buffers.is_empty() {
            return Err(CaptureError::NoCameras);
        }

        let pre_roll_nanos = u64::try_from(pre_roll.as_nanos()).unwrap_or(u64::MAX);
        let start = Timestamp::from_nanos(at.as_nanos().saturating_sub(pre_roll_nanos));

        let mut streams = Vec::with_capacity(self.buffers.len());
        for buffer in &self.buffers {
            let camera_id = buffer.descriptor().camera_id.clone();
            let Some((buffered_from, buffered_to)) = buffer.buffered_span() else {
                return Err(CaptureError::NoFramesBuffered { camera_id });
            };

            let clip = buffer
                .extract(start, at)
                .ok_or(CaptureError::WindowIsEmpty {
                    camera_id,
                    start,
                    end: at,
                    buffered_from,
                    buffered_to,
                })?;

            streams.push(clip);
        }

        Ok(ShotExtraction::new(at, start, pre_roll, streams))
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

    #[test]
    fn the_window_start_is_floored_at_the_session_origin() {
        let mut session = CaptureSession::new();
        session.add_camera(descriptor("cam"), config()).unwrap();
        for i in 0..5u64 {
            session.push(frame("cam", i, i * 1_000_000)).unwrap();
        }

        let extraction = session
            .trigger(Timestamp::from_nanos(4_000_000), Duration::from_secs(30))
            .unwrap();
        assert_eq!(extraction.requested_start(), Timestamp::ZERO);
        assert!(
            extraction.full_pre_roll_available(),
            "the buffer reaches the origin, which is as far back as there is"
        );
    }
}
