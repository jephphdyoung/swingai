use std::sync::Arc;

use swingai_core::{CameraId, PixelFormat, Timestamp};

/// One frame as it left a camera, on the capture-session clock.
///
/// # Payload ownership
///
/// The pixels live behind an [`Arc`], so extracting a clip from a ring buffer
/// clones handles rather than images. A 5-second two-camera buffer at 240fps is
/// multiple gigabytes; copying it to answer a trigger would defeat the point of
/// keeping it in memory. [`payload_handle`](Self::payload_handle) exists so that
/// sharing is testable rather than merely intended.
///
/// # What is deliberately not here
///
/// No vendor fields. Exposure, gain, temperature and raw device timestamps are
/// camera-specific, and a generic frame that grows a field per camera model
/// stops being generic. They belong in the stream's `metadata` map in the
/// manifest, which is the contract's declared extension point.
///
/// [`sequence`](Self::sequence) is the number the source reported, and it means
/// something only within this one stream — see the crate docs. It is not a time
/// reference and must not be compared across cameras.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapturedFrame {
    camera_id: CameraId,
    sequence: u64,
    timestamp: Timestamp,
    width: u32,
    height: u32,
    pixel_format: PixelFormat,
    payload: Arc<[u8]>,
}

impl CapturedFrame {
    pub fn new(
        camera_id: CameraId,
        sequence: u64,
        timestamp: Timestamp,
        width: u32,
        height: u32,
        pixel_format: PixelFormat,
        payload: impl Into<Arc<[u8]>>,
    ) -> Self {
        Self {
            camera_id,
            sequence,
            timestamp,
            width,
            height,
            pixel_format,
            payload: payload.into(),
        }
    }

    #[must_use]
    pub fn camera_id(&self) -> &CameraId {
        &self.camera_id
    }

    /// The stream-local sequence number the source reported. Used to detect
    /// dropped frames and nothing else.
    #[must_use]
    pub const fn sequence(&self) -> u64 {
        self.sequence
    }

    #[must_use]
    pub const fn timestamp(&self) -> Timestamp {
        self.timestamp
    }

    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    #[must_use]
    pub const fn height(&self) -> u32 {
        self.height
    }

    #[must_use]
    pub fn pixel_format(&self) -> &PixelFormat {
        &self.pixel_format
    }

    #[must_use]
    pub fn payload(&self) -> &[u8] {
        &self.payload
    }

    /// The shared handle to the pixels, for a caller that wants to hold the
    /// image without copying it — and for tests that assert extraction really
    /// did share rather than duplicate.
    #[must_use]
    pub fn payload_handle(&self) -> &Arc<[u8]> {
        &self.payload
    }

    #[must_use]
    pub fn payload_len(&self) -> u64 {
        self.payload.len() as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(sequence: u64, payload: Vec<u8>) -> CapturedFrame {
        CapturedFrame::new(
            CameraId::new("cam").unwrap(),
            sequence,
            Timestamp::from_nanos(sequence * 1_000),
            2,
            2,
            PixelFormat::new(PixelFormat::MONO8).unwrap(),
            payload,
        )
    }

    #[test]
    fn a_frame_reports_what_it_was_built_from() {
        let frame = frame(7, vec![1, 2, 3, 4]);
        assert_eq!(frame.camera_id().as_str(), "cam");
        assert_eq!(frame.sequence(), 7);
        assert_eq!(frame.timestamp(), Timestamp::from_nanos(7_000));
        assert_eq!((frame.width(), frame.height()), (2, 2));
        assert_eq!(frame.pixel_format().as_str(), PixelFormat::MONO8);
        assert_eq!(frame.payload(), &[1, 2, 3, 4]);
        assert_eq!(frame.payload_len(), 4);
    }

    #[test]
    fn cloning_shares_the_pixels_rather_than_copying_them() {
        let original = frame(1, vec![9; 64]);
        let copy = original.clone();
        assert!(
            Arc::ptr_eq(original.payload_handle(), copy.payload_handle()),
            "a cloned frame must point at the same pixels"
        );
    }
}
