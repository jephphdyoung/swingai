use swingai_core::{CameraId, CameraView, PixelFormat, ValidationErrors};

use crate::CapturedFrame;

/// What a camera is and what it is configured to produce — everything a ring
/// buffer needs in order to notice that a frame does not belong to it.
///
/// `nominal_fps` is the *configured* rate, not the achieved one. The achieved
/// rate is measured from timestamps by
/// [`FrameSequence::measured_fps`](swingai_core::FrameSequence::measured_fps),
/// and the two differing is normal rather than a fault. Nothing in this crate
/// uses `nominal_fps` to decide which frames belong to a time range.
#[derive(Debug, Clone, PartialEq)]
pub struct CameraDescriptor {
    pub camera_id: CameraId,
    pub view: CameraView,
    pub width: u32,
    pub height: u32,
    pub pixel_format: PixelFormat,
    pub nominal_fps: f64,
}

impl CameraDescriptor {
    pub fn validate(&self) -> ValidationErrors {
        let mut errors = ValidationErrors::new();

        if self.width == 0 {
            errors.push("width", "must be at least 1");
        }
        if self.height == 0 {
            errors.push("height", "must be at least 1");
        }
        if !self.nominal_fps.is_finite() || self.nominal_fps <= 0.0 {
            errors.push(
                "nominal_fps",
                format!("must be a positive finite number, got {}", self.nominal_fps),
            );
        }

        errors
    }
}

/// A synchronous source of frames for one camera.
///
/// Pull-based and blocking-free by construction: a caller drives it, so a test
/// can run a whole capture in microseconds and a simulator can run one without
/// pacing itself against a wall clock. The real acquisition loop will own its
/// own thread and hand frames to a buffer; this trait is the shape that loop
/// produces.
///
/// [`next_frame`](Self::next_frame) returns `None` at end of source, which for a
/// live camera means the stream was stopped. It cannot report a failure, because
/// nothing in this crate can fail: a synthetic source has no device to lose. A
/// real source needs a fallible signature, and widening `Option` to `Result` at
/// that point is a small, obvious change — inventing an error type now with no
/// implementer to produce it would only guess at what a device failure looks
/// like.
pub trait FrameSource {
    /// The camera this source speaks for. Constant for the life of the source;
    /// a ring buffer holds it and rejects any frame that disagrees.
    fn descriptor(&self) -> &CameraDescriptor;

    /// The next frame, or `None` when the source is exhausted.
    ///
    /// Frames must come out in increasing timestamp *and* increasing sequence
    /// order. Missing sequence numbers are how a source reports dropped frames.
    fn next_frame(&mut self) -> Option<CapturedFrame>;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn descriptor() -> CameraDescriptor {
        CameraDescriptor {
            camera_id: CameraId::new("cam").unwrap(),
            view: CameraView::DownTheLine,
            width: 160,
            height: 120,
            pixel_format: PixelFormat::new(PixelFormat::MONO8).unwrap(),
            nominal_fps: 240.0,
        }
    }

    #[test]
    fn a_sensible_descriptor_validates() {
        assert!(descriptor().validate().is_empty());
    }

    #[test]
    fn zero_dimensions_are_rejected() {
        let errors = CameraDescriptor {
            width: 0,
            height: 0,
            ..descriptor()
        }
        .validate();
        assert_eq!(errors.len(), 2);
    }

    #[test]
    fn a_nonsense_frame_rate_is_rejected() {
        for fps in [0.0, -240.0, f64::NAN, f64::INFINITY] {
            let errors = CameraDescriptor {
                nominal_fps: fps,
                ..descriptor()
            }
            .validate();
            assert_eq!(errors.len(), 1, "fps {fps} should be rejected");
        }
    }
}
