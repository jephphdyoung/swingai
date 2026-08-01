use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::{Timestamp, ValidationErrors};

/// The timing spine of one recorded stream: how many frames were stored, when
/// the first and last of them happened, and how many never arrived.
///
/// Field names match the JSON contract exactly, because
/// [`CameraStreamManifest`](../../swingai_contracts/struct.CameraStreamManifest.html)
/// flattens this into the stream object rather than nesting it.
///
/// Counts are unsigned, so a negative frame count is impossible to represent;
/// zero is rejected by [`validate`](Self::validate).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FrameSequence {
    /// Frames actually stored. Excludes anything counted in
    /// `dropped_frame_count`.
    pub frame_count: u32,
    /// The rate the camera was configured for. The rate it actually achieved is
    /// [`measured_fps`](Self::measured_fps), and the two will differ.
    pub nominal_fps: f64,
    pub first_timestamp_ns: Timestamp,
    pub last_timestamp_ns: Timestamp,
    /// Frames the camera reported but that never reached storage.
    pub dropped_frame_count: u32,
}

impl FrameSequence {
    /// Wall time from the first stored frame to the last. `None` if the
    /// timestamps are out of order — check [`validate`](Self::validate) first.
    #[must_use]
    pub fn duration(&self) -> Option<Duration> {
        self.last_timestamp_ns
            .duration_since(self.first_timestamp_ns)
    }

    /// The rate the camera actually delivered, from the timestamps rather than
    /// the configured rate. `None` for a single-frame sequence, where there is
    /// no interval to measure.
    ///
    /// This is `(frame_count - 1) / span` and so ignores dropped frames — it is
    /// the rate frames *landed* at, which is what a timestamp lookup cares
    /// about.
    #[must_use]
    pub fn measured_fps(&self) -> Option<f64> {
        if self.frame_count < 2 {
            return None;
        }
        let seconds = self.duration()?.as_secs_f64();
        if seconds <= 0.0 {
            return None;
        }
        Some(f64::from(self.frame_count - 1) / seconds)
    }

    /// Frames the camera claims it produced: stored plus dropped.
    #[must_use]
    pub fn reported_frame_count(&self) -> u64 {
        u64::from(self.frame_count) + u64::from(self.dropped_frame_count)
    }

    /// Everything wrong with this sequence, collected.
    pub fn validate(&self) -> ValidationErrors {
        let mut errors = ValidationErrors::new();

        if self.frame_count == 0 {
            errors.push(
                "frame_count",
                "must be at least 1; an empty stream is not a capture",
            );
        }

        if !self.nominal_fps.is_finite() || self.nominal_fps <= 0.0 {
            errors.push(
                "nominal_fps",
                format!("must be a positive finite number, got {}", self.nominal_fps),
            );
        }

        let first = self.first_timestamp_ns;
        let last = self.last_timestamp_ns;
        if last < first {
            errors.push(
                "last_timestamp_ns",
                format!(
                    "is {} before first_timestamp_ns ({} vs {})",
                    -last.nanos_since(first),
                    last.as_nanos(),
                    first.as_nanos()
                ),
            );
        } else if self.frame_count > 1 && last == first {
            errors.push(
                "last_timestamp_ns",
                format!(
                    "equals first_timestamp_ns, but frame_count is {}; \
                     more than one frame cannot share an instant",
                    self.frame_count
                ),
            );
        }

        errors
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sequence() -> FrameSequence {
        FrameSequence {
            frame_count: 1244,
            nominal_fps: 249.3,
            first_timestamp_ns: Timestamp::from_nanos(128_471_215_400_000),
            last_timestamp_ns: Timestamp::from_nanos(128_476_201_360_690),
            dropped_frame_count: 0,
        }
    }

    #[test]
    fn a_real_sequence_validates() {
        assert!(sequence().validate().is_empty());
    }

    #[test]
    fn measured_fps_is_close_to_nominal() {
        let measured = sequence().measured_fps().unwrap();
        assert!((measured - 249.3).abs() < 0.01, "measured {measured}");
    }

    #[test]
    fn zero_frames_is_rejected() {
        let sequence = FrameSequence {
            frame_count: 0,
            ..sequence()
        };
        let errors = sequence.validate();
        assert_eq!(errors.len(), 1);
        assert_eq!(errors.iter().next().unwrap().path(), "frame_count");
    }

    #[test]
    fn last_before_first_is_rejected() {
        let sequence = FrameSequence {
            last_timestamp_ns: Timestamp::from_nanos(128_471_215_399_999),
            ..sequence()
        };
        let errors = sequence.validate();
        assert_eq!(errors.len(), 1);
        let error = errors.iter().next().unwrap();
        assert_eq!(error.path(), "last_timestamp_ns");
        assert!(error.message().contains("before first_timestamp_ns"));
    }

    #[test]
    fn many_frames_cannot_share_one_instant() {
        let sequence = FrameSequence {
            last_timestamp_ns: Timestamp::from_nanos(128_471_215_400_000),
            ..sequence()
        };
        let errors = sequence.validate();
        assert_eq!(errors.len(), 1);
        assert!(errors.to_string().contains("cannot share an instant"));
    }

    #[test]
    fn a_single_frame_may_share_its_instant() {
        let sequence = FrameSequence {
            frame_count: 1,
            last_timestamp_ns: Timestamp::from_nanos(128_471_215_400_000),
            ..sequence()
        };
        assert!(sequence.validate().is_empty());
        assert_eq!(sequence.measured_fps(), None);
    }

    #[test]
    fn a_nonsense_frame_rate_is_rejected() {
        for fps in [0.0, -60.0, f64::NAN, f64::INFINITY] {
            let sequence = FrameSequence {
                nominal_fps: fps,
                ..sequence()
            };
            assert!(
                !sequence.validate().is_empty(),
                "fps {fps} should be rejected"
            );
        }
    }

    #[test]
    fn dropped_frames_count_towards_what_the_camera_reported() {
        let sequence = FrameSequence {
            frame_count: 1243,
            dropped_frame_count: 1,
            ..sequence()
        };
        assert_eq!(sequence.reported_frame_count(), 1244);
    }

    #[test]
    fn a_negative_frame_count_cannot_be_deserialized() {
        let json = r#"{
            "frame_count": -1,
            "nominal_fps": 249.3,
            "first_timestamp_ns": 0,
            "last_timestamp_ns": 1,
            "dropped_frame_count": 0
        }"#;
        let error = serde_json::from_str::<FrameSequence>(json).unwrap_err();
        assert!(error.to_string().contains("invalid value"), "{error}");
    }
}
