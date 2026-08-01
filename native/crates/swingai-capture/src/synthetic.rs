use std::collections::BTreeSet;
use std::time::Duration;

use swingai_core::{CameraId, CameraView, PixelFormat, Timestamp, ValidationErrors};

use crate::{CameraDescriptor, CapturedFrame, FrameSource};

/// How a [`SyntheticSource`] behaves. Two sources built from equal configs
/// produce byte-identical frames at identical timestamps, every run.
///
/// # Slots, not deliveries
///
/// `frame_count` is the number of instants the camera *exposes* at — its
/// lifetime — and `missing_sequences` names the ones that never arrive. So a
/// source with `frame_count: 100` and two missing sequences delivers 98 frames
/// spanning the same wall time as 100 would have. That is what a dropped frame
/// is: the moment still happened, the image did not. Modelling it as "98 frames,
/// evenly spaced" would quietly turn a drop into a slower camera and hide
/// exactly the failure this crate exists to make visible.
#[derive(Debug, Clone, PartialEq)]
pub struct SyntheticSourceConfig {
    pub camera_id: CameraId,
    pub view: CameraView,
    pub width: u32,
    pub height: u32,
    /// Timestamp of the first exposure slot, on the capture-session clock.
    pub first_timestamp: Timestamp,
    /// Spacing between consecutive exposure slots.
    pub frame_interval: Duration,
    /// The rate the camera is *configured* for, which need not be exactly
    /// `1 / frame_interval` — a 240fps camera whose period is 4_166_666ns is
    /// running at 240.000019fps, and the manifest records both so the difference
    /// stays visible.
    pub nominal_fps: f64,
    /// Number of exposure slots, including any that go missing.
    pub frame_count: u32,
    /// Sequence number of the first slot. Deliberately configurable, and
    /// deliberately different per camera in the simulator: sequence numbers are
    /// stream-local, and two cameras sharing a numbering would invite somebody
    /// to align on it.
    pub first_sequence: u64,
    /// Slots that are exposed but never delivered.
    pub missing_sequences: BTreeSet<u64>,
}

impl SyntheticSourceConfig {
    /// A gapless source starting at the session origin, sequence 0, with
    /// `nominal_fps` derived from the interval.
    pub fn new(
        camera_id: CameraId,
        view: CameraView,
        width: u32,
        height: u32,
        frame_interval: Duration,
        frame_count: u32,
    ) -> Self {
        let nominal_fps = if frame_interval.is_zero() {
            0.0
        } else {
            1.0 / frame_interval.as_secs_f64()
        };

        Self {
            camera_id,
            view,
            width,
            height,
            first_timestamp: Timestamp::ZERO,
            frame_interval,
            nominal_fps,
            frame_count,
            first_sequence: 0,
            missing_sequences: BTreeSet::new(),
        }
    }

    #[must_use]
    pub fn descriptor(&self) -> CameraDescriptor {
        CameraDescriptor {
            camera_id: self.camera_id.clone(),
            view: self.view,
            width: self.width,
            height: self.height,
            pixel_format: PixelFormat::new(PixelFormat::MONO8)
                .expect("mono8 is a valid pixel format"),
            nominal_fps: self.nominal_fps,
        }
    }

    /// Sequence numbers this config covers: `first_sequence` for `frame_count`.
    fn sequence_range(&self) -> std::ops::Range<u64> {
        self.first_sequence..self.first_sequence + u64::from(self.frame_count)
    }

    /// How many frames actually come out — slots minus planned drops.
    #[must_use]
    pub fn delivered_frame_count(&self) -> u32 {
        let range = self.sequence_range();
        let missing = self
            .missing_sequences
            .iter()
            .filter(|sequence| range.contains(sequence))
            .count();
        self.frame_count - u32::try_from(missing).unwrap_or(self.frame_count)
    }

    /// Timestamp of the last exposure slot, or `None` if the arithmetic
    /// overflows the session clock.
    #[must_use]
    pub fn last_timestamp(&self) -> Option<Timestamp> {
        let slots = u64::from(self.frame_count.checked_sub(1)?);
        let interval = u64::try_from(self.frame_interval.as_nanos()).ok()?;
        self.first_timestamp
            .checked_add(Duration::from_nanos(slots.checked_mul(interval)?))
    }

    pub fn validate(&self) -> ValidationErrors {
        let mut errors = self.descriptor().validate();

        if self.frame_count == 0 {
            errors.push("frame_count", "must be at least 1");
        }
        if self.frame_interval.is_zero() {
            errors.push(
                "frame_interval",
                "must be greater than zero; two frames cannot share an instant",
            );
        }
        if self.frame_count > 0 && !self.frame_interval.is_zero() && self.last_timestamp().is_none()
        {
            errors.push(
                "frame_count",
                "the last frame's timestamp overflows the session clock",
            );
        }

        let range = self.sequence_range();
        for sequence in &self.missing_sequences {
            if !range.contains(sequence) {
                errors.push(
                    "missing_sequences",
                    format!(
                        "{sequence} is outside this source's sequence range ({}..{})",
                        range.start, range.end
                    ),
                );
            }
        }

        if self.frame_count > 0 && self.delivered_frame_count() == 0 {
            errors.push(
                "missing_sequences",
                "every slot is missing, so the source would deliver nothing",
            );
        }

        errors
    }
}

/// A deterministic stand-in for a camera.
///
/// Never sleeps and never reads a clock: `next_frame` computes the timestamp
/// from the slot index, so a test runs a six-second capture instantly and a
/// rerun produces the same bytes. Pixel values are a pure function of
/// `(sequence, x, y)`, which makes "did extraction return the frames I think it
/// did" checkable by looking at the pixels.
#[derive(Debug, Clone)]
pub struct SyntheticSource {
    config: SyntheticSourceConfig,
    descriptor: CameraDescriptor,
    next_slot: u32,
}

impl SyntheticSource {
    pub fn new(config: SyntheticSourceConfig) -> Result<Self, ValidationErrors> {
        config.validate().into_result()?;
        let descriptor = config.descriptor();
        Ok(Self {
            config,
            descriptor,
            next_slot: 0,
        })
    }

    #[must_use]
    pub fn config(&self) -> &SyntheticSourceConfig {
        &self.config
    }

    /// The pixels a given sequence number carries. Public so a test can assert
    /// that an extracted frame is the frame it claims to be, without threading
    /// the source through.
    #[must_use]
    pub fn payload_for(width: u32, height: u32, sequence: u64) -> Vec<u8> {
        let seed = sequence as u8;
        let mut payload = Vec::with_capacity(width as usize * height as usize);
        for y in 0..height {
            for x in 0..width {
                payload.push(
                    (x as u8)
                        .wrapping_mul(3)
                        .wrapping_add((y as u8).wrapping_mul(5))
                        .wrapping_add(seed.wrapping_mul(7)),
                );
            }
        }
        payload
    }

    fn timestamp_for_slot(&self, slot: u32) -> Timestamp {
        let interval = u64::try_from(self.config.frame_interval.as_nanos())
            .expect("a validated frame interval fits in u64 nanoseconds");
        self.config
            .first_timestamp
            .checked_add(Duration::from_nanos(u64::from(slot) * interval))
            .expect("validated at construction not to overflow the session clock")
    }
}

impl FrameSource for SyntheticSource {
    fn descriptor(&self) -> &CameraDescriptor {
        &self.descriptor
    }

    fn next_frame(&mut self) -> Option<CapturedFrame> {
        loop {
            if self.next_slot >= self.config.frame_count {
                return None;
            }

            let slot = self.next_slot;
            self.next_slot += 1;

            let sequence = self.config.first_sequence + u64::from(slot);
            // A planned drop consumes its exposure slot without delivering a
            // frame, so the next frame's timestamp still lands where the camera
            // would have put it.
            if self.config.missing_sequences.contains(&sequence) {
                continue;
            }

            return Some(CapturedFrame::new(
                self.config.camera_id.clone(),
                sequence,
                self.timestamp_for_slot(slot),
                self.config.width,
                self.config.height,
                self.descriptor.pixel_format.clone(),
                Self::payload_for(self.config.width, self.config.height, sequence),
            ));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> SyntheticSourceConfig {
        SyntheticSourceConfig::new(
            CameraId::new("cam").unwrap(),
            CameraView::FaceOn,
            4,
            2,
            Duration::from_nanos(4_166_666),
            10,
        )
    }

    #[test]
    fn a_gapless_source_delivers_every_slot() {
        let mut source = SyntheticSource::new(config()).unwrap();
        let mut delivered = 0;
        while source.next_frame().is_some() {
            delivered += 1;
        }
        assert_eq!(delivered, 10);
        assert_eq!(config().delivered_frame_count(), 10);
    }

    #[test]
    fn payloads_are_the_configured_size_and_vary_by_sequence() {
        let mut source = SyntheticSource::new(config()).unwrap();
        let first = source.next_frame().unwrap();
        let second = source.next_frame().unwrap();
        assert_eq!(first.payload().len(), 8);
        assert_ne!(first.payload(), second.payload());
    }

    #[test]
    fn a_zero_length_interval_is_rejected() {
        let config = SyntheticSourceConfig {
            frame_interval: Duration::ZERO,
            ..config()
        };
        assert!(SyntheticSource::new(config).is_err());
    }

    #[test]
    fn a_missing_sequence_outside_the_range_is_rejected() {
        let config = SyntheticSourceConfig {
            missing_sequences: BTreeSet::from([99]),
            ..config()
        };
        let errors = SyntheticSource::new(config).unwrap_err();
        assert!(
            errors.to_string().contains("outside this source's"),
            "{errors}"
        );
    }

    #[test]
    fn a_source_that_would_deliver_nothing_is_rejected() {
        let config = SyntheticSourceConfig {
            missing_sequences: (0..10).collect(),
            ..config()
        };
        let errors = SyntheticSource::new(config).unwrap_err();
        assert!(errors.to_string().contains("deliver nothing"), "{errors}");
    }

    #[test]
    fn the_last_slot_timestamp_accounts_for_every_slot() {
        let config = config();
        assert_eq!(
            config.last_timestamp(),
            Some(Timestamp::from_nanos(9 * 4_166_666))
        );
    }
}
