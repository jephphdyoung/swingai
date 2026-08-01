use std::fmt;
use std::time::Duration;

use serde::{Deserialize, Serialize};

/// An instant on the **SwingAI capture-session clock**, in whole nanoseconds
/// since that session's monotonic origin.
///
/// The origin is defined by the capture runtime when a session starts, and **the
/// persisted origin is zero** — every `*_timestamp_ns` in a contract document is
/// an offset from it. Timestamps are therefore comparable within a session and
/// meaningless across sessions.
///
/// Unsigned, so a negative value cannot be represented or deserialized. Nothing
/// in a session happens before its own origin, and a negative timestamp in a
/// document means a clock-domain conversion went wrong upstream — better to fail
/// at the seam than to propagate it.
///
/// # Converting device clocks
///
/// Cameras and audio devices report time on **their own clocks**, which may use a
/// different epoch, a different tick frequency, or a counter that is not
/// nanoseconds at all. A machine-vision SDK's frame stamp does not inherently
/// share this clock domain, and must not be assumed to. The capture runtime is
/// responsible for converting each device's raw stamp into the capture-session
/// domain — measuring the offset and rate rather than assuming them — before
/// anything is written to a manifest.
///
/// Raw device timestamps may be retained alongside, in a stream's `metadata` map,
/// for diagnosing drift. They must never be used directly as a primary
/// `*_timestamp_ns` value.
///
/// # Why nanoseconds
///
/// A 240fps frame period is 4.17ms, so millisecond integers would quantize frame
/// timing to a quarter of a frame. `u64` nanoseconds spans 584 years, and a
/// session's offsets stay below 2^53 for its first 104 days, so values remain
/// exactly representable in JSON consumers that parse numbers as doubles.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Timestamp(u64);

impl Timestamp {
    /// The capture-session origin.
    pub const ZERO: Self = Self(0);

    #[must_use]
    pub const fn from_nanos(nanos: u64) -> Self {
        Self(nanos)
    }

    #[must_use]
    pub const fn as_nanos(self) -> u64 {
        self.0
    }

    #[must_use]
    pub fn as_secs_f64(self) -> f64 {
        self.0 as f64 / 1_000_000_000.0
    }

    /// Elapsed time since `earlier`, or `None` if `earlier` is actually later.
    #[must_use]
    pub fn duration_since(self, earlier: Self) -> Option<Duration> {
        self.0.checked_sub(earlier.0).map(Duration::from_nanos)
    }

    /// Signed difference in nanoseconds. Saturates rather than wrapping, and is
    /// the one place a negative number is legitimate — a *difference* can point
    /// either way even though an instant cannot be before the origin.
    #[must_use]
    pub fn nanos_since(self, earlier: Self) -> i64 {
        let delta = i128::from(self.0) - i128::from(earlier.0);
        delta.clamp(i128::from(i64::MIN), i128::from(i64::MAX)) as i64
    }

    /// Signed difference in milliseconds. For humans and log lines — the exact
    /// value is the nanosecond one.
    #[must_use]
    pub fn millis_since(self, earlier: Self) -> f64 {
        self.nanos_since(earlier) as f64 / 1_000_000.0
    }

    #[must_use]
    pub fn checked_add(self, duration: Duration) -> Option<Self> {
        let nanos = u64::try_from(duration.as_nanos()).ok()?;
        self.0.checked_add(nanos).map(Self)
    }
}

impl fmt::Display for Timestamp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}ns", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serializes_as_a_bare_integer() {
        let json = serde_json::to_string(&Timestamp::from_nanos(4_985_960_690)).unwrap();
        assert_eq!(json, "4985960690");
    }

    #[test]
    fn deserializes_from_a_bare_integer() {
        let timestamp: Timestamp = serde_json::from_str("4985960690").unwrap();
        assert_eq!(timestamp.as_nanos(), 4_985_960_690);
    }

    #[test]
    fn the_session_origin_is_zero() {
        let timestamp: Timestamp = serde_json::from_str("0").unwrap();
        assert_eq!(timestamp, Timestamp::ZERO);
    }

    #[test]
    fn a_negative_timestamp_is_rejected() {
        for bad in ["-1", "-4985960690"] {
            let error = serde_json::from_str::<Timestamp>(bad).unwrap_err();
            assert!(
                error.to_string().contains("invalid value"),
                "{bad}: {error}"
            );
        }
    }

    #[test]
    fn a_fractional_timestamp_is_rejected() {
        assert!(serde_json::from_str::<Timestamp>("1.5").is_err());
    }

    #[test]
    fn duration_since_is_none_when_the_order_is_wrong() {
        let early = Timestamp::from_nanos(10);
        let late = Timestamp::from_nanos(20);
        assert_eq!(late.duration_since(early), Some(Duration::from_nanos(10)));
        assert_eq!(early.duration_since(late), None);
    }

    #[test]
    fn signed_difference_survives_the_wrong_order() {
        let early = Timestamp::from_nanos(10);
        let late = Timestamp::from_nanos(20);
        assert_eq!(early.nanos_since(late), -10);
        assert!((early.millis_since(late) - -0.000_01).abs() < 1e-12);
    }

    #[test]
    fn a_difference_too_large_for_i64_saturates_rather_than_wrapping() {
        let huge = Timestamp::from_nanos(u64::MAX);
        assert_eq!(huge.nanos_since(Timestamp::ZERO), i64::MAX);
        assert_eq!(Timestamp::ZERO.nanos_since(huge), i64::MIN);
    }

    #[test]
    fn timestamps_order_by_instant() {
        assert!(Timestamp::from_nanos(1) < Timestamp::from_nanos(2));
    }
}
