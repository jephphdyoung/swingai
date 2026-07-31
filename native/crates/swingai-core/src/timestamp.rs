use std::fmt;
use std::time::Duration;

use serde::{Deserialize, Serialize};

/// An instant on the monotonic capture clock, in whole nanoseconds.
///
/// Nanoseconds because that is what the underlying clocks report
/// (`QueryPerformanceCounter`, `clock_gettime`, machine-vision frame stamps) and
/// because a 240fps frame period is 4.17ms — millisecond integers would quantize
/// frame timing to a quarter of a frame.
///
/// `i64` spans ±292 years. A monotonic clock read stays below 2^53 for the first
/// 104 days of uptime, so values also survive JSON consumers that parse numbers
/// as doubles.
///
/// The zero point is whatever the OS clock counts from. Only *differences*
/// between timestamps are meaningful, and only within one capture session — a
/// reboot resets the origin. Wall-clock time is carried separately, as RFC 3339
/// strings, and is never used to correlate streams.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Timestamp(i64);

impl Timestamp {
    pub const ZERO: Self = Self(0);

    #[must_use]
    pub const fn from_nanos(nanos: i64) -> Self {
        Self(nanos)
    }

    #[must_use]
    pub const fn as_nanos(self) -> i64 {
        self.0
    }

    #[must_use]
    pub fn as_secs_f64(self) -> f64 {
        self.0 as f64 / 1_000_000_000.0
    }

    /// Elapsed time since `earlier`, or `None` if `earlier` is actually later.
    #[must_use]
    pub fn duration_since(self, earlier: Self) -> Option<Duration> {
        let delta = self.0.checked_sub(earlier.0)?;
        u64::try_from(delta).ok().map(Duration::from_nanos)
    }

    /// Signed difference in nanoseconds, saturating rather than wrapping.
    #[must_use]
    pub fn nanos_since(self, earlier: Self) -> i64 {
        self.0.saturating_sub(earlier.0)
    }

    /// Signed difference in milliseconds. For humans and log lines — the exact
    /// value is the nanosecond one.
    #[must_use]
    pub fn millis_since(self, earlier: Self) -> f64 {
        self.nanos_since(earlier) as f64 / 1_000_000.0
    }

    #[must_use]
    pub fn checked_add(self, duration: Duration) -> Option<Self> {
        let nanos = i64::try_from(duration.as_nanos()).ok()?;
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
        let json = serde_json::to_string(&Timestamp::from_nanos(128_471_215_400_000)).unwrap();
        assert_eq!(json, "128471215400000");
    }

    #[test]
    fn deserializes_from_a_bare_integer() {
        let timestamp: Timestamp = serde_json::from_str("128471215400000").unwrap();
        assert_eq!(timestamp.as_nanos(), 128_471_215_400_000);
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
    fn timestamps_order_by_instant() {
        assert!(Timestamp::from_nanos(1) < Timestamp::from_nanos(2));
    }
}
