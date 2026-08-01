use std::fmt;

use serde::{Deserialize, Serialize};
use time::OffsetDateTime;
use time::format_description::well_known::Rfc3339;

use crate::ValidationError;

/// A wall-clock instant, validated as RFC 3339 on the way in.
///
/// This is the *other* kind of time in the contracts, and it is not
/// interchangeable with [`Timestamp`](crate::Timestamp). Wall clock is for
/// humans and for filing — "which session was this, roughly when did it happen".
/// It jumps at NTP corrections and daylight saving, so it must never be used to
/// correlate streams or to place an event within a swing. That is
/// [`Timestamp`](crate::Timestamp)'s job.
///
/// The original string is kept verbatim rather than reformatted, so a document
/// round-trips byte-identically and a writer's choice of precision or offset
/// survives. Parsing is delegated to `time`'s `Rfc3339` implementation, which is
/// RFC 3339 exactly rather than a superset — an ISO 8601 or Temporal parser
/// would accept strings the contract does not allow.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct Rfc3339Timestamp(String);

impl Rfc3339Timestamp {
    pub fn new(value: impl Into<String>) -> Result<Self, ValidationError> {
        let value = value.into();
        Self::parse_str(&value)?;
        Ok(Self(value))
    }

    fn parse_str(value: &str) -> Result<OffsetDateTime, ValidationError> {
        OffsetDateTime::parse(value, &Rfc3339).map_err(|error| {
            ValidationError::new(
                "created_at",
                format!(
                    "{value:?} is not an RFC 3339 date-time ({error}); \
                     expected something like \"2026-07-31T14:22:05.412Z\" \
                     or \"2026-07-31T16:22:05.412+02:00\""
                ),
            )
        })
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// The parsed instant. Re-parses on each call — this is a filing field, not
    /// something on any hot path, and storing both halves would let them drift.
    pub fn to_offset_date_time(&self) -> OffsetDateTime {
        Self::parse_str(&self.0).expect("validated at construction")
    }
}

impl TryFrom<String> for Rfc3339Timestamp {
    type Error = ValidationError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl From<Rfc3339Timestamp> for String {
    fn from(value: Rfc3339Timestamp) -> Self {
        value.0
    }
}

impl fmt::Display for Rfc3339Timestamp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_utc_timestamps_are_accepted() {
        for good in [
            "2026-07-31T14:22:05Z",
            "2026-07-31T14:22:05.412Z",
            "2026-07-31T14:22:05.412345678Z",
            "2024-02-29T00:00:00Z", // a real leap day
        ] {
            assert!(Rfc3339Timestamp::new(good).is_ok(), "{good} should parse");
        }
    }

    #[test]
    fn valid_offsets_are_accepted() {
        for good in [
            "2026-07-31T16:22:05.412+02:00",
            "2026-07-31T09:22:05-05:00",
            "2026-07-31T14:22:05+00:00",
        ] {
            assert!(Rfc3339Timestamp::new(good).is_ok(), "{good} should parse");
        }
    }

    #[test]
    fn the_required_rejections_are_rejected() {
        for bad in [
            "yesterday",
            "2026-99-99T00:00:00Z",
            "2026-07-31 12:00:00", // space separator and no offset
        ] {
            let error = Rfc3339Timestamp::new(bad).unwrap_err().to_string();
            assert!(
                error.contains("not an RFC 3339 date-time"),
                "{bad}: {error}"
            );
        }
    }

    #[test]
    fn a_naive_date_time_without_an_offset_is_rejected() {
        // RFC 3339 requires an offset. A local time here is how two machines end
        // up disagreeing about when a session happened.
        assert!(Rfc3339Timestamp::new("2026-07-31T14:22:05").is_err());
    }

    #[test]
    fn an_impossible_calendar_date_is_rejected() {
        assert!(Rfc3339Timestamp::new("2026-02-30T00:00:00Z").is_err());
        assert!(Rfc3339Timestamp::new("2026-13-01T00:00:00Z").is_err());
        assert!(Rfc3339Timestamp::new("2026-07-31T25:00:00Z").is_err());
    }

    #[test]
    fn an_empty_string_is_rejected() {
        assert!(Rfc3339Timestamp::new("").is_err());
        assert!(serde_json::from_str::<Rfc3339Timestamp>("\"\"").is_err());
    }

    #[test]
    fn the_original_spelling_survives_a_round_trip() {
        // Not normalised to Z or to a fixed precision — the writer's form is kept.
        for original in ["2026-07-31T16:22:05.412+02:00", "2026-07-31T14:22:05Z"] {
            let parsed = Rfc3339Timestamp::new(original).unwrap();
            let json = serde_json::to_string(&parsed).unwrap();
            assert_eq!(json, format!("\"{original}\""));
            assert_eq!(
                serde_json::from_str::<Rfc3339Timestamp>(&json).unwrap(),
                parsed
            );
        }
    }

    #[test]
    fn an_invalid_value_fails_at_deserialization() {
        let error = serde_json::from_str::<Rfc3339Timestamp>("\"yesterday\"").unwrap_err();
        assert!(error.to_string().contains("not an RFC 3339 date-time"));
    }

    #[test]
    fn offsets_are_understood_not_just_checked() {
        let utc = Rfc3339Timestamp::new("2026-07-31T14:22:05Z").unwrap();
        let plus_two = Rfc3339Timestamp::new("2026-07-31T16:22:05+02:00").unwrap();
        assert_eq!(utc.to_offset_date_time(), plus_two.to_offset_date_time());
    }
}
