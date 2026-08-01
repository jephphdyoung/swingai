//! Wall-clock time, used for exactly two things: stamping a manifest's
//! `created_at`, and naming the directory it lands in.
//!
//! Neither correlates anything. Wall clock jumps at NTP corrections and daylight
//! saving, so it cannot place an event within a swing or line two cameras up —
//! that is the capture-session [`Timestamp`](swingai_core::Timestamp)'s job, and
//! the two are separate types precisely so this file cannot be misused.

use swingai_core::{Rfc3339Timestamp, ShotId, ValidationError};
use time::OffsetDateTime;
use time::format_description::well_known::Rfc3339;

/// The current UTC instant as an RFC 3339 string.
///
/// Formatting is `time`'s, not ours: RFC 3339 has enough corners (leap seconds,
/// subsecond precision, offset spelling) that hand-rolling it is how a contract
/// field ends up rejected by its own validator.
#[must_use]
pub fn now_utc() -> Rfc3339Timestamp {
    let formatted = OffsetDateTime::now_utc()
        .format(&Rfc3339)
        .expect("a UTC instant always formats as RFC 3339");
    Rfc3339Timestamp::new(formatted).expect("`time` emits RFC 3339 that the contract accepts")
}

/// A shot id derived from a creation time: the same instant with `:` swapped for
/// `-`.
///
/// Colons are legal in RFC 3339 and illegal in a Windows filename, and
/// [`ShotId`] rejects them for that reason. Everything else about the spelling —
/// the date arithmetic, the subsecond digits, the offset — comes from the
/// formatted timestamp, so this function does no date handling of its own and
/// the id always names the instant the manifest claims.
///
/// Two shots within the same formatted instant would collide; the shot writer
/// refuses to overwrite an existing directory, so that surfaces as an error
/// rather than a lost capture.
pub fn shot_id_for(created_at: &Rfc3339Timestamp) -> Result<ShotId, ValidationError> {
    ShotId::new(created_at.as_str().replace(':', "-"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_current_instant_is_a_valid_contract_timestamp() {
        let now = now_utc();
        assert!(now.as_str().ends_with('Z'), "{now}");
        // Re-parsing is the real check: the contract type validated it already,
        // but this proves the value survives a round trip through the seam.
        assert_eq!(Rfc3339Timestamp::new(now.as_str()).unwrap(), now);
    }

    #[test]
    fn a_shot_id_is_the_creation_time_made_directory_safe() {
        let created_at = Rfc3339Timestamp::new("2026-07-31T14:22:05.412Z").unwrap();
        let shot_id = shot_id_for(&created_at).unwrap();
        assert_eq!(shot_id.as_str(), "2026-07-31T14-22-05.412Z");
    }

    #[test]
    fn an_offset_creation_time_also_yields_a_usable_id() {
        let created_at = Rfc3339Timestamp::new("2026-07-31T16:22:05+02:00").unwrap();
        assert_eq!(
            shot_id_for(&created_at).unwrap().as_str(),
            "2026-07-31T16-22-05+02-00"
        );
    }

    #[test]
    fn the_generated_id_is_accepted_by_the_id_rules() {
        // ShotId is the single authority on what an id may contain; this only
        // proves the derivation satisfies it rather than restating the rules.
        assert!(shot_id_for(&now_utc()).is_ok());
    }
}
