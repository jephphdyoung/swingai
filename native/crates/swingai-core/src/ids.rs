use std::fmt;

use serde::{Deserialize, Serialize};

use crate::ValidationError;

/// Characters that would make an id unusable as a directory name, or ambiguous
/// in a log line.
fn check_id(kind: &'static str, value: &str) -> Result<(), ValidationError> {
    if value.is_empty() {
        return Err(ValidationError::new(kind, "must not be empty"));
    }
    if value == "." || value == ".." {
        return Err(ValidationError::new(
            kind,
            format!("{value:?} is not usable as a directory name"),
        ));
    }
    if let Some(bad) = value.chars().find(|c| {
        matches!(c, '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|') || c.is_control()
    }) {
        return Err(ValidationError::new(
            kind,
            format!("contains {bad:?}, which is not allowed in an id"),
        ));
    }
    Ok(())
}

macro_rules! string_id {
    ($name:ident, $kind:literal, $doc:literal) => {
        #[doc = $doc]
        ///
        /// Constructed through [`new`](Self::new), which rejects anything that
        /// could not also serve as a directory name — ids end up in paths.
        #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
        #[serde(try_from = "String", into = "String")]
        pub struct $name(String);

        impl $name {
            pub fn new(value: impl Into<String>) -> Result<Self, ValidationError> {
                let value = value.into();
                check_id($kind, &value)?;
                Ok(Self(value))
            }

            pub fn as_str(&self) -> &str {
                &self.0
            }
        }

        impl TryFrom<String> for $name {
            type Error = ValidationError;

            fn try_from(value: String) -> Result<Self, Self::Error> {
                Self::new(value)
            }
        }

        impl From<$name> for String {
            fn from(value: $name) -> Self {
                value.0
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(&self.0)
            }
        }
    };
}

string_id!(
    ShotId,
    "shot_id",
    "Identifies one captured shot across capture, storage and analysis."
);

string_id!(
    CameraId,
    "camera_id",
    "Identifies one camera in the booth. A stable logical name chosen by the operator (`fox-dtl`), not the hardware serial — a camera can be replaced without renaming the position it occupies."
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_normal_id_is_accepted() {
        let id = ShotId::new("2026-07-31T14-22-05Z-3f9a").unwrap();
        assert_eq!(id.as_str(), "2026-07-31T14-22-05Z-3f9a");
        assert_eq!(id.to_string(), "2026-07-31T14-22-05Z-3f9a");
    }

    #[test]
    fn an_empty_id_is_rejected() {
        let error = ShotId::new("").unwrap_err();
        assert_eq!(error.path(), "shot_id");
        assert!(error.message().contains("empty"));
    }

    #[test]
    fn path_separators_are_rejected() {
        assert!(ShotId::new("shots/one").is_err());
        assert!(ShotId::new("shots\\one").is_err());
        assert!(CameraId::new("../escape").is_err());
        assert!(ShotId::new("..").is_err());
    }

    #[test]
    fn control_characters_are_rejected() {
        assert!(CameraId::new("fox\ndtl").is_err());
    }

    #[test]
    fn ids_round_trip_as_plain_strings() {
        let id = CameraId::new("fox-dtl").unwrap();
        let json = serde_json::to_string(&id).unwrap();
        assert_eq!(json, "\"fox-dtl\"");
        assert_eq!(serde_json::from_str::<CameraId>(&json).unwrap(), id);
    }

    #[test]
    fn an_invalid_id_fails_at_deserialization() {
        let error = serde_json::from_str::<ShotId>("\"\"").unwrap_err();
        assert!(error.to_string().contains("must not be empty"));
    }
}
