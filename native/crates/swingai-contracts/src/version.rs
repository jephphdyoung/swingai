use std::fmt;

use serde::{Deserialize, Serialize};
use swingai_core::ValidationError;

/// A contract version, `MAJOR.MINOR`, accepted only when it is exactly the
/// version this build speaks.
///
/// **Strict on purpose, for now.** An earlier draft accepted any matching major
/// on the theory that minor versions are additive and unknown fields would be
/// carried through. That guarantee was not real: unknown fields were preserved
/// only at the document root, so a field added inside a stream or an event would
/// have been silently dropped by an older reader — the worst kind of
/// compatibility, the kind that looks like it works.
///
/// Until there is a concrete need for cross-version reading, both sides move
/// together and a mismatch is an error you can see. A later ADR can introduce
/// real minor-version compatibility, with the preservation to back it up.
///
/// The const parameters are the supported version, so the supported value is
/// part of the type: `SchemaVersion<1, 0>` accepts `"1.0"` and nothing else.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct SchemaVersion<const MAJOR: u32, const MINOR: u32>;

/// The version of `schemas/capture-manifest.schema.json` this build speaks.
pub type CaptureManifestVersion = SchemaVersion<1, 0>;

/// The version of `schemas/analysis-result.schema.json` this build speaks.
pub type AnalysisResultVersion = SchemaVersion<1, 0>;

impl<const MAJOR: u32, const MINOR: u32> SchemaVersion<MAJOR, MINOR> {
    /// The only version this build reads or writes.
    pub const CURRENT: Self = Self;

    pub const fn major(self) -> u32 {
        MAJOR
    }

    pub const fn minor(self) -> u32 {
        MINOR
    }

    fn parse(value: &str) -> Result<Self, ValidationError> {
        let unsupported = |detail: &str| {
            ValidationError::new(
                "schema_version",
                format!(
                    "unsupported schema_version {value:?}: {detail}. This build reads and \
                     writes {MAJOR}.{MINOR} only — update SwingAI, or have the writer emit \
                     {MAJOR}.{MINOR}."
                ),
            )
        };

        let Some((major, minor)) = value.split_once('.') else {
            return Err(unsupported("expected \"MAJOR.MINOR\""));
        };

        let (Ok(major), Ok(minor)) = (major.parse::<u32>(), minor.parse::<u32>()) else {
            return Err(unsupported("both parts must be numbers"));
        };

        if (major, minor) != (MAJOR, MINOR) {
            return Err(unsupported("version mismatch"));
        }

        Ok(Self)
    }
}

impl<const MAJOR: u32, const MINOR: u32> Default for SchemaVersion<MAJOR, MINOR> {
    fn default() -> Self {
        Self::CURRENT
    }
}

impl<const MAJOR: u32, const MINOR: u32> TryFrom<String> for SchemaVersion<MAJOR, MINOR> {
    type Error = ValidationError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::parse(&value)
    }
}

impl<const MAJOR: u32, const MINOR: u32> From<SchemaVersion<MAJOR, MINOR>> for String {
    fn from(value: SchemaVersion<MAJOR, MINOR>) -> Self {
        value.to_string()
    }
}

impl<const MAJOR: u32, const MINOR: u32> fmt::Display for SchemaVersion<MAJOR, MINOR> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{MAJOR}.{MINOR}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type V1_0 = SchemaVersion<1, 0>;

    #[test]
    fn the_supported_version_round_trips() {
        let json = serde_json::to_string(&V1_0::CURRENT).unwrap();
        assert_eq!(json, "\"1.0\"");
        assert_eq!(serde_json::from_str::<V1_0>(&json).unwrap(), V1_0::CURRENT);
        assert_eq!(V1_0::CURRENT.major(), 1);
        assert_eq!(V1_0::CURRENT.minor(), 0);
    }

    #[test]
    fn a_newer_minor_is_rejected() {
        for bad in ["\"1.1\"", "\"1.4\""] {
            let error = serde_json::from_str::<V1_0>(bad).unwrap_err().to_string();
            assert!(
                error.contains("unsupported schema_version"),
                "{bad}: {error}"
            );
            assert!(error.contains("1.0 only"), "{bad}: {error}");
        }
    }

    #[test]
    fn a_newer_major_is_rejected() {
        let error = serde_json::from_str::<V1_0>("\"2.0\"")
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("unsupported schema_version \"2.0\""),
            "{error}"
        );
        assert!(error.contains("update SwingAI"), "{error}");
    }

    #[test]
    fn an_older_version_is_rejected_too() {
        assert!(serde_json::from_str::<V1_0>("\"0.9\"").is_err());
    }

    #[test]
    fn malformed_versions_are_rejected() {
        for bad in [
            "\"1\"",
            "\"1.0.0\"",
            "\"one.zero\"",
            "\"\"",
            "\"1.x\"",
            "\"v1.0\"",
        ] {
            assert!(
                serde_json::from_str::<V1_0>(bad).is_err(),
                "{bad} should fail"
            );
        }
    }

    #[test]
    fn every_rejection_says_what_is_supported() {
        for bad in ["\"1.1\"", "\"2.0\"", "\"nonsense\""] {
            let error = serde_json::from_str::<V1_0>(bad).unwrap_err().to_string();
            assert!(error.contains("1.0 only"), "{bad}: {error}");
        }
    }
}
