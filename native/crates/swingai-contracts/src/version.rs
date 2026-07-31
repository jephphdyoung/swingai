use std::fmt;

use serde::{Deserialize, Serialize};
use swingai_core::ValidationError;

/// A contract version, `MAJOR.MINOR`, gated at deserialization on the major
/// version this build understands.
///
/// The const parameter is the supported major. `SchemaVersion<1>` accepts `1.0`,
/// `1.7`, anything `1.x` — new minors are additive and unknown fields survive in
/// the `extra` maps — and refuses `2.0` outright, because a major bump means a
/// field this build depends on may have changed meaning.
///
/// Gating at deserialization rather than in a separate check is deliberate: it
/// makes an unsupported document impossible to hold in a typed value, so no
/// caller can forget to look.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct SchemaVersion<const SUPPORTED_MAJOR: u32> {
    major: u32,
    minor: u32,
}

/// The version of `schemas/capture-manifest.schema.json` this build speaks.
pub type CaptureManifestVersion = SchemaVersion<1>;

/// The version of `schemas/analysis-result.schema.json` this build speaks.
pub type AnalysisResultVersion = SchemaVersion<1>;

impl<const SUPPORTED_MAJOR: u32> SchemaVersion<SUPPORTED_MAJOR> {
    /// The newest minor version this build writes.
    pub const CURRENT: Self = Self {
        major: SUPPORTED_MAJOR,
        minor: 0,
    };

    pub const fn new(minor: u32) -> Self {
        Self {
            major: SUPPORTED_MAJOR,
            minor,
        }
    }

    pub const fn major(self) -> u32 {
        self.major
    }

    pub const fn minor(self) -> u32 {
        self.minor
    }

    /// True when the document was written by something newer than this build.
    /// Not an error — the unknown parts are preserved, not understood.
    pub const fn is_newer_than_current(self) -> bool {
        self.minor > Self::CURRENT.minor
    }

    fn parse(value: &str) -> Result<Self, ValidationError> {
        let (major, minor) = value.split_once('.').ok_or_else(|| {
            ValidationError::new(
                "schema_version",
                format!("expected \"MAJOR.MINOR\", got {value:?}"),
            )
        })?;

        let parse_part = |part: &str, which: &str| {
            part.parse::<u32>().map_err(|_| {
                ValidationError::new(
                    "schema_version",
                    format!("{which} version in {value:?} is not a number"),
                )
            })
        };

        let major = parse_part(major, "major")?;
        let minor = parse_part(minor, "minor")?;

        if major != SUPPORTED_MAJOR {
            return Err(ValidationError::new(
                "schema_version",
                format!(
                    "unsupported schema_version {value:?}: this build reads major version \
                     {SUPPORTED_MAJOR} only. Upgrade SwingAI, or have the writer emit \
                     {SUPPORTED_MAJOR}.x."
                ),
            ));
        }

        Ok(Self { major, minor })
    }
}

impl<const SUPPORTED_MAJOR: u32> Default for SchemaVersion<SUPPORTED_MAJOR> {
    fn default() -> Self {
        Self::CURRENT
    }
}

impl<const SUPPORTED_MAJOR: u32> TryFrom<String> for SchemaVersion<SUPPORTED_MAJOR> {
    type Error = ValidationError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::parse(&value)
    }
}

impl<const SUPPORTED_MAJOR: u32> From<SchemaVersion<SUPPORTED_MAJOR>> for String {
    fn from(value: SchemaVersion<SUPPORTED_MAJOR>) -> Self {
        value.to_string()
    }
}

impl<const SUPPORTED_MAJOR: u32> fmt::Display for SchemaVersion<SUPPORTED_MAJOR> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}", self.major, self.minor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type V1 = SchemaVersion<1>;

    #[test]
    fn the_current_version_round_trips() {
        let json = serde_json::to_string(&V1::CURRENT).unwrap();
        assert_eq!(json, "\"1.0\"");
        assert_eq!(serde_json::from_str::<V1>(&json).unwrap(), V1::CURRENT);
    }

    #[test]
    fn a_newer_minor_is_accepted() {
        let version: V1 = serde_json::from_str("\"1.7\"").unwrap();
        assert_eq!(version.minor(), 7);
        assert!(version.is_newer_than_current());
    }

    #[test]
    fn a_newer_major_is_rejected_with_an_actionable_message() {
        let error = serde_json::from_str::<V1>("\"2.0\"")
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("unsupported schema_version \"2.0\""),
            "{error}"
        );
        assert!(error.contains("major version 1 only"), "{error}");
        assert!(error.contains("Upgrade SwingAI"), "{error}");
    }

    #[test]
    fn an_older_major_is_rejected_too() {
        assert!(serde_json::from_str::<V1>("\"0.9\"").is_err());
    }

    #[test]
    fn malformed_versions_are_rejected() {
        for bad in ["\"1\"", "\"1.0.0\"", "\"one.zero\"", "\"\"", "\"1.x\""] {
            assert!(
                serde_json::from_str::<V1>(bad).is_err(),
                "{bad} should fail"
            );
        }
    }
}
