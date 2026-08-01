use std::fmt;

use serde::{Deserialize, Serialize};

use crate::ValidationError;

/// Names Windows reserves for DOS devices. Reserved with *any* extension and in
/// any case, so `CON`, `con.txt` and `Con.capture` all name the console rather
/// than a file — creating one fails, or worse, silently writes to the device.
const RESERVED_DEVICE_NAMES: [&str; 4] = ["CON", "PRN", "AUX", "NUL"];

/// `COM1`–`COM9` and `LPT1`–`LPT9` are reserved the same way. `COM10` is not,
/// and neither is a bare `COM`.
fn is_numbered_device(upper: &str) -> bool {
    let Some(suffix) = upper
        .strip_prefix("COM")
        .or_else(|| upper.strip_prefix("LPT"))
    else {
        return false;
    };
    matches!(suffix.as_bytes(), [b'1'..=b'9'])
}

/// Whether `value` names a DOS device once its extension is stripped.
fn names_a_reserved_device(value: &str) -> bool {
    // Windows looks at the part before the first `.`, so `NUL.capture` is still
    // the null device.
    let stem = value.split('.').next().unwrap_or(value);
    let upper = stem.to_ascii_uppercase();
    RESERVED_DEVICE_NAMES.contains(&upper.as_str()) || is_numbered_device(&upper)
}

/// Everything that would make an id unusable as a directory name, or ambiguous
/// in a log line.
///
/// The rules are Windows' rather than Linux's, deliberately: the booth is a
/// Windows machine (ADR 0001) and a capture folder is meant to be copied
/// between hosts, so the stricter platform sets the rule. An id that Linux
/// would accept and Windows would not is a shot that cannot be filed on the
/// machine that recorded it.
///
/// This is the single authority on the question. The shot writer turns a
/// [`ShotId`] straight into a directory name and adds no checks of its own —
/// a second copy of these rules would be a second copy to disagree with.
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
    // Windows strips these when creating a file, so `shot.` and `shot` would
    // become the same directory — and an id that does not round-trip through
    // the filesystem is not an id.
    if value.ends_with('.') || value.ends_with(' ') {
        return Err(ValidationError::new(
            kind,
            format!(
                "{value:?} ends in a period or a space, which Windows silently strips \
                 from a directory name"
            ),
        ));
    }
    if names_a_reserved_device(value) {
        return Err(ValidationError::new(
            kind,
            format!(
                "{value:?} names a reserved Windows device (CON, PRN, AUX, NUL, \
                 COM1-COM9, LPT1-LPT9 — in any case, with or without an extension), \
                 so it cannot be used as a directory name"
            ),
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

    /// Every DOS device name, in the spellings that actually turn up.
    const RESERVED: &[&str] = &[
        "CON", "PRN", "AUX", "NUL", // the four bare names
        "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9", "LPT1", "LPT2",
        "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
    ];

    #[test]
    fn every_reserved_device_family_is_rejected() {
        for name in RESERVED {
            let error = ShotId::new(*name)
                .expect_err(&format!("{name} should be rejected as a device name"));
            assert!(
                error.message().contains("reserved Windows device"),
                "{name}"
            );
            assert!(
                CameraId::new(*name).is_err(),
                "{name} must be rejected for a camera id too"
            );
        }
    }

    #[test]
    fn reserved_names_are_matched_without_regard_to_case() {
        for name in ["Con", "con", "cOn", "com1", "Com1", "nUl", "lpt9", "AuX"] {
            assert!(ShotId::new(name).is_err(), "{name} should be rejected");
        }
    }

    #[test]
    fn an_extension_does_not_rescue_a_reserved_name() {
        // Windows resolves the device from the stem, so the extension is
        // decoration -- this is the spelling people reach for when the bare
        // name is refused.
        for name in [
            "CON.txt",
            "nul.capture",
            "COM1.data",
            "aux.tar.gz",
            "LPT3.",
            "prn.",
        ] {
            assert!(ShotId::new(name).is_err(), "{name} should be rejected");
        }
    }

    #[test]
    fn a_trailing_period_or_space_is_rejected() {
        for name in ["shot.", "shot ", "2026-07-31T14-22-05Z.", "a b "] {
            let error = ShotId::new(name).unwrap_err();
            assert!(
                error.message().contains("period or a space"),
                "{name}: {error}"
            );
        }
    }

    #[test]
    fn names_that_merely_resemble_reserved_ones_are_accepted() {
        // The rule has to be narrow, or it eats ordinary ids: `COM10` is not a
        // device, and neither is anything with the name as a prefix or inside
        // a longer word.
        for name in [
            "CON-shot",
            "COM10",
            "COM0",
            "camera.1",
            "shot-name",
            "console",
            "NULL",
            "my-con",
            "LPT",
            "COM",
            "aux-cam",
        ] {
            assert!(ShotId::new(name).is_ok(), "{name} should be accepted");
            assert!(CameraId::new(name).is_ok(), "{name} should be accepted");
        }
    }

    #[test]
    fn a_reserved_name_fails_at_deserialization_too() {
        let error = serde_json::from_str::<CameraId>("\"nul\"").unwrap_err();
        assert!(
            error.to_string().contains("reserved Windows device"),
            "{error}"
        );
    }

    #[test]
    fn the_ids_the_capture_runtime_actually_generates_are_accepted() {
        // The wall-clock shot id and the booth's camera names, so a rule change
        // that broke them would fail here rather than at a capture.
        for name in [
            "2026-07-31T14-22-05.412Z",
            "2026-07-31T16-22-05+02-00",
            "sim-dtl",
            "sim-face-on",
            "fox-dtl",
        ] {
            assert!(ShotId::new(name).is_ok(), "{name} should be accepted");
        }
    }
}
