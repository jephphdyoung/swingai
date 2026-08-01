use std::fmt;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use swingai_core::ValidationError;

/// A forward-slash relative path inside a capture or analysis folder, resolved
/// against the document that names it.
///
/// **The spelling is host-independent.** A contract path always uses `/`, never
/// `\`, whichever OS wrote it. A backslash is rejected outright rather than
/// normalised, because normalising is guesswork: on Linux `clips\shot.mkv` is a
/// single legal filename containing a backslash, so silently reinterpreting it as
/// a directory separator would invent a path the writer never meant. Rejecting at
/// the seam makes a Windows writer emitting native separators a loud bug in that
/// writer.
///
/// That one rule also disposes of the whole family of Windows absolute forms —
/// `C:\...`, UNC `\\server\share`, extended-length `\\?\C:\...`, and rooted
/// `\captures\...` — leaving only the drive-relative `C:rel` spelling to reject
/// separately.
///
/// Relative on purpose: a shot folder is meant to be copied to another machine,
/// archived, or mounted at a different point in a container, and an absolute path
/// baked into a manifest breaks all three. `..` is rejected for the same reason,
/// and because a document read from elsewhere should not be able to name files
/// outside its own directory.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct RelativePath(String);

impl RelativePath {
    pub fn new(value: impl Into<String>) -> Result<Self, ValidationError> {
        let value = value.into();

        if value.is_empty() {
            return Err(ValidationError::new("path", "must not be empty"));
        }

        // First, because it also covers "\\server\share", "\\?\C:\..." and
        // "\rooted" — and because the message should name the real problem.
        if value.contains('\\') {
            return Err(ValidationError::new(
                "path",
                format!(
                    "{value:?} contains a backslash; contract paths are forward-slash \
                     relative paths regardless of the host OS (use \"a/b/c.mkv\")"
                ),
            ));
        }

        if value.starts_with('/') {
            return Err(ValidationError::new(
                "path",
                format!("{value:?} is absolute; paths are relative to the document"),
            ));
        }

        // Both "C:/captures/shot.mkv" and the drive-relative "C:captures" form.
        let bytes = value.as_bytes();
        if bytes.len() >= 2 && bytes[1] == b':' && bytes[0].is_ascii_alphabetic() {
            return Err(ValidationError::new(
                "path",
                format!("{value:?} names a drive; paths are relative to the document"),
            ));
        }

        if value.split('/').any(|component| component == "..") {
            return Err(ValidationError::new(
                "path",
                format!("{value:?} escapes the document's directory with \"..\""),
            ));
        }

        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// The path as this platform reads it.
    ///
    /// Forward slashes are separators on Linux and on Windows alike, so a
    /// contract path resolves natively on both without rewriting.
    pub fn to_path_buf(&self) -> PathBuf {
        PathBuf::from(&self.0)
    }

    /// Resolve against the directory holding the document that named it.
    pub fn resolve_against(&self, base: &Path) -> PathBuf {
        base.join(&self.0)
    }
}

impl TryFrom<String> for RelativePath {
    type Error = ValidationError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl From<RelativePath> for String {
    fn from(value: RelativePath) -> Self {
        value.0
    }
}

impl fmt::Display for RelativePath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every spelling the contract must refuse, with why it is refused.
    const REJECTED: &[&str] = &[
        "C:\\captures\\shot.mkv",        // Windows absolute
        "C:captures\\shot.mkv",          // Windows drive-relative
        "\\\\server\\share\\shot.mkv",   // UNC
        "\\\\?\\C:\\captures\\shot.mkv", // extended-length
        "\\captures\\shot.mkv",          // rooted on the current drive
        "/captures/shot.mkv",            // Unix absolute
        "../shot.mkv",                   // traversal
        "clips/../../shot.mkv",          // traversal, mid-path
        "clips\\shot.mkv",               // native separator, otherwise harmless
        "C:/captures/shot.mkv",          // drive with forward slashes
        "..",                            // traversal, bare
        "",                              // empty
    ];

    #[test]
    fn every_non_portable_spelling_is_rejected() {
        for bad in REJECTED {
            assert!(
                RelativePath::new(*bad).is_err(),
                "{bad:?} should be rejected"
            );
        }
    }

    #[test]
    fn rejection_messages_name_the_actual_problem() {
        let backslash = RelativePath::new("clips\\shot.mkv").unwrap_err();
        assert!(backslash.message().contains("backslash"), "{backslash}");

        let unix_absolute = RelativePath::new("/captures/shot.mkv").unwrap_err();
        assert!(
            unix_absolute.message().contains("absolute"),
            "{unix_absolute}"
        );

        let drive = RelativePath::new("C:captures").unwrap_err();
        assert!(drive.message().contains("drive"), "{drive}");

        let traversal = RelativePath::new("clips/../../shot.mkv").unwrap_err();
        assert!(traversal.message().contains("escapes"), "{traversal}");
    }

    #[test]
    fn forward_slash_relative_paths_are_accepted() {
        for good in [
            "streams/down_the_line.mkv",
            "artifacts/overlay.mp4",
            "clip.mkv",
            "a/b/c/d.png",
            "weird..name.mkv", // ".." inside a component, not a component
            "..hidden/file.mkv",
            "frame_%06d.png",
        ] {
            assert!(RelativePath::new(good).is_ok(), "{good} should be accepted");
        }
    }

    #[test]
    fn paths_round_trip_as_plain_strings() {
        let path = RelativePath::new("streams/face_on.mkv").unwrap();
        let json = serde_json::to_string(&path).unwrap();
        assert_eq!(json, "\"streams/face_on.mkv\"");
        assert_eq!(serde_json::from_str::<RelativePath>(&json).unwrap(), path);
    }

    #[test]
    fn a_backslash_path_fails_at_deserialization() {
        let error = serde_json::from_str::<RelativePath>("\"clips\\\\shot.mkv\"").unwrap_err();
        assert!(error.to_string().contains("backslash"), "{error}");
    }

    #[test]
    fn a_forward_slash_path_resolves_natively_on_this_platform() {
        // The point of forbidding backslashes: one spelling in the document
        // resolves to the same sequence of components on either host OS, using
        // that host's own separator when displayed.
        let path = RelativePath::new("streams/face_on.mkv").unwrap();
        let base = Path::new("shots").join("2026-07-31T14-22-05Z-3f9a");
        let resolved = path.resolve_against(&base);

        let components: Vec<String> = resolved
            .components()
            .map(|component| component.as_os_str().to_string_lossy().into_owned())
            .collect();
        assert_eq!(
            components,
            [
                "shots",
                "2026-07-31T14-22-05Z-3f9a",
                "streams",
                "face_on.mkv"
            ],
            "a contract path must split into the same components on every host"
        );

        assert!(resolved.starts_with(&base));
        assert_eq!(resolved.file_name().unwrap(), "face_on.mkv");
        assert_eq!(resolved.extension().unwrap(), "mkv");
        assert_eq!(path.to_path_buf().components().count(), 2);
    }
}
