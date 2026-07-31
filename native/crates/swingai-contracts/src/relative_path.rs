use std::fmt;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use swingai_core::ValidationError;

/// A path inside a capture or analysis folder, relative to the document that
/// names it.
///
/// Relative on purpose: a shot folder is meant to be copied to another machine,
/// archived, or mounted at a different point in a container, and an absolute
/// path baked into a manifest breaks all three. Escaping the folder with `..` is
/// rejected for the same reason, and because a document read from elsewhere
/// should not be able to name files outside its own directory.
///
/// Stored as the original string. Separators are not rewritten, so a manifest
/// written on Windows round-trips byte-identically; [`to_path_buf`](Self::to_path_buf)
/// hands the interpretation to the platform.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct RelativePath(String);

impl RelativePath {
    pub fn new(value: impl Into<String>) -> Result<Self, ValidationError> {
        let value = value.into();

        if value.is_empty() {
            return Err(ValidationError::new("path", "must not be empty"));
        }

        if value.starts_with('/') || value.starts_with('\\') {
            return Err(ValidationError::new(
                "path",
                format!("{value:?} is absolute; paths are relative to the document"),
            ));
        }

        // "C:\..." and the "C:rel" drive-relative form are both absolute enough
        // to break a moved capture folder.
        let bytes = value.as_bytes();
        if bytes.len() >= 2 && bytes[1] == b':' && (bytes[0] as char).is_ascii_alphabetic() {
            return Err(ValidationError::new(
                "path",
                format!("{value:?} names a drive; paths are relative to the document"),
            ));
        }

        if value.split(['/', '\\']).any(|component| component == "..") {
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

    /// The path as this platform reads it. On Windows both separators work; on
    /// Linux a backslash is an ordinary character, so a manifest written on
    /// Windows with backslashes will not resolve here — writers should emit `/`.
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

    #[test]
    fn ordinary_relative_paths_are_accepted() {
        for good in [
            "streams/down_the_line.mkv",
            "artifacts/overlay.mp4",
            "clip.mkv",
            "a/b/c/d.png",
            "weird..name.mkv",
            "..hidden/file.mkv",
        ] {
            assert!(RelativePath::new(good).is_ok(), "{good} should be accepted");
        }
    }

    #[test]
    fn absolute_paths_are_rejected() {
        for bad in [
            "/var/lib/swingai/clip.mkv",
            "\\\\server\\share\\clip.mkv",
            "C:/captures/clip.mkv",
            "c:captures\\clip.mkv",
        ] {
            assert!(RelativePath::new(bad).is_err(), "{bad} should be rejected");
        }
    }

    #[test]
    fn escaping_the_document_directory_is_rejected() {
        for bad in [
            "../other/clip.mkv",
            "streams/../../clip.mkv",
            "..",
            "a\\..\\b",
        ] {
            assert!(RelativePath::new(bad).is_err(), "{bad} should be rejected");
        }
    }

    #[test]
    fn an_empty_path_is_rejected() {
        assert!(RelativePath::new("").is_err());
        assert!(serde_json::from_str::<RelativePath>("\"\"").is_err());
    }

    #[test]
    fn paths_round_trip_as_plain_strings_without_rewriting_separators() {
        let path = RelativePath::new("streams/face_on.mkv").unwrap();
        let json = serde_json::to_string(&path).unwrap();
        assert_eq!(json, "\"streams/face_on.mkv\"");
        assert_eq!(serde_json::from_str::<RelativePath>(&json).unwrap(), path);
    }

    #[test]
    fn resolving_joins_onto_the_document_directory() {
        let path = RelativePath::new("streams/face_on.mkv").unwrap();
        let resolved = path.resolve_against(Path::new("shots/abc"));
        assert_eq!(resolved, Path::new("shots/abc").join("streams/face_on.mkv"));
    }
}
