use std::fmt;

use serde::{Deserialize, Serialize};

use crate::ValidationError;

/// How the stored frames are laid out.
///
/// An open set, unlike [`CameraView`](crate::CameraView): machine-vision cameras
/// invent formats, and a vendor-specific one should not require a contract
/// revision. The [`is_known`](Self::is_known) constants are the ones we expect to
/// see; anything else parses fine and is passed through.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct PixelFormat(String);

impl PixelFormat {
    pub const MONO8: &'static str = "mono8";
    pub const MONO16: &'static str = "mono16";
    pub const BAYER_RG8: &'static str = "bayer_rg8";
    pub const BAYER_BG8: &'static str = "bayer_bg8";
    pub const BGR8: &'static str = "bgr8";
    pub const RGB8: &'static str = "rgb8";
    pub const YUV420P: &'static str = "yuv420p";
    pub const NV12: &'static str = "nv12";

    const KNOWN: [&'static str; 8] = [
        Self::MONO8,
        Self::MONO16,
        Self::BAYER_RG8,
        Self::BAYER_BG8,
        Self::BGR8,
        Self::RGB8,
        Self::YUV420P,
        Self::NV12,
    ];

    pub fn new(value: impl Into<String>) -> Result<Self, ValidationError> {
        let value = value.into();
        if value.is_empty() {
            return Err(ValidationError::new("pixel_format", "must not be empty"));
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Whether this is a format we have a decoder story for. A `false` here is
    /// not an error — it is a prompt to check before assuming the frames can be
    /// read.
    pub fn is_known(&self) -> bool {
        Self::KNOWN.contains(&self.0.as_str())
    }
}

impl TryFrom<String> for PixelFormat {
    type Error = ValidationError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl From<PixelFormat> for String {
    fn from(value: PixelFormat) -> Self {
        value.0
    }
}

impl fmt::Display for PixelFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_known_format_is_recognised() {
        assert!(PixelFormat::new(PixelFormat::MONO8).unwrap().is_known());
    }

    #[test]
    fn a_vendor_format_parses_but_is_not_known() {
        let format = PixelFormat::new("fox_packed12").unwrap();
        assert!(!format.is_known());
        assert_eq!(format.as_str(), "fox_packed12");
    }

    #[test]
    fn an_empty_format_is_rejected() {
        assert!(PixelFormat::new("").is_err());
        assert!(serde_json::from_str::<PixelFormat>("\"\"").is_err());
    }

    #[test]
    fn formats_round_trip_as_plain_strings() {
        let format = PixelFormat::new(PixelFormat::BAYER_RG8).unwrap();
        let json = serde_json::to_string(&format).unwrap();
        assert_eq!(json, "\"bayer_rg8\"");
        assert_eq!(serde_json::from_str::<PixelFormat>(&json).unwrap(), format);
    }
}
