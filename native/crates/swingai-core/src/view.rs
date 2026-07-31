use std::fmt;

use serde::{Deserialize, Serialize};

/// Where a camera is pointed from.
///
/// A closed set, deliberately. The Python side spells down-the-line `dtl`
/// (`utils/swing_pairing.py`, `data/annotations.json`); this contract spells it
/// out in full and *rejects* the abbreviation rather than guessing, so a
/// mismatched bridge fails loudly at the seam instead of silently filing a swing
/// under the wrong view.
///
/// [`Unknown`](Self::Unknown) is for a camera whose position was never
/// configured — it is an honest answer, not a parse fallback.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CameraView {
    FaceOn,
    DownTheLine,
    Overhead,
    /// A close, high-rate camera aimed at the ball, for impact only.
    Impact,
    Unknown,
}

impl CameraView {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::FaceOn => "face_on",
            Self::DownTheLine => "down_the_line",
            Self::Overhead => "overhead",
            Self::Impact => "impact",
            Self::Unknown => "unknown",
        }
    }
}

impl fmt::Display for CameraView {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn views_round_trip_in_snake_case() {
        for (view, json) in [
            (CameraView::FaceOn, "\"face_on\""),
            (CameraView::DownTheLine, "\"down_the_line\""),
            (CameraView::Overhead, "\"overhead\""),
            (CameraView::Impact, "\"impact\""),
            (CameraView::Unknown, "\"unknown\""),
        ] {
            assert_eq!(serde_json::to_string(&view).unwrap(), json);
            assert_eq!(serde_json::from_str::<CameraView>(json).unwrap(), view);
            assert_eq!(format!("\"{view}\""), json);
        }
    }

    #[test]
    fn the_python_abbreviation_is_rejected_rather_than_guessed() {
        let error = serde_json::from_str::<CameraView>("\"dtl\"").unwrap_err();
        assert!(error.to_string().contains("down_the_line"));
    }
}
