use std::collections::BTreeMap;
use std::fmt;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use swingai_core::{ShotId, Timestamp, ValidationError, ValidationErrors};

use crate::{AnalysisResultVersion, ContractError, RelativePath};

/// What an analyzer concluded about one captured shot.
///
/// Mirrors `schemas/analysis-result.schema.json`. Written by the Python
/// analysis pipeline, read by the Rust runtime.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnalysisResult {
    pub schema_version: AnalysisResultVersion,
    /// The shot this describes. Matches the capture manifest, and is how the two
    /// documents are joined — there is no other link.
    pub shot_id: ShotId,
    pub analyzer: AnalyzerInfo,
    /// RFC 3339 wall-clock time the analysis finished. See the note on
    /// [`CaptureManifest::created_at`](crate::CaptureManifest::created_at).
    pub created_at: String,
    pub status: AnalysisStatus,
    pub events: Vec<SwingEvent>,
    #[serde(default)]
    pub warnings: Vec<Diagnostic>,
    #[serde(default)]
    pub errors: Vec<Diagnostic>,
    /// Scalar measurements keyed by name. Deliberately a flat open map:
    /// biomechanics is not modelled in this contract, and pinning a schema on it
    /// now would be guessing at what the coaching work needs.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub measurements: BTreeMap<String, Measurement>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub artifacts: Vec<Artifact>,
    /// Top-level fields written by a newer minor version, preserved on
    /// round-trip.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

impl AnalysisResult {
    /// Parse and check in one step. See
    /// [`CaptureManifest::from_json_str`](crate::CaptureManifest::from_json_str).
    pub fn from_json_str(json: &str) -> Result<Self, ContractError> {
        let result: Self = serde_json::from_str(json)?;
        result.validate().into_result()?;
        Ok(result)
    }

    pub fn to_json_string_pretty(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// The first event with this name. Names are free-form strings, so this is a
    /// lookup and not a guarantee — see [`SwingEvent::name`].
    #[must_use]
    pub fn event(&self, name: &str) -> Option<&SwingEvent> {
        self.events.iter().find(|event| event.name == name)
    }

    pub fn validate(&self) -> ValidationErrors {
        let mut errors = ValidationErrors::new();

        if self.created_at.trim().is_empty() {
            errors.push("created_at", "must not be empty");
        }
        errors.extend_at("analyzer", self.analyzer.validate());

        if self.status == AnalysisStatus::Ok && !self.errors.is_empty() {
            errors.push(
                "status",
                format!(
                    "is \"ok\" but {} error(s) are reported; use \"partial\" or \"failed\"",
                    self.errors.len()
                ),
            );
        }

        for (index, event) in self.events.iter().enumerate() {
            errors.extend_at(&format!("events[{index}]"), event.validate());
        }
        for (index, warning) in self.warnings.iter().enumerate() {
            errors.extend_at(&format!("warnings[{index}]"), warning.validate());
        }
        for (index, error) in self.errors.iter().enumerate() {
            errors.extend_at(&format!("errors[{index}]"), error.validate());
        }
        for (name, measurement) in &self.measurements {
            errors.extend_at(&format!("measurements.{name}"), measurement.validate());
        }
        for (index, artifact) in self.artifacts.iter().enumerate() {
            errors.extend_at(&format!("artifacts[{index}]"), artifact.validate());
        }

        errors
    }
}

/// Which analyzer produced a result. Results are only comparable within one
/// name and version — the detector changes faster than this contract does.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnalyzerInfo {
    pub name: String,
    pub version: String,
}

impl AnalyzerInfo {
    pub fn validate(&self) -> ValidationErrors {
        let mut errors = ValidationErrors::new();
        if self.name.trim().is_empty() {
            errors.push("name", "must not be empty");
        }
        if self.version.trim().is_empty() {
            errors.push("version", "must not be empty");
        }
        errors
    }
}

/// How the analysis went overall.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnalysisStatus {
    /// Everything asked for was produced. `errors` must be empty.
    Ok,
    /// Usable output with something missing — the reason is in `warnings`.
    /// The common case today: club tracking failing, so P2/P6/P8 are absent
    /// rather than guessed.
    Partial,
    /// No usable output. `events` may be empty.
    Failed,
}

impl fmt::Display for AnalysisStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Ok => "ok",
            Self::Partial => "partial",
            Self::Failed => "failed",
        })
    }
}

/// One detected instant in the swing.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SwingEvent {
    /// Event label, e.g. `P1` or `impact`.
    ///
    /// A free string on purpose. The two Python detection paths disagree about
    /// whether the velocity peak is P6 or P7 (see `CLAUDE.md`); this contract
    /// carries whatever the analyzer says it found and leaves the reconciliation
    /// to the analyzers, rather than baking one side's naming into the wire
    /// format.
    pub name: String,
    /// The analyzer's best estimate of when the event happened.
    pub timestamp_ns: Timestamp,
    /// Plausible bounds, when the analyzer can state them. A wide range on a
    /// confident detection is information — it is how "P4 is somewhere in this
    /// 100ms" gets said without pretending to a single frame.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub range: Option<TimestampRange>,
    pub confidence: Confidence,
    /// Analyzer-specific extras — which view resolved it, which rule fired.
    /// Opaque here and round-tripped unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Map<String, Value>>,
}

impl SwingEvent {
    pub fn validate(&self) -> ValidationErrors {
        let mut errors = ValidationErrors::new();

        if self.name.trim().is_empty() {
            errors.push("name", "must not be empty");
        }

        if let Some(range) = &self.range {
            errors.extend_at("range", range.validate());
            if range.is_ordered() && !range.contains(self.timestamp_ns) {
                errors.push(
                    "timestamp_ns",
                    format!(
                        "{} lies outside its own range ({}..{})",
                        self.timestamp_ns.as_nanos(),
                        range.start_timestamp_ns.as_nanos(),
                        range.end_timestamp_ns.as_nanos()
                    ),
                );
            }
        }

        errors
    }
}

/// Plausible bounds on an event, inclusive at both ends.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TimestampRange {
    pub start_timestamp_ns: Timestamp,
    pub end_timestamp_ns: Timestamp,
}

impl TimestampRange {
    #[must_use]
    pub const fn is_ordered(self) -> bool {
        self.start_timestamp_ns.as_nanos() <= self.end_timestamp_ns.as_nanos()
    }

    #[must_use]
    pub fn contains(self, timestamp: Timestamp) -> bool {
        self.start_timestamp_ns <= timestamp && timestamp <= self.end_timestamp_ns
    }

    pub fn validate(self) -> ValidationErrors {
        let mut errors = ValidationErrors::new();
        if !self.is_ordered() {
            errors.push(
                "end_timestamp_ns",
                format!(
                    "is before start_timestamp_ns ({} vs {})",
                    self.end_timestamp_ns.as_nanos(),
                    self.start_timestamp_ns.as_nanos()
                ),
            );
        }
        errors
    }
}

/// How sure the analyzer is, from 0.0 to 1.0 inclusive.
///
/// Validated at construction and at deserialization, so an out-of-range value
/// cannot exist as a typed value. Worth being strict about: `docs/STATUS.md`
/// records confidences saturating at 1.00 and carrying no information, and a
/// number that cannot even be out of range is one less thing to suspect when
/// that gets investigated.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(try_from = "f64", into = "f64")]
pub struct Confidence(f64);

impl Confidence {
    pub const CERTAIN: Self = Self(1.0);
    pub const NONE: Self = Self(0.0);

    pub fn new(value: f64) -> Result<Self, ValidationError> {
        if !value.is_finite() {
            return Err(ValidationError::new(
                "confidence",
                format!("must be a finite number, got {value}"),
            ));
        }
        if !(0.0..=1.0).contains(&value) {
            return Err(ValidationError::new(
                "confidence",
                format!("must be between 0.0 and 1.0 inclusive, got {value}"),
            ));
        }
        Ok(Self(value))
    }

    #[must_use]
    pub const fn get(self) -> f64 {
        self.0
    }
}

impl TryFrom<f64> for Confidence {
    type Error = ValidationError;

    fn try_from(value: f64) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl From<Confidence> for f64 {
    fn from(value: Confidence) -> Self {
        value.0
    }
}

impl fmt::Display for Confidence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.2}", self.0)
    }
}

/// A warning or an error from the analyzer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Diagnostic {
    /// Stable machine-readable identifier, e.g. `club_track_failed`. Stable
    /// because the runtime will eventually branch on it; the message will not
    /// be.
    pub code: String,
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context: Option<Map<String, Value>>,
}

impl Diagnostic {
    pub fn validate(&self) -> ValidationErrors {
        let mut errors = ValidationErrors::new();
        if self.code.trim().is_empty() {
            errors.push("code", "must not be empty");
        }
        if self.message.trim().is_empty() {
            errors.push("message", "must not be empty");
        }
        errors
    }
}

/// A scalar the analyzer computed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Measurement {
    pub value: f64,
    /// Free-form unit label, e.g. `ms`, `deg`, `ratio`. Optional but strongly
    /// encouraged — a bare number in a coaching report is a bug waiting to be
    /// misread.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unit: Option<String>,
}

impl Measurement {
    pub fn validate(&self) -> ValidationErrors {
        let mut errors = ValidationErrors::new();
        if !self.value.is_finite() {
            errors.push(
                "value",
                format!("must be a finite number, got {}", self.value),
            );
        }
        if let Some(unit) = &self.unit
            && unit.trim().is_empty()
        {
            errors.push("unit", "must not be empty when present");
        }
        errors
    }
}

/// A file the analysis produced, alongside the result document.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Artifact {
    /// What this file is, e.g. `overlay_video` or `chart`. An open set — the
    /// runtime shows what it recognises and lists the rest.
    pub kind: String,
    pub path: RelativePath,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

impl Artifact {
    pub fn validate(&self) -> ValidationErrors {
        let mut errors = ValidationErrors::new();
        if self.kind.trim().is_empty() {
            errors.push("kind", "must not be empty");
        }
        if let Some(description) = &self.description
            && description.trim().is_empty()
        {
            errors.push("description", "must not be empty when present");
        }
        errors
    }
}
