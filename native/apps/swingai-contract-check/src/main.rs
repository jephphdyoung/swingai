//! Validates SwingAI contract documents.
//!
//! ```text
//! swingai-contract-check capture  <path>
//! swingai-contract-check analysis <path>
//! ```
//!
//! This does not run JSON Schema validation. Deserializing into the
//! `swingai-contracts` types already enforces required fields, types, the schema
//! version, timestamp representation and path relativity; the explicit semantic
//! checks cover the rest. Adding a schema validator would mean a second
//! implementation of the same rules that could disagree with the first — the
//! Python test in `tests/test_schema_examples.py` checks the examples against the
//! schemas, which is where that belongs.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use swingai_contracts::{AnalysisResult, CameraView, CaptureManifest, ContractError};

const USAGE: &str = "\
usage: swingai-contract-check <capture|analysis> <path>

  capture  <path>   validate a capture manifest
  analysis <path>   validate an analysis result

Exits 0 when the document is valid, 1 when it is not, 2 on bad arguments.";

fn main() -> ExitCode {
    let mut args = std::env::args_os().skip(1);
    let (Some(kind), Some(path)) = (args.next(), args.next()) else {
        eprintln!("{USAGE}");
        return ExitCode::from(2);
    };

    if args.next().is_some() {
        eprintln!("error: too many arguments\n\n{USAGE}");
        return ExitCode::from(2);
    }

    let path = PathBuf::from(path);
    let outcome = match kind.to_str() {
        Some("capture") => check_capture(&path),
        Some("analysis") => check_analysis(&path),
        other => {
            eprintln!(
                "error: unknown document kind {:?}\n\n{USAGE}",
                other.unwrap_or("<invalid utf-8>")
            );
            return ExitCode::from(2);
        }
    };

    match outcome {
        Ok(summary) => {
            println!("{} {summary}", path.display());
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("{}: {error}", path.display());
            ExitCode::FAILURE
        }
    }
}

/// Read plus parse plus validate, with the two failure kinds kept apart so the
/// message says which happened.
enum CheckError {
    Read(std::io::Error),
    Contract(ContractError),
}

impl std::fmt::Display for CheckError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Read(error) => write!(f, "could not read the file: {error}"),
            Self::Contract(error) => write!(f, "{error}"),
        }
    }
}

fn read(path: &Path) -> Result<String, CheckError> {
    std::fs::read_to_string(path).map_err(CheckError::Read)
}

fn check_capture(path: &Path) -> Result<String, CheckError> {
    let json = read(path)?;
    let manifest = CaptureManifest::from_json_str(&json).map_err(CheckError::Contract)?;

    let mut summary = format!(
        "is a valid capture manifest (schema {}, shot {}, {} stream(s))",
        manifest.schema_version,
        manifest.shot_id,
        manifest.streams.len()
    );

    for stream in &manifest.streams {
        let duration_ms = stream
            .frames
            .last_timestamp_ns
            .millis_since(stream.frames.first_timestamp_ns);
        let measured = stream
            .frames
            .measured_fps()
            .map_or_else(|| "n/a".to_owned(), |fps| format!("{fps:.1}"));
        summary.push_str(&format!(
            "\n  {} [{}] {}x{} {} — {} frames over {:.0}ms ({} fps measured, {} nominal), \
             {} dropped",
            stream.camera_id,
            stream.view,
            stream.width,
            stream.height,
            stream.pixel_format,
            stream.frames.frame_count,
            duration_ms,
            measured,
            stream.frames.nominal_fps,
            stream.frames.dropped_frame_count,
        ));

        if !stream.pixel_format.is_known() {
            summary.push_str(&format!(
                "\n    note: pixel format {} is not one this build knows how to decode",
                stream.pixel_format
            ));
        }
        if stream.view == CameraView::Unknown {
            summary.push_str("\n    note: camera view is \"unknown\"");
        }
    }

    Ok(summary)
}

fn check_analysis(path: &Path) -> Result<String, CheckError> {
    let json = read(path)?;
    let result = AnalysisResult::from_json_str(&json).map_err(CheckError::Contract)?;

    let mut summary = format!(
        "is a valid analysis result (schema {}, shot {}, {} v{}, status {})",
        result.schema_version,
        result.shot_id,
        result.analyzer.name,
        result.analyzer.version,
        result.status,
    );
    summary.push_str(&format!(
        "\n  {} event(s), {} warning(s), {} error(s), {} measurement(s), {} artifact(s)",
        result.events.len(),
        result.warnings.len(),
        result.errors.len(),
        result.measurements.len(),
        result.artifacts.len(),
    ));

    for event in &result.events {
        summary.push_str(&format!(
            "\n  {:<6} {:>18}ns  confidence {}",
            event.name,
            event.timestamp_ns.as_nanos(),
            event.confidence
        ));
    }
    for warning in &result.warnings {
        summary.push_str(&format!(
            "\n  warning [{}] {}",
            warning.code, warning.message
        ));
    }

    Ok(summary)
}
