//! Contract tests against the checked-in examples and against the failures the
//! contracts are supposed to catch.
//!
//! The examples in `schemas/examples/` are the shared fixture: the Rust types
//! parse them here, and `tests/test_schema_examples.py` validates the same files
//! against the JSON Schemas. If the two ever disagree, one of them is wrong about
//! the contract, which is the point of checking both.

use std::path::{Path, PathBuf};

use serde_json::{Value, json};
use swingai_contracts::{
    AnalysisResult, AnalysisStatus, CameraView, CaptureManifest, Confidence, ContractError,
    MediaSource, RelativePath, Timestamp,
};

fn repo_root() -> PathBuf {
    // crates/swingai-contracts -> crates -> native -> repo root
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(3)
        .expect("crate is nested three levels under the repo root")
        .to_path_buf()
}

fn example(name: &str) -> String {
    let path = repo_root().join("schemas/examples").join(name);
    std::fs::read_to_string(&path).unwrap_or_else(|error| panic!("reading {path:?}: {error}"))
}

fn capture_example() -> String {
    example("capture-manifest.example.json")
}

fn analysis_example() -> String {
    example("analysis-result.example.json")
}

/// Parse an example to JSON, apply an edit, and re-render. Lets a test say
/// exactly which field it is breaking without restating the whole document.
fn tweaked(json: &str, edit: impl FnOnce(&mut Value)) -> String {
    let mut value: Value = serde_json::from_str(json).expect("example is valid JSON");
    edit(&mut value);
    serde_json::to_string(&value).expect("re-rendering cannot fail")
}

fn expect_invalid(error: ContractError) -> String {
    match error {
        ContractError::Invalid(errors) => errors.to_string(),
        ContractError::Parse(error) => {
            panic!("expected a validation failure, got a parse error: {error}")
        }
    }
}

fn expect_parse_failure(error: ContractError) -> String {
    match error {
        ContractError::Parse(error) => error.to_string(),
        ContractError::Invalid(errors) => {
            panic!("expected a parse failure, got validation errors: {errors}")
        }
    }
}

// --- the examples ------------------------------------------------------------

#[test]
fn capture_example_deserializes() {
    let manifest =
        CaptureManifest::from_json_str(&capture_example()).expect("example should be valid");

    assert_eq!(manifest.schema_version.to_string(), "1.0");
    assert_eq!(manifest.shot_id.as_str(), "2026-07-31T14-22-05Z-3f9a");
    assert_eq!(manifest.streams.len(), 2);
    assert!(manifest.trigger_timestamp_ns.is_some());

    let dtl = manifest
        .stream_for_view(CameraView::DownTheLine)
        .expect("one down-the-line camera");
    assert_eq!(dtl.camera_id.as_str(), "fox-dtl");
    assert_eq!(dtl.serial_number.as_deref(), Some("61fc"));
    assert_eq!(dtl.frames.frame_count, 1244);
    assert_eq!(dtl.frames.dropped_frame_count, 0);
    assert!(matches!(dtl.media, MediaSource::Video { .. }));
    assert_eq!(dtl.media.path().as_str(), "streams/down_the_line.mkv");

    // The measured rate comes from the timestamps, not the configured rate.
    let measured = dtl.frames.measured_fps().expect("more than one frame");
    assert!((measured - 249.3).abs() < 0.01, "measured {measured}");

    let face_on = manifest
        .stream_for_view(CameraView::FaceOn)
        .expect("one face-on camera");
    assert_eq!(face_on.frames.dropped_frame_count, 1);
    assert_eq!(face_on.gaps.len(), 1);
    assert_eq!(face_on.gaps[0].missing_frame_count, 1);
    assert_eq!(face_on.frames.reported_frame_count(), 1244);
}

#[test]
fn analysis_example_deserializes() {
    let result =
        AnalysisResult::from_json_str(&analysis_example()).expect("example should be valid");

    assert_eq!(result.schema_version.to_string(), "1.0");
    assert_eq!(result.shot_id.as_str(), "2026-07-31T14-22-05Z-3f9a");
    assert_eq!(result.analyzer.name, "swingai.p_positions");
    assert_eq!(result.status, AnalysisStatus::Partial);
    assert_eq!(result.events.len(), 7);
    assert_eq!(result.warnings.len(), 2);
    assert!(result.errors.is_empty());

    let impact = result.event("P7").expect("P7 is present");
    assert!((impact.confidence.get() - 0.94).abs() < f64::EPSILON);
    let range = impact.range.expect("P7 carries a range");
    assert!(range.contains(impact.timestamp_ns));

    assert_eq!(result.measurements["tempo_ratio"].value, 3.0);
    assert_eq!(
        result.measurements["tempo_ratio"].unit.as_deref(),
        Some("ratio")
    );
}

#[test]
fn the_two_examples_describe_the_same_shot() {
    let manifest = CaptureManifest::from_json_str(&capture_example()).unwrap();
    let result = AnalysisResult::from_json_str(&analysis_example()).unwrap();
    assert_eq!(manifest.shot_id, result.shot_id);

    // Every event must fall inside the window the cameras actually recorded —
    // this is the check that the shared monotonic clock is really shared.
    let first = manifest
        .streams
        .iter()
        .map(|stream| stream.frames.first_timestamp_ns)
        .min()
        .unwrap();
    let last = manifest
        .streams
        .iter()
        .map(|stream| stream.frames.last_timestamp_ns)
        .max()
        .unwrap();
    for event in &result.events {
        assert!(
            first <= event.timestamp_ns && event.timestamp_ns <= last,
            "{} at {} falls outside the capture window",
            event.name,
            event.timestamp_ns
        );
    }
}

// --- round trips -------------------------------------------------------------

#[test]
fn capture_round_trip_preserves_everything() {
    let original = CaptureManifest::from_json_str(&capture_example()).unwrap();
    let rendered = original.to_json_string_pretty().unwrap();
    let reparsed = CaptureManifest::from_json_str(&rendered).unwrap();
    assert_eq!(original, reparsed);

    // And the camera-specific metadata survives as written, key for key.
    let value: Value = serde_json::from_str(&rendered).unwrap();
    assert_eq!(value["streams"][0]["metadata"]["exposure_us"], json!(1200));
    assert_eq!(
        value["streams"][0]["metadata"]["trigger_mode"],
        json!("free_run")
    );
}

#[test]
fn analysis_round_trip_preserves_everything() {
    let original = AnalysisResult::from_json_str(&analysis_example()).unwrap();
    let rendered = original.to_json_string_pretty().unwrap();
    let reparsed = AnalysisResult::from_json_str(&rendered).unwrap();
    assert_eq!(original, reparsed);

    let value: Value = serde_json::from_str(&rendered).unwrap();
    assert_eq!(
        value["events"][4]["metadata"]["method"],
        json!("hand_speed_peak")
    );
    assert_eq!(
        value["warnings"][1]["context"]["delta_ns"],
        json!(140_000_000i64)
    );
}

#[test]
fn timestamps_survive_a_round_trip_exactly() {
    // The failure this guards against is a writer or reader treating nanoseconds
    // as a float somewhere; 128_474_265_400_000 would survive that, but a value
    // with low-order digits would not.
    let odd = 128_474_265_400_017i64;
    let json = tweaked(&analysis_example(), |value| {
        value["events"][0]["timestamp_ns"] = json!(odd);
        value["events"][0]["range"] = json!({
            "start_timestamp_ns": odd - 1,
            "end_timestamp_ns": odd + 1,
        });
    });

    let result = AnalysisResult::from_json_str(&json).unwrap();
    assert_eq!(result.events[0].timestamp_ns, Timestamp::from_nanos(odd));

    let reparsed = AnalysisResult::from_json_str(&result.to_json_string_pretty().unwrap()).unwrap();
    assert_eq!(reparsed.events[0].timestamp_ns, Timestamp::from_nanos(odd));
}

// --- schema versions ---------------------------------------------------------

#[test]
fn an_unsupported_capture_schema_version_fails_clearly() {
    let json = tweaked(&capture_example(), |value| {
        value["schema_version"] = json!("2.0");
    });
    let message = expect_parse_failure(CaptureManifest::from_json_str(&json).unwrap_err());
    assert!(
        message.contains("unsupported schema_version \"2.0\""),
        "{message}"
    );
    assert!(message.contains("major version 1 only"), "{message}");
}

#[test]
fn an_unsupported_analysis_schema_version_fails_clearly() {
    let json = tweaked(&analysis_example(), |value| {
        value["schema_version"] = json!("7.3");
    });
    let message = expect_parse_failure(AnalysisResult::from_json_str(&json).unwrap_err());
    assert!(
        message.contains("unsupported schema_version \"7.3\""),
        "{message}"
    );
}

#[test]
fn a_newer_minor_version_is_accepted_and_its_extra_fields_are_preserved() {
    let json = tweaked(&capture_example(), |value| {
        value["schema_version"] = json!("1.4");
        value["ambient_temperature_c"] = json!(21.5);
    });

    let manifest = CaptureManifest::from_json_str(&json).expect("a newer minor is readable");
    assert!(manifest.schema_version.is_newer_than_current());
    assert_eq!(manifest.extra["ambient_temperature_c"], json!(21.5));

    let rendered = manifest.to_json_string_pretty().unwrap();
    let value: Value = serde_json::from_str(&rendered).unwrap();
    assert_eq!(
        value["ambient_temperature_c"],
        json!(21.5),
        "a round trip through an older build must not delete what it did not understand"
    );
}

// --- timestamps and frame counts ---------------------------------------------

#[test]
fn last_timestamp_before_first_fails() {
    let json = tweaked(&capture_example(), |value| {
        value["streams"][0]["last_timestamp_ns"] = json!(128_471_215_399_999i64);
    });
    let message = expect_invalid(CaptureManifest::from_json_str(&json).unwrap_err());
    assert!(
        message.contains("streams[0].last_timestamp_ns"),
        "{message}"
    );
    assert!(message.contains("before first_timestamp_ns"), "{message}");
}

#[test]
fn many_frames_sharing_one_instant_fails() {
    let json = tweaked(&capture_example(), |value| {
        value["streams"][0]["last_timestamp_ns"] =
            value["streams"][0]["first_timestamp_ns"].clone();
    });
    let message = expect_invalid(CaptureManifest::from_json_str(&json).unwrap_err());
    assert!(message.contains("cannot share an instant"), "{message}");
}

#[test]
fn a_negative_frame_count_is_rejected_by_representation() {
    let json = tweaked(&capture_example(), |value| {
        value["streams"][0]["frame_count"] = json!(-1);
    });
    // u32 cannot hold it, so this never reaches validation.
    let message = expect_parse_failure(CaptureManifest::from_json_str(&json).unwrap_err());
    assert!(message.contains("invalid value"), "{message}");
}

#[test]
fn a_fractional_frame_count_is_rejected_by_representation() {
    let json = tweaked(&capture_example(), |value| {
        value["streams"][0]["frame_count"] = json!(1244.5);
    });
    assert!(CaptureManifest::from_json_str(&json).is_err());
}

#[test]
fn a_zero_frame_count_is_rejected_by_validation() {
    let json = tweaked(&capture_example(), |value| {
        value["streams"][0]["frame_count"] = json!(0);
    });
    let message = expect_invalid(CaptureManifest::from_json_str(&json).unwrap_err());
    assert!(message.contains("streams[0].frame_count"), "{message}");
}

#[test]
fn a_nonsense_frame_rate_is_rejected() {
    let json = tweaked(&capture_example(), |value| {
        value["streams"][0]["nominal_fps"] = json!(0.0);
    });
    let message = expect_invalid(CaptureManifest::from_json_str(&json).unwrap_err());
    assert!(message.contains("streams[0].nominal_fps"), "{message}");
}

#[test]
fn a_gap_outside_the_streams_own_span_fails() {
    let json = tweaked(&capture_example(), |value| {
        value["streams"][1]["gaps"][0]["start_timestamp_ns"] = json!(1i64);
        value["streams"][1]["gaps"][0]["end_timestamp_ns"] = json!(2i64);
    });
    let message = expect_invalid(CaptureManifest::from_json_str(&json).unwrap_err());
    assert!(message.contains("streams[1].gaps[0]"), "{message}");
    assert!(
        message.contains("outside the stream's own span"),
        "{message}"
    );
}

#[test]
fn a_backwards_gap_fails() {
    let json = tweaked(&capture_example(), |value| {
        let start = value["streams"][1]["gaps"][0]["start_timestamp_ns"].clone();
        let end = value["streams"][1]["gaps"][0]["end_timestamp_ns"].clone();
        value["streams"][1]["gaps"][0]["start_timestamp_ns"] = end;
        value["streams"][1]["gaps"][0]["end_timestamp_ns"] = start;
    });
    let message = expect_invalid(CaptureManifest::from_json_str(&json).unwrap_err());
    assert!(
        message.contains("streams[1].gaps[0].end_timestamp_ns"),
        "{message}"
    );
}

#[test]
fn gaps_that_exceed_the_dropped_frame_count_fail() {
    let json = tweaked(&capture_example(), |value| {
        value["streams"][1]["gaps"][0]["missing_frame_count"] = json!(9);
    });
    let message = expect_invalid(CaptureManifest::from_json_str(&json).unwrap_err());
    assert!(message.contains("streams[1].gaps"), "{message}");
    assert!(message.contains("dropped_frame_count is 1"), "{message}");
}

#[test]
fn an_event_range_in_the_wrong_order_fails() {
    let json = tweaked(&analysis_example(), |value| {
        value["events"][0]["range"] = json!({
            "start_timestamp_ns": 128_473_085_400_000i64,
            "end_timestamp_ns": 128_473_045_400_000i64,
        });
    });
    let message = expect_invalid(AnalysisResult::from_json_str(&json).unwrap_err());
    assert!(
        message.contains("events[0].range.end_timestamp_ns"),
        "{message}"
    );
}

#[test]
fn an_event_outside_its_own_range_fails() {
    let json = tweaked(&analysis_example(), |value| {
        value["events"][0]["timestamp_ns"] = json!(128_473_500_000_000i64);
    });
    let message = expect_invalid(AnalysisResult::from_json_str(&json).unwrap_err());
    assert!(message.contains("events[0].timestamp_ns"), "{message}");
    assert!(message.contains("outside its own range"), "{message}");
}

// --- confidence --------------------------------------------------------------

#[test]
fn confidence_above_one_fails() {
    let json = tweaked(&analysis_example(), |value| {
        value["events"][0]["confidence"] = json!(1.01);
    });
    let message = expect_parse_failure(AnalysisResult::from_json_str(&json).unwrap_err());
    assert!(message.contains("between 0.0 and 1.0"), "{message}");
}

#[test]
fn confidence_below_zero_fails() {
    let json = tweaked(&analysis_example(), |value| {
        value["events"][0]["confidence"] = json!(-0.001);
    });
    assert!(AnalysisResult::from_json_str(&json).is_err());
}

#[test]
fn a_non_finite_confidence_cannot_be_constructed() {
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(Confidence::new(bad).is_err(), "{bad} should be rejected");
    }
}

#[test]
fn the_endpoints_of_the_confidence_range_are_allowed() {
    assert_eq!(Confidence::new(0.0).unwrap(), Confidence::NONE);
    assert_eq!(Confidence::new(1.0).unwrap(), Confidence::CERTAIN);
}

// --- status consistency ------------------------------------------------------

#[test]
fn status_ok_alongside_reported_errors_fails() {
    let json = tweaked(&analysis_example(), |value| {
        value["status"] = json!("ok");
        value["errors"] = json!([{ "code": "pose_failed", "message": "no landmarks" }]);
    });
    let message = expect_invalid(AnalysisResult::from_json_str(&json).unwrap_err());
    assert!(message.contains("status"), "{message}");
    assert!(message.contains("\"partial\" or \"failed\""), "{message}");
}

#[test]
fn a_failed_analysis_with_no_events_is_valid() {
    let json = json!({
        "schema_version": "1.0",
        "shot_id": "2026-07-31T14-22-05Z-3f9a",
        "analyzer": { "name": "swingai.p_positions", "version": "0.3.0" },
        "created_at": "2026-07-31T14:22:19.883Z",
        "status": "failed",
        "events": [],
        "errors": [{ "code": "no_pose", "message": "MediaPipe found no person in either view" }]
    })
    .to_string();

    let result = AnalysisResult::from_json_str(&json).expect("a failure is still a valid document");
    assert_eq!(result.status, AnalysisStatus::Failed);
    assert!(result.events.is_empty());
    assert_eq!(result.errors.len(), 1);
}

// --- views and metadata ------------------------------------------------------

#[test]
fn the_python_view_abbreviation_is_rejected_rather_than_guessed() {
    let json = tweaked(&capture_example(), |value| {
        value["streams"][0]["view"] = json!("dtl");
    });
    let message = expect_parse_failure(CaptureManifest::from_json_str(&json).unwrap_err());
    assert!(message.contains("down_the_line"), "{message}");
}

#[test]
fn unknown_optional_metadata_does_not_break_parsing() {
    let json = tweaked(&capture_example(), |value| {
        value["streams"][0]["metadata"]["fox_specific_thing"] = json!({ "nested": [1, 2, 3] });
        value["streams"][0]["some_field_a_future_version_adds"] = json!("hello");
    });

    let manifest = CaptureManifest::from_json_str(&json).expect("unknown fields are tolerated");
    let metadata = manifest.streams[0].metadata.as_ref().unwrap();
    assert_eq!(
        metadata["fox_specific_thing"],
        json!({ "nested": [1, 2, 3] })
    );
}

#[test]
fn an_image_sequence_stream_is_accepted() {
    let json = tweaked(&capture_example(), |value| {
        value["streams"][0]["media"] = json!({
            "kind": "image_sequence",
            "path": "streams/down_the_line",
            "pattern": "frame_%06d.png",
        });
    });

    let manifest = CaptureManifest::from_json_str(&json).unwrap();
    match &manifest.streams[0].media {
        MediaSource::ImageSequence { path, pattern } => {
            assert_eq!(path.as_str(), "streams/down_the_line");
            assert_eq!(pattern, "frame_%06d.png");
        }
        other => panic!("expected an image sequence, got {other:?}"),
    }
}

// --- paths -------------------------------------------------------------------

#[test]
fn relative_artifact_paths_work_and_resolve_against_the_document() {
    let result = AnalysisResult::from_json_str(&analysis_example()).unwrap();
    let overlay = result
        .artifacts
        .iter()
        .find(|artifact| artifact.kind == "overlay_video")
        .expect("the example ships an overlay artifact");

    assert_eq!(overlay.path.as_str(), "artifacts/overlay.mp4");

    let shot_dir = Path::new("shots").join("2026-07-31T14-22-05Z-3f9a");
    assert_eq!(
        overlay.path.resolve_against(&shot_dir),
        shot_dir.join("artifacts/overlay.mp4"),
    );
}

#[test]
fn an_absolute_artifact_path_fails() {
    let json = tweaked(&analysis_example(), |value| {
        value["artifacts"][0]["path"] = json!("/var/lib/swingai/overlay.mp4");
    });
    let message = expect_parse_failure(AnalysisResult::from_json_str(&json).unwrap_err());
    assert!(message.contains("is absolute"), "{message}");
}

#[test]
fn a_windows_absolute_media_path_fails() {
    let json = tweaked(&capture_example(), |value| {
        value["streams"][0]["media"]["path"] = json!("D:\\captures\\clip.mkv");
    });
    let message = expect_parse_failure(CaptureManifest::from_json_str(&json).unwrap_err());
    assert!(message.contains("names a drive"), "{message}");
}

#[test]
fn a_media_path_escaping_the_shot_folder_fails() {
    let json = tweaked(&capture_example(), |value| {
        value["streams"][0]["media"]["path"] = json!("../../elsewhere/clip.mkv");
    });
    let message = expect_parse_failure(CaptureManifest::from_json_str(&json).unwrap_err());
    assert!(message.contains("escapes"), "{message}");
}

#[test]
fn a_relative_path_is_constructible_directly() {
    assert!(RelativePath::new("streams/face_on.mkv").is_ok());
    assert!(RelativePath::new("/streams/face_on.mkv").is_err());
}

// --- identifiers -------------------------------------------------------------

#[test]
fn duplicate_camera_ids_fail() {
    let json = tweaked(&capture_example(), |value| {
        value["streams"][1]["camera_id"] = value["streams"][0]["camera_id"].clone();
    });
    let message = expect_invalid(CaptureManifest::from_json_str(&json).unwrap_err());
    assert!(message.contains("streams[1].camera_id"), "{message}");
    assert!(message.contains("more than one stream"), "{message}");
}

#[test]
fn a_capture_with_no_streams_fails() {
    let json = tweaked(&capture_example(), |value| {
        value["streams"] = json!([]);
    });
    let message = expect_invalid(CaptureManifest::from_json_str(&json).unwrap_err());
    assert!(message.contains("at least one camera stream"), "{message}");
}

#[test]
fn a_shot_id_that_could_not_be_a_directory_name_fails() {
    let json = tweaked(&capture_example(), |value| {
        value["shot_id"] = json!("../escape");
    });
    assert!(CaptureManifest::from_json_str(&json).is_err());
}

// --- everything at once ------------------------------------------------------

#[test]
fn validation_reports_every_problem_at_once() {
    let json = tweaked(&capture_example(), |value| {
        value["streams"][0]["width"] = json!(0);
        value["streams"][0]["frame_count"] = json!(0);
        value["streams"][1]["nominal_fps"] = json!(-1.0);
        value["created_at"] = json!("");
    });

    let message = expect_invalid(CaptureManifest::from_json_str(&json).unwrap_err());
    for expected in [
        "created_at",
        "streams[0].width",
        "streams[0].frame_count",
        "streams[1].nominal_fps",
    ] {
        assert!(
            message.contains(expected),
            "{expected} missing from:\n{message}"
        );
    }
}
