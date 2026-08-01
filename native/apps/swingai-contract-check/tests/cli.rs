//! Drives the built binary, because exit codes and which stream a message lands
//! on are the whole interface — a caller in a build script or a capture run only
//! sees those.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn repo_root() -> PathBuf {
    // apps/swingai-contract-check -> apps -> native -> repo root
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(3)
        .expect("crate is nested three levels under the repo root")
        .to_path_buf()
}

fn run(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_swingai-contract-check"))
        .args(args)
        .output()
        .expect("the checker should be runnable")
}

fn stdout(output: &Output) -> String {
    String::from_utf8_lossy(&output.stdout).into_owned()
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

fn example(name: &str) -> String {
    repo_root()
        .join("schemas/examples")
        .join(name)
        .to_string_lossy()
        .into_owned()
}

/// Write a broken document to a scratch file and hand back its path.
fn temp_json(name: &str, contents: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!("swingai-contract-check-{name}.json"));
    std::fs::write(&path, contents).expect("scratch file is writable");
    path
}

#[test]
fn a_valid_capture_manifest_succeeds() {
    let output = run(&["capture", &example("capture-manifest.example.json")]);
    assert!(output.status.success(), "{}", stderr(&output));

    let text = stdout(&output);
    assert!(text.contains("is a valid capture manifest"), "{text}");
    assert!(text.contains("fox-dtl [down_the_line]"), "{text}");
    assert!(text.contains("1 dropped"), "{text}");
}

#[test]
fn a_valid_analysis_result_succeeds() {
    let output = run(&["analysis", &example("analysis-result.example.json")]);
    assert!(output.status.success(), "{}", stderr(&output));

    let text = stdout(&output);
    assert!(text.contains("is a valid analysis result"), "{text}");
    assert!(text.contains("status partial"), "{text}");
    assert!(text.contains("warning [club_track_failed]"), "{text}");
}

#[test]
fn an_invalid_document_exits_one_and_says_what_is_wrong() {
    let path = temp_json(
        "invalid-capture",
        r#"{
            "schema_version": "1.0",
            "shot_id": "shot-1",
            "created_at": "2026-07-31T14:22:05.412Z",
            "streams": [{
                "camera_id": "cam-1",
                "view": "face_on",
                "media": { "kind": "video", "path": "clip.mkv" },
                "width": 0,
                "height": 1080,
                "pixel_format": "mono8",
                "frame_count": 10,
                "nominal_fps": 249.3,
                "first_timestamp_ns": 2000,
                "last_timestamp_ns": 1000,
                "dropped_frame_count": 0
            }]
        }"#,
    );

    let output = run(&["capture", path.to_str().unwrap()]);
    assert_eq!(output.status.code(), Some(1));
    assert!(stdout(&output).is_empty(), "errors belong on stderr");

    let text = stderr(&output);
    assert!(text.contains("2 problems"), "{text}");
    assert!(text.contains("streams[0].width"), "{text}");
    assert!(text.contains("streams[0].last_timestamp_ns"), "{text}");
}

#[test]
fn an_unsupported_schema_version_exits_one_with_an_actionable_message() {
    // A newer *minor* too — version matching is exact for now, and the CLI must
    // say so rather than implying a compatibility range that does not exist.
    for bad in ["9.0", "1.1", "1.4"] {
        let path = temp_json(
            &format!("version-{}", bad.replace('.', "-")),
            &format!(r#"{{ "schema_version": "{bad}" }}"#),
        );
        let output = run(&["analysis", path.to_str().unwrap()]);

        assert_eq!(output.status.code(), Some(1), "for {bad}");
        let text = stderr(&output);
        assert!(
            text.contains(&format!("unsupported schema_version {bad:?}")),
            "{bad}: {text}"
        );
        assert!(text.contains("1.0 only"), "{bad}: {text}");
        assert!(text.contains("update SwingAI"), "{bad}: {text}");
    }
}

#[test]
fn an_invalid_created_at_exits_one() {
    let path = temp_json(
        "bad-created-at",
        r#"{
            "schema_version": "1.0",
            "shot_id": "shot-1",
            "created_at": "yesterday",
            "streams": []
        }"#,
    );
    let output = run(&["capture", path.to_str().unwrap()]);

    assert_eq!(output.status.code(), Some(1));
    assert!(
        stderr(&output).contains("not an RFC 3339 date-time"),
        "{}",
        stderr(&output)
    );
}

#[test]
fn a_non_portable_path_exits_one() {
    let path = temp_json(
        "windows-path",
        r#"{
            "schema_version": "1.0",
            "shot_id": "shot-1",
            "created_at": "2026-07-31T14:22:05.412Z",
            "streams": [{
                "camera_id": "cam-1",
                "view": "face_on",
                "media": { "kind": "video", "path": "streams\\clip.mkv" },
                "width": 1440,
                "height": 1080,
                "pixel_format": "mono8",
                "frame_count": 10,
                "nominal_fps": 249.3,
                "first_timestamp_ns": 0,
                "last_timestamp_ns": 36101083,
                "dropped_frame_count": 0
            }]
        }"#,
    );
    let output = run(&["capture", path.to_str().unwrap()]);

    assert_eq!(output.status.code(), Some(1));
    assert!(stderr(&output).contains("backslash"), "{}", stderr(&output));
}

#[test]
fn a_negative_timestamp_exits_one() {
    let path = temp_json(
        "negative-timestamp",
        r#"{
            "schema_version": "1.0",
            "shot_id": "shot-1",
            "created_at": "2026-07-31T14:22:05.412Z",
            "streams": [{
                "camera_id": "cam-1",
                "view": "face_on",
                "media": { "kind": "video", "path": "streams/clip.mkv" },
                "width": 1440,
                "height": 1080,
                "pixel_format": "mono8",
                "frame_count": 10,
                "nominal_fps": 249.3,
                "first_timestamp_ns": -1,
                "last_timestamp_ns": 36101083,
                "dropped_frame_count": 0
            }]
        }"#,
    );
    let output = run(&["capture", path.to_str().unwrap()]);

    assert_eq!(output.status.code(), Some(1));
    assert!(
        stderr(&output).contains("invalid value"),
        "{}",
        stderr(&output)
    );
}

#[test]
fn a_missing_file_exits_one_and_names_it() {
    let output = run(&["capture", "definitely/not/here.json"]);
    assert_eq!(output.status.code(), Some(1));

    let text = stderr(&output);
    assert!(text.contains("definitely/not/here.json"), "{text}");
    assert!(text.contains("could not read the file"), "{text}");
}

#[test]
fn malformed_json_is_reported_as_a_parse_failure() {
    let path = temp_json("not-json", "{ this is not json");
    let output = run(&["capture", path.to_str().unwrap()]);

    assert_eq!(output.status.code(), Some(1));
    assert!(
        stderr(&output).contains("could not parse"),
        "{}",
        stderr(&output)
    );
}

#[test]
fn bad_usage_exits_two_and_prints_usage() {
    for args in [
        vec![],
        vec!["capture"],
        vec!["nonsense", "some.json"],
        vec!["capture", "a.json", "b.json"],
    ] {
        let output = run(&args);
        assert_eq!(output.status.code(), Some(2), "for args {args:?}");
        assert!(
            stderr(&output).contains("usage: swingai-contract-check"),
            "for args {args:?}"
        );
    }
}
