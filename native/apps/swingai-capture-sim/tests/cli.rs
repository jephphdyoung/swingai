//! Drives the built binary. The simulator's output is a directory and an exit
//! code, so that is what gets checked — and the manifest it produces is reparsed
//! with the contract types rather than taken on the report's word.

use std::path::{Path, PathBuf};
use std::process::Output;
use std::time::{SystemTime, UNIX_EPOCH};

use swingai_contracts::{CameraView, CaptureManifest, MediaSource, Timestamp};

/// A self-deleting output directory. Same reasoning as the capture crate's: a
/// unique path and a guaranteed cleanup is a struct and a `Drop`.
#[derive(Debug)]
struct TempDir {
    path: PathBuf,
}

impl TempDir {
    fn new(label: &str) -> Self {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("the system clock is after 1970")
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "swingai-capture-sim-{label}-{}-{nanos}",
            std::process::id()
        ));
        std::fs::create_dir_all(&path).expect("creatable");
        Self { path }
    }

    fn path(&self) -> &Path {
        &self.path
    }

    /// The one shot directory the simulator wrote.
    fn only_shot(&self) -> PathBuf {
        let mut entries: Vec<PathBuf> = std::fs::read_dir(&self.path)
            .expect("readable")
            .map(|entry| entry.expect("an entry").path())
            .collect();
        assert_eq!(entries.len(), 1, "expected exactly one shot in {entries:?}");
        entries.pop().expect("checked above")
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

fn run(args: &[&str]) -> Output {
    std::process::Command::new(env!("CARGO_BIN_EXE_swingai-capture-sim"))
        .args(args)
        .output()
        .expect("the simulator should be runnable")
}

fn stdout(output: &Output) -> String {
    String::from_utf8_lossy(&output.stdout).into_owned()
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

fn manifest(shot: &Path) -> CaptureManifest {
    let json = std::fs::read_to_string(shot.join("capture-manifest.json"))
        .expect("the manifest is where the layout says");
    CaptureManifest::from_json_str(&json).expect("the contract accepts the generated manifest")
}

#[test]
fn a_default_run_writes_a_shot_the_contract_accepts() {
    let temp = TempDir::new("default");
    let output = run(&["--output", temp.path().to_str().unwrap()]);
    assert!(output.status.success(), "{}", stderr(&output));

    let shot = temp.only_shot();
    let manifest = manifest(&shot);

    assert_eq!(manifest.streams.len(), 2);
    assert_eq!(
        manifest.trigger_timestamp_ns,
        Some(Timestamp::from_nanos(5_000_000_000)),
        "the default trigger is 5000ms on the session clock"
    );

    let dtl = manifest
        .stream_for_view(CameraView::DownTheLine)
        .expect("a down-the-line stream");
    let face_on = manifest
        .stream_for_view(CameraView::FaceOn)
        .expect("a face-on stream");

    assert_eq!(dtl.camera_id.as_str(), "sim-dtl");
    assert_eq!(face_on.camera_id.as_str(), "sim-face-on");
    assert_eq!(dtl.frames.dropped_frame_count, 0);
    assert_eq!(
        face_on.frames.dropped_frame_count, 2,
        "the scenario plans one two-frame drop, and it must land in the window"
    );
    assert_eq!(face_on.gaps.len(), 1);
    assert!(face_on.gaps[0].after_frame_index.is_some());

    // Two cameras started at different instants return different counts for the
    // same window; that is the point of the scenario.
    assert_ne!(dtl.frames.frame_count, face_on.frames.frame_count);
    assert_ne!(
        dtl.frames.first_timestamp_ns,
        face_on.frames.first_timestamp_ns
    );
}

#[test]
fn the_frames_the_manifest_promises_are_on_disk() {
    let temp = TempDir::new("frames");
    let output = run(&[
        "--output",
        temp.path().to_str().unwrap(),
        "--trigger-ms",
        "800",
        "--pre-roll-ms",
        "200",
    ]);
    assert!(output.status.success(), "{}", stderr(&output));

    let shot = temp.only_shot();
    for stream in &manifest(&shot).streams {
        let MediaSource::ImageSequence { path, pattern } = &stream.media else {
            panic!("the simulator writes image sequences");
        };
        assert_eq!(pattern, "frame_%06d.pgm");

        let directory = path.resolve_against(&shot);
        let stored = std::fs::read_dir(&directory).expect("readable").count();
        assert_eq!(
            stored,
            stream.frames.frame_count as usize,
            "{} should hold exactly what the manifest claims",
            directory.display()
        );

        let first = directory.join("frame_000000.pgm");
        let bytes = std::fs::read(&first).expect("the first frame is readable");
        let header = format!("P5\n{} {}\n255\n", stream.width, stream.height);
        assert!(bytes.starts_with(header.as_bytes()), "{first:?}");
        assert_eq!(
            bytes.len(),
            header.len() + (stream.width * stream.height) as usize
        );
    }
}

#[test]
fn the_report_says_what_was_captured() {
    let temp = TempDir::new("report");
    let output = run(&["--output", temp.path().to_str().unwrap()]);
    let text = stdout(&output);

    for expected in [
        "shot directory",
        "manifest check   valid",
        "trigger          5000000000ns",
        "full pre-roll    yes",
        "sim-dtl [down_the_line]",
        "sim-face-on [face_on]",
        "streams/face_on/",
        "gap after stored frame",
        "(wall clock, for filing only)",
    ] {
        assert!(text.contains(expected), "missing {expected:?} in:\n{text}");
    }
}

#[test]
fn a_pre_roll_longer_than_retention_is_reported_as_incomplete() {
    let temp = TempDir::new("short-retention");
    let output = run(&[
        "--output",
        temp.path().to_str().unwrap(),
        "--retention-ms",
        "100",
    ]);
    assert!(output.status.success(), "{}", stderr(&output));

    let text = stdout(&output);
    assert!(
        text.contains("full pre-roll    no"),
        "a short buffer must say so rather than quietly returning less:\n{text}"
    );
    // And it still writes the shot, with whatever it did have.
    assert!(manifest(&temp.only_shot()).streams[0].frames.frame_count > 0);
}

#[test]
fn the_same_scenario_produces_the_same_pixels_every_run() {
    let temp = TempDir::new("deterministic");
    for shot_id in ["run-a", "run-b"] {
        let output = run(&[
            "--output",
            temp.path().to_str().unwrap(),
            "--shot-id",
            shot_id,
            "--trigger-ms",
            "600",
            "--pre-roll-ms",
            "200",
        ]);
        assert!(output.status.success(), "{}", stderr(&output));
    }

    let frame = |shot: &str| {
        std::fs::read(
            temp.path()
                .join(shot)
                .join("streams/down_the_line/frame_000007.pgm"),
        )
        .expect("the frame exists in both runs")
    };
    assert_eq!(
        frame("run-a"),
        frame("run-b"),
        "only the wall clock differs between runs"
    );
}

#[test]
fn an_existing_shot_id_is_refused_rather_than_overwritten() {
    let temp = TempDir::new("collision");
    let args = [
        "--output",
        temp.path().to_str().unwrap(),
        "--shot-id",
        "same-shot",
    ];

    assert!(run(&args).status.success());
    let second = run(&args);

    assert_eq!(second.status.code(), Some(1));
    assert!(
        stderr(&second).contains("already exists"),
        "{}",
        stderr(&second)
    );
}

#[test]
fn bad_arguments_exit_two_and_print_usage() {
    for args in [
        vec![],
        vec!["--output"],
        vec!["--nonsense", "x"],
        vec!["--output", "/tmp", "--trigger-ms", "soon"],
        vec!["--output", "/tmp", "--pre-roll-ms", "0"],
        vec!["--output", "/tmp", "--shot-id", "bad/id"],
    ] {
        let output = run(&args);
        assert_eq!(output.status.code(), Some(2), "for {args:?}");
        assert!(
            stderr(&output).contains("usage: swingai-capture-sim"),
            "for {args:?}: {}",
            stderr(&output)
        );
    }
}

#[test]
fn help_exits_zero() {
    let output = run(&["--help"]);
    assert!(output.status.success());
    assert!(stdout(&output).contains("--pre-roll-ms"));
}
