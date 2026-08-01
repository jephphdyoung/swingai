//! The writer is the only thing in this crate that touches the filesystem and
//! the only thing that produces a document another process will read, so both
//! halves get checked: the layout on disk, and the manifest against the contract
//! that owns it.

mod support;

use std::path::Path;
use std::time::Duration;

use support::{TempDir, add_and_run, missing, ring, source_config};
use swingai_capture::{
    CameraId, CameraView, CaptureSession, CapturedFrame, PixelFormat, Rfc3339Timestamp,
    ShotExtraction, ShotId, Timestamp, WriteError, now_utc, shot_id_for, write_shot,
};
use swingai_contracts::{CaptureManifest, MediaSource};

const MS: u64 = 1_000_000;

fn created_at() -> Rfc3339Timestamp {
    Rfc3339Timestamp::new("2026-07-31T14:22:05.412Z").expect("a valid RFC 3339 instant")
}

fn shot_id() -> ShotId {
    ShotId::new("2026-07-31T14-22-05.412Z").expect("a valid shot id")
}

fn generous() -> swingai_capture::RingBufferConfig {
    ring(Duration::from_secs(10), 1 << 24)
}

/// Down-the-line clean, face-on with a two-frame hole in the middle of the
/// window — the scenario the simulator runs, in miniature.
fn extraction() -> ShotExtraction {
    let mut session = CaptureSession::new();
    add_and_run(
        &mut session,
        source_config("dtl", CameraView::DownTheLine, 40),
        generous(),
    );

    let mut face_on = source_config("face-on", CameraView::FaceOn, 40);
    face_on.first_timestamp = Timestamp::from_nanos(500_000);
    face_on.first_sequence = 5_000;
    face_on.missing_sequences = missing([5_022, 5_023]);
    add_and_run(&mut session, face_on, generous());

    session
        .trigger(Timestamp::from_nanos(300 * MS), Duration::from_millis(100))
        .expect("both cameras cover the window")
}

fn pgm_header(path: &Path) -> (String, usize) {
    let bytes = std::fs::read(path).expect("the frame is readable");
    // Three ASCII lines, then the payload.
    let header_end = bytes
        .iter()
        .enumerate()
        .filter(|(_, byte)| **byte == b'\n')
        .nth(2)
        .map(|(index, _)| index + 1)
        .expect("a PGM header has three newlines");
    (
        String::from_utf8(bytes[..header_end].to_vec()).expect("the header is ASCII"),
        bytes.len() - header_end,
    )
}

#[test]
fn the_expected_directory_layout_is_written() {
    let temp = TempDir::new("layout");
    let written = write_shot(temp.path(), &shot_id(), &created_at(), &extraction())
        .expect("the shot is writable");

    assert_eq!(written.directory, temp.path().join(shot_id().as_str()));
    assert_eq!(
        TempDir::entries(&written.directory),
        ["capture-manifest.json", "streams"]
    );
    assert_eq!(
        TempDir::entries(&written.directory.join("streams")),
        ["down_the_line", "face_on"]
    );
    assert_eq!(
        written.stream_directories,
        ["down_the_line", "face_on"],
        "reported in manifest order"
    );

    // Contiguous from zero, whatever the source's own numbering was.
    let face_on = TempDir::entries(&written.directory.join("streams/face_on"));
    assert_eq!(face_on.first().unwrap(), "frame_000000.pgm");
    assert_eq!(
        face_on.len(),
        written.manifest.streams[1].frames.frame_count as usize
    );
    assert!(
        !face_on.iter().any(|name| name.contains("5022")),
        "source sequence numbers must not reach a filename"
    );
}

#[test]
fn frames_are_valid_pgm_with_the_payload_their_header_promises() {
    let temp = TempDir::new("pgm");
    let written =
        write_shot(temp.path(), &shot_id(), &created_at(), &extraction()).expect("writable");

    let (header, payload_len) = pgm_header(
        &written
            .directory
            .join("streams/down_the_line/frame_000000.pgm"),
    );

    assert_eq!(
        header,
        format!("P5\n{} {}\n255\n", support::WIDTH, support::HEIGHT)
    );
    assert_eq!(payload_len, support::FRAME_BYTES as usize);
}

#[test]
fn every_frame_in_a_stream_is_written_not_just_the_first() {
    let temp = TempDir::new("all-frames");
    let written =
        write_shot(temp.path(), &shot_id(), &created_at(), &extraction()).expect("writable");

    let stream = &written.manifest.streams[0];
    for index in 0..stream.frames.frame_count {
        let path = written
            .directory
            .join(format!("streams/down_the_line/frame_{index:06}.pgm"));
        let (_, payload_len) = pgm_header(&path);
        assert_eq!(payload_len, support::FRAME_BYTES as usize, "{path:?}");
    }
}

#[test]
fn manifest_paths_are_forward_slash_whatever_the_host() {
    let temp = TempDir::new("paths");
    let written =
        write_shot(temp.path(), &shot_id(), &created_at(), &extraction()).expect("writable");

    let json = std::fs::read_to_string(&written.manifest_path).expect("readable");
    assert!(!json.contains('\\'), "no backslash may reach the document");
    assert!(json.contains("\"streams/down_the_line\""), "{json}");

    for stream in &written.manifest.streams {
        let MediaSource::ImageSequence { path, pattern } = &stream.media else {
            panic!("the writer stores image sequences");
        };
        assert!(path.as_str().starts_with("streams/"));
        assert_eq!(pattern, "frame_%06d.pgm");
        assert_eq!(stream.pixel_format.as_str(), PixelFormat::MONO8);
    }
}

#[test]
fn manifest_counts_and_timestamps_describe_the_stored_frames() {
    let temp = TempDir::new("counts");
    let extraction = extraction();
    let written =
        write_shot(temp.path(), &shot_id(), &created_at(), &extraction).expect("writable");

    for (clip, stream) in extraction.streams().iter().zip(&written.manifest.streams) {
        let directory = written.directory.join("streams").join(stream.view.as_str());
        let stored = TempDir::entries(&directory).len();

        assert_eq!(stream.frames.frame_count as usize, stored);
        assert_eq!(stream.frames.frame_count as usize, clip.frames().len());
        assert_eq!(stream.frames.first_timestamp_ns, clip.first_timestamp());
        assert_eq!(stream.frames.last_timestamp_ns, clip.last_timestamp());
        assert_eq!(
            stream.frames.dropped_frame_count,
            clip.dropped_frame_count()
        );
    }

    // The clip, not the source's lifetime: both cameras ran for 400ms and the
    // window asked for 100ms of it.
    assert_eq!(
        written.manifest.trigger_timestamp_ns,
        Some(Timestamp::from_nanos(300 * MS))
    );
    let dtl = &written.manifest.streams[0].frames;
    assert_eq!(dtl.first_timestamp_ns, Timestamp::from_nanos(200 * MS));
    assert_eq!(dtl.last_timestamp_ns, Timestamp::from_nanos(300 * MS));
    assert_eq!(dtl.frame_count, 11);
}

#[test]
fn a_gap_is_recorded_against_the_stored_frame_index() {
    let temp = TempDir::new("gaps");
    let written =
        write_shot(temp.path(), &shot_id(), &created_at(), &extraction()).expect("writable");

    let face_on = &written.manifest.streams[1];
    assert_eq!(face_on.frames.dropped_frame_count, 2);
    assert_eq!(face_on.gaps.len(), 1);

    let gap = &face_on.gaps[0];
    let index = gap.after_frame_index.expect("the writer always sets it");
    assert!(
        (index as usize) < face_on.frames.frame_count as usize,
        "the index must address a frame that was actually stored"
    );
    assert!(
        written
            .directory
            .join(format!("streams/face_on/frame_{index:06}.pgm"))
            .exists(),
        "and that frame must be on disk"
    );
    assert_eq!(gap.missing_frame_count, 2);
    assert!(gap.end_timestamp_ns > gap.start_timestamp_ns);
}

#[test]
fn the_generated_manifest_round_trips_through_the_contract() {
    let temp = TempDir::new("round-trip");
    let written =
        write_shot(temp.path(), &shot_id(), &created_at(), &extraction()).expect("writable");

    let json = std::fs::read_to_string(&written.manifest_path).expect("readable");
    let reparsed = CaptureManifest::from_json_str(&json).expect("the contract accepts it");

    assert_eq!(reparsed, written.manifest);
    assert!(reparsed.validate().is_empty());
    assert_eq!(reparsed.shot_id, shot_id());
    assert_eq!(reparsed.created_at, created_at());
    assert!(
        reparsed.stream_for_view(CameraView::DownTheLine).is_some(),
        "and it is queryable the way a reader would query it"
    );
}

#[test]
fn an_existing_shot_directory_is_never_overwritten() {
    let temp = TempDir::new("existing");
    let destination = temp.path().join(shot_id().as_str());
    std::fs::create_dir_all(&destination).expect("creatable");
    std::fs::write(destination.join("evidence.txt"), b"an earlier capture").expect("writable");

    let error =
        write_shot(temp.path(), &shot_id(), &created_at(), &extraction()).expect_err("refused");

    assert!(matches!(error, WriteError::DestinationExists(_)));
    assert_eq!(
        std::fs::read_to_string(destination.join("evidence.txt")).unwrap(),
        "an earlier capture",
        "the earlier capture must survive untouched"
    );
}

#[test]
fn a_failed_write_leaves_nothing_that_looks_like_a_shot() {
    let temp = TempDir::new("failure");

    // A frame whose payload does not match its dimensions. The ring buffer
    // checks dimensions, not payload length, so this reaches the writer — and
    // the writer must refuse it rather than emit a truncated PGM.
    let mut session = CaptureSession::new();
    session
        .add_camera(
            support::descriptor("dtl", CameraView::DownTheLine),
            generous(),
        )
        .unwrap();
    for sequence in 0..3u64 {
        session
            .push(CapturedFrame::new(
                CameraId::new("dtl").unwrap(),
                sequence,
                Timestamp::from_nanos(sequence * 10 * MS),
                support::WIDTH,
                support::HEIGHT,
                support::mono8(),
                // One byte short of what 4x2 implies.
                vec![0u8; support::FRAME_BYTES as usize - 1],
            ))
            .unwrap();
    }
    let extraction = session
        .trigger(Timestamp::from_nanos(20 * MS), Duration::from_millis(20))
        .unwrap();

    let error = write_shot(temp.path(), &shot_id(), &created_at(), &extraction)
        .expect_err("a short payload must not be written");
    assert!(matches!(error, WriteError::PayloadSize { .. }));

    assert!(
        !temp.path().join(shot_id().as_str()).exists(),
        "no shot directory may appear for a capture that failed"
    );
    assert_eq!(
        TempDir::entries(temp.path()),
        Vec::<String>::new(),
        "and no staging directory may be left behind either"
    );
}

#[test]
fn a_non_mono8_stream_is_refused_before_anything_is_renamed_into_place() {
    let temp = TempDir::new("format");

    let mut descriptor = support::descriptor("dtl", CameraView::DownTheLine);
    descriptor.pixel_format = PixelFormat::new(PixelFormat::MONO16).unwrap();

    let mut session = CaptureSession::new();
    session.add_camera(descriptor, generous()).unwrap();
    for sequence in 0..3u64 {
        session
            .push(CapturedFrame::new(
                CameraId::new("dtl").unwrap(),
                sequence,
                Timestamp::from_nanos(sequence * 10 * MS),
                support::WIDTH,
                support::HEIGHT,
                PixelFormat::new(PixelFormat::MONO16).unwrap(),
                vec![0u8; support::FRAME_BYTES as usize * 2],
            ))
            .unwrap();
    }
    let extraction = session
        .trigger(Timestamp::from_nanos(20 * MS), Duration::from_millis(20))
        .unwrap();

    let error = write_shot(temp.path(), &shot_id(), &created_at(), &extraction)
        .expect_err("the PGM writer only speaks mono8");
    assert!(matches!(error, WriteError::UnsupportedPixelFormat { .. }));
    assert_eq!(TempDir::entries(temp.path()), Vec::<String>::new());
}

#[test]
fn a_stale_staging_directory_does_not_block_a_later_capture() {
    let temp = TempDir::new("stale");
    let staging = temp.path().join(format!(".{}.partial", shot_id()));
    std::fs::create_dir_all(staging.join("streams")).expect("creatable");
    std::fs::write(staging.join("junk"), b"from a crash").expect("writable");

    let written = write_shot(temp.path(), &shot_id(), &created_at(), &extraction())
        .expect("the leftover is ours to clear");

    assert!(!staging.exists(), "the staging directory is consumed");
    assert!(!written.directory.join("junk").exists());
    assert_eq!(
        TempDir::entries(temp.path()),
        [shot_id().as_str().to_owned()]
    );
}

#[test]
fn two_cameras_sharing_a_view_get_distinct_directories() {
    let temp = TempDir::new("shared-view");

    let mut session = CaptureSession::new();
    for id in ["face-a", "face-b"] {
        add_and_run(
            &mut session,
            source_config(id, CameraView::FaceOn, 20),
            generous(),
        );
    }
    let extraction = session
        .trigger(Timestamp::from_nanos(150 * MS), Duration::from_millis(50))
        .unwrap();

    let written =
        write_shot(temp.path(), &shot_id(), &created_at(), &extraction).expect("writable");

    assert_eq!(
        written.stream_directories,
        ["face_on-face-a", "face_on-face-b"],
        "a shared view must not collide into one directory"
    );
    assert_eq!(
        TempDir::entries(&written.directory.join("streams")),
        ["face_on-face-a", "face_on-face-b"]
    );
    assert!(
        CaptureManifest::from_json_str(&std::fs::read_to_string(&written.manifest_path).unwrap())
            .is_ok()
    );
}

#[test]
fn a_wall_clock_shot_id_names_a_directory_that_can_actually_be_created() {
    let temp = TempDir::new("generated-id");
    let created_at = now_utc();
    let shot_id = shot_id_for(&created_at).expect("derivable");

    let written = write_shot(temp.path(), &shot_id, &created_at, &extraction()).expect("writable");

    assert_eq!(written.directory.file_name().unwrap(), shot_id.as_str());
    assert_eq!(written.manifest.created_at, created_at);
}
