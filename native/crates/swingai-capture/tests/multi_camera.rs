//! The property the booth actually depends on: two cameras that agree about
//! nothing except the clock still yield a synchronized clip.

mod support;

use std::time::Duration;

use support::{add_and_run, descriptor, missing, paced_frame, ring, source_config};
use swingai_capture::{
    CameraView, CaptureError, CaptureSession, CapturedFrame, ShotExtraction, StreamClip, Timestamp,
};

const MS: u64 = 1_000_000;

fn generous() -> swingai_capture::RingBufferConfig {
    ring(Duration::from_secs(10), 1 << 24)
}

fn clip<'a>(extraction: &'a ShotExtraction, camera_id: &str) -> &'a StreamClip {
    extraction
        .streams()
        .iter()
        .find(|clip| clip.descriptor().camera_id.as_str() == camera_id)
        .unwrap_or_else(|| panic!("{camera_id} should be in the extraction"))
}

fn timestamps(clip: &StreamClip) -> Vec<u64> {
    clip.frames()
        .iter()
        .map(|frame| frame.timestamp().as_nanos())
        .collect()
}

/// Two cameras, the second started 500µs after the first, both at 100fps.
fn offset_session() -> CaptureSession {
    let mut session = CaptureSession::new();

    add_and_run(
        &mut session,
        source_config("dtl", CameraView::DownTheLine, 30),
        generous(),
    );

    let mut face_on = source_config("face-on", CameraView::FaceOn, 30);
    face_on.first_timestamp = Timestamp::from_nanos(500_000);
    add_and_run(&mut session, face_on, generous());

    session
}

#[test]
fn two_streams_with_different_starting_offsets_extract_by_time() {
    let session = offset_session();
    let extraction = session
        .trigger(Timestamp::from_nanos(100 * MS), Duration::from_millis(50))
        .expect("both cameras have data across the window");

    assert_eq!(extraction.requested_start(), Timestamp::from_nanos(50 * MS));

    // The down-the-line camera lands on 10ms boundaries, so both ends are on a
    // frame: 50ms through 100ms inclusive.
    assert_eq!(
        timestamps(clip(&extraction, "dtl")),
        [50 * MS, 60 * MS, 70 * MS, 80 * MS, 90 * MS, 100 * MS]
    );

    // The face-on camera is 500µs late, so its last in-window frame is at
    // 90.5ms — 100.5ms is past the trigger. Five frames, not six, and that is
    // the correct answer rather than a rounding decision.
    assert_eq!(
        timestamps(clip(&extraction, "face-on")),
        [
            50 * MS + 500_000,
            60 * MS + 500_000,
            70 * MS + 500_000,
            80 * MS + 500_000,
            90 * MS + 500_000,
        ]
    );

    for clip in extraction.streams() {
        assert!(clip.first_timestamp() >= extraction.requested_start());
        assert!(clip.last_timestamp() <= extraction.trigger_timestamp());
    }
}

#[test]
fn a_different_frame_count_on_one_camera_does_not_move_the_other() {
    let baseline = {
        let mut session = CaptureSession::new();
        add_and_run(
            &mut session,
            source_config("dtl", CameraView::DownTheLine, 30),
            generous(),
        );
        add_and_run(
            &mut session,
            source_config("face-on", CameraView::FaceOn, 30),
            generous(),
        );
        session
            .trigger(Timestamp::from_nanos(200 * MS), Duration::from_millis(50))
            .unwrap()
    };

    let with_a_longer_partner = {
        let mut session = CaptureSession::new();
        add_and_run(
            &mut session,
            source_config("dtl", CameraView::DownTheLine, 30),
            generous(),
        );
        add_and_run(
            &mut session,
            source_config("face-on", CameraView::FaceOn, 500),
            generous(),
        );
        session
            .trigger(Timestamp::from_nanos(200 * MS), Duration::from_millis(50))
            .unwrap()
    };

    assert_eq!(
        timestamps(clip(&baseline, "dtl")),
        timestamps(clip(&with_a_longer_partner, "dtl")),
        "one camera running longer must not change what the other returns"
    );
}

#[test]
fn a_gap_in_one_camera_does_not_shift_the_other() {
    let extract = |face_on_missing: Option<[u64; 2]>| {
        let mut session = CaptureSession::new();
        add_and_run(
            &mut session,
            source_config("dtl", CameraView::DownTheLine, 40),
            generous(),
        );

        let mut face_on = source_config("face-on", CameraView::FaceOn, 40);
        if let Some(sequences) = face_on_missing {
            face_on.missing_sequences = missing(sequences);
        }
        add_and_run(&mut session, face_on, generous());

        session
            .trigger(Timestamp::from_nanos(300 * MS), Duration::from_millis(100))
            .unwrap()
    };

    let clean = extract(None);
    let holed = extract(Some([22, 23]));

    assert_eq!(
        clip(&clean, "dtl").frames(),
        clip(&holed, "dtl").frames(),
        "the untouched camera must be byte-identical, frames and pixels"
    );

    let damaged = clip(&holed, "face-on");
    assert_eq!(damaged.dropped_frame_count(), 2);
    assert_eq!(damaged.gaps().len(), 1);
    assert_eq!(
        clip(&clean, "face-on").frames().len() - 2,
        damaged.frames().len(),
        "and the damaged one is short by exactly the frames it lost"
    );

    // The frames after the hole are still at the instants they always were.
    let after_the_hole: Vec<u64> = damaged
        .frames()
        .iter()
        .filter(|frame| frame.sequence() >= 24)
        .map(|frame| frame.timestamp().as_nanos())
        .collect();
    assert_eq!(after_the_hole.first(), Some(&(240 * MS)));
}

#[test]
fn a_trigger_before_any_frame_exists_is_an_actionable_error() {
    let mut session = CaptureSession::new();
    session
        .add_camera(descriptor("dtl", CameraView::DownTheLine), generous())
        .unwrap();

    let error = session
        .trigger(Timestamp::from_nanos(100 * MS), Duration::from_millis(50))
        .unwrap_err();

    assert!(matches!(error, CaptureError::NoFramesBuffered { .. }));
    let message = error.to_string();
    assert!(message.contains("dtl"), "{message}");
    assert!(
        message.contains("before capture produced anything"),
        "{message}"
    );
}

#[test]
fn a_trigger_before_this_cameras_first_frame_names_what_it_does_hold() {
    let mut session = CaptureSession::new();
    let mut config = source_config("dtl", CameraView::DownTheLine, 10);
    config.first_timestamp = Timestamp::from_nanos(1_000 * MS);
    add_and_run(&mut session, config, generous());

    let error = session
        .trigger(Timestamp::from_nanos(100 * MS), Duration::from_millis(50))
        .unwrap_err();

    assert!(matches!(error, CaptureError::WindowIsEmpty { .. }));
    let message = error.to_string();
    assert!(
        message.contains("no frames in the requested window"),
        "{message}"
    );
    assert!(message.contains("1000000000ns"), "{message}");
}

#[test]
fn an_incomplete_pre_roll_is_reported_rather_than_quietly_shortened() {
    let mut session = CaptureSession::new();

    // Starts at 200ms, so a 150ms pre-roll from a 250ms trigger reaches back
    // further than this camera has ever run.
    let mut late = source_config("late", CameraView::FaceOn, 30);
    late.first_timestamp = Timestamp::from_nanos(200 * MS);
    add_and_run(&mut session, late, generous());

    add_and_run(
        &mut session,
        source_config("early", CameraView::DownTheLine, 60),
        generous(),
    );

    let extraction = session
        .trigger(Timestamp::from_nanos(250 * MS), Duration::from_millis(150))
        .unwrap();

    assert!(!extraction.full_pre_roll_available());
    assert!(!clip(&extraction, "late").full_pre_roll_available());
    assert!(clip(&extraction, "early").full_pre_roll_available());
    assert_eq!(
        clip(&extraction, "late").buffered_from(),
        Timestamp::from_nanos(200 * MS),
        "and it says how far back it does reach"
    );

    // The frames it did have are still returned — a short clip, not no clip.
    assert!(!clip(&extraction, "late").frames().is_empty());
}

#[test]
fn retention_that_evicted_the_pre_roll_reports_the_same_way() {
    let mut session = CaptureSession::new();
    // 60ms of retention cannot answer a 150ms pre-roll, however long the camera
    // has been running.
    add_and_run(
        &mut session,
        source_config("dtl", CameraView::DownTheLine, 60),
        ring(Duration::from_millis(60), 1 << 24),
    );

    let extraction = session
        .trigger(Timestamp::from_nanos(590 * MS), Duration::from_millis(150))
        .unwrap();

    assert!(!extraction.full_pre_roll_available());
    assert_eq!(
        clip(&extraction, "dtl").buffered_from(),
        Timestamp::from_nanos(530 * MS)
    );
}

#[test]
fn a_duplicate_camera_id_is_rejected() {
    let mut session = CaptureSession::new();
    session
        .add_camera(descriptor("dtl", CameraView::DownTheLine), generous())
        .unwrap();

    let error = session
        .add_camera(descriptor("dtl", CameraView::FaceOn), generous())
        .unwrap_err();

    assert!(matches!(error, CaptureError::DuplicateCamera { .. }));
    assert_eq!(session.camera_count(), 1, "and nothing was replaced");
}

#[test]
fn a_frame_can_only_reach_its_own_camera() {
    let mut session = CaptureSession::new();
    session
        .add_camera(descriptor("dtl", CameraView::DownTheLine), generous())
        .unwrap();

    let error = session.push(paced_frame("face-on", 0)).unwrap_err();
    assert!(matches!(error, CaptureError::UnknownCamera { .. }));
}

#[test]
fn more_than_two_cameras_are_representable() {
    let views = [
        ("dtl", CameraView::DownTheLine),
        ("face-on", CameraView::FaceOn),
        ("overhead", CameraView::Overhead),
        ("impact", CameraView::Impact),
    ];

    let mut session = CaptureSession::new();
    for (index, (id, view)) in views.iter().enumerate() {
        let mut config = source_config(id, *view, 40);
        // Each camera started at a slightly different moment, as four
        // independently started cameras would.
        config.first_timestamp = Timestamp::from_nanos(index as u64 * 250_000);
        add_and_run(&mut session, config, generous());
    }
    assert_eq!(session.camera_count(), 4);

    let extraction = session
        .trigger(Timestamp::from_nanos(300 * MS), Duration::from_millis(100))
        .unwrap();

    assert_eq!(extraction.streams().len(), 4);
    for clip in extraction.streams() {
        assert!(!clip.frames().is_empty());
        assert!(clip.first_timestamp() >= extraction.requested_start());
        assert!(clip.last_timestamp() <= extraction.trigger_timestamp());
    }
}

#[test]
fn push_order_across_cameras_does_not_change_the_result() {
    // Buffers are per-camera, so interleaving cannot matter — asserted rather
    // than assumed, because a shared buffer would break exactly here.
    let sequential = offset_session()
        .trigger(Timestamp::from_nanos(100 * MS), Duration::from_millis(50))
        .unwrap();

    let interleaved = {
        let mut session = CaptureSession::new();
        session
            .add_camera(descriptor("dtl", CameraView::DownTheLine), generous())
            .unwrap();
        session
            .add_camera(descriptor("face-on", CameraView::FaceOn), generous())
            .unwrap();

        for sequence in 0..30u64 {
            session.push(paced_frame("dtl", sequence)).unwrap();
            session
                .push(CapturedFrame::new(
                    swingai_capture::CameraId::new("face-on").unwrap(),
                    sequence,
                    Timestamp::from_nanos(sequence * 10 * MS + 500_000),
                    support::WIDTH,
                    support::HEIGHT,
                    support::mono8(),
                    vec![sequence as u8; support::FRAME_BYTES as usize],
                ))
                .unwrap();
        }

        session
            .trigger(Timestamp::from_nanos(100 * MS), Duration::from_millis(50))
            .unwrap()
    };

    for camera_id in ["dtl", "face-on"] {
        assert_eq!(
            timestamps(clip(&sequential, camera_id)),
            timestamps(clip(&interleaved, camera_id)),
            "{camera_id} should not care what order the cameras were drained in"
        );
    }
}
