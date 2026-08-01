//! The synthetic source is the substitute for a camera in every other test, so
//! "the same configuration produces the same capture" has to be true before any
//! of them mean anything.

mod support;

use std::time::Duration;

use support::{INTERVAL, missing, source_config};
use swingai_capture::{CameraView, CapturedFrame, FrameSource, SyntheticSource, Timestamp};

fn drain(config: swingai_capture::SyntheticSourceConfig) -> Vec<CapturedFrame> {
    let mut source = SyntheticSource::new(config).expect("a valid configuration");
    let mut frames = Vec::new();
    while let Some(frame) = source.next_frame() {
        frames.push(frame);
    }
    frames
}

#[test]
fn the_same_configuration_yields_identical_frames() {
    let config = source_config("cam", CameraView::FaceOn, 50);
    let first = drain(config.clone());
    let second = drain(config);

    assert_eq!(first.len(), 50);
    assert_eq!(
        first, second,
        "identical configurations must produce identical frames, pixels included"
    );
}

#[test]
fn the_starting_timestamp_and_interval_are_honored() {
    let mut config = source_config("cam", CameraView::FaceOn, 4);
    config.first_timestamp = Timestamp::from_nanos(1_000_000);

    let timestamps: Vec<u64> = drain(config)
        .iter()
        .map(|frame| frame.timestamp().as_nanos())
        .collect();

    let interval = INTERVAL.as_nanos() as u64;
    assert_eq!(
        timestamps,
        [
            1_000_000,
            1_000_000 + interval,
            1_000_000 + 2 * interval,
            1_000_000 + 3 * interval,
        ]
    );
}

#[test]
fn the_starting_sequence_number_is_honored_and_stays_stream_local() {
    let mut config = source_config("cam", CameraView::DownTheLine, 3);
    config.first_sequence = 5_000;

    let sequences: Vec<u64> = drain(config).iter().map(CapturedFrame::sequence).collect();
    assert_eq!(sequences, [5_000, 5_001, 5_002]);
}

#[test]
fn planned_gaps_are_emitted_as_missing_sequence_numbers() {
    let mut config = source_config("cam", CameraView::FaceOn, 10);
    config.missing_sequences = missing([3, 4, 7]);

    let frames = drain(config.clone());
    let sequences: Vec<u64> = frames.iter().map(CapturedFrame::sequence).collect();

    assert_eq!(sequences, [0, 1, 2, 5, 6, 8, 9]);
    assert_eq!(config.delivered_frame_count(), 7);
}

#[test]
fn a_dropped_frame_costs_its_slot_rather_than_shifting_the_ones_after_it() {
    // The exposure still happened; only the image is missing. So frame 5 lands
    // where frame 5 always would have, not where frame 3 would have.
    let mut config = source_config("cam", CameraView::FaceOn, 10);
    config.missing_sequences = missing([3, 4]);

    let interval = INTERVAL.as_nanos() as u64;
    let frames = drain(config);
    let after_gap = frames
        .iter()
        .find(|frame| frame.sequence() == 5)
        .expect("frame 5 is delivered");

    assert_eq!(after_gap.timestamp(), Timestamp::from_nanos(5 * interval));
}

#[test]
fn the_end_of_the_source_is_deterministic() {
    let mut source =
        SyntheticSource::new(source_config("cam", CameraView::FaceOn, 3)).expect("valid");

    for _ in 0..3 {
        assert!(source.next_frame().is_some());
    }
    for _ in 0..5 {
        assert!(
            source.next_frame().is_none(),
            "an exhausted source stays exhausted"
        );
    }
}

#[test]
fn a_trailing_planned_gap_still_ends_the_source_cleanly() {
    let mut config = source_config("cam", CameraView::FaceOn, 5);
    config.missing_sequences = missing([3, 4]);

    let frames = drain(config);
    assert_eq!(frames.len(), 3, "the last two slots deliver nothing");
    assert_eq!(frames.last().unwrap().sequence(), 2);
}

#[test]
fn frames_carry_the_configured_shape_and_a_full_payload() {
    let frames = drain(source_config("cam", CameraView::Overhead, 2));
    for frame in &frames {
        assert_eq!(frame.width(), support::WIDTH);
        assert_eq!(frame.height(), support::HEIGHT);
        assert_eq!(frame.payload_len(), support::FRAME_BYTES);
        assert_eq!(frame.pixel_format().as_str(), "mono8");
    }
    assert_ne!(
        frames[0].payload(),
        frames[1].payload(),
        "consecutive frames must be distinguishable by their pixels"
    );
}

#[test]
fn the_descriptor_reports_the_configured_rate_not_the_derived_one() {
    let mut config = source_config("cam", CameraView::FaceOn, 2);
    config.frame_interval = Duration::from_nanos(4_166_666);
    config.nominal_fps = 240.0;

    let source = SyntheticSource::new(config).expect("valid");
    assert!((source.descriptor().nominal_fps - 240.0).abs() < f64::EPSILON);
}
