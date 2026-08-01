//! The ring buffer is where a capture goes wrong quietly if it is going to.
//! These tests are mostly about what it *refuses*.

mod support;

use std::sync::Arc;
use std::time::Duration;

use support::{FRAME_BYTES, HEIGHT, WIDTH, camera, descriptor, frame, mono8, paced_frame, ring};
use swingai_capture::{
    CameraView, CaptureError, CapturedFrame, FrameRingBuffer, PixelFormat, PreRollWindow, Timestamp,
};

const MS: u64 = 1_000_000;

fn buffer(retention_ms: u64, max_payload_bytes: u64) -> FrameRingBuffer {
    FrameRingBuffer::new(
        descriptor("cam", CameraView::FaceOn),
        ring(Duration::from_millis(retention_ms), max_payload_bytes),
    )
    .expect("a valid buffer configuration")
}

/// Fill with `count` frames at 10ms spacing, sequence == index.
fn filled(buffer: &mut FrameRingBuffer, count: u64) {
    for sequence in 0..count {
        buffer
            .push(paced_frame("cam", sequence))
            .expect("in-order frames are accepted");
    }
}

#[test]
fn frames_come_back_in_timestamp_order() {
    let mut buffer = buffer(1_000, 1 << 20);
    filled(&mut buffer, 10);

    let clip = buffer
        .extract(PreRollWindow::between(
            Timestamp::ZERO,
            Timestamp::from_nanos(90 * MS),
        ))
        .expect("the whole span");

    let timestamps: Vec<u64> = clip
        .frames()
        .iter()
        .map(|frame| frame.timestamp().as_nanos())
        .collect();
    let mut sorted = timestamps.clone();
    sorted.sort_unstable();
    assert_eq!(timestamps, sorted);
    assert_eq!(timestamps.len(), 10);
}

#[test]
fn frames_older_than_the_retention_window_are_evicted() {
    // 45ms of retention at 10ms spacing keeps the newest five.
    let mut buffer = buffer(45, 1 << 20);
    filled(&mut buffer, 20);

    assert_eq!(buffer.len(), 5);
    let (oldest, newest) = buffer.buffered_span().expect("frames are retained");
    assert_eq!(oldest, Timestamp::from_nanos(150 * MS));
    assert_eq!(newest, Timestamp::from_nanos(190 * MS));
}

#[test]
fn a_frame_exactly_at_the_retention_edge_survives() {
    let mut buffer = buffer(50, 1 << 20);
    filled(&mut buffer, 6);

    // Newest is 50ms; retention is 50ms; the frame at 0ms is exactly at the edge.
    assert_eq!(buffer.len(), 6);
    assert_eq!(buffer.buffered_span().unwrap().0, Timestamp::ZERO);
}

#[test]
fn the_byte_limit_evicts_even_when_retention_would_keep_the_frames() {
    // Retention would hold everything; the cap holds three frames.
    let mut buffer = buffer(10_000, FRAME_BYTES * 3);
    filled(&mut buffer, 50);

    assert_eq!(buffer.len(), 3);
    assert_eq!(buffer.payload_bytes(), FRAME_BYTES * 3);
    assert_eq!(
        buffer.buffered_span().unwrap().1,
        Timestamp::from_nanos(490 * MS)
    );
}

#[test]
fn a_byte_limit_too_small_for_one_frame_is_an_error_rather_than_a_silent_no_op() {
    let mut buffer = buffer(1_000, FRAME_BYTES - 1);
    let error = buffer.push(paced_frame("cam", 0)).unwrap_err();

    assert!(matches!(error, CaptureError::ByteLimitTooSmall { .. }));
    assert!(buffer.is_empty(), "nothing was silently retained either");
    let message = error.to_string();
    assert!(message.contains("cannot hold even one"), "{message}");
    assert!(
        message.contains("max_payload_bytes"),
        "the message must name the knob to turn: {message}"
    );
}

#[test]
fn repeated_eviction_leaves_ordering_and_accounting_intact() {
    // Far more pushes than the buffer can hold, so the deque wraps repeatedly.
    let mut buffer = buffer(35, FRAME_BYTES * 8);
    filled(&mut buffer, 1_000);

    let clip = buffer
        .extract(PreRollWindow::between(
            Timestamp::ZERO,
            Timestamp::from_nanos(u64::MAX),
        ))
        .expect("something is retained");

    let sequences: Vec<u64> = clip.frames().iter().map(CapturedFrame::sequence).collect();
    assert_eq!(sequences, [996, 997, 998, 999]);
    assert_eq!(
        buffer.payload_bytes(),
        buffer.len() as u64 * FRAME_BYTES,
        "byte accounting must survive a thousand evictions"
    );

    // And the pixels still identify the frames they came from.
    for frame in clip.frames() {
        assert_eq!(frame.payload()[0], frame.sequence() as u8);
    }
}

#[test]
fn both_boundary_frames_are_included() {
    let mut buffer = buffer(1_000, 1 << 20);
    filled(&mut buffer, 10);

    let clip = buffer
        .extract(PreRollWindow::between(
            Timestamp::from_nanos(30 * MS),
            Timestamp::from_nanos(60 * MS),
        ))
        .expect("a middle slice");

    let sequences: Vec<u64> = clip.frames().iter().map(CapturedFrame::sequence).collect();
    assert_eq!(
        sequences,
        [3, 4, 5, 6],
        "the range is inclusive at both ends"
    );
    assert_eq!(clip.first_timestamp(), Timestamp::from_nanos(30 * MS));
    assert_eq!(clip.last_timestamp(), Timestamp::from_nanos(60 * MS));
}

#[test]
fn a_window_between_two_frames_is_empty_rather_than_rounded_to_the_nearest() {
    let mut buffer = buffer(1_000, 1 << 20);
    filled(&mut buffer, 10);

    // Strictly between frame 3 (30ms) and frame 4 (40ms).
    assert!(
        buffer
            .extract(PreRollWindow::between(
                Timestamp::from_nanos(31 * MS),
                Timestamp::from_nanos(39 * MS)
            ))
            .is_none()
    );
}

#[test]
fn a_timestamp_moving_backward_is_rejected() {
    let mut buffer = buffer(1_000, 1 << 20);
    buffer.push(frame("cam", 0, 50 * MS)).unwrap();

    let error = buffer.push(frame("cam", 1, 40 * MS)).unwrap_err();
    assert!(matches!(error, CaptureError::TimestampWentBackward { .. }));
    assert!(
        error
            .to_string()
            .contains("not converting its device clock"),
        "{error}"
    );
    assert_eq!(buffer.len(), 1);
}

#[test]
fn a_duplicate_timestamp_is_rejected() {
    let mut buffer = buffer(1_000, 1 << 20);
    buffer.push(frame("cam", 0, 50 * MS)).unwrap();

    let error = buffer.push(frame("cam", 1, 50 * MS)).unwrap_err();
    assert!(matches!(error, CaptureError::DuplicateTimestamp { .. }));
    assert_eq!(buffer.len(), 1);
}

#[test]
fn a_repeated_or_decreasing_sequence_number_is_rejected() {
    for offered in [3u64, 4] {
        let mut buffer = buffer(1_000, 1 << 20);
        buffer.push(frame("cam", 4, 10 * MS)).unwrap();

        let error = buffer.push(frame("cam", offered, 20 * MS)).unwrap_err();
        assert!(
            matches!(error, CaptureError::SequenceWentBackward { .. }),
            "sequence {offered} after 4 should be rejected: {error}"
        );
        assert_eq!(buffer.len(), 1);
    }
}

#[test]
fn a_resolution_change_mid_stream_is_rejected() {
    let mut buffer = buffer(1_000, 1 << 20);
    let wrong_size = CapturedFrame::new(
        camera("cam"),
        0,
        Timestamp::ZERO,
        WIDTH + 1,
        HEIGHT,
        mono8(),
        vec![0u8; (WIDTH as usize + 1) * HEIGHT as usize],
    );

    let error = buffer.push(wrong_size).unwrap_err();
    assert!(matches!(error, CaptureError::DimensionsChanged { .. }));
    assert!(error.to_string().contains("a new stream"), "{error}");
}

#[test]
fn a_pixel_format_change_mid_stream_is_rejected() {
    let mut buffer = buffer(1_000, 1 << 20);
    let wrong_format = CapturedFrame::new(
        camera("cam"),
        0,
        Timestamp::ZERO,
        WIDTH,
        HEIGHT,
        PixelFormat::new(PixelFormat::MONO16).unwrap(),
        vec![0u8; FRAME_BYTES as usize],
    );

    let error = buffer.push(wrong_format).unwrap_err();
    assert!(matches!(error, CaptureError::PixelFormatChanged { .. }));
}

#[test]
fn a_sequence_gap_is_detected_and_bounded_by_the_frames_around_it() {
    let mut buffer = buffer(1_000, 1 << 20);
    buffer.push(frame("cam", 0, 0)).unwrap();
    buffer.push(frame("cam", 1, 10 * MS)).unwrap();
    // Sequences 2 and 3 never arrive.
    buffer.push(frame("cam", 4, 40 * MS)).unwrap();
    buffer.push(frame("cam", 5, 50 * MS)).unwrap();

    let clip = buffer
        .extract(PreRollWindow::between(
            Timestamp::ZERO,
            Timestamp::from_nanos(50 * MS),
        ))
        .expect("the whole span");

    assert_eq!(clip.gaps().len(), 1);
    let gap = clip.gaps()[0];
    assert_eq!(gap.missing_frame_count, 2);
    assert_eq!(gap.start_timestamp, Timestamp::from_nanos(10 * MS));
    assert_eq!(gap.end_timestamp, Timestamp::from_nanos(40 * MS));
    assert_eq!(
        gap.after_frame_index, 1,
        "the index is into the clip's stored frames, not the source's numbering"
    );
    assert_eq!(clip.dropped_frame_count(), 2);
    assert_eq!(buffer.dropped_frame_total(), 2);
}

#[test]
fn a_gap_outside_the_extracted_window_is_not_reported() {
    let mut buffer = buffer(1_000, 1 << 20);
    buffer.push(frame("cam", 0, 0)).unwrap();
    buffer.push(frame("cam", 3, 30 * MS)).unwrap(); // gap at the front
    buffer.push(frame("cam", 4, 40 * MS)).unwrap();
    buffer.push(frame("cam", 5, 50 * MS)).unwrap();

    let clip = buffer
        .extract(PreRollWindow::between(
            Timestamp::from_nanos(40 * MS),
            Timestamp::from_nanos(50 * MS),
        ))
        .expect("the tail");

    assert!(
        clip.gaps().is_empty(),
        "the hole is before this clip starts"
    );
    assert_eq!(clip.dropped_frame_count(), 0);
    assert_eq!(
        buffer.dropped_frame_total(),
        2,
        "but the stream's lifetime total still remembers it"
    );
}

#[test]
fn gap_indexes_are_relative_to_the_clip_not_the_buffer() {
    let mut buffer = buffer(1_000, 1 << 20);
    for sequence in 0..5u64 {
        buffer.push(paced_frame("cam", sequence)).unwrap();
    }
    buffer.push(frame("cam", 7, 70 * MS)).unwrap(); // two missing after seq 4

    let whole = buffer
        .extract(PreRollWindow::between(
            Timestamp::ZERO,
            Timestamp::from_nanos(70 * MS),
        ))
        .unwrap();
    assert_eq!(whole.gaps()[0].after_frame_index, 4);

    let tail = buffer
        .extract(PreRollWindow::between(
            Timestamp::from_nanos(30 * MS),
            Timestamp::from_nanos(70 * MS),
        ))
        .unwrap();
    assert_eq!(
        tail.gaps()[0].after_frame_index,
        1,
        "the same hole, counted from the start of a shorter clip"
    );
}

#[test]
fn extraction_shares_payloads_instead_of_copying_them() {
    let mut buffer = buffer(1_000, 1 << 20);
    filled(&mut buffer, 4);

    let first = buffer
        .extract(PreRollWindow::between(
            Timestamp::ZERO,
            Timestamp::from_nanos(30 * MS),
        ))
        .unwrap();
    let second = buffer
        .extract(PreRollWindow::between(
            Timestamp::ZERO,
            Timestamp::from_nanos(30 * MS),
        ))
        .unwrap();

    for (a, b) in first.frames().iter().zip(second.frames()) {
        assert!(
            Arc::ptr_eq(a.payload_handle(), b.payload_handle()),
            "two extractions of the same frame must point at one allocation"
        );
    }
    assert!(
        Arc::strong_count(first.frames()[0].payload_handle()) >= 3,
        "the buffer and both clips hold the same pixels"
    );
}

#[test]
fn eviction_after_extraction_leaves_the_clip_intact() {
    // The clip owns its frames through the Arc, so retention moving on must not
    // invalidate a clip already handed out.
    let mut buffer = buffer(35, 1 << 20);
    filled(&mut buffer, 4);
    let clip = buffer
        .extract(PreRollWindow::between(
            Timestamp::ZERO,
            Timestamp::from_nanos(30 * MS),
        ))
        .unwrap();

    filled_from(&mut buffer, 4, 100);
    assert_eq!(clip.frames().len(), 4);
    assert_eq!(clip.frames()[0].payload()[0], 0);
}

fn filled_from(buffer: &mut FrameRingBuffer, start: u64, count: u64) {
    for sequence in start..start + count {
        buffer.push(paced_frame("cam", sequence)).unwrap();
    }
}
