//! Shared fixtures. Included by several test binaries, so not everything here is
//! used by every one of them.
#![allow(dead_code)]

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use swingai_capture::{
    CameraDescriptor, CameraId, CameraView, CaptureSession, CapturedFrame, FrameSource,
    PixelFormat, RingBufferConfig, SyntheticSource, SyntheticSourceConfig, Timestamp,
};

/// Small enough that a test can hold thousands of frames without noticing.
pub const WIDTH: u32 = 4;
pub const HEIGHT: u32 = 2;
pub const FRAME_BYTES: u64 = (WIDTH * HEIGHT) as u64;

/// 100fps — a round 10ms period keeps the arithmetic in the assertions readable.
pub const INTERVAL: Duration = Duration::from_millis(10);

pub fn camera(id: &str) -> CameraId {
    CameraId::new(id).expect("a valid camera id")
}

pub fn mono8() -> PixelFormat {
    PixelFormat::new(PixelFormat::MONO8).expect("mono8 is valid")
}

pub fn descriptor(id: &str, view: CameraView) -> CameraDescriptor {
    CameraDescriptor {
        camera_id: camera(id),
        view,
        width: WIDTH,
        height: HEIGHT,
        pixel_format: mono8(),
        nominal_fps: 100.0,
    }
}

pub fn ring(retention: Duration, max_payload_bytes: u64) -> RingBufferConfig {
    RingBufferConfig::new(retention, max_payload_bytes)
}

/// A frame with a payload keyed to its sequence number, so a test can tell which
/// frame it is holding by looking at the pixels.
pub fn frame(id: &str, sequence: u64, nanos: u64) -> CapturedFrame {
    CapturedFrame::new(
        camera(id),
        sequence,
        Timestamp::from_nanos(nanos),
        WIDTH,
        HEIGHT,
        mono8(),
        vec![sequence as u8; FRAME_BYTES as usize],
    )
}

/// A frame at `sequence * INTERVAL`, the common case.
pub fn paced_frame(id: &str, sequence: u64) -> CapturedFrame {
    frame(id, sequence, sequence * INTERVAL.as_nanos() as u64)
}

pub fn source_config(id: &str, view: CameraView, frame_count: u32) -> SyntheticSourceConfig {
    SyntheticSourceConfig::new(camera(id), view, WIDTH, HEIGHT, INTERVAL, frame_count)
}

/// Drain a source into a session, so a test reads as "this camera ran".
pub fn run_source(session: &mut CaptureSession, config: SyntheticSourceConfig) {
    let mut source = SyntheticSource::new(config).expect("a valid source configuration");
    while let Some(frame) = source.next_frame() {
        session
            .push(frame)
            .expect("the session accepts its own frames");
    }
}

/// Register a camera and run a source for it in one step.
pub fn add_and_run(
    session: &mut CaptureSession,
    config: SyntheticSourceConfig,
    ring_config: RingBufferConfig,
) {
    let source = SyntheticSource::new(config.clone()).expect("a valid source configuration");
    session
        .add_camera(source.descriptor().clone(), ring_config)
        .expect("a camera can be added once");
    run_source(session, config);
}

pub fn missing(sequences: impl IntoIterator<Item = u64>) -> BTreeSet<u64> {
    sequences.into_iter().collect()
}

/// A directory under the system temp dir that deletes itself.
///
/// No `tempfile` dependency for this: the tests need a unique path and a
/// guaranteed cleanup, which is a struct and a `Drop`. Uniqueness comes from the
/// process id, a per-test label and the wall clock, so parallel test binaries
/// cannot collide — and no shared counter is needed to arrange it.
#[derive(Debug)]
pub struct TempDir {
    path: PathBuf,
}

impl TempDir {
    pub fn new(label: &str) -> Self {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("the system clock is after 1970")
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "swingai-capture-{label}-{}-{nanos}",
            std::process::id()
        ));
        std::fs::create_dir_all(&path).expect("the temp directory is creatable");
        Self { path }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Everything directly inside, sorted, for asserting on a layout.
    pub fn entries(path: &Path) -> Vec<String> {
        let mut names: Vec<String> = std::fs::read_dir(path)
            .unwrap_or_else(|error| panic!("reading {}: {error}", path.display()))
            .map(|entry| {
                entry
                    .expect("a directory entry")
                    .file_name()
                    .to_string_lossy()
                    .into_owned()
            })
            .collect();
        names.sort();
        names
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}
