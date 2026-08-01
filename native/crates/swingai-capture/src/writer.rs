use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use swingai_contracts::{
    CameraStreamManifest, CaptureManifest, CaptureManifestVersion, MediaSource, RelativePath,
    SequenceGap,
};
use swingai_core::{PixelFormat, Rfc3339Timestamp, ShotId};

use crate::{CapturedFrame, ShotExtraction, StreamClip, WriteError};

/// The printf-style pattern recorded in the manifest for every stream.
pub const FRAME_FILENAME_PATTERN: &str = "frame_%06d.pgm";

/// Directory holding the per-stream frame directories, relative to the manifest.
const STREAMS_DIRECTORY: &str = "streams";

/// The file a reader looks for first.
const MANIFEST_FILENAME: &str = "capture-manifest.json";

/// What [`write_shot`] produced.
#[derive(Debug, Clone)]
pub struct WrittenShot {
    /// The shot directory, now complete.
    pub directory: PathBuf,
    /// The manifest inside it.
    pub manifest_path: PathBuf,
    /// The manifest as written, already validated and re-parsed.
    pub manifest: CaptureManifest,
    /// Each stream's directory name under `streams/`, in manifest order.
    pub stream_directories: Vec<String>,
}

/// Write one extracted shot as a self-contained directory.
///
/// ```text
/// <output_root>/<shot_id>/
/// ├── capture-manifest.json
/// └── streams/
///     ├── down_the_line/frame_000000.pgm ...
///     └── face_on/frame_000000.pgm ...
/// ```
///
/// # Nothing half-written is left looking finished
///
/// Everything goes into a sibling `.<shot_id>.partial` directory first, and that
/// is renamed into place only after every frame is on disk and the manifest has
/// passed the contract's own validation and a full reparse. A rename within one
/// directory is the closest thing to atomic that a filesystem offers, so a
/// reader either finds a whole shot or finds nothing. If any step fails the
/// staging directory is removed, so a failed capture cannot be mistaken for a
/// short one.
///
/// An existing shot directory is never overwritten: it is the record of
/// something that happened once, and a silent replacement would destroy a
/// capture to save a retry.
pub fn write_shot(
    output_root: &Path,
    shot_id: &ShotId,
    created_at: &Rfc3339Timestamp,
    extraction: &ShotExtraction,
) -> Result<WrittenShot, WriteError> {
    let destination = output_root.join(shot_id.as_str());
    if destination.exists() {
        return Err(WriteError::DestinationExists(destination));
    }

    let staging = output_root.join(format!(".{shot_id}.partial"));
    // Left over from an earlier failure of this same shot; it is ours to clear.
    if staging.exists() {
        remove_tree(&staging)?;
    }
    create_dir(&staging)?;

    match stage_shot(&staging, shot_id, created_at, extraction) {
        Ok((manifest, stream_directories)) => {
            rename(&staging, &destination)?;
            Ok(WrittenShot {
                manifest_path: destination.join(MANIFEST_FILENAME),
                directory: destination,
                manifest,
                stream_directories,
            })
        }
        Err(error) => {
            // Best effort: the original failure is what the caller needs to see,
            // and a leftover `.partial` directory is inert — it is not a shot.
            let _ = fs::remove_dir_all(&staging);
            Err(error)
        }
    }
}

/// Everything that happens inside the staging directory, so the caller can clean
/// up on any failure with one `match`.
fn stage_shot(
    staging: &Path,
    shot_id: &ShotId,
    created_at: &Rfc3339Timestamp,
    extraction: &ShotExtraction,
) -> Result<(CaptureManifest, Vec<String>), WriteError> {
    let directories = stream_directory_names(extraction.streams());
    let streams_root = staging.join(STREAMS_DIRECTORY);
    create_dir(&streams_root)?;

    let mut streams = Vec::with_capacity(extraction.streams().len());
    for (clip, directory) in extraction.streams().iter().zip(&directories) {
        let stream_root = streams_root.join(directory);
        create_dir(&stream_root)?;
        write_frames(&stream_root, clip)?;
        streams.push(stream_manifest(clip, directory)?);
    }

    let manifest = CaptureManifest {
        schema_version: CaptureManifestVersion::CURRENT,
        shot_id: shot_id.clone(),
        created_at: created_at.clone(),
        trigger_timestamp_ns: Some(extraction.trigger_timestamp()),
        streams,
    };

    // Validate before writing, and reparse what was serialized: this writer's
    // whole job is to produce a document the other side can read, so proving it
    // round-trips is part of writing it rather than a test-only nicety.
    manifest.validate().into_result()?;
    let json = manifest
        .to_json_string_pretty()
        .map_err(|error| WriteError::Manifest(error.into()))?;
    let reparsed = CaptureManifest::from_json_str(&json)?;

    write_bytes(&staging.join(MANIFEST_FILENAME), json.as_bytes())?;

    Ok((reparsed, directories))
}

/// One directory per stream, named for the view.
///
/// The contract permits two cameras to share a view — a second face-on angle is
/// a plausible booth — so a view that appears more than once is qualified with
/// the camera id rather than silently colliding. Camera ids are already
/// constrained to be directory-safe by [`CameraId`], so nothing is escaped here.
fn stream_directory_names(streams: &[StreamClip]) -> Vec<String> {
    let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
    for clip in streams {
        *counts.entry(clip.descriptor().view.as_str()).or_default() += 1;
    }

    streams
        .iter()
        .map(|clip| {
            let view = clip.descriptor().view.as_str();
            if counts.get(view).copied().unwrap_or_default() > 1 {
                format!("{view}-{}", clip.descriptor().camera_id)
            } else {
                view.to_owned()
            }
        })
        .collect()
}

/// Frames are numbered contiguously from zero *within the clip*.
///
/// Not by source sequence number: the clip starts wherever the trigger's window
/// started, and a dropped frame would leave a hole in the numbering that a
/// reader would have to guess about. The source's sequence numbers stay in
/// memory, where they do their one job — the gaps in the manifest — and never
/// reach a filename.
fn write_frames(stream_root: &Path, clip: &StreamClip) -> Result<(), WriteError> {
    for (index, frame) in clip.frames().iter().enumerate() {
        let path = stream_root.join(format!("frame_{index:06}.pgm"));
        write_pgm(&path, frame)?;
    }
    Ok(())
}

/// Binary PGM (`P5`): a three-line ASCII header, then one byte per pixel.
///
/// No image crate, because there is nothing to encode — the payload is already
/// 8-bit grayscale, and the header is fifteen bytes of text. Adding a dependency
/// to write it would be paying for a decision (which codec) that ADR 0001
/// deliberately defers.
fn write_pgm(path: &Path, frame: &CapturedFrame) -> Result<(), WriteError> {
    if frame.pixel_format().as_str() != PixelFormat::MONO8 {
        return Err(WriteError::UnsupportedPixelFormat {
            camera_id: frame.camera_id().clone(),
            pixel_format: frame.pixel_format().clone(),
        });
    }

    let expected = frame.width() as usize * frame.height() as usize;
    if frame.payload().len() != expected {
        return Err(WriteError::PayloadSize {
            camera_id: frame.camera_id().clone(),
            expected,
            actual: frame.payload().len(),
        });
    }

    let file = File::create(path).map_err(|source| WriteError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let mut writer = BufWriter::new(file);

    let write = |writer: &mut BufWriter<File>, frame: &CapturedFrame| -> std::io::Result<()> {
        write!(writer, "P5\n{} {}\n255\n", frame.width(), frame.height())?;
        writer.write_all(frame.payload())?;
        writer.flush()
    };

    write(&mut writer, frame).map_err(|source| WriteError::Io {
        path: path.to_path_buf(),
        source,
    })
}

fn stream_manifest(clip: &StreamClip, directory: &str) -> Result<CameraStreamManifest, WriteError> {
    let descriptor = clip.descriptor();

    Ok(CameraStreamManifest {
        camera_id: descriptor.camera_id.clone(),
        // A synthetic source has no hardware behind it. A real one fills this in.
        serial_number: None,
        view: descriptor.view,
        media: MediaSource::ImageSequence {
            path: RelativePath::new(format!("{STREAMS_DIRECTORY}/{directory}"))?,
            pattern: FRAME_FILENAME_PATTERN.to_owned(),
        },
        width: descriptor.width,
        height: descriptor.height,
        pixel_format: descriptor.pixel_format.clone(),
        // Describes the extracted clip, never the source's whole lifetime.
        frames: clip.frame_sequence(),
        gaps: clip
            .gaps()
            .iter()
            .map(|gap| SequenceGap {
                start_timestamp_ns: gap.start_timestamp,
                end_timestamp_ns: gap.end_timestamp,
                missing_frame_count: gap.missing_frame_count,
                // Local to the stored frames, not the source's numbering.
                after_frame_index: Some(gap.after_frame_index),
            })
            .collect(),
        metadata: None,
    })
}

fn create_dir(path: &Path) -> Result<(), WriteError> {
    fs::create_dir_all(path).map_err(|source| WriteError::Io {
        path: path.to_path_buf(),
        source,
    })
}

fn remove_tree(path: &Path) -> Result<(), WriteError> {
    fs::remove_dir_all(path).map_err(|source| WriteError::Io {
        path: path.to_path_buf(),
        source,
    })
}

fn rename(from: &Path, to: &Path) -> Result<(), WriteError> {
    fs::rename(from, to).map_err(|source| WriteError::Io {
        path: to.to_path_buf(),
        source,
    })
}

fn write_bytes(path: &Path, bytes: &[u8]) -> Result<(), WriteError> {
    fs::write(path, bytes).map_err(|source| WriteError::Io {
        path: path.to_path_buf(),
        source,
    })
}

#[cfg(test)]
mod tests {
    use swingai_core::{CameraId, CameraView, Timestamp};

    use super::*;
    use crate::{CameraDescriptor, ClipGap};

    fn clip(id: &str, view: CameraView) -> StreamClip {
        let descriptor = CameraDescriptor {
            camera_id: CameraId::new(id).unwrap(),
            view,
            width: 2,
            height: 2,
            pixel_format: PixelFormat::new(PixelFormat::MONO8).unwrap(),
            nominal_fps: 100.0,
        };
        let frames = vec![CapturedFrame::new(
            CameraId::new(id).unwrap(),
            0,
            Timestamp::ZERO,
            2,
            2,
            PixelFormat::new(PixelFormat::MONO8).unwrap(),
            vec![0u8; 4],
        )];
        StreamClip::new(
            descriptor,
            frames,
            Vec::new(),
            Timestamp::ZERO,
            Timestamp::ZERO,
        )
    }

    #[test]
    fn distinct_views_get_their_own_directory_names() {
        let streams = [
            clip("a", CameraView::DownTheLine),
            clip("b", CameraView::FaceOn),
        ];
        assert_eq!(
            stream_directory_names(&streams),
            ["down_the_line", "face_on"]
        );
    }

    #[test]
    fn a_shared_view_is_qualified_with_the_camera_id() {
        let streams = [
            clip("cam-1", CameraView::FaceOn),
            clip("cam-2", CameraView::FaceOn),
            clip("cam-3", CameraView::DownTheLine),
        ];
        assert_eq!(
            stream_directory_names(&streams),
            ["face_on-cam-1", "face_on-cam-2", "down_the_line"]
        );
    }

    #[test]
    fn a_gap_becomes_a_contract_gap_with_its_local_index() {
        let descriptor = CameraDescriptor {
            camera_id: CameraId::new("cam").unwrap(),
            view: CameraView::FaceOn,
            width: 2,
            height: 2,
            pixel_format: PixelFormat::new(PixelFormat::MONO8).unwrap(),
            nominal_fps: 100.0,
        };
        let frames = (0..3)
            .map(|i| {
                CapturedFrame::new(
                    CameraId::new("cam").unwrap(),
                    i,
                    Timestamp::from_nanos(i * 10_000_000),
                    2,
                    2,
                    PixelFormat::new(PixelFormat::MONO8).unwrap(),
                    vec![0u8; 4],
                )
            })
            .collect();
        let gaps = vec![ClipGap {
            start_timestamp: Timestamp::from_nanos(10_000_000),
            end_timestamp: Timestamp::from_nanos(20_000_000),
            missing_frame_count: 2,
            after_frame_index: 1,
        }];
        let clip = StreamClip::new(descriptor, frames, gaps, Timestamp::ZERO, Timestamp::ZERO);

        let stream = stream_manifest(&clip, "face_on").unwrap();
        assert_eq!(stream.gaps.len(), 1);
        assert_eq!(stream.gaps[0].after_frame_index, Some(1));
        assert_eq!(stream.frames.dropped_frame_count, 2);
        assert_eq!(stream.frames.frame_count, 3);
        assert_eq!(
            stream.media,
            MediaSource::ImageSequence {
                path: RelativePath::new("streams/face_on").unwrap(),
                pattern: FRAME_FILENAME_PATTERN.to_owned(),
            }
        );
    }

    #[test]
    fn the_clip_and_not_the_source_lifetime_is_what_the_manifest_describes() {
        let clip = clip("cam", CameraView::FaceOn);
        let sequence = clip.frame_sequence();
        assert_eq!(sequence.frame_count, 1);
        assert_eq!(sequence.first_timestamp_ns, Timestamp::ZERO);
        assert_eq!(sequence.last_timestamp_ns, Timestamp::ZERO);
        assert!(sequence.validate().is_empty());
        // One frame has no interval to measure; the nominal rate still stands.
        assert_eq!(sequence.measured_fps(), None);
        assert!((sequence.nominal_fps - 100.0).abs() < f64::EPSILON);
    }
}
