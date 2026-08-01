use std::fmt;
use std::path::PathBuf;

use swingai_contracts::ContractError;
use swingai_core::{CameraId, PixelFormat, Timestamp, ValidationError, ValidationErrors};

/// Why a frame could not be buffered, or a trigger could not be answered.
///
/// Every variant names the camera, because a two-camera booth's first question
/// about any capture fault is "which one".
#[derive(Debug)]
pub enum CaptureError {
    /// A camera id was configured twice in one session.
    DuplicateCamera { camera_id: CameraId },
    /// A frame arrived for a camera this session was never told about.
    UnknownCamera { camera_id: CameraId },
    /// A frame reached the wrong buffer.
    CameraMismatch {
        expected: CameraId,
        actual: CameraId,
    },
    /// Time went backwards within one stream, which means the source is not
    /// emitting session-clock timestamps — most likely a raw device stamp got
    /// through unconverted.
    TimestampWentBackward {
        camera_id: CameraId,
        previous: Timestamp,
        offered: Timestamp,
    },
    /// Two frames from one camera claim the same instant. A stream where this is
    /// legal has no usable timeline, and the manifest rejects it too.
    DuplicateTimestamp {
        camera_id: CameraId,
        timestamp: Timestamp,
    },
    /// A sequence number repeated or went backwards, so gap detection can no
    /// longer be trusted for this stream.
    SequenceWentBackward {
        camera_id: CameraId,
        previous: u64,
        offered: u64,
    },
    /// Frame dimensions changed mid-stream. A buffer holds one camera in one
    /// configuration; a resolution change is a new stream.
    DimensionsChanged {
        camera_id: CameraId,
        expected: (u32, u32),
        actual: (u32, u32),
    },
    /// Pixel format changed mid-stream.
    PixelFormatChanged {
        camera_id: CameraId,
        expected: PixelFormat,
        actual: PixelFormat,
    },
    /// The byte cap cannot hold even one frame, so the buffer would evict
    /// everything it was given and answer every trigger with nothing.
    ByteLimitTooSmall {
        camera_id: CameraId,
        frame_bytes: u64,
        max_payload_bytes: u64,
    },
    /// A trigger arrived before the session had any cameras.
    NoCameras,
    /// A trigger arrived before this camera had produced anything.
    NoFramesBuffered { camera_id: CameraId },
    /// The camera has frames, but none inside the requested window — typically a
    /// trigger before the camera started, or a pre-roll entirely behind what
    /// retention still holds.
    WindowIsEmpty {
        camera_id: CameraId,
        start: Timestamp,
        end: Timestamp,
        buffered_from: Timestamp,
        buffered_to: Timestamp,
    },
    /// A camera or ring-buffer configuration was not usable.
    InvalidConfig(ValidationErrors),
}

impl fmt::Display for CaptureError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateCamera { camera_id } => write!(
                f,
                "camera {camera_id} is already configured in this session; camera ids must be \
                 unique so a frame can be routed to exactly one buffer"
            ),
            Self::UnknownCamera { camera_id } => write!(
                f,
                "no buffer is configured for camera {camera_id}; add it to the session before \
                 pushing its frames"
            ),
            Self::CameraMismatch { expected, actual } => write!(
                f,
                "this buffer holds camera {expected}, but the frame came from {actual}"
            ),
            Self::TimestampWentBackward {
                camera_id,
                previous,
                offered,
            } => write!(
                f,
                "camera {camera_id} offered a frame at {offered} after one at {previous}, \
                 {}ms earlier; capture-session timestamps only move forward, so the source is \
                 not converting its device clock",
                previous.millis_since(*offered)
            ),
            Self::DuplicateTimestamp {
                camera_id,
                timestamp,
            } => write!(
                f,
                "camera {camera_id} offered two frames at {timestamp}; more than one frame \
                 cannot share an instant"
            ),
            Self::SequenceWentBackward {
                camera_id,
                previous,
                offered,
            } => write!(
                f,
                "camera {camera_id} offered sequence {offered} after {previous}; sequence \
                 numbers must increase, or dropped frames cannot be detected"
            ),
            Self::DimensionsChanged {
                camera_id,
                expected,
                actual,
            } => write!(
                f,
                "camera {camera_id} is configured for {}x{} but sent a {}x{} frame; a \
                 resolution change is a new stream, not a new frame",
                expected.0, expected.1, actual.0, actual.1
            ),
            Self::PixelFormatChanged {
                camera_id,
                expected,
                actual,
            } => write!(
                f,
                "camera {camera_id} is configured for {expected} but sent a {actual} frame"
            ),
            Self::ByteLimitTooSmall {
                camera_id,
                frame_bytes,
                max_payload_bytes,
            } => write!(
                f,
                "camera {camera_id}'s byte limit is {max_payload_bytes} bytes, which cannot \
                 hold even one {frame_bytes}-byte frame; raise max_payload_bytes above one \
                 frame or the buffer retains nothing"
            ),
            Self::NoCameras => f.write_str(
                "the session has no cameras configured, so a trigger has nothing to extract",
            ),
            Self::NoFramesBuffered { camera_id } => write!(
                f,
                "camera {camera_id} has buffered no frames; the trigger fired before capture \
                 produced anything"
            ),
            Self::WindowIsEmpty {
                camera_id,
                start,
                end,
                buffered_from,
                buffered_to,
            } => write!(
                f,
                "camera {camera_id} has no frames in the requested window [{start}, {end}]; it \
                 holds {buffered_from} to {buffered_to}"
            ),
            Self::InvalidConfig(errors) => {
                let count = errors.len();
                let noun = if count == 1 { "problem" } else { "problems" };
                write!(f, "unusable configuration, {count} {noun}:\n{errors}")
            }
        }
    }
}

impl std::error::Error for CaptureError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidConfig(errors) => Some(errors),
            _ => None,
        }
    }
}

impl From<ValidationErrors> for CaptureError {
    fn from(errors: ValidationErrors) -> Self {
        Self::InvalidConfig(errors)
    }
}

/// Why a shot directory could not be written.
#[derive(Debug)]
pub enum WriteError {
    /// The shot directory already exists. Never overwritten: a shot directory is
    /// the record of something that happened once.
    DestinationExists(PathBuf),
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    /// A frame's payload is not the size its dimensions imply, so a PGM built
    /// from it would be silently truncated or padded.
    PayloadSize {
        camera_id: CameraId,
        expected: usize,
        actual: usize,
    },
    /// The writer stores 8-bit grayscale PGM, so it can only accept `mono8`.
    UnsupportedPixelFormat {
        camera_id: CameraId,
        pixel_format: PixelFormat,
    },
    /// A generated path is not a legal contract path.
    Path(ValidationError),
    /// The generated manifest failed the contract's own checks. A bug in this
    /// writer, surfaced before anything is renamed into place.
    Manifest(ContractError),
}

impl fmt::Display for WriteError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DestinationExists(path) => write!(
                f,
                "{} already exists; a shot directory is never overwritten",
                path.display()
            ),
            Self::Io { path, source } => write!(f, "{}: {source}", path.display()),
            Self::PayloadSize {
                camera_id,
                expected,
                actual,
            } => write!(
                f,
                "camera {camera_id} sent a {actual}-byte payload where its dimensions imply \
                 {expected} bytes"
            ),
            Self::UnsupportedPixelFormat {
                camera_id,
                pixel_format,
            } => write!(
                f,
                "camera {camera_id} is {pixel_format}; the shot writer stores 8-bit grayscale \
                 PGM and can only write {}",
                PixelFormat::MONO8
            ),
            Self::Path(error) => write!(f, "generated path is not a contract path: {error}"),
            Self::Manifest(error) => write!(f, "the generated manifest is not valid: {error}"),
        }
    }
}

impl std::error::Error for WriteError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Path(error) => Some(error),
            Self::Manifest(error) => Some(error),
            _ => None,
        }
    }
}

impl From<ValidationError> for WriteError {
    fn from(error: ValidationError) -> Self {
        Self::Path(error)
    }
}

impl From<ContractError> for WriteError {
    fn from(error: ContractError) -> Self {
        Self::Manifest(error)
    }
}

impl From<ValidationErrors> for WriteError {
    fn from(errors: ValidationErrors) -> Self {
        Self::Manifest(ContractError::Invalid(errors))
    }
}
