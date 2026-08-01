//! Runs a deterministic two-camera capture and writes one shot directory.
//!
//! ```text
//! swingai-capture-sim --output <directory> [--pre-roll-ms 3000] [--trigger-ms 5000]
//! ```
//!
//! **There are no cameras involved.** Both streams come from
//! [`SyntheticSource`], which computes frames and timestamps from a
//! configuration rather than reading a device, so the run takes no wall-clock
//! time and produces the same pixels every time. What it demonstrates is the
//! part that will not change when real cameras arrive: two independent streams
//! on one capture-session clock, a trigger, a pre-roll extracted by timestamp
//! from each of them separately, and a `capture-manifest.json` that the contract
//! types accept.
//!
//! The scenario deliberately makes the streams disagree — different start
//! instants, different sequence numbering, different frame counts, and a planned
//! two-frame drop on one camera — because a capture model that only works when
//! both cameras behave identically has not been tested at all.

use std::collections::BTreeSet;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Duration;

use swingai_capture::{
    CameraId, CameraView, CaptureSession, FrameSource, RingBufferConfig, ShotExtraction, ShotId,
    StreamClip, SyntheticSource, SyntheticSourceConfig, Timestamp, WrittenShot, now_utc,
    shot_id_for, write_shot,
};

const USAGE: &str = "\
usage: swingai-capture-sim --output <directory> [options]

  --output <directory>   where the shot directory is created (required)
  --trigger-ms <ms>      when the trigger fires, on the capture-session
                         clock (default 5000)
  --pre-roll-ms <ms>     how far back the trigger reaches (default 3000)
  --retention-ms <ms>    per-camera ring-buffer retention
                         (default: pre-roll + 1000)
  --shot-id <id>         shot id, and the directory name
                         (default: derived from the wall clock)

Writes one shot directory of deterministic synthetic frames. No cameras, no
microphone, no real time: the sources compute their own timestamps.

Exits 0 on success, 1 on failure, 2 on bad arguments.";

/// 240fps, the rate the booth is being designed around. The period is not
/// exactly 1/240s, which is the point: `nominal_fps` is what the camera was
/// asked for and the measured rate comes from the timestamps.
const FRAME_INTERVAL: Duration = Duration::from_nanos(4_166_666);
const NOMINAL_FPS: f64 = 240.0;

/// Small enough that a 4-second two-camera buffer is tens of megabytes rather
/// than gigabytes. Nothing here depends on the resolution.
const WIDTH: u32 = 160;
const HEIGHT: u32 = 120;

/// Safety cap per camera. Generous for this scenario — retention is what binds.
const MAX_PAYLOAD_BYTES: u64 = 256 * 1024 * 1024;

/// The face-on camera starts a little after the down-the-line one, as two
/// independently started cameras would.
const FACE_ON_START_OFFSET: Duration = Duration::from_micros(1_700);

/// Face-on numbers its frames from here. Different from the other camera on
/// purpose: sequence numbers are stream-local, and a shared numbering would
/// invite somebody to align on it.
const FACE_ON_FIRST_SEQUENCE: u64 = 5_000;

/// Frames the sources keep producing after the trigger, as a live capture would.
const POST_TRIGGER_TAIL: Duration = Duration::from_millis(100);

fn main() -> ExitCode {
    let options = match Options::parse(std::env::args().skip(1)) {
        Ok(Some(options)) => options,
        Ok(None) => {
            println!("{USAGE}");
            return ExitCode::SUCCESS;
        }
        Err(message) => {
            eprintln!("error: {message}\n\n{USAGE}");
            return ExitCode::from(2);
        }
    };

    match run(&options) {
        Ok(report) => {
            print!("{report}");
            ExitCode::SUCCESS
        }
        Err(message) => {
            eprintln!("error: {message}");
            ExitCode::FAILURE
        }
    }
}

#[derive(Debug)]
struct Options {
    output: PathBuf,
    trigger: Duration,
    pre_roll: Duration,
    retention: Duration,
    shot_id: Option<ShotId>,
    /// Exposure slots each source runs for, derived from `trigger`. Computed
    /// during parsing so a trigger too large to build a scenario for is reported
    /// as the bad argument it is, rather than surfacing later as a failed run.
    slots: u32,
}

impl Options {
    /// Hand-rolled, like `swingai-contract-check`: five flags do not justify a
    /// CLI framework, and this stays a fixture rather than a product surface.
    ///
    /// `Ok(None)` means help was asked for.
    fn parse(args: impl Iterator<Item = String>) -> Result<Option<Self>, String> {
        let mut output = None;
        let mut trigger_ms = 5_000u64;
        let mut pre_roll_ms = 3_000u64;
        let mut retention_ms = None;
        let mut shot_id = None;

        let mut args = args.peekable();
        while let Some(flag) = args.next() {
            let mut value = || args.next().ok_or_else(|| format!("{flag} needs a value"));

            match flag.as_str() {
                "--help" | "-h" => return Ok(None),
                "--output" => output = Some(PathBuf::from(value()?)),
                "--trigger-ms" => trigger_ms = parse_millis("--trigger-ms", &value()?)?,
                "--pre-roll-ms" => pre_roll_ms = parse_millis("--pre-roll-ms", &value()?)?,
                "--retention-ms" => {
                    retention_ms = Some(parse_millis("--retention-ms", &value()?)?);
                }
                "--shot-id" => {
                    let raw = value()?;
                    shot_id = Some(ShotId::new(raw).map_err(|error| error.to_string())?);
                }
                other => return Err(format!("unknown argument {other:?}")),
            }
        }

        let output = output.ok_or("--output is required")?;
        if pre_roll_ms == 0 {
            return Err("--pre-roll-ms must be greater than zero".to_owned());
        }
        if trigger_ms == 0 {
            return Err("--trigger-ms must be greater than zero".to_owned());
        }

        // Default retention comfortably exceeds the pre-roll, so the default run
        // demonstrates a complete window while frames still age out. Checked,
        // because a pre-roll near u64::MAX would otherwise wrap the default into
        // something smaller than the pre-roll it is meant to exceed.
        let retention_ms = match retention_ms {
            Some(explicit) => explicit,
            None => pre_roll_ms.checked_add(1_000).ok_or_else(|| {
                format!(
                    "--pre-roll-ms {pre_roll_ms} leaves no room for the default retention \
                     of pre-roll + 1000ms; pass --retention-ms explicitly"
                )
            })?,
        };

        let trigger = Duration::from_millis(trigger_ms);

        Ok(Some(Self {
            output,
            trigger,
            pre_roll: Duration::from_millis(pre_roll_ms),
            retention: Duration::from_millis(retention_ms),
            shot_id,
            slots: slot_count(trigger)?,
        }))
    }
}

fn parse_millis(flag: &str, value: &str) -> Result<u64, String> {
    value
        .parse()
        .map_err(|_| format!("{flag} expects whole milliseconds, got {value:?}"))
}

fn run(options: &Options) -> Result<String, String> {
    let trigger = Timestamp::from_nanos(duration_nanos(options.trigger));
    let slots = options.slots;

    let down_the_line = SyntheticSourceConfig {
        first_sequence: 0,
        nominal_fps: NOMINAL_FPS,
        ..SyntheticSourceConfig::new(
            CameraId::new("sim-dtl").map_err(|error| error.to_string())?,
            CameraView::DownTheLine,
            WIDTH,
            HEIGHT,
            FRAME_INTERVAL,
            slots,
        )
    };

    // One frame shorter, and started slightly later: two cameras brought up by
    // software do not agree on either, and the extraction must not care.
    let face_on_start = Timestamp::from_nanos(duration_nanos(FACE_ON_START_OFFSET));
    let mut face_on = SyntheticSourceConfig {
        first_timestamp: face_on_start,
        first_sequence: FACE_ON_FIRST_SEQUENCE,
        nominal_fps: NOMINAL_FPS,
        frame_count: slots - 1,
        ..SyntheticSourceConfig::new(
            CameraId::new("sim-face-on").map_err(|error| error.to_string())?,
            CameraView::FaceOn,
            WIDTH,
            HEIGHT,
            FRAME_INTERVAL,
            slots,
        )
    };
    face_on.missing_sequences = planned_gap(&face_on, options);

    let mut sources = [
        SyntheticSource::new(down_the_line).map_err(|errors| errors.to_string())?,
        SyntheticSource::new(face_on).map_err(|errors| errors.to_string())?,
    ];

    let mut session = CaptureSession::new();
    let ring = RingBufferConfig::new(options.retention, MAX_PAYLOAD_BYTES);
    for source in &sources {
        session
            .add_camera(source.descriptor().clone(), ring)
            .map_err(|error| error.to_string())?;
    }

    // Push order across cameras is irrelevant — each frame only ever touches its
    // own buffer — so the sources are drained one after the other.
    for source in &mut sources {
        while let Some(frame) = source.next_frame() {
            session.push(frame).map_err(|error| error.to_string())?;
        }
    }

    let extraction = session
        .trigger(trigger, options.pre_roll)
        .map_err(|error| error.to_string())?;

    let created_at = now_utc();
    let shot_id = match &options.shot_id {
        Some(shot_id) => shot_id.clone(),
        None => shot_id_for(&created_at).map_err(|error| error.to_string())?,
    };

    let written = write_shot(&options.output, &shot_id, &created_at, &extraction)
        .map_err(|error| error.to_string())?;

    Ok(report(&extraction, &written))
}

/// Enough exposure slots to reach the trigger and keep going briefly after it.
///
/// Fails rather than saturates on an absurd trigger: a slot count that does not
/// fit in `u32` is not a scenario this can run, and clamping to `u32::MAX` would
/// turn "you asked for something impossible" into a run that quietly generates
/// four billion frames. The limit is the config field's own type, not a number
/// picked here.
fn slot_count(trigger: Duration) -> Result<u32, String> {
    let span = duration_nanos(trigger)
        .checked_add(duration_nanos(POST_TRIGGER_TAIL))
        .ok_or_else(|| {
            format!(
                "--trigger-ms {:.0} is too large to place on the session clock",
                trigger.as_secs_f64() * 1_000.0
            )
        })?;

    let slots = span / duration_nanos(FRAME_INTERVAL) + 1;
    u32::try_from(slots).map_err(|_| {
        format!(
            "--trigger-ms {:.0} needs {slots} exposure slots at {NOMINAL_FPS}fps, \
             more than a frame count can hold",
            trigger.as_secs_f64() * 1_000.0
        )
    })
}

/// Two consecutive frames dropped halfway through the requested window, so the
/// gap always lands inside the extracted clip rather than only with the default
/// arguments.
///
/// Empty when the source is too short to hold a gap: one needs a delivered frame
/// on each side, so the slot has to be neither the first nor within two of the
/// last.
fn planned_gap(config: &SyntheticSourceConfig, options: &Options) -> BTreeSet<u64> {
    let Some(last_slot) = u64::from(config.frame_count).checked_sub(1) else {
        return BTreeSet::new();
    };
    let Some(highest_usable) = last_slot.checked_sub(2) else {
        return BTreeSet::new();
    };
    if highest_usable < 1 {
        return BTreeSet::new();
    }

    let midpoint =
        duration_nanos(options.trigger).saturating_sub(duration_nanos(options.pre_roll) / 2);
    let start = config.first_timestamp.as_nanos();
    let slot =
        (midpoint.saturating_sub(start) / duration_nanos(FRAME_INTERVAL)).clamp(1, highest_usable);

    // `first_sequence + slot + 1` cannot overflow: the source's own validation
    // proves `first_sequence + frame_count - 1` is representable, and
    // `slot + 1 <= frame_count - 2`.
    BTreeSet::from([
        config.first_sequence + slot,
        config.first_sequence + slot + 1,
    ])
}

fn duration_nanos(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

fn millis(timestamp: Timestamp) -> f64 {
    timestamp.as_secs_f64() * 1_000.0
}

/// Why the pre-roll is or is not complete.
///
/// The two ways of being short are worth telling apart. A window reaching before
/// the session origin means that much history never existed for anyone, and no
/// buffer setting would have helped; a camera not reaching the window start
/// means retention or a late start, which is fixable. Printing a bare "no" for
/// both would hide which one the operator is looking at.
fn availability(extraction: &ShotExtraction) -> String {
    let window = extraction.window();

    if window.reaches_before_origin() {
        return format!(
            "no — the requested {:.0}ms reaches back past the session origin; only {:.0}ms \
             of session exists before the trigger",
            window.pre_roll().as_secs_f64() * 1_000.0,
            window.expressible_span().as_secs_f64() * 1_000.0,
        );
    }

    if extraction.full_pre_roll_available() {
        return "yes".to_owned();
    }

    let short: Vec<&str> = extraction
        .streams()
        .iter()
        .filter(|clip| !clip.full_pre_roll_available())
        .map(|clip| clip.descriptor().camera_id.as_str())
        .collect();

    format!(
        "no — {} did not reach the start of the window",
        short.join(", ")
    )
}

fn report(extraction: &ShotExtraction, written: &WrittenShot) -> String {
    let manifest = &written.manifest;
    let mut report = String::new();

    report.push_str(&format!(
        "shot directory   {}\n\
         manifest         {}\n\
         manifest check   valid — parsed and revalidated as schema {}, {} stream(s)\n\
         shot id          {}\n\
         created at       {}  (wall clock, for filing only)\n",
        written.directory.display(),
        written.manifest_path.display(),
        manifest.schema_version,
        manifest.streams.len(),
        manifest.shot_id,
        manifest.created_at,
    ));

    let window = extraction.window();
    report.push_str(&format!(
        "trigger          {} ({:.3}ms on the capture-session clock)\n\
         pre-roll         {:.0}ms requested — extracted from {} to {} ({:.0}ms)\n\
         full pre-roll    {}\n",
        extraction.trigger_timestamp(),
        millis(extraction.trigger_timestamp()),
        window.pre_roll().as_secs_f64() * 1_000.0,
        window.start(),
        window.end(),
        window.expressible_span().as_secs_f64() * 1_000.0,
        availability(extraction),
    ));

    for (clip, directory) in extraction.streams().iter().zip(&written.stream_directories) {
        report.push_str(&stream_report(clip, directory));
    }

    report
}

fn stream_report(clip: &StreamClip, directory: &str) -> String {
    let descriptor = clip.descriptor();
    let sequence = clip.frame_sequence();
    let measured = sequence
        .measured_fps()
        .map_or_else(|| "n/a".to_owned(), |fps| format!("{fps:.3}"));

    let mut report = format!(
        "\n  {} [{}] {}x{} {} — streams/{}/\n\
         \x20   stored frames  {}\n\
         \x20   first / last   {} / {}  ({:.3}ms of swing)\n\
         \x20   frame rate     {measured} measured, {:.1} nominal\n\
         \x20   dropped        {} in {} gap(s)\n\
         \x20   full pre-roll  {}  (buffer reaches back to {})\n",
        descriptor.camera_id,
        descriptor.view,
        descriptor.width,
        descriptor.height,
        descriptor.pixel_format,
        directory,
        sequence.frame_count,
        sequence.first_timestamp_ns,
        sequence.last_timestamp_ns,
        clip.last_timestamp().millis_since(clip.first_timestamp()),
        descriptor.nominal_fps,
        sequence.dropped_frame_count,
        clip.gaps().len(),
        yes_no(clip.full_pre_roll_available()),
        clip.buffered_from(),
    );

    for gap in clip.gaps() {
        report.push_str(&format!(
            "\x20     gap after stored frame {:06}: {} -> {}, {} frame(s) missing\n",
            gap.after_frame_index, gap.start_timestamp, gap.end_timestamp, gap.missing_frame_count,
        ));
    }

    report
}

fn yes_no(value: bool) -> &'static str {
    if value { "yes" } else { "no" }
}
