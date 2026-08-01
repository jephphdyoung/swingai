"""The example contract documents must validate against their JSON Schemas.

The Rust types in ``native/crates/swingai-contracts`` parse the same two files
(``native/crates/swingai-contracts/tests/contracts.rs``). Checking both sides
against one fixture is what keeps the schema and the implementation from
drifting apart: the schema is what the Python pipeline will validate against,
the Rust types are what the capture runtime writes, and a disagreement between
them is exactly the bug this seam is prone to.

See ``docs/adr/0001-hybrid-rust-python-runtime.md``.
"""

import json
from pathlib import Path

import pytest

jsonschema = pytest.importorskip("jsonschema")

SCHEMAS = Path(__file__).resolve().parent.parent / "schemas"

CONTRACTS = [
    ("capture-manifest", "capture-manifest.schema.json", "capture-manifest.example.json"),
    ("analysis-result", "analysis-result.schema.json", "analysis-result.example.json"),
]


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _validator(schema_name):
    schema = _load(SCHEMAS / schema_name)
    cls = jsonschema.validators.validator_for(schema)
    cls.check_schema(schema)
    # The format checker only enforces "date-time" when rfc3339-validator is
    # installed, and it usually is not -- which is exactly why the schema also
    # carries an explicit pattern. Passing the checker anyway costs nothing and
    # tightens things where the extra is present.
    return cls(schema, format_checker=cls.FORMAT_CHECKER)


@pytest.mark.parametrize("name,schema_name,example_name", CONTRACTS, ids=[c[0] for c in CONTRACTS])
def test_example_validates_against_its_schema(name, schema_name, example_name):
    validator = _validator(schema_name)
    example = _load(SCHEMAS / "examples" / example_name)
    errors = sorted(validator.iter_errors(example), key=lambda e: list(e.absolute_path))
    assert not errors, "\n".join(
        f"{'/'.join(str(p) for p in e.absolute_path) or '<root>'}: {e.message}" for e in errors
    )


@pytest.mark.parametrize("name,schema_name,example_name", CONTRACTS, ids=[c[0] for c in CONTRACTS])
def test_schema_itself_is_valid(name, schema_name, example_name):
    _validator(schema_name)  # check_schema raises if the schema is malformed


def test_the_two_examples_describe_the_same_shot():
    capture = _load(SCHEMAS / "examples" / "capture-manifest.example.json")
    analysis = _load(SCHEMAS / "examples" / "analysis-result.example.json")
    assert capture["shot_id"] == analysis["shot_id"]


def test_capture_manifest_rejects_a_frame_index_masquerading_as_a_timestamp():
    """Timestamps are integer nanoseconds, not seconds and not frame numbers.

    A float here is the shape of the mistake worth catching: someone writing
    seconds-as-float, or milliseconds with a fractional part.
    """
    validator = _validator("capture-manifest.schema.json")
    manifest = _load(SCHEMAS / "examples" / "capture-manifest.example.json")
    manifest["streams"][0]["first_timestamp_ns"] = 128471.2154
    assert list(validator.iter_errors(manifest))


# --- RFC 3339 wall-clock times ------------------------------------------------

INVALID_DATE_TIMES = [
    "yesterday",
    "2026-99-99T00:00:00Z",
    "2026-07-31 12:00:00",  # space separator, no offset
    "2026-07-31T14:22:05",  # no offset
    "2026-13-01T00:00:00Z",  # month 13
    "2026-07-31T25:00:00Z",  # hour 25
    "",
]

VALID_DATE_TIMES = [
    "2026-07-31T14:22:05Z",
    "2026-07-31T14:22:05.412Z",
    "2026-07-31T14:22:05.412345678Z",
    "2026-07-31T16:22:05.412+02:00",
    "2026-07-31T09:22:05-05:00",
]


@pytest.mark.parametrize("value", INVALID_DATE_TIMES)
def test_capture_manifest_rejects_an_invalid_created_at(value):
    validator = _validator("capture-manifest.schema.json")
    manifest = _load(SCHEMAS / "examples" / "capture-manifest.example.json")
    manifest["created_at"] = value
    assert list(validator.iter_errors(manifest)), f"{value!r} should be rejected"


@pytest.mark.parametrize("value", INVALID_DATE_TIMES)
def test_analysis_result_rejects_an_invalid_created_at(value):
    validator = _validator("analysis-result.schema.json")
    result = _load(SCHEMAS / "examples" / "analysis-result.example.json")
    result["created_at"] = value
    assert list(validator.iter_errors(result)), f"{value!r} should be rejected"


@pytest.mark.parametrize("value", VALID_DATE_TIMES)
def test_capture_manifest_accepts_valid_utc_and_offset_created_at(value):
    validator = _validator("capture-manifest.schema.json")
    manifest = _load(SCHEMAS / "examples" / "capture-manifest.example.json")
    manifest["created_at"] = value
    assert not list(validator.iter_errors(manifest)), f"{value!r} should be accepted"


# --- session-relative, nonnegative timestamps ---------------------------------


def test_the_capture_example_starts_at_the_session_origin():
    """The persisted session origin is zero, and the example demonstrates it."""
    manifest = _load(SCHEMAS / "examples" / "capture-manifest.example.json")
    assert min(s["first_timestamp_ns"] for s in manifest["streams"]) == 0


@pytest.mark.parametrize(
    "pointer",
    [
        ("streams", 0, "first_timestamp_ns"),
        ("streams", 0, "last_timestamp_ns"),
        ("trigger_timestamp_ns",),
    ],
)
def test_capture_manifest_rejects_a_negative_timestamp(pointer):
    validator = _validator("capture-manifest.schema.json")
    manifest = _load(SCHEMAS / "examples" / "capture-manifest.example.json")
    target = manifest
    for key in pointer[:-1]:
        target = target[key]
    target[pointer[-1]] = -1
    assert list(validator.iter_errors(manifest))


def test_capture_manifest_rejects_a_negative_gap_timestamp():
    validator = _validator("capture-manifest.schema.json")
    manifest = _load(SCHEMAS / "examples" / "capture-manifest.example.json")
    manifest["streams"][1]["gaps"][0]["start_timestamp_ns"] = -1
    assert list(validator.iter_errors(manifest))


def test_analysis_result_rejects_a_negative_timestamp():
    validator = _validator("analysis-result.schema.json")
    result = _load(SCHEMAS / "examples" / "analysis-result.example.json")
    result["events"][0]["timestamp_ns"] = -1
    assert list(validator.iter_errors(result))


def test_analysis_result_rejects_a_negative_range_bound():
    validator = _validator("analysis-result.schema.json")
    result = _load(SCHEMAS / "examples" / "analysis-result.example.json")
    result["events"][0]["range"]["start_timestamp_ns"] = -5
    assert list(validator.iter_errors(result))


def test_analysis_events_fall_inside_the_captured_window():
    """The two documents share one capture-session clock, so this must hold."""
    capture = _load(SCHEMAS / "examples" / "capture-manifest.example.json")
    analysis = _load(SCHEMAS / "examples" / "analysis-result.example.json")
    first = min(s["first_timestamp_ns"] for s in capture["streams"])
    last = max(s["last_timestamp_ns"] for s in capture["streams"])
    for event in analysis["events"]:
        assert first <= event["timestamp_ns"] <= last, event["name"]


# --- schema versions ----------------------------------------------------------


@pytest.mark.parametrize("version", ["1.1", "1.4", "2.0", "0.9", "1", "1.0.0", ""])
def test_schemas_accept_only_the_exact_supported_version(version):
    """Strict for now -- there is no minor-version compatibility story yet, so a
    mismatch is an error rather than a silent partial read."""
    for schema_name, example_name in [
        ("capture-manifest.schema.json", "capture-manifest.example.json"),
        ("analysis-result.schema.json", "analysis-result.example.json"),
    ]:
        validator = _validator(schema_name)
        document = _load(SCHEMAS / "examples" / example_name)
        document["schema_version"] = version
        assert list(validator.iter_errors(document)), f"{schema_name} {version!r}"


def test_capture_manifest_requires_at_least_one_stream():
    validator = _validator("capture-manifest.schema.json")
    manifest = _load(SCHEMAS / "examples" / "capture-manifest.example.json")
    manifest["streams"] = []
    assert list(validator.iter_errors(manifest))


# --- portable paths -----------------------------------------------------------

NON_PORTABLE_PATHS = [
    "C:\\captures\\shot.mkv",  # Windows absolute
    "C:captures\\shot.mkv",  # Windows drive-relative
    "\\\\server\\share\\shot.mkv",  # UNC
    "\\\\?\\C:\\captures\\shot.mkv",  # extended-length
    "\\captures\\shot.mkv",  # rooted on the current drive
    "/captures/shot.mkv",  # Unix absolute
    "../shot.mkv",  # traversal
    "clips/../../shot.mkv",  # traversal, mid-path
    "clips\\shot.mkv",  # native separator, otherwise harmless
    "C:/captures/shot.mkv",  # drive with forward slashes
    "",  # empty
]


@pytest.mark.parametrize("path", NON_PORTABLE_PATHS)
def test_capture_manifest_rejects_a_non_portable_media_path(path):
    """Contract paths are forward-slash and host-independent. Backslashes are
    rejected rather than normalised: on Linux ``clips\\shot.mkv`` is one legal
    filename, so reinterpreting it as a separator would invent a path nobody
    wrote."""
    validator = _validator("capture-manifest.schema.json")
    manifest = _load(SCHEMAS / "examples" / "capture-manifest.example.json")
    manifest["streams"][0]["media"]["path"] = path
    assert list(validator.iter_errors(manifest)), f"{path!r} should be rejected"


@pytest.mark.parametrize("path", NON_PORTABLE_PATHS)
def test_analysis_result_rejects_a_non_portable_artifact_path(path):
    validator = _validator("analysis-result.schema.json")
    result = _load(SCHEMAS / "examples" / "analysis-result.example.json")
    result["artifacts"][0]["path"] = path
    assert list(validator.iter_errors(result)), f"{path!r} should be rejected"


@pytest.mark.parametrize(
    "path",
    [
        "streams/down_the_line.mkv",
        "artifacts/overlay.mp4",
        "clip.mkv",
        "a/b/c/d.png",
        "weird..name.mkv",
        "..hidden/file.mkv",
    ],
)
def test_capture_manifest_accepts_forward_slash_relative_paths(path):
    validator = _validator("capture-manifest.schema.json")
    manifest = _load(SCHEMAS / "examples" / "capture-manifest.example.json")
    manifest["streams"][0]["media"]["path"] = path
    assert not list(validator.iter_errors(manifest)), f"{path!r} should be accepted"


def test_capture_manifest_rejects_the_python_view_abbreviation():
    """``utils/swing_pairing.py`` says ``dtl``; the contract says
    ``down_the_line``. The bridge maps between them, and the contract refuses to
    guess -- so a missing mapping fails loudly instead of misfiling a swing."""
    validator = _validator("capture-manifest.schema.json")
    manifest = _load(SCHEMAS / "examples" / "capture-manifest.example.json")
    manifest["streams"][0]["view"] = "dtl"
    assert list(validator.iter_errors(manifest))


@pytest.mark.parametrize("confidence", [-0.001, 1.001])
def test_analysis_result_rejects_confidence_outside_the_unit_interval(confidence):
    validator = _validator("analysis-result.schema.json")
    result = _load(SCHEMAS / "examples" / "analysis-result.example.json")
    result["events"][0]["confidence"] = confidence
    assert list(validator.iter_errors(result))


def test_analysis_result_rejects_an_unknown_status():
    validator = _validator("analysis-result.schema.json")
    result = _load(SCHEMAS / "examples" / "analysis-result.example.json")
    result["status"] = "mostly_fine"
    assert list(validator.iter_errors(result))


def test_metadata_and_context_maps_stay_open():
    """``metadata`` and ``context`` are the contract's extension points -- the
    only places unknown keys are guaranteed to survive a round trip. Everywhere
    else, the schema version has to move."""
    capture_validator = _validator("capture-manifest.schema.json")
    manifest = _load(SCHEMAS / "examples" / "capture-manifest.example.json")
    manifest["streams"][0]["metadata"]["fox_specific_thing"] = {"nested": [1, 2, 3]}
    assert not list(capture_validator.iter_errors(manifest))

    analysis_validator = _validator("analysis-result.schema.json")
    result = _load(SCHEMAS / "examples" / "analysis-result.example.json")
    result["warnings"][0]["context"]["anything"] = [1, "two", {"three": 3}]
    result["events"][0]["metadata"]["rule_id"] = "onset-v3"
    assert not list(analysis_validator.iter_errors(result))


def test_raw_device_clock_values_are_kept_in_metadata_not_in_timestamp_fields():
    """Camera stamps may use another epoch or tick frequency; they get converted
    into the capture-session clock before being written, and the raw values ride
    along in metadata for drift diagnosis only."""
    manifest = _load(SCHEMAS / "examples" / "capture-manifest.example.json")
    for stream in manifest["streams"]:
        device = stream["metadata"]["device_clock"]
        assert device["tick_frequency_hz"] != 1_000_000_000
        assert device["first_frame_ticks"] != stream["first_timestamp_ns"]
