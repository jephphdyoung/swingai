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
    # format_checker so "date-time" on created_at is actually enforced rather
    # than being decoration.
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


def test_capture_manifest_requires_at_least_one_stream():
    validator = _validator("capture-manifest.schema.json")
    manifest = _load(SCHEMAS / "examples" / "capture-manifest.example.json")
    manifest["streams"] = []
    assert list(validator.iter_errors(manifest))


def test_capture_manifest_rejects_an_absolute_media_path():
    validator = _validator("capture-manifest.schema.json")
    manifest = _load(SCHEMAS / "examples" / "capture-manifest.example.json")
    manifest["streams"][0]["media"]["path"] = "/var/lib/swingai/clip.mkv"
    assert list(validator.iter_errors(manifest))


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


def test_analysis_result_rejects_an_absolute_artifact_path():
    validator = _validator("analysis-result.schema.json")
    result = _load(SCHEMAS / "examples" / "analysis-result.example.json")
    result["artifacts"][0]["path"] = "C:\\swingai\\overlay.mp4"
    assert list(validator.iter_errors(result))


def test_unknown_optional_metadata_is_permitted():
    """Forward compatibility: a newer writer's extra fields must not make an
    older reader reject the document."""
    validator = _validator("capture-manifest.schema.json")
    manifest = _load(SCHEMAS / "examples" / "capture-manifest.example.json")
    manifest["ambient_temperature_c"] = 21.5
    manifest["streams"][0]["metadata"]["fox_specific_thing"] = {"nested": [1, 2, 3]}
    assert not list(validator.iter_errors(manifest))
