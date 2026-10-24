"""scripts/catalog.py: bootstrap is idempotent and registers in place."""

import importlib.util
import os

import pytest

from _catalog_fixtures import dataset  # noqa: F401  (fixture)
from utils import catalog as cm
from utils.frame_access import file_sha256

pytest.importorskip("jsonschema")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _cli():
    spec = importlib.util.spec_from_file_location("catalog_cli",
                                                  os.path.join(REPO, "scripts", "catalog.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_bootstrap_registers_known_bytes_once_and_never_touches_them(dataset, monkeypatch,
                                                                    capsys):
    cli = _cli()
    tw = dataset.videos["bob_fo"]                         # stands in for videos/user/TW_face.mp4
    monkeypatch.setitem(cli.EXISTING, file_sha256(tw), cli.EXISTING.pop(
        "b3fd238757d239b2164e8b90683f950590ccf31d08276def12b206b80b2945b3"))
    before = file_sha256(tw)
    assert cli.main(["bootstrap"]) == 0
    assert cli.main(["bootstrap"]) == 0
    assert "nothing changed" in capsys.readouterr().out
    cat = cm.load_catalog()
    assert set(cat.golfers) == {"grant-waite", "tiger-woods"}
    (src,) = cat.sources.values()
    assert src["golfer_id"] == "tiger-woods"              # reference golfer, file stays in user/
    assert src["location"] == {"root": "user", "path": "bob.avi"}
    assert src["provenance"]["title"].startswith("Tiger Woods")
    assert cat.collection("grant-waite-full-swing-v1")["status"] == "draft"
    assert file_sha256(tw) == before

    assert cli.main(["register", dataset.videos["gw_dtl"], "--golfer", "grant-waite",
                     "--view", "down_the_line", "--date", "2011-05"]) == 0
    assert cli.main(["register", dataset.videos["gw_dtl"], "--golfer", "nobody"]) == 2
    assert cli.main(["verify"]) == 0
    with open(dataset.videos["gw_dtl"], "r+b") as f:
        f.seek(64)
        f.write(b"\xff" * 8)
    cm._sha_memo.clear()
    assert cli.main(["verify", "--no-search"]) == 1


def test_verify_reports_a_tampered_frozen_snapshot(dataset, capsys):
    import json
    cli = _cli()
    cat = cm.load_catalog()
    cm.add_golfer(cat, "Grant Waite", "reference", "right")
    src, _ = cm.register_source(cat, dataset.videos["gw_dtl"], golfer_id="grant-waite")
    shot = cm.add_shot(cat, golfer_id="grant-waite", source_id=src["source_id"],
                       camera_view="down_the_line", start_frame=0, end_frame=10,
                       swing_type="full")
    cm.create_collection(cat, "gw-v1", "grant-waite", "GW")
    cm.add_to_collection(cat, "gw-v1", shot["shot_id"])
    cm.freeze_collection(cat, "gw-v1", by="tester")
    cm.save_catalog(cat)
    assert cli.main(["verify"]) == 0
    assert "snapshot intact" in capsys.readouterr().out
    path = os.path.join(cat.root, cat.collection("gw-v1")["snapshot"]["path"])
    doc = json.load(open(path))
    doc["frozen_by"] = "someone else"
    json.dump(doc, open(path, "w"))
    assert cli.main(["verify"]) == 1
    assert "BROKEN" in capsys.readouterr().out


def test_set_sync_records_the_exposure_alignment(dataset, capsys):
    cli = _cli()
    cat = cm.load_catalog()
    cm.add_golfer(cat, "Alice", "user", "right", golfer_id="alice")
    dtl, _ = cm.register_source(cat, dataset.videos["alice_dtl"], golfer_id="alice")
    fo, _ = cm.register_source(cat, dataset.videos["alice_fo"], golfer_id="alice")
    shot = cm.add_shot(cat, golfer_id="alice", source_id=dtl["source_id"],
                       camera_view="down_the_line", start_frame=0, end_frame=20,
                       swing_type="full")["shot_id"]
    cm.add_stream(cat, shot, source_id=fo["source_id"], camera_view="face_on", start_frame=0,
                  end_frame=20, same_swing_method="co-triggered pair", asserted_by="alice")
    cm.set_multi_view(cat, shot, "same_swing", "verified", "same impact sound", "alice")
    cm.save_catalog(cat)
    assert cli.main(["set-sync", shot, "--status", "verified", "--method", "LED flash",
                     "--by", "alice", "--offset-ms", "-37.5", "--uncertainty-ms", "0.05",
                     "--exposure", "bounded", "--exposure-uncertainty-ms", "0.02"]) == 0
    sync = cm.load_catalog().shot(shot)["multi_view"]["synchronization"]
    assert (sync["offset_ms"], sync["uncertainty_ms"], sync["exposure_alignment"],
            sync["exposure_uncertainty_ms"]) == (-37.5, 0.05, "bounded", 0.02)
    assert "exposure alignment bounded ± 0.02 ms" in capsys.readouterr().out
    # Bounded without a number is refused and nothing is saved.
    assert cli.main(["set-sync", shot, "--status", "verified", "--method", "LED flash",
                     "--by", "alice", "--offset-ms", "-37.5", "--uncertainty-ms", "0.05",
                     "--exposure", "bounded"]) == 2
    assert cm.load_catalog().shot(shot)["multi_view"]["synchronization"][
        "exposure_uncertainty_ms"] == 0.02
