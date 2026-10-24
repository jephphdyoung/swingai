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
