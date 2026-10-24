"""A throwaway dataset: reference/user/output/data directories with tiny videos."""

import pytest

from _dual_view_fixtures import write_video
from utils import annotations as ann
from utils import catalog as cm
from utils import frame_access as fa


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    ref, user, out, data = (tmp_path / d for d in ("reference", "user", "output", "data"))
    for d in (ref, user, out, data):
        d.mkdir()
    monkeypatch.setenv("SWINGAI_REFERENCE_DIR", str(ref))
    monkeypatch.setenv("SWINGAI_USER_DIR", str(user))
    monkeypatch.setenv("SWINGAI_OUTPUT_DIR", str(out))
    monkeypatch.setenv("SWINGAI_DATA_DIR", str(data))
    monkeypatch.setattr(ann, "ANNOTATIONS_PATH", str(data / "annotations.json"))
    fa._memo.clear()
    fa._readers.clear()
    cm._sha_memo.clear()

    class D:
        pass
    d = D()
    d.tmp, d.ref, d.user, d.out, d.data = tmp_path, ref, user, out, data
    (user / "alice").mkdir()
    d.videos = {
        "gw_dtl": write_video(ref / "GW_DTL.avi", 40, 30.0),
        "gw_fo": write_video(ref / "GW_faceon.avi", 40, 30.0, (240, 320)),
        "gw_long": write_video(ref / "GW_compilation.avi", 120, 30.0),
        "alice_dtl": write_video(user / "alice" / "dtl.avi", 60, 60.0),
        "alice_fo": write_video(user / "alice" / "fo.avi", 60, 60.0, (300, 240)),
        "bob_fo": write_video(user / "bob.avi", 50, 60.0),
    }
    yield d
    fa._memo.clear()
    fa._readers.clear()


def golfers(cat):
    cm.add_golfer(cat, "Grant Waite", "reference", "right")
    cm.add_golfer(cat, "Tiger Woods", "reference", "right")
    cm.add_golfer(cat, "Alice", "user", "right")
    cm.add_golfer(cat, "Bob", "user", "left")
    return cat


def register(cat, d, key, golfer, view, **kw):
    rec, created = cm.register_source(cat, d.videos[key], golfer_id=golfer, camera_view=view,
                                      **kw)
    assert created
    return rec["source_id"]
