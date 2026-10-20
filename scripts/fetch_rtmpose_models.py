"""Download the RTMPose + YOLOX ONNX models into models/rtmpose/ and verify them.

    python scripts/fetch_rtmpose_models.py            # idempotent

Sources are OpenMMLab's mmdeploy ONNX exports (the same files rtmlib uses for
its Halpe26 "balanced" mode). Both the archive and the extracted model are
checked against pinned SHA-256s, so a changed upstream file is an error rather
than a silently different model.
"""

import hashlib
import io
import os
import sys
import urllib.request
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.pose_rtmpose import MODELS, default_model_dir  # noqa: E402

BASE = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def fetch(model_dir=None):
    model_dir = model_dir or default_model_dir()
    os.makedirs(model_dir, exist_ok=True)
    for spec in MODELS.values():
        dest = os.path.join(model_dir, spec["file"])
        if os.path.exists(dest) and _sha256(open(dest, "rb").read()) == spec["onnx_sha256"]:
            print(f"ok      {spec['file']}")
            continue
        url = BASE + spec["archive"]
        print(f"fetch   {url}")
        blob = urllib.request.urlopen(url).read()
        if _sha256(blob) != spec["archive_sha256"]:
            raise SystemExit(f"archive checksum mismatch: {spec['archive']}")
        zf = zipfile.ZipFile(io.BytesIO(blob))
        member = next(n for n in zf.namelist() if n.endswith("end2end.onnx"))
        data = zf.read(member)
        if _sha256(data) != spec["onnx_sha256"]:
            raise SystemExit(f"model checksum mismatch: {member}")
        tmp = dest + ".tmp"
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, dest)
        print(f"wrote   {dest}")


if __name__ == "__main__":
    fetch(sys.argv[1] if len(sys.argv) > 1 else None)
