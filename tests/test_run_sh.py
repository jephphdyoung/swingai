"""run.sh must never modify the video libraries it mounts.

Checks the podman command it would run (SWINGAI_DRY_RUN=1) and the script
text itself: source footage read-only, no SELinux relabel option on any
mount, no recursive chmod/chcon, and the env-var overrides still honoured.
"""

import os
import re
import shutil
import stat
import subprocess

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_SH = os.path.join(REPO, "run.sh")

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


@pytest.fixture
def dirs(tmp_path):
    out = {}
    for name in ("user", "reference", "output"):
        d = tmp_path / name
        d.mkdir()
        clip = d / "clip.mp4"
        clip.write_bytes(b"not really a video")
        os.chmod(clip, 0o640)
        os.chmod(d, 0o750)
        out[name] = d
    return out


def dry_run(dirs, **extra):
    env = dict(os.environ, SWINGAI_DRY_RUN="1",
               SWINGAI_USER_DIR=str(dirs["user"]),
               SWINGAI_REFERENCE_DIR=str(dirs["reference"]),
               SWINGAI_OUTPUT_DIR=str(dirs["output"]))
    env.update(extra)
    res = subprocess.run(["bash", RUN_SH], env=env, capture_output=True, text=True)
    return res, res.stdout.splitlines()


def mounts(argv):
    """{container_path: (host_path, options)} for every -v."""
    found = {}
    for i, arg in enumerate(argv):
        if arg == "-v":
            parts = argv[i + 1].split(":")
            host, container = parts[0], parts[1]
            found[container] = (host, parts[2] if len(parts) > 2 else "")
    return found


def test_source_video_dirs_are_read_only_and_never_relabelled(dirs):
    res, argv = dry_run(dirs)
    assert res.returncode == 0, res.stderr
    m = mounts(argv)
    for target, name in (("/app/videos/user", "user"), ("/app/videos/reference", "reference")):
        host, opts = m[target]
        assert host == os.path.realpath(dirs[name])
        assert "ro" in opts.split(",")
        assert not set(opts.split(",")) & {"z", "Z", "rw", "U"}, opts


def test_only_output_and_data_are_writable(dirs):
    _, argv = dry_run(dirs)
    m = mounts(argv)
    assert set(m) == {"/app/videos/user", "/app/videos/reference",
                      "/app/videos/output", "/app/data"}
    for target in ("/app/videos/output", "/app/data"):
        assert "ro" not in m[target][1].split(",")
    assert m["/app/videos/output"][0] == os.path.realpath(dirs["output"])
    assert m["/app/data"][0] == os.path.join(REPO, "data")


def test_no_mount_requests_a_relabel(dirs):
    _, argv = dry_run(dirs)
    for host, opts in mounts(argv).values():
        assert "z" not in opts.lower().split(","), (host, opts)


def test_selinux_handled_by_label_disable_not_relabelling(dirs):
    _, argv = dry_run(dirs)
    i = argv.index("--security-opt")
    assert argv[i + 1] == "label=disable"


def test_script_never_chmods_or_chcons():
    code = [ln.split("#", 1)[0] for ln in open(RUN_SH).read().splitlines()]
    body = "\n".join(code)
    assert not re.search(r"\b(chmod|chcon|chown|restorecon|setfacl)\b", body)
    assert not re.search(r":[^\s\"']*\b[zZ]\b", body)


def test_dry_run_leaves_permissions_untouched(dirs):
    before = {p: stat.S_IMODE(os.stat(p).st_mode)
              for d in dirs.values() for p in (d, d / "clip.mp4")}
    dry_run(dirs)
    after = {p: stat.S_IMODE(os.stat(p).st_mode) for p in before}
    assert after == before


def test_missing_user_dir_is_an_error_not_created(dirs, tmp_path):
    missing = tmp_path / "nope"
    res, _ = dry_run(dirs, SWINGAI_USER_DIR=str(missing))
    assert res.returncode != 0
    assert not missing.exists()


def test_port_override(dirs):
    _, argv = dry_run(dirs, SWINGAI_PORT="8600")
    assert argv[argv.index("-p") + 1] == "8600:8501"


def test_replaces_a_stale_container_of_the_same_name(dirs):
    _, argv = dry_run(dirs)
    assert "--replace" in argv
