"""scripts/verify_gpu.py judging logic: both networks, each on its own evidence.

No GPU needed: the evidence is synthetic. The real-GPU run of the same checks
is in test_gpu_rtmpose.py.
"""

import copy
import importlib.util
import os

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("verify_gpu",
                                               os.path.join(REPO, "scripts", "verify_gpu.py"))
vg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vg)

CUDA, CPU = vg.CUDA, vg.CPU


def node(provider, op, dur):
    return {"cat": "Node", "name": f"/model/{op}_0_kernel_time", "dur": dur,
            "args": {"provider": provider, "op_name": op}}


def good(net="pose"):
    """Evidence shaped like the 2026-10-21 RTX 5070 Ti run."""
    return {
        "providers": [CUDA, CPU],
        "profile": vg.tally_profile([node(CUDA, "Conv", 900), node(CUDA, "MatMul", 90),
                                     node(CPU, "Reshape", 5), node(CPU, "Gather", 5)]),
        "utilization": {"idle": 0.0, "cuda": 82.0, "cpu": 0.0, "samples": [20, 20, 20]},
        "ms": {"cuda": 1.9, "cpu": 4.0},
        "parity_px": 1.7 if net == "pose" else 0.6,
    }


def test_tally_profile_counts_nodes_time_and_ops_per_provider():
    events = [node(CUDA, "Conv", 100), node(CUDA, "Conv", 50), node(CPU, "Shape", 2),
              {"cat": "Session", "name": "model_run", "dur": 999},
              {"cat": "Node", "name": "/model/Conv_0_fence_before", "dur": 7,
               "args": {"provider": CUDA, "op_name": "Conv"}}]
    t = vg.tally_profile(events)
    assert t[CUDA] == {"nodes": 2, "us": 150, "ops": {"Conv": 2}}
    assert t[CPU] == {"nodes": 1, "us": 2, "ops": {"Shape": 1}}


def test_parse_utilization_skips_non_numbers():
    assert vg.parse_utilization("0\n87\n[N/A]\n 97 \n") == [0, 87, 97]


@pytest.mark.parametrize("net", vg.NETWORKS)
def test_good_evidence_passes(net):
    assert vg.judge_network(net, good(net)) == []


def test_both_networks_are_required():
    assert vg.judge({"detector": good("detector"), "pose": good("pose")}) == []
    fails = vg.judge({"pose": good("pose")})
    assert fails == ["detector: not verified (no evidence collected)"]


def test_a_failing_detector_is_not_masked_by_a_passing_pose_network():
    bad = good("detector")
    bad["providers"] = [CPU]
    fails = vg.judge({"detector": bad, "pose": good("pose")})
    assert fails and all(f.startswith("detector:") for f in fails)


@pytest.mark.parametrize("net", vg.NETWORKS)
@pytest.mark.parametrize("breakage, message", [
    (lambda r: r.update(providers=[CPU]), "CUDA is not first"),
    (lambda r: r.update(providers=[]), "CUDA is not first"),
    # Session says CUDA, but every node ran on the CPU (silent fallback).
    (lambda r: r.update(profile=vg.tally_profile([node(CPU, "Conv", 900)])),
     "no profiled node executed on CUDA"),
    (lambda r: r.update(profile=vg.tally_profile([node(CUDA, "Relu", 10),
                                                  node(CPU, "Reshape", 90)])),
     "only 10% of node time ran on CUDA"),
    (lambda r: r.update(profile=vg.tally_profile([node(CUDA, "Conv", 990),
                                                  node(CPU, "MatMul", 5)])),
     "compute ops executed on the CPU: MatMul"),
    # No kernel activity although the profile claims CUDA.
    (lambda r: r["utilization"].update(cuda=3.0), "not clearly above"),
    # The GPU was busy anyway (another process): the rise isn't ours.
    (lambda r: r["utilization"].update(idle=70.0), "not clearly above"),
    # The CPU-provider control lights the GPU up just as much: not our kernels.
    (lambda r: r["utilization"].update(cpu=75.0), "not clearly above"),
    (lambda r: r["utilization"].update(cuda=None), "no NVML utilization samples"),
    (lambda r: r.update(parity_px=None), "no CUDA/CPU parity measured"),
    (lambda r: r.update(parity_px=float("inf")), "outputs differ"),
    (lambda r: r.update(parity_px=12.0), "outputs differ by 12.0 px"),
])
def test_each_broken_check_fails_naming_the_network(net, breakage, message):
    r = copy.deepcopy(good(net))
    breakage(r)
    fails = vg.judge_network(net, r)
    assert fails and all(f.startswith(f"{net}: ") for f in fails), fails
    assert any(message in f for f in fails), fails


def test_describe_reports_every_piece_of_evidence():
    text = vg.describe("detector", good("detector"))
    for s in ("[detector]", "session providers", "of node time on CUDA",
              "ops executed on CPU", "GPU kernel activity", "CUDA run 82.0%", "per call",
              "CUDA vs CPU output"):
        assert s in text
