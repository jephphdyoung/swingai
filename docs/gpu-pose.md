# GPU pose estimation (RTMPose)

SwingAI has two pose backends behind the same `PoseBackend` interface
(`utils/pose_data.py`):

| | MediaPipe (default) | RTMPose GPU |
|---|---|---|
| Model | Pose Landmarker (`pose_landmarker.task`), VIDEO mode | RTMPose-m Body7 → Halpe26 (256×192) + YOLOX-m HumanArt person detector |
| Runs on | CPU (XNNPACK) | NVIDIA GPU, ONNX Runtime CUDA EP |
| Image | `swingai` (`Dockerfile`) | `swingai-gpu` (`Dockerfile.gpu`) |
| Confidence | visibility, a probability; threshold **0.5** | SimCC peak height, uncalibrated; threshold **0.3** |
| Hand points (pinky/index/thumb) | yes | **none**: reported unavailable |
| Depth (z) | relative, hip-centred | none |

The default stays MediaPipe. The benchmark
([benchmarks/pose-backends-2026-10-20.md](benchmarks/pose-backends-2026-10-20.md)) has not yet
been scored against human-verified joints, so it does not show RTMPose to be more
accurate.

---

## Running it

```bash
SWINGAI_GPU=1 ./run.sh              # app with both backends, http://localhost:8501
SWINGAI_GPU=1 ./run.sh verify-gpu   # prove inference runs on the GPU, then exit
./run.sh                            # CPU image, MediaPipe only -- unchanged
```

In the **Dual-View Comparison** tab pick **Pose model → RTMPose GPU**. Each model's results
are cached separately (see below). If a video hasn't been processed by the selected
model, the tab offers **Extract poses**. In the CPU image, selecting RTMPose shows an
error saying why it isn't available. It never falls back to MediaPipe under the RTMPose
label.

`SWINGAI_GPU=1 ./run.sh` does, in order:

1. **CDI spec**: generates `~/.local/share/swingai/cdi/nvidia.yaml` with
   `nvidia-ctk cdi generate` (as your user, no sudo) when it's missing or doesn't match the
   running driver, and passes it with `podman --cdi-spec-dir`. Why: the system spec in
   `/etc/cdi/nvidia.yaml` still referenced driver 610.57.04 after the upgrade to 615.78.08,
   and every GPU container failed with `cannot stat libEGL_nvidia.so.610.57.04`. Regenerating
   the system spec needs root and goes stale again on the next driver update. The per-user spec
   is refreshed automatically.
2. **Build** `swingai-gpu`.
3. **Fetch/verify models** into `models/rtmpose/` (`scripts/fetch_rtmpose_models.py`;
   idempotent, SHA-256 checked, git-ignored, persistent across runs).
4. **Run** with `--device nvidia.com/gpu=all`, models mounted **read-only**, and the same
   footage policy as the CPU image: user/reference read-only, no `chmod`/`chcon`, no `:z`/`:Z`,
   `--security-opt label=disable`. Only `videos/output` (pose cache) and `data/` are writable.

Without an NVIDIA driver or `nvidia-ctk`, `SWINGAI_GPU=1` stops with a clear error. CPU
mode needs neither.

## Environment (verified 2026-10-20)

| Component | Version |
|---|---|
| Host | Fedora 44, kernel 7.2.11-200.fc44 |
| GPU | NVIDIA GeForce RTX 5070 Ti, 16 GB, compute capability 12.0 (Blackwell) |
| Driver | 615.78.08 (CUDA user-mode 13.4) |
| NVIDIA Container Toolkit | 1.17.4 |
| Podman | 5.8.9, rootless |
| Base image | `nvidia/cuda:13.4.3-cudnn-runtime-ubuntu24.04@sha256:3a6966a4…` (pinned by digest) |
| CUDA runtime / cuBLAS / cuDNN | 13.4.92 / 13.8.0.4 / 9.27.0.42 |
| ONNX Runtime | `onnxruntime-gpu==1.31.0` |
| Python packages | identical to the CPU image via `constraints.txt` (MediaPipe 0.10.35, OpenCV 4.13.0.92, NumPy 2.4.6, Streamlit 1.58.0) |

**Why CUDA 13, not 12.8:** the PyPI `onnxruntime-gpu` 1.31 is built against CUDA 13 + cuDNN 9.
On a CUDA 12.8 base, its CUDA provider fails to load (`libcublasLt.so.13: cannot open shared
object file`) and ONNX Runtime silently falls back to CPU. The backend refuses that
(`CUDA provider not active`) instead of quietly running on the CPU.

**Why `constraints.txt`:** an unpinned GPU build resolved MediaPipe 1.1.1, OpenCV 5.0,
NumPy 2.5 and Streamlit 1.66, while the CPU image runs 0.10.35 / 4.13 / 2.4 / 1.58.
Both Dockerfiles now install with `-c constraints.txt` (a `pip freeze` of the CPU image).

### Runtime choice: ONNX Runtime + our own pre/post-processing

- **ONNX Runtime CUDA EP** runs the official mmdeploy ONNX exports directly. CUDA 13 kernels
  run on sm_120 (verified below). TensorRT is also available in the wheel but needs per-GPU
  engine builds; it's left for later if speed matters.
- **Pre/post-processing is ours** (`utils/pose_rtmpose.py`), written from the models' own
  `pipeline.json`, not imported from `rtmlib`. It's pure numpy/OpenCV, so the coordinate
  transforms are unit-tested without a GPU. It was cross-checked against `rtmlib` 0.0.13
  on 15 frames from each of the four main videos: identical geometry. The one difference
  is that **`rtmlib` feeds the BGR frame normalised with RGB statistics**, contradicting
  the model's `pipeline.json` (`"to_rgb": true`). With RGB input the two agree to 0.000 px
  on two videos and 0.6–1.7 px on the others, which is one SimCC bin from float64-vs-float32
  normalisation. Fed BGR as `rtmlib` does by default, they differ by up to 5.7 px.

### Models

| Role | File | Source archive (OpenMMLab mmdeploy export) | ONNX SHA-256 |
|---|---|---|---|
| Pose | `rtmpose-m_body7-halpe26_256x192.onnx` | `rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.zip` | `26f3a19e…b0bf2ecf` |
| Detector | `yolox-m_humanart_640.onnx` | `yolox_m_8xb8-300e_humanart-c2c7a14a.zip` | `3dea6513…2f5f23` |

Base URL `https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/`. These are
`rtmlib`'s Halpe26 "balanced" models. Full hashes are in `utils/pose_rtmpose.py::MODELS`.

### Pipeline per frame

1. YOLOX-m on the full frame (keep-ratio resize to 640, pad 114, BGR, no normalisation).
2. Golfer = the box overlapping the previous frame's golfer (IoU ≥ 0.3), else the largest
   person with score ≥ 0.5. No person → the frame is `detected=False`.
3. Box × 1.25, aspect fixed to 192:256, affine crop (RGB, ImageNet mean/std).
4. SimCC decode (argmax / 2, score = min of x/y peaks), then the exact inverse of the crop
   affine maps keypoints back to **original-frame pixels**.
5. Halpe26 → SwingAI names (`HALPE26_FOR`): body joints 1:1, `foot_index` ← big toe
   (approximate), pinky/index/thumb → unavailable. Nothing is carried forward or filled in.

### Cache identity

A pose cache entry is keyed by the video (name, size, mtime) and the backend identity:

```
mediapipe-0.10.35-<model sha>-env<runtime>
rtmpose-m-halpe26-<pose sha>+yolox-m-<det sha>-cfg<preprocessing hash>-ort1.31.0-cuda-env<runtime>
```

`cfg` hashes every preprocessing/inference setting (`CONFIG`). `env` hashes Python, libc,
OpenCV and NumPy versions, which was added after measuring that **the same MediaPipe version
and model give different landmarks in the CPU image (Debian, glibc 2.41) and the GPU image
(Ubuntu, glibc 2.39)**: median 0.3 px, up to 28 px where video-mode tracking diverges. Each
environment therefore keeps its own MediaPipe cache.

### Confidence thresholds are per backend

MediaPipe's visibility is a probability that the landmark is visible. RTMPose's SimCC score
is the height of the decoded peak: uncalibrated and not a probability. They are not
interchangeable, so each `PoseSequence` carries its backend's default
(`default_min_confidence`, persisted in the cache) and the UI slider defaults to it.
In practice RTMPose scores ≥ 0.3 almost everywhere, **including the occluded DTL lead arm**,
where MediaPipe reports low visibility. A confident RTMPose point is not evidence that the
point is right. Only the labelled evaluation can show that.

## Verification

`SWINGAI_GPU=1 ./run.sh verify-gpu` (`scripts/verify_gpu.py`), 2026-10-20:

```
onnxruntime 1.31.0; available providers ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
GPU: NVIDIA GeForce RTX 5070 Ti, 615.78.08, 12.0, 16303 MiB
CUDA sessions: providers ['CUDAExecutionProvider', 'CPUExecutionProvider']; init 4.04s
video: videos/user/Jeff Young - 2026-06-20 150221 Fox Face on right aaf0.mp4 (60 frames sampled, 1440x1080)
nvidia-smi compute apps while sessions live: '1, 908'
pose graph node executions by provider: {'CUDAExecutionProvider': 555, 'CPUExecutionProvider': 6}
share of node time on CUDA: 99.8%
per-frame (detector + pose): CUDA 15.0 ms, CPU 303.5 ms (20.2x)
CUDA vs CPU keypoint difference: max 2.13 px, median 1.66 px over 20 frames
golfer detected in 60/60 sampled frames
PASS: RTMPose inference executed on the GPU
```

The evidence is the process's GPU memory, ONNX Runtime's per-node profile (which provider
actually executed each operator), the speed-up, and CPU/GPU output parity. The 6 CPU nodes
are shape ops ONNX Runtime deliberately keeps on the host. The script exits non-zero if any
of these fail.

GPU tests (`tests/test_gpu_rtmpose.py`) skip unless a working CUDA provider and the models
are present:

```bash
podman --cdi-spec-dir ~/.local/share/swingai/cdi run --rm --device nvidia.com/gpu=all \
  --security-opt label=disable -v "$PWD":/app swingai-gpu python -m pytest tests/ -q
```

## Limitations

- **Accuracy is not yet established.** Joint-level ground truth needs a person; see the
  benchmark report for the labelling step.
- End to end, RTMPose processes 60–74 fps against MediaPipe's 76–87 fps. Inference is
  ~15 ms/frame on the GPU, but each frame is also decoded, converted and letterboxed on the
  CPU, and the detector runs on every frame. Batching or tracking the box between detector
  runs would help if speed matters.
- RTMPose is per-frame (no temporal smoothing). On 240fps captures it shows more
  frame-to-frame jitter than MediaPipe's tracked VIDEO mode.
- No hand points, so nothing that needs pinky/index/thumb is available under RTMPose (no
  current measurement uses them).
- Hip and foot keypoint definitions differ between the model families; compare values
  within one backend, not across.
- Model init is ~4 s per video (sessions are created per extraction run).
