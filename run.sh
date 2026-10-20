#!/bin/bash
set -euo pipefail

# Usage:
#   ./run.sh                     CPU image (MediaPipe), the default
#   SWINGAI_GPU=1 ./run.sh       GPU image: adds RTMPose on the NVIDIA GPU
#   SWINGAI_GPU=1 ./run.sh verify-gpu   prove inference runs on the GPU, then exit
COMMAND="${1:-app}"

# === CONFIGURATION ===
GPU="${SWINGAI_GPU:-0}"
if [ "$GPU" = "1" ]; then
  IMAGE_NAME="swingai-gpu"
  DOCKERFILE="Dockerfile.gpu"
else
  IMAGE_NAME="swingai"
  DOCKERFILE="Dockerfile"
fi
PORT="${SWINGAI_PORT:-8501}"
CONTAINER_NAME="swingai_container"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Video directories. Override any of these to point at footage outside the
# repo, e.g. SWINGAI_USER_DIR=~/Golf/Videos/2026-07-27/SwingCatalyst ./run.sh
# The same variables are read by utils/paths.py inside the container, so the
# host and container agree on where footage lives.
REFERENCE_DIR="${SWINGAI_REFERENCE_DIR:-$REPO_ROOT/videos/reference}"
USER_DIR="${SWINGAI_USER_DIR:-$REPO_ROOT/videos/user}"
OUTPUT_DIR="${SWINGAI_OUTPUT_DIR:-$REPO_ROOT/videos/output}"
DATA_DIR="$REPO_ROOT/data"
# RTMPose weights (GPU only). Persistent across runs; mounted read-only.
MODEL_DIR="${SWINGAI_MODEL_DIR:-$REPO_ROOT/models}"
# NVIDIA CDI spec, generated per user so it needs no sudo and can be refreshed
# when the driver changes (a stale /etc/cdi spec breaks every GPU container).
CDI_DIR="${SWINGAI_CDI_DIR:-${XDG_DATA_HOME:-$HOME/.local/share}/swingai/cdi}"

# SWINGAI_DRY_RUN=1 prints the podman command instead of building and running.
DRY_RUN="${SWINGAI_DRY_RUN:-0}"

# === MOUNT POLICY ===
# Source footage is never modified: no chmod, no chcon, and no :z/:Z relabel
# option on any mount (:Z relabels the *host* directory even with :ro).
#
# Instead the container runs with SELinux separation disabled for this one
# container (--security-opt label=disable), so it can read files with their
# existing labels. Under rootless podman the container's root is your own
# UID, so ordinary owner permissions are enough -- nothing needs a+rw.
#
#   user, reference  -> read-only
#   models (GPU)     -> read-only (fetched/verified in a separate step)
#   output, data     -> read-write (rendered videos, pose cache, annotations)

# === PREPARE FOLDERS ===
# Only the output directory is created; video directories must already exist.
if [ "$DRY_RUN" != "1" ]; then
  mkdir -p "$OUTPUT_DIR"
  if [ "$GPU" = "1" ]; then mkdir -p "$MODEL_DIR"; fi
fi

# Resolve to absolute paths (also dereferences symlinks, which podman bind
# mounts will not follow on their own).
REFERENCE_DIR="$(readlink -f "$REFERENCE_DIR")"
USER_DIR="$(readlink -f "$USER_DIR")"
OUTPUT_DIR="$(readlink -f "$OUTPUT_DIR")"
DATA_DIR="$(readlink -f "$DATA_DIR")"
MODEL_DIR="$(readlink -f "$MODEL_DIR")"

for d in "$REFERENCE_DIR" "$USER_DIR"; do
  if [ ! -d "$d" ]; then
    echo "❌ Not a directory: $d" >&2
    exit 1
  fi
done
if [ "$DRY_RUN" != "1" ]; then
  for d in "$OUTPUT_DIR" "$DATA_DIR"; do
    if [ ! -w "$d" ]; then
      echo "❌ Not writable by $(id -un): $d" >&2
      exit 1
    fi
  done
fi

PODMAN_GLOBAL=()
GPU_ARGS=()
if [ "$GPU" = "1" ]; then
  PODMAN_GLOBAL=(--cdi-spec-dir "$CDI_DIR")
  GPU_ARGS=(--device nvidia.com/gpu=all -v "$MODEL_DIR":/app/models:ro)
fi

MOUNTS=(
  -v "$REFERENCE_DIR":/app/videos/reference:ro
  -v "$USER_DIR":/app/videos/user:ro
  -v "$OUTPUT_DIR":/app/videos/output
  -v "$DATA_DIR":/app/data
)

if [ "$COMMAND" = "verify-gpu" ]; then
  if [ "$GPU" != "1" ]; then
    echo "❌ verify-gpu needs SWINGAI_GPU=1" >&2
    exit 1
  fi
  RUN_ARGS=(run --rm --security-opt label=disable "${GPU_ARGS[@]}" "${MOUNTS[@]}"
            "$IMAGE_NAME" python scripts/verify_gpu.py)
else
  RUN_ARGS=(run --rm --replace --name "$CONTAINER_NAME" --security-opt label=disable
            "${GPU_ARGS[@]}" -p "$PORT":8501 "${MOUNTS[@]}" "$IMAGE_NAME")
fi

if [ "$DRY_RUN" = "1" ]; then
  printf '%s\n' podman "${PODMAN_GLOBAL[@]}" "${RUN_ARGS[@]}"
  exit 0
fi

# === GPU PREFLIGHT ===
if [ "$GPU" = "1" ]; then
  if ! command -v nvidia-smi >/dev/null || ! command -v nvidia-ctk >/dev/null; then
    echo "❌ SWINGAI_GPU=1 needs the NVIDIA driver (nvidia-smi) and NVIDIA Container" >&2
    echo "   Toolkit (nvidia-ctk). Run without SWINGAI_GPU for the CPU (MediaPipe) app." >&2
    exit 1
  fi
  DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
  if ! grep -q "libcuda.so.$DRIVER" "$CDI_DIR/nvidia.yaml" 2>/dev/null; then
    echo "🔧 Generating NVIDIA CDI spec for driver $DRIVER in $CDI_DIR"
    mkdir -p "$CDI_DIR"
    nvidia-ctk cdi generate --output="$CDI_DIR/nvidia.yaml" >/dev/null 2>&1 || {
      echo "❌ nvidia-ctk cdi generate failed" >&2; exit 1; }
  fi
fi

# === BUILD IMAGE ===
echo "🔧 Building image '$IMAGE_NAME' ($DOCKERFILE)..."
if ! podman build -f "$REPO_ROOT/$DOCKERFILE" -t "$IMAGE_NAME" "$REPO_ROOT"; then
  echo "❌ Build failed. Aborting."
  exit 1
fi

if [ "$GPU" = "1" ]; then
  echo "📦 Fetching/verifying RTMPose models in $MODEL_DIR"
  podman run --rm --security-opt label=disable -v "$MODEL_DIR":/app/models "$IMAGE_NAME" \
    python scripts/fetch_rtmpose_models.py /app/models/rtmpose
fi

echo "📁 reference: $REFERENCE_DIR (read-only)"
echo "📁 user:      $USER_DIR (read-only)"
echo "📁 output:    $OUTPUT_DIR"
echo "📁 data:      $DATA_DIR"
if [ "$GPU" = "1" ]; then echo "📁 models:    $MODEL_DIR (read-only)"; fi

# === RUN CONTAINER ===
# NOTE: data/ MUST be mounted — annotations are saved to data/annotations.json.
# Without this mount they live only in the container's --rm layer and are lost
# on restart.
if [ "$COMMAND" != "verify-gpu" ]; then
  echo "🚀 Running '$IMAGE_NAME' container on http://localhost:$PORT ..."
fi
podman "${PODMAN_GLOBAL[@]}" "${RUN_ARGS[@]}"
