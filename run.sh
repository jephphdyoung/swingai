#!/bin/bash
set -euo pipefail

# === CONFIGURATION ===
IMAGE_NAME="swingai"
PORT=8501
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

# === BUILD IMAGE ===
echo "🔧 Building image '$IMAGE_NAME'..."
if ! podman build -t "$IMAGE_NAME" "$REPO_ROOT"; then
  echo "❌ Build failed. Aborting."
  exit 1
fi

# === PREPARE FOLDERS ===
mkdir -p "$OUTPUT_DIR"

# Resolve to absolute paths (also dereferences symlinks, which podman bind
# mounts will not follow on their own).
REFERENCE_DIR="$(readlink -f "$REFERENCE_DIR")"
USER_DIR="$(readlink -f "$USER_DIR")"
OUTPUT_DIR="$(readlink -f "$OUTPUT_DIR")"

for d in "$REFERENCE_DIR" "$USER_DIR"; do
  if [ ! -d "$d" ]; then
    echo "❌ Not a directory: $d"
    exit 1
  fi
done

chmod -R a+rw "$USER_DIR" "$REFERENCE_DIR" "$OUTPUT_DIR" "$DATA_DIR" 2>/dev/null || true
if command -v chcon >/dev/null 2>&1; then
  chcon -Rt svirt_sandbox_file_t \
    "$USER_DIR" "$REFERENCE_DIR" "$OUTPUT_DIR" "$DATA_DIR" 2>/dev/null || true
fi

echo "📁 reference: $REFERENCE_DIR"
echo "📁 user:      $USER_DIR"
echo "📁 output:    $OUTPUT_DIR"

# === RUN CONTAINER ===
# NOTE: data/ MUST be mounted — annotations are saved to data/annotations.json.
# Without this mount they live only in the container's --rm layer and are lost
# on restart.
echo "🚀 Running '$IMAGE_NAME' container on http://localhost:$PORT ..."
podman run --rm \
  --name "$CONTAINER_NAME" \
  -p "$PORT":8501 \
  -v "$REFERENCE_DIR":/app/videos/reference:Z \
  -v "$USER_DIR":/app/videos/user:Z \
  -v "$OUTPUT_DIR":/app/videos/output:Z \
  -v "$DATA_DIR":/app/data:Z \
  "$IMAGE_NAME"
