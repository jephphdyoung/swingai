#!/bin/bash
set -euo pipefail

# === CONFIGURATION ===
IMAGE_NAME="swingai"
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
#   output, data     -> read-write (rendered videos, pose cache, annotations)

# === PREPARE FOLDERS ===
# Only the output directory is created; video directories must already exist.
if [ "$DRY_RUN" != "1" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

# Resolve to absolute paths (also dereferences symlinks, which podman bind
# mounts will not follow on their own).
REFERENCE_DIR="$(readlink -f "$REFERENCE_DIR")"
USER_DIR="$(readlink -f "$USER_DIR")"
OUTPUT_DIR="$(readlink -f "$OUTPUT_DIR")"
DATA_DIR="$(readlink -f "$DATA_DIR")"

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

RUN_ARGS=(
  run --rm --replace
  --name "$CONTAINER_NAME"
  --security-opt label=disable
  -p "$PORT":8501
  -v "$REFERENCE_DIR":/app/videos/reference:ro
  -v "$USER_DIR":/app/videos/user:ro
  -v "$OUTPUT_DIR":/app/videos/output
  -v "$DATA_DIR":/app/data
  "$IMAGE_NAME"
)

if [ "$DRY_RUN" = "1" ]; then
  printf '%s\n' podman "${RUN_ARGS[@]}"
  exit 0
fi

# === BUILD IMAGE ===
echo "🔧 Building image '$IMAGE_NAME'..."
if ! podman build -t "$IMAGE_NAME" "$REPO_ROOT"; then
  echo "❌ Build failed. Aborting."
  exit 1
fi

echo "📁 reference: $REFERENCE_DIR (read-only)"
echo "📁 user:      $USER_DIR (read-only)"
echo "📁 output:    $OUTPUT_DIR"
echo "📁 data:      $DATA_DIR"

# === RUN CONTAINER ===
# NOTE: data/ MUST be mounted — annotations are saved to data/annotations.json.
# Without this mount they live only in the container's --rm layer and are lost
# on restart.
echo "🚀 Running '$IMAGE_NAME' container on http://localhost:$PORT ..."
podman "${RUN_ARGS[@]}"
