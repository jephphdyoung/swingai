#!/bin/bash

# === CONFIGURATION ===
IMAGE_NAME="swingai"
PORT=8501
CONTAINER_NAME="swingai_container"

# Optional: change these if your folder names differ
SAMPLES_DIR="$(pwd)/sample_videos"
MY_VIDEOS_DIR="$(pwd)/my_videos"
DATA_DIR="$(pwd)/data"
OUTPUT_DIR="$(pwd)"

# === BUILD IMAGE ===
echo "🔧 Building image '$IMAGE_NAME'..."
podman build -t "$IMAGE_NAME" .

if [ $? -ne 0 ]; then
  echo "❌ Build failed. Aborting."
  exit 1
fi
# prepare video folder
chmod -R a+rw my_videos sample_videos data
chcon -Rt svirt_sandbox_file_t my_videos
chcon -Rt svirt_sandbox_file_t sample_videos
chcon -Rt svirt_sandbox_file_t data
# === RUN CONTAINER ===
echo "🚀 Running '$IMAGE_NAME' container..."
# NOTE: data/ MUST be mounted — annotations are saved to data/annotations.json.
# Without this mount they live only in the container's --rm layer and are lost on restart.
podman run --rm \
  --name "$CONTAINER_NAME" \
  -p "$PORT":8501 \
  -v "$SAMPLES_DIR":/app/sample_videos:Z \
  -v "$MY_VIDEOS_DIR":/app/my_videos:Z \
  -v "$DATA_DIR":/app/data:Z \
  -v "$OUTPUT_DIR":/app/output \
  "$IMAGE_NAME"
