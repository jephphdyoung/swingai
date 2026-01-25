# SwingAI: Golf Swing Comparison Tool

SwingAI is a containerized tool that uses MediaPipe and DTW (Dynamic Time Warping) to compare a golfer’s swing against a reference video. It extracts body landmarks, aligns pose sequences, and generates a side-by-side comparison video.

## Features

- Extracts body pose landmarks using MediaPipe
- Aligns videos with FastDTW
- Generates side-by-side comparison video
- Streamlit UI for selecting videos
- Fully containerized (via Docker or Podman)

---

## 📁 Project Structure

swingai/
│
├── main.py
├── streamlit_app.py
├── pose_landmarker.task   # MediaPipe model (required)
├── requirements.txt
├── Dockerfile
├── sample_videos/
│   ├── GW_faceon.mp4
│   └── GW_DTL.mp4
├── my_videos/
│   └── TW_face.mp4
└── utils/
    ├── pose_extractor.py
    ├── video_sync.py
    └── video_renderer.py

---

## 🐳 Run in a Container

### 1. Build the container

podman build -t swingai .
# or use Docker:
# docker build -t swingai .

### 2. Run the app

Mount your video folders and run:

podman run --rm -p 8501:8501 \
  -v "$PWD/sample_videos:/app/sample_videos" \
  -v "$PWD/my_videos:/app/my_videos" \
  -v "$PWD:/app/output" \
  swingai

Then visit: http://localhost:8501

---

## ▶️ Run Without UI (CLI Mode)

podman run --rm \
  -v "$PWD/sample_videos:/app/sample_videos" \
  -v "$PWD/my_videos:/app/my_videos" \
  swingai python main.py my_videos/TW_face.mp4 sample_videos/GW_faceon.mp4

---

## 💻 Local Dev (No Container)

### 1. Install dependencies

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

### 2. Run the analyzer

python main.py my_videos/TW_face.mp4 sample_videos/GW_faceon.mp4

### 3. Run the UI

streamlit run streamlit_app.py

---

## 📦 Sample Videos

Use these sample links to populate your folders:

- Grant Waite Caddie View: https://www.youtube.com/watch?v=6sD5asBLNrw
- Tiger Woods Iron Shot: https://www.youtube.com/watch?v=pccwKQsEeO4

Download them using `yt-dlp`, or use the downloader container if you've built one.

---

## 📂 Output

The comparison video will be saved as:

comparison_output.mp4

If running in a container, the file appears in your mounted volume.

---

## ✅ Requirements

- Python 3.8+
- ffmpeg (only required for local dev with yt-dlp)
- Streamlit (for UI)
- MediaPipe
- OpenCV
- fastdtw

---
## TODOs

*in the user video. p1 is around the 2-2.5 second mark
in the ref video , p1 is around the 2sec mark

* highlight club shaft and club head
* show joint angles
* show clubface angle
* pause as p1- p8 positions
* show delta between videos
* show sequencing differences
* playback view that will allow users for frame/step by step playback. with p1-p8 markers
* calculate swing cadence. time from start to top of swing (ms) divided by top of swing to impact (ms)
* is it possible to calculate/estimate weight distrubition? And the diretion, and accleration of it?
---

## 📜 License

MIT License – do whatever you want with this, just don’t blame us if it slices.
