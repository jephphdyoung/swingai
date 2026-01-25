# Quick start
```bash

python3 -m venv venv
source venv/bin/activate
#pip install -r requirements.txt
pip install opencv-python mediapipe numpy
```
# ensure ffmpeg is installed: `ffmpeg -version`
```bash
python extract_p1_p4_p7.py --input /path/to/video.mp4 --outdir ./out --prefix user
```