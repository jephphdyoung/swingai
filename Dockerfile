FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y ffmpeg libgl1-mesa-glx libgles2-mesa libegl1-mesa libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
