# Automated Egg Counting System

An end-to-end computer vision system for detecting and counting eggs using **YOLOv8** object detection. Supports image detection, video processing with line-crossing counting, and RTSP live stream monitoring through a web-based dashboard.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Inference](#inference)
- [Web Application](#web-application)
- [Deployment](#deployment)
  - [Docker Hub — pre-built images](#docker-hub--pre-built-images)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [License](#license)

---

## Overview

This system provides three operational modes:

| Mode | Description |
|------|-------------|
| **Image** | Upload a single image, detect all eggs, return annotated result with count |
| **Video** | Upload a video file, track eggs across frames, count each egg exactly once as it crosses a configurable ROI (Region of Interest) line |
| **Stream** | Connect to an RTSP camera feed for real-time egg detection and counting |

The detection model is **YOLOv8** trained on a single-class egg dataset. Tracking uses a bbox-aware centroid tracker with Hungarian assignment that combines IoU and centroid distance, so overlapping or stacked eggs keep distinct IDs and each one is counted independently.

---

## Project Structure

```
Automated-Egg-Counting-System/
├── app/                            # FastAPI web application
│   ├── main.py                     # App entrypoint
│   ├── config.py                   # Settings (env-based)
│   ├── core/
│   │   ├── detector.py             # YOLOv8 model loading & inference
│   │   ├── tracker.py              # Bbox-aware centroid tracker (Hungarian)
│   │   ├── counter.py              # EggCounter: ROI line-crossing logic
│   │   ├── line_counter.py         # Trackerless conveyor counter
│   │   ├── annotator.py            # Frame annotation (bboxes, trails, dashboard)
│   │   ├── video_processor.py     # Background video/stream processor
│   │   ├── model_cache.py          # Thread-safe YOLO model cache
│   │   ├── runtime_config.py       # Live runtime configuration
│   │   └── exporter.py             # TensorRT export state machine
│   ├── routers/
│   │   ├── image.py                # POST /api/image/detect
│   │   ├── video.py                # Video upload, playback, counting
│   │   ├── stream.py               # RTSP stream management
│   │   ├── config_router.py        # GET/PATCH /api/config
│   │   ├── export_router.py        # TensorRT export endpoints
│   │   └── health_router.py        # GET /health
│   └── static/                     # Frontend (HTML/CSS/JS)
│       ├── index.html              # Image detection page
│       ├── video.html              # Video processing page
│       ├── stream.html             # Live stream page
│       ├── css/style.css
│       └── js/
├── detect_and_count.py             # Standalone CLI for image/video inference
├── best.pt                         # Trained model weights
├── requirements.txt                # Python deps (GPU / CUDA 12.6)
├── requirements-cpu.txt            # Python deps (CPU only)
├── Dockerfile                      # GPU container image
├── Dockerfile.cpu                  # CPU container image
├── docker-compose.yml              # Docker Compose with GPU support
├── start.bat                       # Windows quick-start (Docker)
└── .env                            # Environment variables (optional)
```

---

## Prerequisites

- **Python** 3.10+
- **CUDA-compatible GPU** (recommended for inference speed; CPU also supported)
- **FFmpeg** (required for video re-encoding/download feature)
- **Docker** (optional, for containerized deployment)

### Hardware Recommendations

| Task | Minimum | Recommended |
|------|---------|-------------|
| Inference | CPU (slow) | Any CUDA GPU |
| Web App | 4 GB RAM | 8+ GB RAM |

---

## Setup

### 1. Clone the project

```bash
git clone https://github.com/BasimBashir/Automated-Egg-Counting-System.git
cd Automated-Egg-Counting-System
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate
```

### 3. Install dependencies

```bash
# GPU
pip install -r requirements.txt

# CPU only
pip install -r requirements-cpu.txt
```

### 4. Verify GPU availability (optional)

```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

---

## Inference

### CLI — Standalone Script

The `detect_and_count.py` script works without the web server for quick testing.

#### Detect eggs in an image

```bash
python detect_and_count.py path/to/image.jpg
```

With options:

```bash
python detect_and_count.py path/to/image.jpg --conf 0.3 --model best.pt --save output.jpg
```

#### Count eggs in a video

```bash
python detect_and_count.py path/to/video.mp4
```

With options:

```bash
python detect_and_count.py path/to/video.mp4 \
    --conf 0.25 \
    --roi 0.7 \
    --max-distance 40 \
    --max-disappeared 50 \
    --save annotated_output.mp4
```

#### Process an RTSP stream

```bash
python detect_and_count.py rtsp://user:pass@camera-ip:554/stream
```

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `input` | (required) | Path to image, video, or RTSP URL |
| `--model` | `best.pt` | Path to YOLOv8 model weights |
| `--conf` | `0.25` | Detection confidence threshold |
| `--roi` | `0.7` | ROI line position (0.0=top, 1.0=bottom) |
| `--max-distance` | `40` | Max pixel distance for tracker matching |
| `--max-disappeared` | `50` | Frames before dropping a lost track |
| `--save` | `None` | Output path for annotated result |

### Controls (during video/stream playback)

- Press **`q`** to quit

---

## Web Application

The web app provides a browser-based UI with three pages: Image, Video, and Stream.

### Start the server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 5590
```

Then open **http://localhost:5590** in your browser.

### Image Detection Page (`/`)

1. Drag-and-drop or click to upload an image
2. The server runs YOLOv8 detection and returns an annotated image
3. Displays egg count and side-by-side comparison (original vs annotated)
4. Download the annotated image

### Video Processing Page (`/video.html`)

1. Upload a video file (MP4, AVI, MOV, MKV)
2. Click **Play** to start processing
3. Click **Start Counting** to enable the ROI line-crossing counter
4. Watch the live MJPEG feed with bounding boxes, trails, and dashboard overlay
5. When complete, download the annotated output video (re-encoded to H.264)

### Live Stream Page (`/stream.html`)

1. Enter an RTSP URL (or configure via `.env`)
2. Click **Connect** to start the stream
3. Click **Start Counting** to enable counting
4. Adjust ROI position and confidence threshold in real time via sliders
5. Live stats: egg count, FPS

---

## Deployment

### Option 1: Direct (bare metal / VM)

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 5590
```

For production:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 5590 --workers 1
```

### Option 2: Docker

#### Build and run locally

```bash
docker compose up --build
```

This builds the image (tagged as `basim123/egg-counter:latest`), starts the container on port **5590** with GPU passthrough (NVIDIA Container Toolkit required).

#### Quick start (Windows)

Double-click `start.bat` — it builds the container, waits for the server, and opens the browser automatically.

#### Docker Compose configuration

```yaml
services:
  egg-counter:
    image: basim123/egg-counter:latest
    build: .
    ports:
      - "5590:5590"
    volumes:
      - ./app/uploads:/app/app/uploads
      - ./app/outputs:/app/app/outputs
    environment:
      - MODEL_PATH=best.pt
      - ROI_POSITION=0.7
      - CONFIDENCE=0.25
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

---

### Docker Hub — pre-built images

Pre-built images are published on Docker Hub under **`basim123/egg-counter`**:

| Tag | Base | Use when |
|-----|------|----------|
| `latest` | NVIDIA CUDA 12.6.2 | You have an NVIDIA GPU + NVIDIA Container Toolkit |
| `cpu` | Python 3.12 slim | No GPU / any machine |

---

#### Build and push (owner only)

```bash
# GPU image (default)
docker build -t basim123/egg-counter:latest .
docker push basim123/egg-counter:latest

# CPU image
docker build -f Dockerfile.cpu -t basim123/egg-counter:cpu .
docker push basim123/egg-counter:cpu
```

---

#### Pull and run — GPU

Requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

```bash
docker pull basim123/egg-counter:latest
```

```bash
# Linux / macOS
docker run -d \
  --gpus all \
  --name egg-counter \
  -p 5590:5590 \
  -v "$(pwd)/uploads:/app/app/uploads" \
  -v "$(pwd)/outputs:/app/app/outputs" \
  -e MODEL_PATH=best.pt \
  -e ROI_POSITION=0.7 \
  -e CONFIDENCE=0.25 \
  --restart unless-stopped \
  basim123/egg-counter:latest
```

```powershell
# Windows PowerShell
docker run -d `
  --gpus all `
  --name egg-counter `
  -p 5590:5590 `
  -v "${PWD}/uploads:/app/app/uploads" `
  -v "${PWD}/outputs:/app/app/outputs" `
  -e MODEL_PATH=best.pt `
  -e ROI_POSITION=0.7 `
  -e CONFIDENCE=0.25 `
  --restart unless-stopped `
  basim123/egg-counter:latest
```

Open **http://localhost:5590** in your browser.

---

#### Pull and run — CPU

```bash
docker pull basim123/egg-counter:cpu
```

```bash
# Linux / macOS
docker run -d \
  --name egg-counter \
  -p 5590:5590 \
  -v "$(pwd)/uploads:/app/app/uploads" \
  -v "$(pwd)/outputs:/app/app/outputs" \
  -e MODEL_PATH=best.pt \
  -e ROI_POSITION=0.7 \
  -e CONFIDENCE=0.25 \
  --restart unless-stopped \
  basim123/egg-counter:cpu
```

```powershell
# Windows PowerShell
docker run -d `
  --name egg-counter `
  -p 5590:5590 `
  -v "${PWD}/uploads:/app/app/uploads" `
  -v "${PWD}/outputs:/app/app/outputs" `
  -e MODEL_PATH=best.pt `
  -e ROI_POSITION=0.7 `
  -e CONFIDENCE=0.25 `
  --restart unless-stopped `
  basim123/egg-counter:cpu
```

Open **http://localhost:5590** in your browser.

> **Note:** CPU inference is significantly slower than GPU. Video and stream processing will run at reduced FPS.

---

#### RTSP stream (optional)

Pass your camera URL via the `-e RTSP_URL=` flag:

```bash
docker run -d --gpus all -p 5590:5590 \
  -e RTSP_URL=rtsp://user:pass@192.168.1.100:554/stream \
  basim123/egg-counter:latest
```

---

#### Useful container commands

```bash
docker logs -f egg-counter           # tail logs
docker stop egg-counter              # stop
docker rm egg-counter                # remove
docker pull basim123/egg-counter:latest && \
  docker stop egg-counter && docker rm egg-counter && \
  docker run ...                     # update to latest
```

### Option 3: Cloud deployment

#### AWS EC2 / GCP Compute Engine

1. Launch a GPU instance (e.g., `g4dn.xlarge` on AWS, `n1-standard-4` + T4 on GCP)
2. Install NVIDIA drivers and Docker
3. Clone the project and copy `best.pt` into the root
4. Run `docker compose up -d`
5. Open port **5590** in the security group / firewall

#### Behind a reverse proxy (Nginx)

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:5590;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_buffering off;           # Required for MJPEG streaming
        proxy_cache off;
        proxy_read_timeout 3600s;      # Keep stream connections alive
    }
}
```

---

## Configuration

All settings can be configured via environment variables, the `.env` file, or **live at runtime** via the `PATCH /api/config` endpoint:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `best.pt` | Path to YOLOv8 model weights |
| `RTSP_URL` | (empty) | Default RTSP stream URL |
| `ROI_POSITION` | `0.7` | ROI line position (0.0 = top, 1.0 = bottom) |
| `CONFIDENCE` | `0.25` | Detection confidence threshold |
| `NMS_IOU` | `0.45` | Non-max suppression IoU threshold |
| `IMGSZ` | `640` | Inference image size (multiple of 32) |
| `MAX_DISTANCE` | `40` | Max pixel distance for tracker matching |
| `MAX_DISAPPEARED` | `50` | Frames before dropping a lost track |

Runtime overrides via `PATCH /api/config` take effect immediately for new requests — no container restart required. Changing `model_path` reloads the model synchronously and rolls back on failure.

---

## API Reference

### Health

```
GET    /health                           # GPU status, current model, config
```

### Image Detection

```
POST   /api/image/detect
Content-Type: multipart/form-data
Body: file=<image>

Response: image/jpeg (annotated image)
Headers: X-Egg-Count: <number>
```

### Video Processing

```
POST   /api/video/upload                 # Upload video, returns { session_id }
POST   /api/video/{id}/start             # Start playback
POST   /api/video/{id}/stop              # Stop playback
POST   /api/video/{id}/counting/start    # Enable counting
POST   /api/video/{id}/counting/stop     # Disable counting
GET    /api/video/{id}/feed              # MJPEG stream
GET    /api/video/{id}/status            # { egg_count, frame_num, fps, ... }
GET    /api/video/{id}/download          # Download H.264 output
```

### Live Stream

```
POST   /api/stream/start                 # { url: "rtsp://..." }
POST   /api/stream/stop
POST   /api/stream/counting/start
POST   /api/stream/counting/stop
GET    /api/stream/feed                  # MJPEG stream
GET    /api/stream/status                # { egg_count, fps, is_connected, ... }
```

### Configuration

```
GET    /api/config                       # Get current live settings
PATCH  /api/config                       # Update settings (partial, no restart)
```

### TensorRT Export

```
POST   /api/export/tensorrt              # Start background TensorRT export
GET    /api/export/tensorrt              # Poll status; auto-switches model_path on DONE
```

Interactive Swagger docs available at **http://localhost:5590/docs**.

---

## License

See [LICENSE](LICENSE) for terms.
