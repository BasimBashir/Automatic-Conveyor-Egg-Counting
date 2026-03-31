# Automated Egg Counting System

A production-ready egg detection and counting system built with **YOLOv5** and **FastAPI**. Provides a CLI tool, REST API, and web dashboard for detecting eggs in images, processing videos with ROI-line counting, and monitoring live RTSP camera streams in real time.

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Web Dashboard** | Multi-page dark-themed UI for image detection, video processing, and live stream monitoring |
| **REST API** | Full API with Swagger docs for integration into other systems |
| **Image Detection** | Upload an image, get annotated result with egg count |
| **Video Processing** | Upload a video, independently control playback and counting, download H.264 output |
| **Live RTSP Stream** | Connect to Hikvision or any RTSP camera, count eggs in real time |
| **ROI Line Counting** | Centroid tracking counts each egg exactly once as it crosses a configurable line |
| **Visual Annotations** | Motion trails, animated ROI line, crossing flash effects, corner-accent bounding boxes |
| **Docker + GPU** | Single-container deployment with NVIDIA GPU support |
| **Social Media Ready** | ffmpeg H.264 re-encoding with faststart — output plays on WhatsApp, Instagram, etc. |

---

## Quick Start

### Option A: Docker Pull (fastest)

```bash
docker pull basim123/egg-counter-cuda:latest
docker run --gpus all -p 5580:5580 basim123/egg-counter-cuda:latest
```

Open **http://localhost:5580**

### Option B: Docker Build from Source

```bash
git clone https://github.com/BasimBashir/Automatic-Conveyor-Egg-Counting.git
cd Automatic-Conveyor-Egg-Counting

# Build and run (GPU)
docker compose up --build
```

Open **http://localhost:5580**

### Option C: Local Setup

```bash
git clone https://github.com/BasimBashir/Automatic-Conveyor-Egg-Counting.git
cd Automatic-Conveyor-Egg-Counting

# Create virtual environment
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Linux / macOS
source .venv/bin/activate
```

**Install PyTorch (GPU)** — visit https://pytorch.org/get-started/locally/ for your CUDA version:

```bash
# Example: CUDA 12.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# CPU only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**Install remaining dependencies:**

```bash
pip install -r requirements.txt
```

**Start the server:**

```bash
uvicorn app.main:app --host 0.0.0.0 --port 5580
```

Open **http://localhost:5580**

---

## Web Dashboard

### Image Detection (`/`)

Upload an image via drag-and-drop or file picker. Shows original and annotated side-by-side with egg count. Download the annotated result.

### Video Processing (`/video.html`)

1. Upload a video file
2. Click **Play** to start processing (detections are shown immediately)
3. Click **Start Counting** to enable ROI line tracking
4. Watch the live egg count update in the stats panel
5. Click **Stop** when done, then **Download Output (H.264)** for a social-media-ready file

Play and counting are independent controls — you can watch detections without counting, or toggle counting on/off mid-video.

### Live Stream (`/stream.html`)

1. Enter your RTSP URL (or pre-configure it in `.env`)
2. Click **Connect** to start the live feed
3. Click **Start Counting** to begin ROI line tracking
4. Adjust **ROI Position** and **Confidence** sliders in real time

### API Docs (`/docs`)

Auto-generated Swagger UI with all endpoints documented. Try them interactively.

---

## REST API

Base URL: `http://localhost:5580`

### Image

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/image/detect` | Upload image, returns annotated JPEG with `X-Egg-Count` header |

**Example:**
```bash
curl -X POST http://localhost:5580/api/image/detect \
  -F "file=@photo.jpg" \
  -o annotated.jpg -D -
```

### Video

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/video/upload` | Upload video file, returns `session_id` |
| `POST` | `/api/video/{id}/start` | Start processing |
| `POST` | `/api/video/{id}/stop` | Stop processing |
| `POST` | `/api/video/{id}/counting/start` | Enable ROI counting |
| `POST` | `/api/video/{id}/counting/stop` | Disable ROI counting |
| `GET` | `/api/video/{id}/feed` | MJPEG stream of annotated frames |
| `GET` | `/api/video/{id}/status` | Current state (egg_count, frame, fps, etc.) |
| `GET` | `/api/video/{id}/download` | Download H.264 re-encoded output |

**Example:**
```bash
# Upload
curl -X POST http://localhost:5580/api/video/upload -F "file=@video.mp4"
# Returns: {"session_id": "a1b2c3d4", "filename": "video.mp4"}

# Start processing + counting
curl -X POST http://localhost:5580/api/video/a1b2c3d4/start
curl -X POST http://localhost:5580/api/video/a1b2c3d4/counting/start

# Poll status
curl http://localhost:5580/api/video/a1b2c3d4/status

# Download output
curl http://localhost:5580/api/video/a1b2c3d4/download -o output.mp4
```

### Live Stream

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/stream/start` | Start RTSP stream (accepts `{"url": "rtsp://..."}` or uses env) |
| `POST` | `/api/stream/stop` | Disconnect stream |
| `POST` | `/api/stream/counting/start` | Enable ROI counting |
| `POST` | `/api/stream/counting/stop` | Disable ROI counting |
| `GET` | `/api/stream/feed` | MJPEG stream of live feed |
| `GET` | `/api/stream/status` | Current state (egg_count, fps, is_connected) |

**Example:**
```bash
# Connect to camera
curl -X POST http://localhost:5580/api/stream/start \
  -H "Content-Type: application/json" \
  -d '{"url": "rtsp://admin:pass@192.168.1.100:554/stream"}'

# Enable counting
curl -X POST http://localhost:5580/api/stream/counting/start

# View live feed in browser
# Open: http://localhost:5580/api/stream/feed
```

### Config

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/config` | Get current settings |
| `PUT` | `/api/config` | Update settings live |

**Example:**
```bash
# Get current config
curl http://localhost:5580/api/config

# Update ROI and confidence
curl -X PUT http://localhost:5580/api/config \
  -H "Content-Type: application/json" \
  -d '{"roi_position": 0.6, "confidence": 0.3}'
```

---

## CLI Usage

The standalone `detect_and_count.py` script works without the web server:

### Image detection

```bash
python detect_and_count.py path/to/image.jpg
python detect_and_count.py path/to/image.jpg --save result.jpg
```

### Video processing with ROI counting

```bash
python detect_and_count.py path/to/video.mp4 --save output.mp4
python detect_and_count.py path/to/video.mp4 --roi 0.7 --conf 0.3
```

### Live RTSP stream

```bash
python detect_and_count.py "rtsp://user:pass@camera-ip:554/stream"
```

Press **q** to stop during playback.

### CLI options

| Argument | Description | Default |
|----------|-------------|---------|
| `input` | Path to image, video, or RTSP URL | (required) |
| `--save` | Path to save annotated output | None |
| `--conf` | Detection confidence threshold | 0.25 |
| `--model` | Path to YOLOv5 model weights | `best.pt` |
| `--roi` | ROI line position (0.0=top, 1.0=bottom) | 0.7 |
| `--max-distance` | Max pixel distance for tracking | 40 |
| `--max-disappeared` | Frames before dropping a lost track | 50 |

---

## Configuration

All settings can be configured via `.env` file or environment variables:

```env
RTSP_URL=rtsp://user:pass@camera-ip:554/stream
MODEL_PATH=best.pt
ROI_POSITION=0.7
CONFIDENCE=0.25
MAX_DISTANCE=40
MAX_DISAPPEARED=50
```

When using Docker, set these in `docker-compose.yml` or pass them as environment variables.

---

## Project Structure

```
Automated-Egg-Counting-System/
├── app/
│   ├── main.py              # FastAPI app entry point
│   ├── config.py             # Settings from .env
│   ├── core/
│   │   ├── detector.py       # YOLOv5 model loading and inference
│   │   ├── tracker.py        # Centroid-based object tracker
│   │   ├── counter.py        # ROI line crossing logic
│   │   ├── annotator.py      # Visual annotations (trails, bbox, dashboard)
│   │   └── video_processor.py # Background thread processor with MJPEG buffer
│   ├── routers/
│   │   ├── image.py          # Image detection endpoint
│   │   ├── video.py          # Video session management
│   │   ├── stream.py         # RTSP stream endpoints
│   │   └── config_router.py  # Live config GET/PUT
│   └── static/               # Frontend HTML/CSS/JS
├── detect_and_count.py       # Standalone CLI tool
├── best.pt                   # Trained YOLOv5 model weights
├── .env                      # Configuration
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

---

## Docker

### Pull from Docker Hub

```bash
docker pull basim123/egg-counter-cuda:latest
docker run --gpus all -p 5580:5580 basim123/egg-counter-cuda:latest
```

### Build from source

```bash
docker compose up --build
```

This builds and tags the image as `basim123/egg-counter-cuda:latest`.

### GPU support

The `docker-compose.yml` includes NVIDIA GPU reservation. Requires:
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed
- Docker configured with `nvidia` runtime

### CPU-only

Remove the `deploy.resources.reservations` block from `docker-compose.yml` to run on CPU.

---

## Model Training

1. Label egg images using [Roboflow](https://roboflow.com) or CVAT
2. Train YOLOv5 in Google Colab and export `best.pt`
3. Place `best.pt` in the project root directory

---

## Video Overlay Guide

| Element | Meaning |
|---------|---------|
| **Animated dashed red line** | ROI counting line with directional arrows |
| **Amber/yellow dots** | Tracked eggs that haven't crossed the line yet |
| **Green dots** | Eggs that have been counted |
| **Thin gradient trails** | Motion path with dot markers for each tracked egg |
| **Ripple ring** | Expanding ring effect when an egg crosses the line |
| **Corner-accent boxes** | Bounding boxes with confidence % labels |
| **Dashboard panel** | Live egg count, in-frame count, FPS, progress bar |
