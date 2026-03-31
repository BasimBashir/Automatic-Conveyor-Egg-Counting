# Egg Counter Web App — Design Spec

## Overview

A FastAPI web application wrapping the existing YOLOv5 egg detection and counting system. Provides a REST API and multi-page frontend for image detection, video processing, and live RTSP stream monitoring with independent stream/counting controls.

---

## Architecture

**Monolith**: Single FastAPI server serves both the REST API and static HTML/CSS/JS frontend. One Docker container.

---

## Pages

| Page | URL | Purpose |
|------|-----|---------|
| Image Detection | `/` | Upload image, get annotated result + egg count |
| Video Processing | `/video` | Upload video, start/stop playback, start/stop counting, download output |
| Live Stream | `/stream` | RTSP connection, live feed, start/stop counting, live config |
| API Docs | `/docs` | Auto-generated Swagger UI |

---

## API Endpoints

### Image
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/image/detect` | Upload image, returns annotated image + egg count |

### Video
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/video/upload` | Upload video, returns session ID |
| `POST` | `/api/video/{id}/start` | Start processing (detections visible) |
| `POST` | `/api/video/{id}/stop` | Stop processing |
| `POST` | `/api/video/{id}/counting/start` | Enable ROI counting |
| `POST` | `/api/video/{id}/counting/stop` | Disable ROI counting |
| `GET` | `/api/video/{id}/feed` | MJPEG stream of annotated frames |
| `GET` | `/api/video/{id}/status` | egg_count, frame_num, is_counting, is_playing |
| `GET` | `/api/video/{id}/download` | Download H.264 re-encoded video |

### Live Stream
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/stream/start` | Start RTSP stream (optional URL body, fallback to env) |
| `POST` | `/api/stream/stop` | Disconnect stream |
| `POST` | `/api/stream/counting/start` | Enable ROI counting |
| `POST` | `/api/stream/counting/stop` | Disable ROI counting |
| `GET` | `/api/stream/feed` | MJPEG stream of live feed |
| `GET` | `/api/stream/status` | egg_count, is_connected, is_counting, fps |

### Config
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/config` | Get current settings |
| `PUT` | `/api/config` | Update ROI, confidence, max_distance live |

---

## Project Structure

```
app/
├── main.py                  # FastAPI app, mount static, include routers
├── config.py                # Settings from .env
├── core/
│   ├── detector.py          # load_model(), detect_frame()
│   ├── tracker.py           # CentroidTracker class
│   ├── counter.py           # EggCounter (tracker + ROI crossing logic)
│   ├── annotator.py         # Drawing functions (trails, bbox, dashboard, ROI, flash)
│   └── video_processor.py   # VideoProcessor class (background thread, state management)
├── routers/
│   ├── image.py             # POST /api/image/detect
│   ├── video.py             # Video session management
│   ├── stream.py            # Live RTSP stream
│   └── config_router.py     # GET/PUT /api/config
├── static/
│   ├── index.html           # Image detection page
│   ├── video.html           # Video processing page
│   ├── stream.html          # Live stream page
│   ├── css/style.css        # Shared dark theme styles
│   └── js/
│       ├── image.js
│       ├── video.js
│       └── stream.js
├── uploads/                  # Uploaded files (Docker volume)
├── outputs/                  # Processed outputs
├── .env
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

---

## Core Processing

### VideoProcessor Class

Runs in a background thread per session. Manages state:
- `is_playing` — frame reading active
- `is_detecting` — model inference on each frame
- `is_counting` — ROI line crossing logic active
- `egg_count` — total eggs crossed
- `frame_num` — current frame

Loop:
1. Read frame from VideoCapture (file or RTSP)
2. If detecting: run model, get detections
3. If counting: update tracker, check ROI crossings
4. Annotate frame (bboxes, trails, dashboard, flashes)
5. Encode to JPEG, store in `latest_frame` buffer for MJPEG
6. If saving: write to temp file (mp4v codec)

### MJPEG Streaming

- Endpoints return `StreamingResponse` with `multipart/x-mixed-replace`
- Frontend: `<img src="/api/stream/feed">` — no JS needed for video
- Status polled via `setInterval` every 500ms calling `/status`

### Output Re-encoding

On processing completion or user stop:
```bash
ffmpeg -i temp_raw.mp4 -c:v libx264 -preset fast -crf 23 -c:a copy -movflags +faststart output.mp4
```
- Fixes WhatsApp/social media corruption
- `-movflags +faststart` moves moov atom to front

---

## Frontend Design

### Shared
- Dark theme matching annotation color palette
- Top navbar: Image | Video | Stream (active page highlighted)
- Amber/green accent colors from video annotations
- Large glowing green egg count number
- Status dots: green = active, red = stopped

### Image Page (`/`)
- Drag-and-drop zone / file picker
- Side-by-side: original + annotated
- Egg count display + download button

### Video Page (`/video`)
- Upload zone for video file
- 4 control buttons: Play / Stop, Start Counting / Stop Counting
- Center: MJPEG feed
- Side panel: live egg count, frame progress, FPS
- Download button (appears when done)

### Stream Page (`/stream`)
- RTSP URL input (pre-filled from env)
- Connect / Disconnect button
- Start Counting / Stop Counting button
- Center: MJPEG feed
- Side panel: live egg count, stream status, FPS
- Settings: ROI slider, confidence slider (applied live via PUT /api/config)

---

## Docker

### Dockerfile
- Base: `python:3.11-slim`
- Install: ffmpeg, libgl1 (for OpenCV)
- Copy app + model (`best.pt`)
- Install Python deps
- Expose port 8000
- CMD: `uvicorn app.main:app --host 0.0.0.0 --port 8000`

### docker-compose.yml
- Single service: `egg-counter`
- Volumes: `uploads/`, `outputs/`
- Ports: `8000:8000`
- Environment: RTSP_URL, MODEL_PATH, ROI_POSITION, CONFIDENCE
- GPU support via `deploy.resources.reservations.devices` (nvidia runtime)

---

## Config (.env)

```env
RTSP_URL=rtsp://user:pass@camera-ip:554/stream
MODEL_PATH=best.pt
ROI_POSITION=0.7
CONFIDENCE=0.25
MAX_DISTANCE=40
MAX_DISAPPEARED=50
```
