# Automated Egg Counting System

End-to-end computer vision API for detecting and counting eggs on conveyor belts. Built on **YOLOv8**. Supports image detection, single-video processing, and **up to 10 concurrent RTSP cameras or video files** through one shared model in one container. Each conveyor (per stream or per upload) can be configured **top → bottom** or **left → right**.

This README is structured for integrators: people building their own desktop apps, dashboards, mobile clients, or line-control HMIs against this service.

---

## Table of Contents

- [What it is](#what-it-is)
- [Quick start — Docker](#quick-start--docker)
- [Quick start — local Python](#quick-start--local-python)
- [Web UI guide](#web-ui-guide)
- [REST API reference](#rest-api-reference)
- [Building your own client](#building-your-own-client)
- [Conveyor direction](#conveyor-direction)
- [CLI (detect_and_count.py)](#cli-detect_and_countpy)
- [Deployment](#deployment)
- [Configuration reference](#configuration-reference)
- [Project structure](#project-structure)
- [License](#license)

---

## What it is

Three operational modes share one FastAPI service:

| Mode | What it does |
|------|--------------|
| **Image** | One-shot detection on an uploaded image. Returns the annotated image + count. |
| **Video** | One-shot processing of an uploaded video file. Polled MJPEG live preview, H.264 download. |
| **Streams** | Up to **10 fixed slots**. Each slot can carry an RTSP URL or an uploaded video file. One batched inference call serves all 10 slots; results are returned independently per slot. Slot config (URL, conveyor direction, ROI, confidence) is persisted to disk and survives container restarts. |

Detection: YOLOv8 (single-class egg model). Tracking: bbox-aware centroid tracker with Hungarian assignment. Counting: per-direction line-crossing logic.

---

## Quick start — Docker

Pre-built images on Docker Hub (`basim123/egg-counter`):

| Tag | Base | Use when |
|-----|------|----------|
| `latest` | NVIDIA CUDA 12.6.2 | NVIDIA GPU + NVIDIA Container Toolkit available |
| `cpu` | Python 3.12 slim | No GPU |

### Docker Compose (recommended)

```bash
git clone https://github.com/BasimBashir/Automated-Egg-Counting-System.git
cd Automated-Egg-Counting-System
docker compose up -d
```

Opens on `http://localhost:5590`. The `app/data` volume persists slot configuration across restarts.

### Manual docker run — GPU

```bash
docker pull basim123/egg-counter:latest

docker run -d \
  --gpus all \
  --name egg-counter \
  -p 5590:5590 \
  -v "$(pwd)/app/uploads:/app/app/uploads" \
  -v "$(pwd)/app/outputs:/app/app/outputs" \
  -v "$(pwd)/app/data:/app/app/data" \
  -e MODEL_PATH=best.pt \
  --restart unless-stopped \
  basim123/egg-counter:latest
```

PowerShell:

```powershell
docker run -d `
  --gpus all `
  --name egg-counter `
  -p 5590:5590 `
  -v "${PWD}/app/uploads:/app/app/uploads" `
  -v "${PWD}/app/outputs:/app/app/outputs" `
  -v "${PWD}/app/data:/app/app/data" `
  -e MODEL_PATH=best.pt `
  --restart unless-stopped `
  basim123/egg-counter:latest
```

### Manual docker run — CPU

```bash
docker pull basim123/egg-counter:cpu

docker run -d \
  --name egg-counter \
  -p 5590:5590 \
  -v "$(pwd)/app/uploads:/app/app/uploads" \
  -v "$(pwd)/app/outputs:/app/app/outputs" \
  -v "$(pwd)/app/data:/app/app/data" \
  -e MODEL_PATH=best.pt \
  --restart unless-stopped \
  basim123/egg-counter:cpu
```

> CPU inference is slower; batched 10-stream throughput drops accordingly.

### First-boot RTSP seeding (optional)

If `app/data/streams.json` does not exist *and* you set `-e RTSP_URL=rtsp://...`, slot 1 is pre-populated with that URL on first run. On every subsequent boot the file is canonical and the env var is ignored.

### Updating an existing deployment

```bash
docker pull basim123/egg-counter:latest
docker compose up -d
```

Slot configs in `./app/data/streams.json` survive the upgrade.

---

## Quick start — local Python

Requires Python 3.10+, FFmpeg.

```bash
git clone https://github.com/BasimBashir/Automated-Egg-Counting-System.git
cd Automated-Egg-Counting-System
python -m venv venv
# Windows:   venv\Scripts\activate
# Linux/mac: source venv/bin/activate

# GPU:
pip install -r requirements.txt
# OR CPU:
pip install -r requirements-cpu.txt

uvicorn app.main:app --host 0.0.0.0 --port 5590
```

Open `http://localhost:5590`.

---

## Web UI guide

### Image (`/`)

Upload an image → annotated image + count.

### Video (`/video.html`)

Upload an MP4/AVI/MOV/MKV → live MJPEG preview → optionally enable counting → download H.264 result. A **Conveyor direction** toggle (Top→Bottom / Left→Right) is on the upload card; the ROI slider applies along the chosen axis.

### Streams (`/streams.html`)

10-slot grid wall + drill-down.

1. **Grid wall** — overview of all 10 slots. Each tile shows status, count, FPS, direction arrow, and a thumbnail of the live feed.
2. **Click a tile** to drill in. Pick a source (RTSP URL **or** upload a video file), pick direction, set ROI and confidence, optionally check **Enabled on boot** and **Auto-count on start**. Click **Save settings**, then **Connect**, then **Start Counting**.

Grid polls `/api/streams/status` every second.

---

## REST API reference

Base URL: `http://<host>:5590`. All endpoints return JSON unless marked MJPEG / image.

### Conventions

- Slots are integers `1..10`. Other values → `404`.
- `direction` is `"tb"` or `"lr"`. `roi_position` is a float `[0, 1]` interpreted as a fraction along the direction-of-travel axis.
- MJPEG endpoints return `multipart/x-mixed-replace; boundary=frame`. Browsers and Electron handle this natively in `<img src="...">`.

### Health

`GET /health` — GPU availability, current model, runtime config.

### Image detection

`POST /api/image/detect` — multipart upload `file=<image>`. Returns annotated JPEG; `X-Egg-Count` header carries the count.

```bash
curl -X POST http://localhost:5590/api/image/detect \
  -F file=@egg.jpg -o annotated.jpg -D headers.txt
grep X-Egg-Count headers.txt
```

```python
import requests
r = requests.post("http://localhost:5590/api/image/detect",
                  files={"file": open("egg.jpg", "rb")})
open("annotated.jpg", "wb").write(r.content)
print("count:", r.headers["X-Egg-Count"])
```

```javascript
const fd = new FormData();
fd.append("file", file);  // a File from <input type="file">
const r = await fetch("/api/image/detect", { method: "POST", body: fd });
const count = r.headers.get("X-Egg-Count");
const blob = await r.blob();
```

### Video processing (one-shot upload)

```
POST   /api/video/upload                 # multipart: file, direction?, roi_position?
POST   /api/video/{id}/start             # begin playback / inference
POST   /api/video/{id}/stop
POST   /api/video/{id}/counting/start
POST   /api/video/{id}/counting/stop
GET    /api/video/{id}/feed              # MJPEG
GET    /api/video/{id}/status            # { egg_count, frame_num, fps, ... }
GET    /api/video/{id}/download          # H.264 .mp4
```

Upload example:

```bash
curl -X POST http://localhost:5590/api/video/upload \
  -F file=@conveyor.mp4 -F direction=lr -F roi_position=0.5
```

```python
import requests
r = requests.post("http://localhost:5590/api/video/upload",
                  files={"file": open("conveyor.mp4", "rb")},
                  data={"direction": "lr", "roi_position": 0.5})
sid = r.json()["session_id"]
requests.post(f"http://localhost:5590/api/video/{sid}/start")
requests.post(f"http://localhost:5590/api/video/{sid}/counting/start")
```

### Streams (multi-slot)

```
GET    /api/streams                       # all 10 slots: [{slot, config, runtime}]
GET    /api/streams/status                # aggregate runtime keyed by slot
GET    /api/streams/{slot}                # single slot
GET    /api/streams/{slot}/status         # single slot runtime
PUT    /api/streams/{slot}/config         # set full config
PATCH  /api/streams/{slot}/config         # partial update
POST   /api/streams/{slot}/upload         # multipart: file → source becomes type=file
POST   /api/streams/{slot}/start
POST   /api/streams/{slot}/stop
POST   /api/streams/{slot}/counting/start
POST   /api/streams/{slot}/counting/stop
POST   /api/streams/{slot}/reset          # zero this slot's counter
GET    /api/streams/{slot}/feed           # MJPEG
```

**SlotConfig** body:

```json
{
  "source": { "type": "rtsp", "url": "rtsp://user:pass@10.0.0.13:554/cam" },
  "direction": "lr",
  "roi_position": 0.5,
  "confidence": 0.3,
  "enabled": true,
  "count_on_start": true
}
```

Or with a file source:

```json
{
  "source": { "type": "file", "path": "app/uploads/slot3_test.mp4", "filename": "test.mp4" }
}
```

#### Configure slot 3 (left→right), connect, start counting

```bash
curl -X PUT http://localhost:5590/api/streams/3/config \
  -H 'Content-Type: application/json' \
  -d '{
    "source": {"type": "rtsp", "url": "rtsp://user:pw@10.0.0.13:554/cam"},
    "direction": "lr",
    "roi_position": 0.5,
    "confidence": 0.3,
    "enabled": true,
    "count_on_start": true
  }'
curl -X POST http://localhost:5590/api/streams/3/start
```

```python
import requests
BASE = "http://localhost:5590"
requests.put(f"{BASE}/api/streams/3/config", json={
    "source": {"type": "rtsp", "url": "rtsp://user:pw@10.0.0.13:554/cam"},
    "direction": "lr", "roi_position": 0.5, "confidence": 0.3,
    "enabled": True, "count_on_start": True,
})
requests.post(f"{BASE}/api/streams/3/start")
```

```javascript
const BASE = "http://localhost:5590";
await fetch(`${BASE}/api/streams/3/config`, {
  method: "PUT",
  headers: {"Content-Type": "application/json"},
  body: JSON.stringify({
    source: {type: "rtsp", url: "rtsp://user:pw@10.0.0.13:554/cam"},
    direction: "lr", roi_position: 0.5, confidence: 0.3,
    enabled: true, count_on_start: true,
  }),
});
await fetch(`${BASE}/api/streams/3/start`, {method: "POST"});
```

#### Upload a video file as the source for slot 5

```bash
curl -X POST http://localhost:5590/api/streams/5/upload \
  -F file=@benchmark.mp4
curl -X PATCH http://localhost:5590/api/streams/5/config \
  -H 'Content-Type: application/json' \
  -d '{"direction": "tb", "roi_position": 0.7, "enabled": true, "count_on_start": true}'
curl -X POST http://localhost:5590/api/streams/5/start
```

```python
import requests
BASE = "http://localhost:5590"
requests.post(f"{BASE}/api/streams/5/upload",
              files={"file": open("benchmark.mp4", "rb")})
requests.patch(f"{BASE}/api/streams/5/config",
               json={"direction": "tb", "roi_position": 0.7,
                     "enabled": True, "count_on_start": True})
requests.post(f"{BASE}/api/streams/5/start")
```

```javascript
const fd = new FormData();
fd.append("file", file);  // a File from <input type="file">
await fetch("/api/streams/5/upload", {method: "POST", body: fd});
await fetch("/api/streams/5/config", {
  method: "PATCH",
  headers: {"Content-Type": "application/json"},
  body: JSON.stringify({direction: "tb", roi_position: 0.7,
                        enabled: true, count_on_start: true}),
});
await fetch("/api/streams/5/start", {method: "POST"});
```

#### Poll all 10 slots in one round-trip

```bash
curl -s http://localhost:5590/api/streams/status | python -m json.tool
```

```python
import requests, time
while True:
    s = requests.get("http://localhost:5590/api/streams/status").json()
    for slot, runtime in s.items():
        print(slot, runtime["egg_count"], runtime["fps"])
    time.sleep(1)
```

```javascript
setInterval(async () => {
  const s = await (await fetch("/api/streams/status")).json();
  for (const [slot, runtime] of Object.entries(s)) {
    console.log(slot, runtime.egg_count, runtime.fps);
  }
}, 1000);
```

### Configuration

```
GET    /api/config                       # current live settings (globals)
PATCH  /api/config                       # update settings; no restart
```

### TensorRT export

```
POST   /api/export/tensorrt              # start background TensorRT export
GET    /api/export/tensorrt              # poll status; auto-switches model_path on DONE
```

Interactive Swagger docs: `http://localhost:5590/docs`.

---

## Building your own client

### Polling pattern

For dashboards monitoring all 10 slots, poll **`GET /api/streams/status`** once per second. It returns all 10 slots in one round-trip — cheap and complete. The grid wall in the bundled UI uses this exact pattern.

For a single focused slot, poll `GET /api/streams/{slot}/status` at the same cadence.

### Embedding MJPEG in your own UI

| Platform | Pattern |
|----------|---------|
| **Browser / Electron** | `<img src="http://host:5590/api/streams/3/feed">` — the browser handles `multipart/x-mixed-replace` natively. |
| **Qt (C++ / PyQt / PySide)** | Use `QNetworkAccessManager` to fetch the URL and parse multipart boundaries manually into `QImage`. Easiest: drop a `QWebEngineView` pointed at the URL. |
| **.NET / WPF / WinForms** | `WebView2` pointed at the feed URL is the simplest path. Raw multipart parsing is possible with `HttpClient` but tedious. |
| **Mobile (iOS / Android)** | `WKWebView` / `WebView` with the feed URL works out of the box. |

### Error handling and reconnects

`status.error` is the canonical place to surface RTSP / decode failures. The server retries RTSP open with `STREAM_RECONNECT_BACKOFF_S` between attempts — clients should not aggressively re-POST `/start`. A slot with `is_complete: true` is a finished file source; calling `/start` again rewinds it.

### Persistence and safe upgrades

`app/data/streams.json` is the source of truth for slot configuration. A `docker pull && docker compose up -d` preserves it because it lives in the mounted volume on the host. Editing the file directly is supported — but the server only re-reads on container restart.

### Production tips

- Put the service behind a reverse proxy (Nginx example below) and terminate TLS there.
- For aggressive deployments, set `STREAM_BATCH_INTERVAL_MS` lower (e.g. `16` ≈ 60 fps cap) if your GPU has headroom.
- The model is loaded once at startup. Switching `model_path` at runtime via `PATCH /api/config` reloads it synchronously and rolls back on failure.

---

## Conveyor direction

| Direction | ROI line | `roi_position = 0.7` means | Counts when |
|-----------|----------|-----------------------------|-------------|
| `tb` (top → bottom) | Horizontal | 70 % from the top of the frame | Centroid moves down across the line |
| `lr` (left → right) | Vertical | 70 % from the left of the frame | Centroid moves rightwards across the line |

Direction is fixed per slot / per video session — it cannot flip mid-stream (the counter would lose state). For a belt that runs both ways, use two slots pointed at the same camera with different directions and ROI positions.

---

## CLI (`detect_and_count.py`)

```bash
python detect_and_count.py image.jpg                       # detect
python detect_and_count.py video.mp4                       # count (default direction tb)
python detect_and_count.py video.mp4 --direction lr        # left→right
python detect_and_count.py rtsp://user:pw@host:554/cam --conf 0.3 --roi 0.5
```

| Argument | Default | Description |
|----------|---------|-------------|
| `input` | (required) | Path to image, video, or RTSP URL |
| `--model` | `best.pt` | YOLOv8 weights |
| `--conf` | `0.25` | Detection confidence threshold |
| `--direction` | `tb` | `tb` (top → bottom) or `lr` (left → right) |
| `--roi` | `0.7` | ROI position along direction-of-travel (0..1) |
| `--max-distance` | `40` | Tracker match max pixel distance |
| `--max-disappeared` | `50` | Frames before dropping a lost track |
| `--save` | none | Output path for annotated result |

---

## Deployment

### Behind Nginx

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:5590;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_buffering off;           # required for MJPEG
        proxy_cache off;
        proxy_read_timeout 3600s;
    }
}
```

### Cloud (AWS / GCP)

1. Launch a GPU instance (e.g. `g4dn.xlarge` on AWS, `n1-standard-4 + T4` on GCP).
2. Install NVIDIA drivers and Docker.
3. Clone the repo, drop `best.pt` into the root.
4. `docker compose up -d`.
5. Open port `5590` in the firewall.

---

## Configuration reference

All settings are env vars, set in `.env`, or patched live via `PATCH /api/config`:

| Variable | Default | Notes |
|----------|---------|-------|
| `MODEL_PATH` | `best.pt` | YOLOv8 weights file |
| `RTSP_URL` | (empty) | First-boot only — seeds slot 1 if `streams.json` is absent |
| `ROI_POSITION` | `0.7` | Default ROI fraction (used when a slot doesn't specify) |
| `CONFIDENCE` | `0.25` | Default detection threshold |
| `NMS_IOU` | `0.45` | NMS IoU threshold |
| `IMGSZ` | `640` | YOLOv8 inference image size (multiple of 32) |
| `MAX_DISTANCE` | `40` | Tracker centroid match distance |
| `MAX_DISAPPEARED` | `50` | Frames before dropping a lost track |
| `STREAM_BATCH_INTERVAL_MS` | `33` | Batch scheduler tick (~30 fps cap) |
| `STREAM_RECONNECT_BACKOFF_S` | `5` | Seconds between RTSP reconnect attempts |

---

## Project structure

```
Automated-Egg-Counting-System/
├── app/
│   ├── main.py                          # FastAPI entrypoint; scheduler lifecycle
│   ├── config.py                        # Pydantic settings
│   ├── core/
│   │   ├── detector.py                  # YOLOv8 inference wrapper
│   │   ├── model_cache.py               # Singleton model loader
│   │   ├── tracker.py                   # Bbox-aware centroid tracker
│   │   ├── counter.py                   # Tracker-based direction-aware counter
│   │   ├── line_counter.py              # Trackerless direction-aware counter
│   │   ├── annotator.py                 # Frame annotation
│   │   ├── video_processor.py           # Single-video processor (used by /api/video)
│   │   ├── stream_manager.py            # 10-slot registry + streams.json persistence
│   │   ├── stream_slot.py               # Per-slot capture thread + state
│   │   ├── batch_scheduler.py           # Single batched-inference thread
│   │   ├── runtime_config.py            # Live runtime config
│   │   └── exporter.py                  # TensorRT export
│   ├── routers/
│   │   ├── image.py                     # /api/image
│   │   ├── video.py                     # /api/video (single-video)
│   │   ├── streams.py                   # /api/streams (10-slot)
│   │   ├── config_router.py             # /api/config
│   │   ├── export_router.py             # /api/export/tensorrt
│   │   └── health_router.py             # /health
│   ├── static/
│   │   ├── index.html / video.html / streams.html
│   │   └── js/, css/
│   ├── data/                            # streams.json + slot-owned files (volume-mounted)
│   ├── uploads/, outputs/               # video upload + H.264 output dirs
├── detect_and_count.py                  # Standalone CLI
├── best.pt                              # YOLOv8 weights
├── tests/                               # pytest suite
├── requirements.txt / requirements-cpu.txt / requirements-dev.txt
├── Dockerfile / Dockerfile.cpu / docker-compose.yml
└── README.md
```

---

## License

See [LICENSE](LICENSE).
