# Automated Egg Counting System

End-to-end computer vision API for detecting and counting eggs on conveyor belts. Built on **ultralytics YOLO**. Supports image detection, single-video processing, and **up to 10 concurrent RTSP cameras or video files** in one container — each counted by its own **ultralytics `ObjectCounter`** (detection + tracking + line-crossing). Each conveyor (per stream or per upload) can be configured **top → bottom** or **left → right**.

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
| **Streams** | Up to **10 fixed slots**. Each slot can carry an RTSP URL or an uploaded video file and runs its own `ObjectCounter` (independent detection, tracking, and counting). Slot config (URL, conveyor direction, enable/count-on-start flags) is persisted to disk and survives container restarts. |

Detection + tracking + counting are delegated end-to-end to ultralytics **`solutions.ObjectCounter`** (single-class egg model) at model defaults. The conveyor direction orients the counter's region line — always the frame's center line — and objects are counted as they cross it in the travel direction.

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
docker compose logs -f          # watch the first-boot TensorRT build
```

Opens on `http://localhost:5590`. The `app/data` volume persists slot configuration, and the `engine_cache` volume persists the TensorRT engine, across restarts.

**First boot** builds a TensorRT engine for the host GPU (2–6 min on an RTX 3090; longer on smaller cards), caches it to the `engine_cache` volume, and switches `MODEL_PATH` to it. Later boots reuse the cache and start in seconds. Disable with `TRT_AUTO_BUILD=0` (runs the `.pt` directly; also CPU-safe). See [Deployment → TensorRT](#deployment) for details.

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
  -v egg_engine_cache:/app/engine_cache \
  --restart unless-stopped \
  basim123/egg-counter:latest
```

The `egg_engine_cache` named volume holds the TensorRT engine built on first boot, so it survives `docker rm` / re-creation. PowerShell:

```powershell
docker run -d `
  --gpus all `
  --name egg-counter `
  -p 5590:5590 `
  -v "${PWD}/app/uploads:/app/app/uploads" `
  -v "${PWD}/app/outputs:/app/app/outputs" `
  -v "${PWD}/app/data:/app/app/data" `
  -v egg_engine_cache:/app/engine_cache `
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

> CPU inference is slower; per-stream throughput drops accordingly, and the TensorRT auto-build is skipped (the `.pt` runs directly).

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

Upload an MP4/AVI/MOV/MKV → live MJPEG preview → optionally enable counting → download H.264 result. A **Conveyor direction** toggle (Top→Bottom / Left→Right) is on the upload card; the counting line is the frame's center line, oriented by the chosen direction.

### Streams (`/streams.html`)

10-slot grid wall + drill-down.

1. **Grid wall** — overview of all 10 slots. Each tile shows status, count, FPS, direction arrow, and a thumbnail of the live feed.
2. **Click a tile** to drill in. Pick a source (RTSP URL **or** upload a video file), pick conveyor direction, optionally check **Enabled on boot** and **Auto-count on start**. Click **Save settings**, then **Connect**, then **Start Counting**.

Grid polls `/api/streams/status` every second.

---

## REST API reference

> **Building a client on top of the container?** See **[API.md](API.md)** — a standalone, integration-focused reference (every endpoint, request/response shapes, MJPEG consumption, and curl/Python/JS examples) written for developers running the published Docker image without the source repo.

Base URL: `http://<host>:5590`. All endpoints return JSON unless marked MJPEG / image. Interactive Swagger UI at **`http://localhost:5590/docs`**.

### Conventions

- Slots are integers `1..10`. Other values → `404`.
- `direction` is `"tb"` (top → bottom) or `"lr"` (left → right). It orients the counting line, which is always the frame's center line. There is no ROI/confidence knob — counting runs at ObjectCounter's model defaults.
- All JSON bodies use `Content-Type: application/json`. Multipart bodies use `multipart/form-data`.
- MJPEG endpoints return `multipart/x-mixed-replace; boundary=frame`. Browsers and Electron consume them with `<img src="…">`.
- Common error codes: `400` (validation), `404` (unknown slot/session), `503` (manager not ready during boot).

---

### Health

#### `GET /health`

Service health, GPU availability, and current model path. Use as a liveness probe.

**Returns (`200`):**
```json
{
  "status": "ok",
  "gpu_available": true,
  "device_name": "NVIDIA T4",
  "model_path": "best.pt"
}
```

**Example:**
```bash
curl http://localhost:5590/health
```

---

### Image detection

#### `POST /api/image/detect`

One-shot detect-and-count on a single image. Returns the annotated image inline as JPEG; the count is in an HTTP header.

**Body** (`multipart/form-data`):
- `file` — image file (JPG / PNG / BMP / WEBP).

**Returns (`200`):** `image/jpeg` (annotated). Header `X-Egg-Count: <integer>`.

**Status:** `200` on success; `400` if the upload isn't a readable image.

**Example:**
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
fd.append("file", file);          // a File from <input type="file">
const r = await fetch("/api/image/detect", { method: "POST", body: fd });
const count = r.headers.get("X-Egg-Count");
const blob = await r.blob();
```

---

### Video sessions

One-shot processing of an uploaded video. The upload returns a `session_id`; subsequent calls take that ID in the path. A session keeps running until the file ends, you call `/stop`, or the process restarts.

#### `POST /api/video/upload`

Upload a video file. Creates a session keyed by the returned `session_id`. The file is stored under `app/uploads/{session_id}_<filename>` and an annotated H.264 download is rendered to `app/outputs/` once you `/stop` or the file completes.

**Body** (`multipart/form-data`):
- `file` — video (MP4 / AVI / MOV / MKV).
- `direction` — `"tb"` or `"lr"`. Default `"tb"`.

**Returns (`200`):**
```json
{ "session_id": "7f83de97", "filename": "input.mp4", "direction": "tb" }
```

**Status:** `200` on success; `400` for invalid `direction`.

**Example:**
```bash
curl -X POST http://localhost:5590/api/video/upload \
  -F file=@conveyor.mp4 -F direction=lr
```

#### `GET /api/video/{session_id}/config`

Return the current conveyor direction of the session.

**Returns (`200`):**
```json
{ "direction": "tb" }
```

**Status:** `404` if session unknown.

#### `PATCH /api/video/{session_id}/config`

Change direction **after** upload. Safe to call before or during playback; rebuilds the underlying ObjectCounter, which resets `egg_count` for this session.

**Body** (JSON):
```json
{ "direction": "lr" }
```

**Returns (`200`):** the new `{direction}` echo.

**Status:** `200` on success; `400` for invalid values; `404` if session unknown.

**Example:**
```bash
curl -X PATCH http://localhost:5590/api/video/7f83de97/config \
  -H 'Content-Type: application/json' \
  -d '{"direction":"lr"}'
```

#### `POST /api/video/{session_id}/start`

Begin background playback + inference. Idempotent — calling while already playing is a no-op.

**Returns (`200`):** `{ "status": "playing" }`.

#### `POST /api/video/{session_id}/stop`

Stop playback. Finalizes the H.264 re-encode if one was requested.

**Returns (`200`):** `{ "status": "stopped" }`.

#### `POST /api/video/{session_id}/counting/start`

Enable line-crossing counting. Detections are still produced before this call, but the count only increments while counting is enabled.

**Returns (`200`):** `{ "status": "counting" }`.

#### `POST /api/video/{session_id}/counting/stop`

Disable counting (detections still flow into the live preview).

**Returns (`200`):** `{ "status": "not_counting" }`.

#### `GET /api/video/{session_id}/status`

Per-session runtime snapshot. Poll this once a second from a client to drive a progress UI.

**Returns (`200`):**
```json
{
  "is_playing": true, "is_counting": false, "egg_count": 0,
  "frame_num": 142, "total_frames": 1800, "fps": 27.4,
  "is_complete": false, "is_stream": false,
  "direction": "tb", "error": null
}
```

#### `GET /api/video/{session_id}/feed`

Live MJPEG of annotated frames. Open in `<img src="…">` or with an MJPEG viewer.

**Returns:** `multipart/x-mixed-replace; boundary=frame`. Stream stays open as long as the session is playing.

#### `GET /api/video/{session_id}/download`

Returns the H.264-encoded annotated output as a file download. Only available after the session has stopped (or completed); call `POST /stop` first if needed.

**Returns (`200`):** `video/mp4` (`Content-Disposition: attachment; filename=egg_count_<id>.mp4`).

**Status:** `404` if the output isn't ready yet (session still playing without stopping).

---

### Streams (10-slot multi-stream)

Slot configuration persists across container restarts (lives in `app/data/streams.json`).

#### `GET /api/streams`

List all 10 slots with their config and runtime state. Useful for a dashboard initial render.

**Returns (`200`):** array of length 10.
```json
[
  { "slot": 1, "config": { ... } | null, "runtime": { ... } },
  ...
]
```

#### `GET /api/streams/status`

Aggregate runtime for all 10 slots in one round-trip. **This is the polling endpoint for a multi-stream dashboard** — call once per second.

**Returns (`200`):** map keyed by slot.
```json
{
  "1": { "is_connected": true, "is_counting": true, "egg_count": 312,
         "fps": 27.4, "direction": "tb", "error": null,
         "is_stream": true, "is_complete": false,
         "frame_num": 18420, "total_frames": 0 },
  "2": { ... },
  ...
}
```

**Example:**
```bash
curl -s http://localhost:5590/api/streams/status | python -m json.tool
```

```python
import requests, time
BASE = "http://localhost:5590"
while True:
    s = requests.get(f"{BASE}/api/streams/status").json()
    for slot, r in s.items():
        print(slot, r["egg_count"], r["fps"])
    time.sleep(1)
```

#### `GET /api/streams/{slot}`

Single slot — both config and runtime.

**Path params:** `slot` (1..10).

**Returns (`200`):**
```json
{
  "slot": 3,
  "config": { "source": {...}, "direction": "tb",
              "enabled": true, "count_on_start": false } | null,
  "runtime": { ...same shape as in /api/streams/status... }
}
```

**Status:** `404` for slot < 1 or > 10.

#### `GET /api/streams/{slot}/status`

Runtime-only snapshot for one slot. Smaller payload than `GET /api/streams/{slot}`.

**Returns (`200`):** same `runtime` object as above.

#### `PUT /api/streams/{slot}/config`

Set the slot's **full** config. Replaces any prior config and persists to `streams.json`. If the slot has a capture thread running, the new config is applied immediately (counter is rebuilt → count resets).

**Body** (JSON):
```json
{
  "source": { "type": "rtsp", "url": "rtsp://user:pw@10.0.0.13:554/cam" },
  "direction": "lr",
  "enabled": true,
  "count_on_start": true
}
```

Or a file source:
```json
{
  "source": { "type": "file",
              "path": "app/uploads/slot3_test.mp4",
              "filename": "test.mp4" }
}
```

`source` may be `null` to leave the slot unconfigured.

**Returns (`200`):** the resulting config object.

**Status:** `400` on validation errors (bad direction); `404` if slot ∉ 1..10.

**Example:**
```bash
curl -X PUT http://localhost:5590/api/streams/3/config \
  -H 'Content-Type: application/json' \
  -d '{
    "source": {"type": "rtsp", "url": "rtsp://user:pw@10.0.0.13:554/cam"},
    "direction": "lr", "enabled": true, "count_on_start": true
  }'
```

#### `PATCH /api/streams/{slot}/config`

Partial update. Any field omitted from the body keeps its current value. Setting `"source": null` clears the source.

**Body** (any subset):
```json
{ "direction": "lr", "count_on_start": true }
```

**Returns (`200`):** the full new config.

**Status:** `400` on invalid values; `404` for unknown slot.

#### `POST /api/streams/{slot}/upload`

Attach a local video file to the slot. The file is stored under `app/uploads/slot{N}_<filename>` and the slot's `source` switches to `{"type": "file", "path": "...", "filename": "..."}`. Any previous slot-owned file is deleted.

**Body** (`multipart/form-data`):
- `file` — video.

**Returns (`200`):** the slot's new config.

**Example:**
```bash
curl -X POST http://localhost:5590/api/streams/5/upload \
  -F file=@benchmark.mp4
```

#### `POST /api/streams/{slot}/start`

Open the slot's source and begin capture + inference. The slot's `count_on_start` flag determines whether counting auto-enables.

**Returns (`200`):**
- `{ "status": "started" }` — capture thread launched.
- `{ "status": "already_running" }` — slot was already playing (idempotent; preserves count).

**Status:** `400` if the slot has no source configured.

#### `POST /api/streams/{slot}/stop`

Stop the capture thread. Counter state and last frame are preserved; calling `start` again resumes from scratch.

**Returns (`200`):** `{ "status": "stopped" }`.

#### `POST /api/streams/{slot}/counting/start`

Enable counting on an already-playing slot.

**Returns (`200`):** `{ "status": "counting" }`.

**Status:** `400` if the slot isn't playing.

#### `POST /api/streams/{slot}/counting/stop`

Disable counting (detections still flow).

**Returns (`200`):** `{ "status": "not_counting" }`.

#### `POST /api/streams/{slot}/reset`

Zero the slot's count and clear tracker / trail / flash state. Safe to call any time.

**Returns (`200`):** `{ "status": "reset" }`.

#### `GET /api/streams/{slot}/feed`

Annotated MJPEG live feed for the slot. Embed in `<img src="…">`.

**Returns:** `multipart/x-mixed-replace; boundary=frame`. Closes when the slot stops.

**Status:** `400` if the slot isn't playing.

---

### Configuration (global runtime)

These knobs apply to defaults across the service. Per-slot / per-session settings override them.

#### `GET /api/config`

Return current live settings.

**Returns (`200`):**
```json
{
  "rtsp_url": "", "model_path": "best.pt",
  "upload_dir": "app/uploads", "output_dir": "app/outputs"
}
```

#### `PATCH /api/config`

Update settings without restarting. Changing `model_path` triggers a synchronous model reload (rolls back on failure).

**Body** (any subset of the GET response's fields).

**Returns (`200`):** the full new config.

**Example:**
```bash
curl -X PATCH http://localhost:5590/api/config \
  -H 'Content-Type: application/json' \
  -d '{"model_path": "best.engine"}'
```

---

### TensorRT export

Convert the loaded `.pt` model to an `.engine` (TensorRT) file for faster GPU inference. Requires the GPU image and a compatible NVIDIA driver/toolkit.

#### `POST /api/export/tensorrt`

Start a background export. Idempotent — calling while one is in progress returns the current state.

**Returns (`200`):** `{ "state": "IN_PROGRESS", "progress": 0 }` (or `IDLE` → moves to `IN_PROGRESS`).

#### `GET /api/export/tensorrt`

Poll the export state. When `state == "DONE"` the service has already switched `model_path` to the new `.engine` file.

**Returns (`200`):**
```json
{ "state": "DONE", "progress": 100, "engine_path": "best.engine", "error": null }
```
Possible `state` values: `IDLE`, `IN_PROGRESS`, `DONE`, `ERROR`.

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
- Each active stream runs its own `ObjectCounter` (its own inference + tracker), so GPU load scales with the number of active streams — sized for an RTX 3090 with the TensorRT engine. Keep an eye on GPU utilisation as you approach 10 concurrent streams.
- On first boot on a new GPU the container builds a TensorRT engine and caches it to the `engine_cache` volume (keyed by GPU + TRT version + precision + model hash); later boots reuse it. Disable with `TRT_AUTO_BUILD=0`.
- The model is loaded once at startup. Switching `model_path` at runtime via `PATCH /api/config` reloads it synchronously and rolls back on failure.

---

## Conveyor direction

| Direction | Counting line | Counts when |
|-----------|---------------|-------------|
| `tb` (top → bottom) | Horizontal, frame center (y = height / 2) | A tracked object crosses the line moving **down** |
| `lr` (left → right) | Vertical, frame center (x = width / 2) | A tracked object crosses the line moving **right** |

The counting line is always the frame's center line — counting runs at ultralytics ObjectCounter's model defaults, so there is no ROI or confidence knob. Direction is fixed per slot / per video session; it cannot flip mid-stream (the counter would lose state). For a belt that runs both ways, use two slots pointed at the same camera with different directions.

---

## CLI (`detect_and_count.py`)

```bash
python detect_and_count.py image.jpg                       # detect
python detect_and_count.py video.mp4                       # count (default direction tb)
python detect_and_count.py video.mp4 --direction lr        # left→right
python detect_and_count.py rtsp://user:pw@host:554/cam --direction lr
```

| Argument | Default | Description |
|----------|---------|-------------|
| `input` | (required) | Path to image, video, or RTSP URL |
| `--model` | `best.pt` | ultralytics YOLO weights (`.pt` or `.engine`) |
| `--direction` | `tb` | `tb` (top → bottom) or `lr` (left → right) |
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

Settings are env vars set in `.env` (or the shell / compose). `MODEL_PATH` and `RTSP_URL` can also be patched live via `PATCH /api/config`:

| Variable | Default | Notes |
|----------|---------|-------|
| `MODEL_PATH` | `best.pt` | ultralytics YOLO weights (`.pt` or `.engine`). Promoted to the built `.engine` on first boot when `TRT_AUTO_BUILD=1`. |
| `RTSP_URL` | (empty) | First-boot only — seeds slot 1 if `streams.json` is absent |
| `STREAM_RECONNECT_BACKOFF_S` | `5` | Seconds between RTSP reconnect attempts |
| `TRT_AUTO_BUILD` | `1` | Build a TensorRT engine for the host GPU on first boot and cache it (set `0` to run the `.pt` directly / on CPU) |
| `TRT_HALF` | `true` | Build the engine in FP16 |
| `TRT_IMGSZ` | `640` | Engine input size — must match the size `ObjectCounter` infers at |

---

## Project structure

```
Automated-Egg-Counting-System/
├── app/
│   ├── main.py                          # FastAPI entrypoint; per-slot lifecycle
│   ├── config.py                        # Pydantic settings
│   ├── core/
│   │   ├── detector.py                  # ultralytics YOLO inference (image/preview)
│   │   ├── model_cache.py               # Singleton model loader
│   │   ├── counting.py                  # ObjectCounter region builder (tb/lr) + box extractor
│   │   ├── annotator.py                 # Frame annotation (bboxes + region line)
│   │   ├── video_processor.py           # Per-source ObjectCounter loop (video + each stream slot)
│   │   ├── stream_manager.py            # 10-slot registry + streams.json persistence
│   │   ├── stream_slot.py               # Per-slot wrapper around a VideoProcessor
│   │   ├── runtime_config.py            # Live runtime config
│   │   └── exporter.py                  # TensorRT export (/api/export)
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
├── best.pt                              # ultralytics YOLO weights (retrained yolo26s)
├── docker-entrypoint.sh                 # TensorRT engine auto-build on first boot
├── tests/                               # pytest suite
├── requirements.txt / requirements-cpu.txt / requirements-dev.txt
├── Dockerfile / Dockerfile.cpu / docker-compose.yml
├── API.md                               # standalone client-integration API reference
└── README.md
```

---

## License

See [LICENSE](LICENSE).
