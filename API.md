# Egg Counter — Integration API

A self-contained reference for developers building clients (dashboards, desktop/mobile apps, line-control HMIs, backends) on top of the **published Docker image** — you do **not** need the source repo.

The service is an HTTP API that detects and counts eggs on a conveyor belt. Detection, tracking, and counting are delegated end-to-end to ultralytics **`solutions.ObjectCounter`** running a single-class (`eggs`) model that is baked into the image. It exposes three ways to consume it:

- **Image** — one-shot detect-and-count on a still.
- **Video** — process an uploaded video file, with a live MJPEG preview and a downloadable annotated result.
- **Streams** — up to **10 fixed slots**, each an RTSP camera or uploaded file, each counted independently in its own thread.

---

## 1. Run the container

The image is on Docker Hub as **`basim123/egg-counter`** (`:latest` = GPU/CUDA, `:cpu` = CPU-only). The egg model ships inside the image — nothing else to download.

### GPU (recommended) — `docker run`

```bash
docker run -d --gpus all --name egg-counter \
  -p 5590:5590 \
  -v egg_engine_cache:/app/engine_cache \
  -v egg_data:/app/app/data \
  --restart unless-stopped \
  basim123/egg-counter:latest
```

### GPU — `docker-compose.yml`

```yaml
services:
  egg-counter:
    image: basim123/egg-counter:latest
    ports: ["5590:5590"]
    volumes:
      - egg_engine_cache:/app/engine_cache   # persists the TensorRT engine
      - egg_data:/app/app/data               # persists the 10 slot configs
    deploy:
      resources:
        reservations:
          devices: [{ driver: nvidia, count: 1, capabilities: [gpu] }]
    restart: unless-stopped
volumes:
  egg_engine_cache:
  egg_data:
```

### CPU

```bash
docker run -d --name egg-counter -p 5590:5590 \
  -v egg_data:/app/app/data --restart unless-stopped \
  basim123/egg-counter:cpu
```

### First boot & TensorRT

On the **first GPU boot**, the container builds a TensorRT engine tuned to the host GPU, caches it to the `engine_cache` volume, and serves inference from it. This takes **~2–6 min on an RTX 3090** (longer on smaller cards); later boots reuse the cache and start in seconds. Watch it with `docker logs -f egg-counter`. The CPU image skips this and runs the `.pt` directly.

Until the engine is ready the API is already up, but counting/inference calls will be slower or briefly error — gate your client on `GET /health` returning `200`.

### Environment variables

Only infrastructure knobs are configurable — detection/counting run at the model's tuned defaults (there is intentionally **no** confidence/IoU/ROI knob).

| Variable | Default | Purpose |
|----------|---------|---------|
| `RTSP_URL` | (empty) | First-boot only: seeds slot 1 if no saved slot config exists |
| `MODEL_PATH` | `best.pt` | Weights path inside the container (auto-promoted to `best.engine` after the TRT build) |
| `STREAM_RECONNECT_BACKOFF_S` | `5` | Base backoff for RTSP reconnect attempts |
| `TRT_AUTO_BUILD` | `1` | Set `0` to skip the TensorRT build and run the `.pt` directly |
| `TRT_HALF` | `true` | Build the engine in FP16 |
| `TRT_IMGSZ` | `640` | Engine input size (matches a 640×480 sub-stream) |

---

## 2. Conventions

- **Base URL:** `http://<host>:5590`
- **Auth:** none. The API is open — run it on a trusted network or behind your own reverse proxy / auth layer for production.
- **Content types:** JSON in/out (`application/json`) unless an endpoint is marked **image** or **MJPEG**. Uploads use `multipart/form-data`.
- **Interactive docs:** Swagger UI at `http://<host>:5590/docs`, OpenAPI JSON at `/openapi.json`.
- **Slots** are integers `1..10`; anything else → `404`.
- **`direction`** is `"tb"` (top→bottom) or `"lr"` (left→right). It orients the counting line, which is always the frame's **center line**. An egg is counted **once**, when its tracked center crosses that line in the travel direction (`tb` → moving down, `lr` → moving right). Multiple/touching eggs crossing together are each counted independently.
- **`egg_count`** in any status payload is the cumulative number of eggs that have crossed in the travel direction since counting started (or since the last reset).
- **Errors:** `400` validation, `404` unknown slot/session, `409` export already running, `422` model load failed, `503` service still starting.

---

## 3. Endpoints

### 3.1 Health

#### `GET /health`
Liveness/readiness probe.
```json
{ "status": "ok", "gpu_available": true, "device_name": "NVIDIA GeForce RTX 3090", "model_path": "best.engine" }
```

### 3.2 Config

#### `GET /api/config`
```json
{ "rtsp_url": "", "model_path": "best.pt", "upload_dir": "app/uploads", "output_dir": "app/outputs" }
```

#### `PATCH /api/config`
Update `rtsp_url` / `model_path` / dirs without restarting. Changing `model_path` reloads the model synchronously (rolls back on failure → `422`).
```bash
curl -X PATCH http://localhost:5590/api/config \
  -H 'Content-Type: application/json' -d '{"model_path":"best.engine"}'
```

### 3.3 Image detection

#### `POST /api/image/detect`
One-shot detect-and-count on a still. Returns the annotated JPEG inline; the count is in a response header.

- **Body** (`multipart/form-data`): `file` — JPG/PNG/BMP/WEBP.
- **Returns `200`:** `image/jpeg`. Header **`X-Egg-Count: <int>`** (CORS-exposed).
- **`400`** if the upload is not a readable image.

```bash
curl -X POST http://localhost:5590/api/image/detect \
  -F file=@tray.jpg -o annotated.jpg -D - | grep -i x-egg-count
```

### 3.4 Video sessions

One-shot processing of an uploaded video. Upload returns a `session_id` used in all subsequent paths. A session runs until the file ends, you `/stop` it, or the container restarts.

| Method & path | Purpose |
|---|---|
| `POST /api/video/upload` | Upload a file → `{ session_id, filename, direction }`. Body: `file` (multipart), `direction` (`tb`\|`lr`, default `tb`). |
| `POST /api/video/{id}/start` | Begin playback → `{ "status": "playing" }` |
| `POST /api/video/{id}/stop` | Stop → `{ "status": "stopped" }` |
| `POST /api/video/{id}/counting/start` | Enable counting → `{ "status": "counting" }` |
| `POST /api/video/{id}/counting/stop` | Pause counting → `{ "status": "not_counting" }` |
| `GET /api/video/{id}/status` | Runtime snapshot (see [§4](#4-data-shapes)) |
| `GET /api/video/{id}/config` | `{ "direction": "tb" }` |
| `PATCH /api/video/{id}/config` | Body `{ "direction": "lr" }`; rebuilds the counter (resets `egg_count`) → `{ "direction": "lr" }` |
| `GET /api/video/{id}/feed` | **MJPEG** live preview (`multipart/x-mixed-replace; boundary=frame`) |
| `GET /api/video/{id}/download` | Annotated **H.264 `video/mp4`**; available after `/stop` or completion (else `404`) |

```bash
SID=$(curl -s -F file=@belt.mp4 -F direction=lr \
  http://localhost:5590/api/video/upload | jq -r .session_id)
curl -X POST http://localhost:5590/api/video/$SID/start
curl -X POST http://localhost:5590/api/video/$SID/counting/start
curl -s http://localhost:5590/api/video/$SID/status        # poll egg_count / progress
curl -X POST http://localhost:5590/api/video/$SID/stop
curl -o counted.mp4 http://localhost:5590/api/video/$SID/download
```

### 3.5 Streams (10 fixed slots)

Ten persistent slots (`1..10`). Each slot's config is saved to disk and survives restarts; each active slot runs its own ObjectCounter in its own thread.

| Method & path | Purpose |
|---|---|
| `GET /api/streams` | All 10 slots → `[{ slot, config, runtime }, …]` |
| `GET /api/streams/status` | Runtime of every slot in one call → `{ "1": {…}, … }` — **the dashboard poll endpoint** |
| `GET /api/streams/{slot}` | One slot → `{ slot, config, runtime }` |
| `GET /api/streams/{slot}/status` | One slot's runtime only |
| `PUT /api/streams/{slot}/config` | Full config (see [§4](#4-data-shapes)); persists; live-applies if running |
| `PATCH /api/streams/{slot}/config` | Partial update; `"source": null` clears the source |
| `POST /api/streams/{slot}/upload` | Attach a video file to the slot (`file`, multipart) → switches source to that file |
| `POST /api/streams/{slot}/start` | Start capture → `{ "status": "started" }` or `{ "status": "already_running" }` |
| `POST /api/streams/{slot}/stop` | Stop → `{ "status": "stopped" }` |
| `POST /api/streams/{slot}/counting/start` | Enable counting (`400` if not running) |
| `POST /api/streams/{slot}/counting/stop` | Pause counting |
| `POST /api/streams/{slot}/reset` | Zero the count + rebuild the counter → `{ "status": "reset" }` |
| `GET /api/streams/{slot}/feed` | **MJPEG** live preview (`400` if the slot isn't running) |

```bash
# Point slot 3 at an RTSP camera, left→right, auto-start counting on boot
curl -X PUT http://localhost:5590/api/streams/3/config \
  -H 'Content-Type: application/json' -d '{
    "source": {"type":"rtsp","url":"rtsp://user:pw@10.0.0.13:554/cam"},
    "direction":"lr", "enabled":true, "count_on_start":true }'
curl -X POST http://localhost:5590/api/streams/3/start
curl -X POST http://localhost:5590/api/streams/3/counting/start
curl -s http://localhost:5590/api/streams/status            # poll all slots
```

### 3.6 TensorRT export (manual)

Build/refresh a TensorRT engine on the running GPU container (the boot-time auto-build covers the common case; use this to rebuild on demand).

- `POST /api/export/tensorrt` — start a background export. Body `{ "half": true, "imgsz": 640 }`. Returns `202` + status; `409` if one is already running.
- `GET /api/export/tensorrt` — poll status; on `state: "DONE"` the service has already switched `model_path` to the new `.engine`.

```json
{ "state": "DONE", "source_model": "best.pt", "output_path": "best.engine",
  "error": null, "started_at": 1718970000.0, "finished_at": 1718970180.0, "elapsed_s": 180.0 }
```
`state` ∈ `IDLE` | `RUNNING` | `DONE` | `FAILED`.

---

## 4. Data shapes

**Runtime status** (from `GET .../status`; stream slots also include `is_connected`, an alias of `is_playing`):
```json
{
  "is_playing": true,
  "is_connected": true,
  "is_counting": true,
  "egg_count": 312,
  "frame_num": 18420,
  "total_frames": 0,
  "fps": 27.4,
  "is_complete": false,
  "is_stream": true,
  "direction": "lr",
  "error": null
}
```
`total_frames` is `0` for live streams; a positive number for finite video files (use it for a progress bar). `error` is a human-readable string when a source drops or fails, else `null`.

**Slot config** (in `GET /api/streams[/{slot}]` and returned by config writes):
```json
{
  "source": { "type": "rtsp", "url": "rtsp://…" },
  "direction": "tb",
  "enabled": false,
  "count_on_start": false
}
```
- `source` is `{"type":"rtsp","url":"…"}`, `{"type":"file","path":"…","filename":"…"}`, or `null`.
- `enabled` — auto-start this slot when the container boots.
- `count_on_start` — begin counting automatically when the slot starts.

---

## 5. Consuming MJPEG feeds

`/feed` endpoints return `multipart/x-mixed-replace; boundary=frame` (annotated JPEG frames: egg boxes + the center counting line; no on-frame numbers — read counts from the status endpoints).

- **HTML / browser / Electron / WebView:** `<img src="http://host:5590/api/streams/1/feed">`.
- **Qt:** `QNetworkAccessManager` + manual multipart parsing into `QImage`, or a `QWebEngineView` pointed at the URL.
- **OpenCV:** `cv2.VideoCapture("http://host:5590/api/streams/1/feed")`.

The frame stream is for display only; **always read the authoritative count from `GET .../status`** (`egg_count`), not from the pixels.

---

## 6. Client recipes

**Python — drive one stream slot and poll the count**
```python
import requests, time
B = "http://localhost:5590"
requests.put(f"{B}/api/streams/1/config", json={
    "source": {"type": "rtsp", "url": "rtsp://user:pw@cam/stream"},
    "direction": "lr", "enabled": True, "count_on_start": True})
requests.post(f"{B}/api/streams/1/start")
requests.post(f"{B}/api/streams/1/counting/start")
while True:
    s = requests.get(f"{B}/api/streams/1/status").json()
    print("eggs:", s["egg_count"], "fps:", s["fps"], "err:", s["error"])
    time.sleep(1)
```

**JavaScript — image detect + live feed**
```javascript
const B = "http://localhost:5590";
const fd = new FormData(); fd.append("file", fileInput.files[0]);
const r = await fetch(`${B}/api/image/detect`, { method: "POST", body: fd });
document.querySelector("#count").textContent = r.headers.get("X-Egg-Count");
document.querySelector("#img").src = URL.createObjectURL(await r.blob());

// live stream feed (slot 1)
document.querySelector("#feed").src = `${B}/api/streams/1/feed`;
// poll the count alongside it
setInterval(async () => {
  const all = await (await fetch(`${B}/api/streams/status`)).json();
  document.querySelector("#live").textContent = all["1"].egg_count;
}, 1000);
```

---

## 7. Notes & gotchas

- **Counting is per crossing.** Each tracked egg is counted exactly once as it crosses the center line in the travel direction; backward motion is ignored. Resetting (`/reset`) or changing direction rebuilds the tracker and zeroes the count.
- **Direction is fixed per slot/session** — it cannot flip mid-run (the tracker would lose state). For a belt that runs both ways, configure two slots on the same camera with different directions.
- **640×480 @ 30fps** sub-stream is the design target (matches the engine's input size). Higher-resolution sources work but rebuild the engine with `TRT_IMGSZ` to match if you change resolution.
- **GPU scaling:** each active stream runs its own inference, so GPU load grows with the number of active streams. An RTX 3090 comfortably handles the 10 slots at the sub-stream resolution.
- **CORS:** the count headers are exposed (`Access-Control-Expose-Headers`). If you call cross-origin from a browser and need more, front the service with a proxy that adds the headers you require.
