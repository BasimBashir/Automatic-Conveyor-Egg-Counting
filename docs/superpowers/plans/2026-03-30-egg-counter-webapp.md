# Egg Counter Web App — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a FastAPI web app with REST API and multi-page HTML/CSS/JS frontend for egg detection on images, video processing with counting, and live RTSP stream monitoring — all in a single Docker container.

**Architecture:** Monolith FastAPI server. Existing detection/tracking/annotation code from `detect_and_count.py` gets extracted into focused `app/core/` modules. Background threads run `VideoProcessor` instances for video/stream sessions. MJPEG streaming via `StreamingResponse`. ffmpeg re-encodes output to H.264 for social media compatibility.

**Tech Stack:** Python 3.11, FastAPI, Uvicorn, PyTorch, YOLOv5, OpenCV, ffmpeg, Docker

---

## File Map

```
app/
├── __init__.py
├── main.py                  # FastAPI app: mount static, include routers, load model on startup
├── config.py                # Pydantic Settings from .env
├── core/
│   ├── __init__.py
│   ├── detector.py          # load_model(), detect_frame() — single-frame inference
│   ├── tracker.py           # CentroidTracker class (extracted from detect_and_count.py)
│   ├── counter.py           # EggCounter: wraps tracker + ROI crossing + trails + flashes
│   ├── annotator.py         # All draw_* functions (extracted from detect_and_count.py)
│   └── video_processor.py   # VideoProcessor: background thread, state, MJPEG buffer, save
├── routers/
│   ├── __init__.py
│   ├── image.py             # POST /api/image/detect
│   ├── video.py             # Video upload + session endpoints
│   ├── stream.py            # RTSP stream endpoints
│   └── config_router.py     # GET/PUT /api/config
├── static/
│   ├── index.html
│   ├── video.html
│   ├── stream.html
│   ├── css/
│   │   └── style.css
│   └── js/
│       ├── image.js
│       ├── video.js
│       └── stream.js
├── uploads/                 # Created at startup
├── outputs/                 # Created at startup
.env
requirements.txt             # FastAPI deps added
Dockerfile
docker-compose.yml
```

---

## Task 1: Project scaffold and config

**Files:**
- Create: `app/__init__.py`, `app/core/__init__.py`, `app/routers/__init__.py`
- Create: `app/config.py`
- Create: `.env`
- Modify: `requirements.txt`

- [ ] **Step 1: Create package init files**

Create empty `__init__.py` files:
- `app/__init__.py`
- `app/core/__init__.py`
- `app/routers/__init__.py`

- [ ] **Step 2: Create app/config.py**

```python
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    rtsp_url: str = ""
    model_path: str = "best.pt"
    roi_position: float = 0.7
    confidence: float = 0.25
    max_distance: int = 40
    max_disappeared: int = 50
    upload_dir: str = "app/uploads"
    output_dir: str = "app/outputs"

    class Config:
        env_file = ".env"


settings = Settings()
```

- [ ] **Step 3: Create .env**

```env
RTSP_URL=
MODEL_PATH=best.pt
ROI_POSITION=0.7
CONFIDENCE=0.25
MAX_DISTANCE=40
MAX_DISAPPEARED=50
```

- [ ] **Step 4: Update requirements.txt**

```
opencv-python-headless
numpy
pillow
ultralytics
pandas
tqdm
seaborn
gitpython
fastapi
uvicorn[standard]
python-multipart
pydantic-settings
python-dotenv
```

- [ ] **Step 5: Install dependencies**

Run: `pip install -r requirements.txt`

- [ ] **Step 6: Commit**

```bash
git add app/__init__.py app/core/__init__.py app/routers/__init__.py app/config.py .env requirements.txt
git commit -m "feat: scaffold app structure and config"
```

---

## Task 2: Extract core modules from detect_and_count.py

**Files:**
- Create: `app/core/tracker.py`
- Create: `app/core/annotator.py`
- Create: `app/core/detector.py`
- Create: `app/core/counter.py`

- [ ] **Step 1: Create app/core/tracker.py**

Extract the `CentroidTracker` class from `detect_and_count.py` lines 34-99. Copy it exactly as-is:

```python
import numpy as np
from collections import OrderedDict


class CentroidTracker:
    """Centroid-based object tracker for counting objects crossing a line."""

    def __init__(self, max_disappeared=50, max_distance=40):
        self.next_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def update(self, detections):
        if len(detections) == 0:
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    del self.objects[obj_id]
                    del self.disappeared[obj_id]
            return self.objects

        if len(self.objects) == 0:
            for det in detections:
                self._register(det)
            return self.objects

        obj_ids = list(self.objects.keys())
        obj_centroids = np.array(list(self.objects.values()))
        det_centroids = np.array(detections)

        diff = obj_centroids[:, np.newaxis, :] - det_centroids[np.newaxis, :, :]
        dist_matrix = np.sqrt((diff ** 2).sum(axis=2))

        rows = dist_matrix.min(axis=1).argsort()
        cols = dist_matrix.argmin(axis=1)[rows]

        used_rows = set()
        used_cols = set()

        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            if dist_matrix[row, col] > self.max_distance:
                continue
            obj_id = obj_ids[row]
            self.objects[obj_id] = tuple(det_centroids[col])
            self.disappeared[obj_id] = 0
            used_rows.add(row)
            used_cols.add(col)

        for row in range(len(obj_centroids)):
            if row not in used_rows:
                obj_id = obj_ids[row]
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    del self.objects[obj_id]
                    del self.disappeared[obj_id]

        for col in range(len(det_centroids)):
            if col not in used_cols:
                self._register(tuple(det_centroids[col]))

        return self.objects

    def _register(self, centroid):
        self.objects[self.next_id] = centroid
        self.disappeared[self.next_id] = 0
        self.next_id += 1

    def reset(self):
        """Reset all tracking state."""
        self.next_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
```

- [ ] **Step 2: Create app/core/annotator.py**

Extract all drawing functions and `COLORS` dict from `detect_and_count.py` lines 15-269:

```python
import cv2
import numpy as np

COLORS = {
    "panel_bg":     (20, 20, 20),
    "panel_border": (60, 60, 60),
    "accent":       (0, 200, 255),
    "counted":      (0, 230, 118),
    "uncounted":    (255, 180, 0),
    "roi_line":     (80, 80, 255),
    "roi_glow":     (60, 60, 200),
    "flash":        (0, 255, 255),
    "bbox":         (255, 170, 50),
    "bbox_counted": (0, 200, 100),
    "trail":        (200, 120, 0),
    "white":        (255, 255, 255),
    "dim":          (160, 160, 160),
    "very_dim":     (100, 100, 100),
}


def draw_rounded_rect(img, pt1, pt2, color, radius=12, thickness=-1, alpha=0.85):
    overlay = img.copy()
    x1, y1 = pt1
    x2, y2 = pt2
    r = radius
    cv2.rectangle(overlay, (x1 + r, y1), (x2 - r, y2), color, thickness)
    cv2.rectangle(overlay, (x1, y1 + r), (x2, y2 - r), color, thickness)
    cv2.ellipse(overlay, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
    cv2.ellipse(overlay, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    cv2.ellipse(overlay, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)
    cv2.ellipse(overlay, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def draw_trail(img, points, base_color, max_length=20):
    pts = list(points)
    n = len(pts)
    if n < 2:
        return
    for i in range(1, n):
        alpha = i / n
        thickness = max(1, int(alpha * 3))
        r = int(base_color[0] * alpha)
        g = int(base_color[1] * alpha)
        b = int(base_color[2] * alpha)
        cv2.line(img, pts[i - 1], pts[i], (r, g, b), thickness, cv2.LINE_AA)


def draw_roi_line(img, roi_y, width, frame_num):
    cv2.line(img, (0, roi_y), (width, roi_y), COLORS["roi_glow"], 6, cv2.LINE_AA)
    dash_len = 20
    gap_len = 12
    offset = (frame_num * 2) % (dash_len + gap_len)
    x = -offset
    while x < width:
        x1 = max(0, x)
        x2 = min(width, x + dash_len)
        if x2 > x1:
            cv2.line(img, (x1, roi_y), (x2, roi_y), COLORS["roi_line"], 2, cv2.LINE_AA)
        x += dash_len + gap_len
    arrow_spacing = 120
    for ax in range(arrow_spacing // 2, width, arrow_spacing):
        cv2.arrowedLine(
            img, (ax, roi_y - 10), (ax, roi_y + 10),
            COLORS["roi_line"], 2, cv2.LINE_AA, tipLength=0.5
        )


def draw_crossing_flash(img, cx, cy, intensity):
    radius = int(20 + 15 * intensity)
    alpha = intensity * 0.5
    overlay = img.copy()
    cv2.circle(overlay, (cx, cy), radius, COLORS["flash"], -1, cv2.LINE_AA)
    cv2.circle(overlay, (cx, cy), radius + 4, COLORS["counted"], 2, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def draw_dashboard(img, total_count, in_frame, total_tracked, frame_num,
                   total_frames, is_stream, fps_display, width):
    panel_w = 300
    panel_h = 130
    margin = 8
    draw_rounded_rect(
        img, (margin, margin), (margin + panel_w, margin + panel_h),
        COLORS["panel_bg"], radius=10, alpha=0.88
    )
    cv2.line(img, (margin + 10, margin + 2), (margin + panel_w - 10, margin + 2),
             COLORS["accent"], 2, cv2.LINE_AA)
    x0 = margin + 14
    y0 = margin + 30
    count_text = f"{total_count}"
    cv2.putText(img, count_text, (x0, y0 + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, COLORS["counted"], 3, cv2.LINE_AA)
    tw = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 1.4, 3)[0][0]
    cv2.putText(img, "eggs counted", (x0 + tw + 8, y0 + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLORS["dim"], 1, cv2.LINE_AA)
    cv2.line(img, (x0, y0 + 18), (x0 + panel_w - 30, y0 + 18),
             COLORS["panel_border"], 1, cv2.LINE_AA)
    y1 = y0 + 42
    cv2.putText(img, f"In Frame: {in_frame}", (x0, y1),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, COLORS["white"], 1, cv2.LINE_AA)
    cv2.putText(img, f"Tracked: {total_tracked}", (x0 + 140, y1),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, COLORS["accent"], 1, cv2.LINE_AA)
    y2 = y1 + 24
    if is_stream:
        frame_text = f"Frame: {frame_num}"
    else:
        pct = (frame_num / total_frames * 100) if total_frames > 0 else 0
        frame_text = f"Frame: {frame_num}/{total_frames} ({pct:.0f}%)"
    cv2.putText(img, frame_text, (x0, y2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLORS["very_dim"], 1, cv2.LINE_AA)
    cv2.putText(img, f"FPS: {fps_display:.0f}", (x0 + 200, y2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLORS["very_dim"], 1, cv2.LINE_AA)
    if not is_stream and total_frames > 0:
        y3 = y2 + 18
        bar_x1 = x0
        bar_x2 = x0 + panel_w - 30
        bar_w = bar_x2 - bar_x1
        progress = frame_num / total_frames
        cv2.rectangle(img, (bar_x1, y3), (bar_x2, y3 + 4), COLORS["panel_border"], -1)
        cv2.rectangle(img, (bar_x1, y3), (bar_x1 + int(bar_w * progress), y3 + 4),
                      COLORS["accent"], -1)


def draw_bbox(img, x1, y1, x2, y2, counted, conf):
    color = COLORS["bbox_counted"] if counted else COLORS["bbox"]
    corner_len = 8
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
    cv2.line(img, (x1, y1), (x1 + corner_len, y1), color, 2, cv2.LINE_AA)
    cv2.line(img, (x1, y1), (x1, y1 + corner_len), color, 2, cv2.LINE_AA)
    cv2.line(img, (x2, y1), (x2 - corner_len, y1), color, 2, cv2.LINE_AA)
    cv2.line(img, (x2, y1), (x2, y1 + corner_len), color, 2, cv2.LINE_AA)
    cv2.line(img, (x1, y2), (x1 + corner_len, y2), color, 2, cv2.LINE_AA)
    cv2.line(img, (x1, y2), (x1, y2 - corner_len), color, 2, cv2.LINE_AA)
    cv2.line(img, (x2, y2), (x2 - corner_len, y2), color, 2, cv2.LINE_AA)
    cv2.line(img, (x2, y2), (x2, y2 - corner_len), color, 2, cv2.LINE_AA)
    label = f"{conf:.0%}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
    cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 6, y1), color, -1)
    cv2.putText(img, label, (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1, cv2.LINE_AA)


def annotate_detections(frame, detections, objects, counted_ids, trails,
                        flash_events, roi_y, frame_num, total_count,
                        total_frames, is_stream, fps_display):
    """Full-frame annotation: bboxes, trails, ROI line, flashes, dashboard."""
    annotated = frame.copy()
    height, width = annotated.shape[:2]

    # 1. Motion trails
    for obj_id, trail_pts in trails.items():
        if obj_id in counted_ids:
            draw_trail(annotated, trail_pts, COLORS["counted"])
        else:
            draw_trail(annotated, trail_pts, COLORS["trail"])

    # 2. Bounding boxes
    for info in detections:
        cx = (info["x1"] + info["x2"]) // 2
        cy = (info["y1"] + info["y2"]) // 2
        is_counted = False
        for obj_id, (ox, oy) in objects.items():
            if abs(ox - cx) < 5 and abs(oy - cy) < 5:
                is_counted = obj_id in counted_ids
                break
        draw_bbox(annotated, info["x1"], info["y1"],
                  info["x2"], info["y2"], is_counted, info["conf"])

    # 3. ROI line
    if roi_y is not None:
        draw_roi_line(annotated, roi_y, width, frame_num)

    # 4. Crossing flashes
    active_flashes = []
    for (fx, fy, f_start) in flash_events:
        age = frame_num - f_start
        if age < 12:
            intensity = 1.0 - (age / 12.0)
            draw_crossing_flash(annotated, fx, fy, intensity)
            active_flashes.append((fx, fy, f_start))
    flash_events.clear()
    flash_events.extend(active_flashes)

    # 5. Centroid dots
    for obj_id, (cx, cy) in objects.items():
        color = COLORS["counted"] if obj_id in counted_ids else COLORS["uncounted"]
        cv2.circle(annotated, (int(cx), int(cy)), 4, color, -1, cv2.LINE_AA)
        cv2.circle(annotated, (int(cx), int(cy)), 6, color, 1, cv2.LINE_AA)

    # 6. Dashboard
    tracker_total = max(len(objects), 0)
    draw_dashboard(annotated, total_count, len(objects), tracker_total,
                   frame_num, total_frames, is_stream, fps_display, width)

    # 7. ROI label
    if roi_y is not None:
        cv2.putText(annotated, "COUNTING LINE", (width - 200, roi_y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["roi_line"], 2, cv2.LINE_AA)

    return annotated


def annotate_image_detections(frame, det_info):
    """Annotate a single image with detection boxes and count overlay."""
    annotated = frame.copy()
    egg_count = len(det_info)

    for info in det_info:
        draw_bbox(annotated, info["x1"], info["y1"],
                  info["x2"], info["y2"], counted=False, conf=info["conf"])
        cx = (info["x1"] + info["x2"]) // 2
        cy = (info["y1"] + info["y2"]) // 2
        cv2.circle(annotated, (cx, cy), 4, COLORS["accent"], -1, cv2.LINE_AA)

    draw_rounded_rect(annotated, (8, 8), (250, 55), COLORS["panel_bg"], radius=8, alpha=0.85)
    cv2.putText(annotated, f"{egg_count}", (18, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, COLORS["counted"], 3, cv2.LINE_AA)
    tw = cv2.getTextSize(f"{egg_count}", cv2.FONT_HERSHEY_SIMPLEX, 1.3, 3)[0][0]
    cv2.putText(annotated, "eggs detected", (18 + tw + 8, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLORS["dim"], 1, cv2.LINE_AA)

    return annotated, egg_count
```

- [ ] **Step 3: Create app/core/detector.py**

```python
import torch
import pathlib
import warnings
import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)
pathlib.PosixPath = pathlib.WindowsPath

_model = None


def load_model(model_path: str) -> object:
    """Load YOLOv5 model. Cached as singleton."""
    global _model
    if _model is None:
        _model = torch.hub.load("ultralytics/yolov5", "custom", path=model_path)
    return _model


def detect_frame(model, frame: np.ndarray, conf: float = 0.25) -> list[dict]:
    """Run detection on a single frame. Returns list of detection dicts."""
    model.conf = conf
    results = model(frame)
    detections = results.pandas().xyxy[0]
    det_info = []
    for _, det in detections.iterrows():
        det_info.append({
            "x1": int(det["xmin"]),
            "y1": int(det["ymin"]),
            "x2": int(det["xmax"]),
            "y2": int(det["ymax"]),
            "conf": float(det["confidence"]),
        })
    return det_info
```

- [ ] **Step 4: Create app/core/counter.py**

```python
from collections import deque
from app.core.tracker import CentroidTracker


class EggCounter:
    """Wraps CentroidTracker with ROI line crossing logic, trails, and flash events."""

    def __init__(self, roi_y: int, max_disappeared: int = 50,
                 max_distance: int = 40, trail_length: int = 18):
        self.roi_y = roi_y
        self.tracker = CentroidTracker(max_disappeared=max_disappeared,
                                       max_distance=max_distance)
        self.trail_length = trail_length
        self.total_count = 0
        self.counted_ids = set()
        self.prev_positions = {}
        self.trails = {}
        self.flash_events = []

    def update(self, det_info: list[dict]) -> dict:
        """Process detections for one frame. Returns current tracked objects."""
        centroids = []
        for d in det_info:
            cx = (d["x1"] + d["x2"]) // 2
            cy = (d["y1"] + d["y2"]) // 2
            centroids.append((cx, cy))

        objects = self.tracker.update(centroids)
        active_ids = set(objects.keys())

        # Update trails
        for obj_id, (cx, cy) in objects.items():
            if obj_id not in self.trails:
                self.trails[obj_id] = deque(maxlen=self.trail_length)
            self.trails[obj_id].append((int(cx), int(cy)))

        for old_id in list(self.trails.keys()):
            if old_id not in active_ids:
                del self.trails[old_id]

        # Check line crossings
        for obj_id, (cx, cy) in objects.items():
            if obj_id in self.counted_ids:
                continue
            prev_y = self.prev_positions.get(obj_id)
            if prev_y is not None:
                if (prev_y >= self.roi_y > cy) or (prev_y <= self.roi_y < cy):
                    self.total_count += 1
                    self.counted_ids.add(obj_id)
                    self.flash_events.append((int(cx), int(cy)))
            self.prev_positions[obj_id] = cy

        for old_id in list(self.prev_positions.keys()):
            if old_id not in active_ids:
                del self.prev_positions[old_id]

        return objects

    def reset(self):
        """Reset all counting state."""
        self.tracker.reset()
        self.total_count = 0
        self.counted_ids = set()
        self.prev_positions = {}
        self.trails = {}
        self.flash_events = []
```

- [ ] **Step 5: Commit**

```bash
git add app/core/tracker.py app/core/annotator.py app/core/detector.py app/core/counter.py
git commit -m "feat: extract core modules from detect_and_count.py"
```

---

## Task 3: VideoProcessor class

**Files:**
- Create: `app/core/video_processor.py`

- [ ] **Step 1: Create app/core/video_processor.py**

```python
import cv2
import time
import threading
import subprocess
import os
import numpy as np
from collections import deque

from app.core.detector import detect_frame
from app.core.counter import EggCounter
from app.core.annotator import annotate_detections


class VideoProcessor:
    """Background video/stream processor with independent play/count controls."""

    def __init__(self, source: str, model, roi_y: int, confidence: float = 0.25,
                 max_disappeared: int = 50, max_distance: int = 40,
                 save_raw_path: str = None, is_stream: bool = False):
        self.source = source
        self.model = model
        self.roi_y = roi_y
        self.confidence = confidence
        self.is_stream = is_stream
        self.save_raw_path = save_raw_path

        self.counter = EggCounter(roi_y=roi_y, max_disappeared=max_disappeared,
                                  max_distance=max_distance)

        # State flags
        self.is_playing = False
        self.is_counting = False
        self.frame_num = 0
        self.total_frames = 0
        self.fps_source = 30.0
        self.fps_display = 0.0
        self.is_complete = False
        self.error = None

        # MJPEG buffer
        self._latest_frame = None
        self._frame_lock = threading.Lock()

        # Thread
        self._thread = None
        self._stop_event = threading.Event()

        # Video writer
        self._writer = None

    @property
    def egg_count(self) -> int:
        return self.counter.total_count

    @property
    def latest_frame(self) -> bytes:
        with self._frame_lock:
            return self._latest_frame

    def start(self):
        """Start processing in background thread."""
        if self.is_playing:
            return
        self._stop_event.clear()
        self.is_playing = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop processing."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        self.is_playing = False

    def start_counting(self):
        self.is_counting = True

    def stop_counting(self):
        self.is_counting = False

    def get_status(self) -> dict:
        return {
            "is_playing": self.is_playing,
            "is_counting": self.is_counting,
            "egg_count": self.egg_count,
            "frame_num": self.frame_num,
            "total_frames": self.total_frames,
            "fps": round(self.fps_display, 1),
            "is_complete": self.is_complete,
            "is_stream": self.is_stream,
            "error": self.error,
        }

    def _run(self):
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            self.error = f"Could not open: {self.source}"
            self.is_playing = False
            return

        self.fps_source = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.total_frames <= 0:
            self.is_stream = True

        # Setup writer for saving
        if self.save_raw_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(
                self.save_raw_path, fourcc, self.fps_source, (width, height)
            )

        fps_timer = time.time()
        fps_frame_count = 0
        frame_delay = 1.0 / self.fps_source if not self.is_stream else 0

        while not self._stop_event.is_set():
            frame_start = time.time()
            ret, frame = cap.read()
            if not ret:
                if not self.is_stream:
                    self.is_complete = True
                else:
                    self.error = "Stream connection lost"
                break

            self.frame_num += 1
            fps_frame_count += 1

            elapsed = time.time() - fps_timer
            if elapsed >= 0.5:
                self.fps_display = fps_frame_count / elapsed
                fps_frame_count = 0
                fps_timer = time.time()

            # Detect
            det_info = detect_frame(self.model, frame, self.confidence)

            # Count (if enabled)
            objects = {}
            if self.is_counting:
                objects = self.counter.update(det_info)
            else:
                # Still track for display but don't count crossings
                centroids = [((d["x1"]+d["x2"])//2, (d["y1"]+d["y2"])//2)
                             for d in det_info]
                objects = self.counter.tracker.update(centroids)

            # Annotate
            flash_with_frame = [(fx, fy, self.frame_num - i)
                                for i, (fx, fy) in enumerate(
                                    reversed(self.counter.flash_events[-12:]))]
            annotated = annotate_detections(
                frame=frame,
                detections=det_info,
                objects=objects,
                counted_ids=self.counter.counted_ids if self.is_counting else set(),
                trails=self.counter.trails,
                flash_events=flash_with_frame,
                roi_y=self.roi_y if self.is_counting else None,
                frame_num=self.frame_num,
                total_count=self.counter.total_count if self.is_counting else 0,
                total_frames=self.total_frames,
                is_stream=self.is_stream,
                fps_display=self.fps_display,
            )

            # Save raw
            if self._writer:
                self._writer.write(annotated)

            # Encode to JPEG for MJPEG stream
            _, jpeg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
            with self._frame_lock:
                self._latest_frame = jpeg.tobytes()

            # Pace playback for video files
            if not self.is_stream and frame_delay > 0:
                proc_time = time.time() - frame_start
                wait = frame_delay - proc_time
                if wait > 0:
                    time.sleep(wait)

        cap.release()
        if self._writer:
            self._writer.release()
        self.is_playing = False

    @staticmethod
    def reencode_h264(input_path: str, output_path: str):
        """Re-encode video to H.264 with faststart for social media compatibility."""
        cmd = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-movflags", "+faststart",
            "-an",
            output_path,
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        # Clean up raw temp file
        if os.path.exists(input_path):
            os.remove(input_path)
```

- [ ] **Step 2: Commit**

```bash
git add app/core/video_processor.py
git commit -m "feat: add VideoProcessor with background thread and MJPEG buffer"
```

---

## Task 4: FastAPI main app and config router

**Files:**
- Create: `app/main.py`
- Create: `app/routers/config_router.py`

- [ ] **Step 1: Create app/routers/config_router.py**

```python
from fastapi import APIRouter
from pydantic import BaseModel
from app.config import settings

router = APIRouter(prefix="/api/config", tags=["config"])


class ConfigUpdate(BaseModel):
    roi_position: float | None = None
    confidence: float | None = None
    max_distance: int | None = None
    max_disappeared: int | None = None


@router.get("")
def get_config():
    return {
        "roi_position": settings.roi_position,
        "confidence": settings.confidence,
        "max_distance": settings.max_distance,
        "max_disappeared": settings.max_disappeared,
        "rtsp_url": settings.rtsp_url,
    }


@router.put("")
def update_config(update: ConfigUpdate):
    if update.roi_position is not None:
        settings.roi_position = update.roi_position
    if update.confidence is not None:
        settings.confidence = update.confidence
    if update.max_distance is not None:
        settings.max_distance = update.max_distance
    if update.max_disappeared is not None:
        settings.max_disappeared = update.max_disappeared
    return {"status": "ok", "config": get_config()}
```

- [ ] **Step 2: Create app/main.py**

```python
import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.core.detector import load_model
from app.routers import image, video, stream, config_router

app = FastAPI(title="Egg Counter API", version="1.0.0")

# Ensure directories exist
os.makedirs(settings.upload_dir, exist_ok=True)
os.makedirs(settings.output_dir, exist_ok=True)


@app.on_event("startup")
def startup():
    load_model(settings.model_path)


# Routers
app.include_router(image.router)
app.include_router(video.router)
app.include_router(stream.router)
app.include_router(config_router.router)

# Static files (HTML/CSS/JS) — mounted last so API routes take priority
static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
```

- [ ] **Step 3: Commit**

```bash
git add app/main.py app/routers/config_router.py
git commit -m "feat: add FastAPI main app and config router"
```

---

## Task 5: Image router

**Files:**
- Create: `app/routers/image.py`

- [ ] **Step 1: Create app/routers/image.py**

```python
import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import Response

from app.config import settings
from app.core.detector import load_model, detect_frame
from app.core.annotator import annotate_image_detections

router = APIRouter(prefix="/api/image", tags=["image"])


@router.post("/detect")
async def detect_image(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return Response(content="Invalid image", status_code=400)

    model = load_model(settings.model_path)
    det_info = detect_frame(model, frame, settings.confidence)
    annotated, egg_count = annotate_image_detections(frame, det_info)

    _, jpeg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 95])

    return Response(
        content=jpeg.tobytes(),
        media_type="image/jpeg",
        headers={
            "X-Egg-Count": str(egg_count),
            "Access-Control-Expose-Headers": "X-Egg-Count",
        },
    )
```

- [ ] **Step 2: Commit**

```bash
git add app/routers/image.py
git commit -m "feat: add image detection endpoint"
```

---

## Task 6: Video router

**Files:**
- Create: `app/routers/video.py`

- [ ] **Step 1: Create app/routers/video.py**

```python
import os
import uuid
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, FileResponse

from app.config import settings
from app.core.detector import load_model
from app.core.video_processor import VideoProcessor

router = APIRouter(prefix="/api/video", tags=["video"])

# Active video sessions: session_id -> VideoProcessor
_sessions: dict[str, VideoProcessor] = {}


@router.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    session_id = str(uuid.uuid4())[:8]
    filepath = os.path.join(settings.upload_dir, f"{session_id}_{file.filename}")

    with open(filepath, "wb") as f:
        shutil.copyfileobj(file.file, f)

    height = _get_video_height(filepath)
    roi_y = int(height * settings.roi_position)
    raw_output = os.path.join(settings.output_dir, f"{session_id}_raw.mp4")

    model = load_model(settings.model_path)
    processor = VideoProcessor(
        source=filepath,
        model=model,
        roi_y=roi_y,
        confidence=settings.confidence,
        max_disappeared=settings.max_disappeared,
        max_distance=settings.max_distance,
        save_raw_path=raw_output,
        is_stream=False,
    )
    _sessions[session_id] = processor

    return {"session_id": session_id, "filename": file.filename}


@router.post("/{session_id}/start")
def start_video(session_id: str):
    proc = _get_session(session_id)
    proc.start()
    return {"status": "playing"}


@router.post("/{session_id}/stop")
def stop_video(session_id: str):
    proc = _get_session(session_id)
    proc.stop()
    return {"status": "stopped"}


@router.post("/{session_id}/counting/start")
def start_counting(session_id: str):
    proc = _get_session(session_id)
    proc.start_counting()
    return {"status": "counting"}


@router.post("/{session_id}/counting/stop")
def stop_counting(session_id: str):
    proc = _get_session(session_id)
    proc.stop_counting()
    return {"status": "not_counting"}


@router.get("/{session_id}/feed")
def video_feed(session_id: str):
    proc = _get_session(session_id)
    return StreamingResponse(
        _mjpeg_generator(proc),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.get("/{session_id}/status")
def video_status(session_id: str):
    proc = _get_session(session_id)
    return proc.get_status()


@router.get("/{session_id}/download")
def download_video(session_id: str):
    proc = _get_session(session_id)
    raw_path = proc.save_raw_path
    if not raw_path or not os.path.exists(raw_path):
        raise HTTPException(status_code=404, detail="Output not ready")

    output_path = os.path.join(settings.output_dir, f"{session_id}_output.mp4")
    if not os.path.exists(output_path):
        VideoProcessor.reencode_h264(raw_path, output_path)

    return FileResponse(
        output_path,
        media_type="video/mp4",
        filename=f"egg_count_{session_id}.mp4",
    )


def _get_session(session_id: str) -> VideoProcessor:
    proc = _sessions.get(session_id)
    if proc is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return proc


def _get_video_height(filepath: str) -> int:
    import cv2
    cap = cv2.VideoCapture(filepath)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return height if height > 0 else 384


def _mjpeg_generator(proc: VideoProcessor):
    import time
    while proc.is_playing or not proc.is_complete:
        frame_bytes = proc.latest_frame
        if frame_bytes:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame_bytes
                + b"\r\n"
            )
        time.sleep(0.03)
```

- [ ] **Step 2: Commit**

```bash
git add app/routers/video.py
git commit -m "feat: add video upload, processing, and download endpoints"
```

---

## Task 7: Stream router

**Files:**
- Create: `app/routers/stream.py`

- [ ] **Step 1: Create app/routers/stream.py**

```python
import time
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.config import settings
from app.core.detector import load_model
from app.core.video_processor import VideoProcessor

router = APIRouter(prefix="/api/stream", tags=["stream"])

_processor: VideoProcessor | None = None


class StreamStart(BaseModel):
    url: str | None = None


@router.post("/start")
def start_stream(body: StreamStart = StreamStart()):
    global _processor

    url = body.url or settings.rtsp_url
    if not url:
        raise HTTPException(status_code=400, detail="No RTSP URL provided")

    if _processor and _processor.is_playing:
        _processor.stop()

    model = load_model(settings.model_path)
    # For streams we don't know height upfront, use default
    roi_y = int(384 * settings.roi_position)

    _processor = VideoProcessor(
        source=url,
        model=model,
        roi_y=roi_y,
        confidence=settings.confidence,
        max_disappeared=settings.max_disappeared,
        max_distance=settings.max_distance,
        is_stream=True,
    )
    _processor.start()
    return {"status": "connected", "url": url}


@router.post("/stop")
def stop_stream():
    global _processor
    if _processor:
        _processor.stop()
    return {"status": "disconnected"}


@router.post("/counting/start")
def start_counting():
    if not _processor or not _processor.is_playing:
        raise HTTPException(status_code=400, detail="Stream not active")
    _processor.start_counting()
    return {"status": "counting"}


@router.post("/counting/stop")
def stop_counting():
    if not _processor or not _processor.is_playing:
        raise HTTPException(status_code=400, detail="Stream not active")
    _processor.stop_counting()
    return {"status": "not_counting"}


@router.get("/feed")
def stream_feed():
    if not _processor:
        raise HTTPException(status_code=400, detail="Stream not started")
    return StreamingResponse(
        _mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.get("/status")
def stream_status():
    if not _processor:
        return {
            "is_connected": False,
            "is_counting": False,
            "egg_count": 0,
            "fps": 0,
        }
    status = _processor.get_status()
    return {
        "is_connected": status["is_playing"],
        "is_counting": status["is_counting"],
        "egg_count": status["egg_count"],
        "fps": status["fps"],
        "error": status["error"],
    }


def _mjpeg_generator():
    while _processor and _processor.is_playing:
        frame_bytes = _processor.latest_frame
        if frame_bytes:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame_bytes
                + b"\r\n"
            )
        time.sleep(0.03)
```

- [ ] **Step 2: Commit**

```bash
git add app/routers/stream.py
git commit -m "feat: add RTSP live stream endpoints"
```

---

## Task 8: Frontend — shared CSS

**Files:**
- Create: `app/static/css/style.css`

- [ ] **Step 1: Create app/static/css/style.css**

```css
* { margin: 0; padding: 0; box-sizing: border-box; }

:root {
    --bg: #0f1117;
    --surface: #1a1d27;
    --surface-2: #242836;
    --border: #2e3345;
    --text: #e4e4e7;
    --text-dim: #9ca3af;
    --accent: #00c8ff;
    --green: #00e676;
    --red: #ff5252;
    --amber: #ffc400;
    --radius: 12px;
}

body {
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
}

/* ── Navbar ─────────────────────────────────────────────── */
.navbar {
    display: flex;
    align-items: center;
    gap: 32px;
    padding: 16px 32px;
    background: var(--surface);
    border-bottom: 1px solid var(--border);
}

.navbar .logo {
    font-size: 18px;
    font-weight: 700;
    color: var(--green);
    letter-spacing: -0.5px;
}

.navbar a {
    color: var(--text-dim);
    text-decoration: none;
    font-size: 14px;
    font-weight: 500;
    padding: 6px 16px;
    border-radius: 8px;
    transition: all 0.2s;
}

.navbar a:hover { color: var(--text); background: var(--surface-2); }
.navbar a.active { color: var(--accent); background: var(--surface-2); }

/* ── Layout ─────────────────────────────────────────────── */
.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 32px;
}

.page-title {
    font-size: 24px;
    font-weight: 700;
    margin-bottom: 8px;
}

.page-subtitle {
    color: var(--text-dim);
    font-size: 14px;
    margin-bottom: 32px;
}

/* ── Cards ──────────────────────────────────────────────── */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 24px;
    margin-bottom: 24px;
}

/* ── Buttons ────────────────────────────────────────────── */
.btn {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 10px 20px;
    border: none;
    border-radius: 8px;
    font-size: 14px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s;
}

.btn-primary { background: var(--accent); color: #000; }
.btn-primary:hover { filter: brightness(1.15); }

.btn-success { background: var(--green); color: #000; }
.btn-success:hover { filter: brightness(1.15); }

.btn-danger { background: var(--red); color: #fff; }
.btn-danger:hover { filter: brightness(1.15); }

.btn-outline {
    background: transparent;
    color: var(--text);
    border: 1px solid var(--border);
}
.btn-outline:hover { border-color: var(--accent); color: var(--accent); }

.btn:disabled {
    opacity: 0.4;
    cursor: not-allowed;
    filter: none;
}

/* ── Controls Row ───────────────────────────────────────── */
.controls {
    display: flex;
    gap: 12px;
    align-items: center;
    flex-wrap: wrap;
    margin-bottom: 20px;
}

/* ── Upload Zone ────────────────────────────────────────── */
.upload-zone {
    border: 2px dashed var(--border);
    border-radius: var(--radius);
    padding: 48px;
    text-align: center;
    cursor: pointer;
    transition: all 0.3s;
}

.upload-zone:hover, .upload-zone.dragover {
    border-color: var(--accent);
    background: rgba(0, 200, 255, 0.05);
}

.upload-zone .icon {
    font-size: 48px;
    margin-bottom: 12px;
}

.upload-zone .label {
    font-size: 16px;
    font-weight: 600;
    margin-bottom: 4px;
}

.upload-zone .hint {
    font-size: 13px;
    color: var(--text-dim);
}

/* ── Video Feed ─────────────────────────────────────────── */
.feed-container {
    position: relative;
    background: #000;
    border-radius: var(--radius);
    overflow: hidden;
    min-height: 300px;
    display: flex;
    align-items: center;
    justify-content: center;
}

.feed-container img {
    width: 100%;
    height: auto;
    display: block;
}

.feed-placeholder {
    color: var(--text-dim);
    font-size: 14px;
}

/* ── Stats Panel ────────────────────────────────────────── */
.stats-panel {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 16px;
    margin-top: 20px;
}

.stat {
    background: var(--surface-2);
    border-radius: 10px;
    padding: 16px;
    text-align: center;
}

.stat .value {
    font-size: 32px;
    font-weight: 700;
    line-height: 1;
}

.stat .value.green { color: var(--green); }
.stat .value.accent { color: var(--accent); }
.stat .value.amber { color: var(--amber); }

.stat .label {
    font-size: 12px;
    color: var(--text-dim);
    margin-top: 6px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* ── Status Dot ─────────────────────────────────────────── */
.status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    display: inline-block;
}

.status-dot.active { background: var(--green); box-shadow: 0 0 8px var(--green); }
.status-dot.inactive { background: var(--red); }

/* ── Result Section (Image page) ────────────────────────── */
.result-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 24px;
}

.result-grid img {
    width: 100%;
    border-radius: var(--radius);
    border: 1px solid var(--border);
}

/* ── Settings Row ───────────────────────────────────────── */
.setting-row {
    display: flex;
    align-items: center;
    gap: 16px;
    margin-bottom: 12px;
}

.setting-row label {
    font-size: 13px;
    color: var(--text-dim);
    min-width: 100px;
}

.setting-row input[type="range"] {
    flex: 1;
    accent-color: var(--accent);
}

.setting-row .range-value {
    font-size: 13px;
    font-weight: 600;
    color: var(--accent);
    min-width: 50px;
    text-align: right;
}

/* ── Input Fields ───────────────────────────────────────── */
.input {
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px 14px;
    color: var(--text);
    font-size: 14px;
    width: 100%;
    outline: none;
    transition: border-color 0.2s;
}

.input:focus { border-color: var(--accent); }

/* ── Responsive ─────────────────────────────────────────── */
@media (max-width: 768px) {
    .result-grid { grid-template-columns: 1fr; }
    .navbar { gap: 16px; padding: 12px 16px; }
    .container { padding: 16px; }
}
```

- [ ] **Step 2: Commit**

```bash
git add app/static/css/style.css
git commit -m "feat: add shared dark theme CSS"
```

---

## Task 9: Frontend — Image page

**Files:**
- Create: `app/static/index.html`
- Create: `app/static/js/image.js`

- [ ] **Step 1: Create app/static/index.html**

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Egg Counter — Image Detection</title>
    <link rel="stylesheet" href="/css/style.css">
</head>
<body>
    <nav class="navbar">
        <span class="logo">EggCounter</span>
        <a href="/" class="active">Image</a>
        <a href="/video.html">Video</a>
        <a href="/stream.html">Stream</a>
    </nav>

    <div class="container">
        <h1 class="page-title">Image Detection</h1>
        <p class="page-subtitle">Upload an image to detect and count eggs</p>

        <div class="card">
            <div class="upload-zone" id="uploadZone">
                <div class="icon">&#128444;</div>
                <div class="label">Drop image here or click to browse</div>
                <div class="hint">JPG, PNG, BMP, WEBP</div>
                <input type="file" id="fileInput" accept="image/*" hidden>
            </div>
        </div>

        <div id="results" style="display:none;">
            <div class="stats-panel" style="margin-bottom: 24px;">
                <div class="stat">
                    <div class="value green" id="eggCount">0</div>
                    <div class="label">Eggs Detected</div>
                </div>
            </div>

            <div class="result-grid">
                <div>
                    <h3 style="margin-bottom: 12px; color: var(--text-dim);">Original</h3>
                    <img id="originalImg" src="" alt="Original">
                </div>
                <div>
                    <h3 style="margin-bottom: 12px; color: var(--text-dim);">Annotated</h3>
                    <img id="annotatedImg" src="" alt="Annotated">
                </div>
            </div>

            <div class="controls" style="margin-top: 16px;">
                <a id="downloadBtn" class="btn btn-primary" download="annotated.jpg">Download Annotated</a>
            </div>
        </div>
    </div>

    <script src="/js/image.js"></script>
</body>
</html>
```

- [ ] **Step 2: Create app/static/js/image.js**

```javascript
const uploadZone = document.getElementById("uploadZone");
const fileInput = document.getElementById("fileInput");
const results = document.getElementById("results");
const eggCount = document.getElementById("eggCount");
const originalImg = document.getElementById("originalImg");
const annotatedImg = document.getElementById("annotatedImg");
const downloadBtn = document.getElementById("downloadBtn");

uploadZone.addEventListener("click", () => fileInput.click());

uploadZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    uploadZone.classList.add("dragover");
});

uploadZone.addEventListener("dragleave", () => {
    uploadZone.classList.remove("dragover");
});

uploadZone.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadZone.classList.remove("dragover");
    if (e.dataTransfer.files.length) {
        handleFile(e.dataTransfer.files[0]);
    }
});

fileInput.addEventListener("change", () => {
    if (fileInput.files.length) {
        handleFile(fileInput.files[0]);
    }
});

async function handleFile(file) {
    // Show original
    originalImg.src = URL.createObjectURL(file);

    // Upload and detect
    uploadZone.querySelector(".label").textContent = "Processing...";
    const formData = new FormData();
    formData.append("file", file);

    try {
        const resp = await fetch("/api/image/detect", { method: "POST", body: formData });
        const count = resp.headers.get("X-Egg-Count") || "0";
        const blob = await resp.blob();
        const url = URL.createObjectURL(blob);

        annotatedImg.src = url;
        downloadBtn.href = url;
        eggCount.textContent = count;
        results.style.display = "block";
        uploadZone.querySelector(".label").textContent = "Drop another image or click to browse";
    } catch (err) {
        uploadZone.querySelector(".label").textContent = "Error — try again";
        console.error(err);
    }
}
```

- [ ] **Step 3: Commit**

```bash
git add app/static/index.html app/static/js/image.js
git commit -m "feat: add image detection page"
```

---

## Task 10: Frontend — Video page

**Files:**
- Create: `app/static/video.html`
- Create: `app/static/js/video.js`

- [ ] **Step 1: Create app/static/video.html**

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Egg Counter — Video Processing</title>
    <link rel="stylesheet" href="/css/style.css">
</head>
<body>
    <nav class="navbar">
        <span class="logo">EggCounter</span>
        <a href="/">Image</a>
        <a href="/video.html" class="active">Video</a>
        <a href="/stream.html">Stream</a>
    </nav>

    <div class="container">
        <h1 class="page-title">Video Processing</h1>
        <p class="page-subtitle">Upload a video to detect, track, and count eggs crossing the ROI line</p>

        <div class="card" id="uploadCard">
            <div class="upload-zone" id="uploadZone">
                <div class="icon">&#127910;</div>
                <div class="label">Drop video here or click to browse</div>
                <div class="hint">MP4, AVI, MOV, MKV</div>
                <input type="file" id="fileInput" accept="video/*" hidden>
            </div>
        </div>

        <div id="playerSection" style="display:none;">
            <div class="controls">
                <button class="btn btn-success" id="btnPlay">Play</button>
                <button class="btn btn-danger" id="btnStop" disabled>Stop</button>
                <span style="width:1px; height:24px; background:var(--border);"></span>
                <button class="btn btn-primary" id="btnCountStart" disabled>Start Counting</button>
                <button class="btn btn-outline" id="btnCountStop" disabled>Stop Counting</button>
                <span style="flex:1;"></span>
                <span class="status-dot inactive" id="statusDot"></span>
                <span style="font-size:13px; color:var(--text-dim);" id="statusLabel">Stopped</span>
            </div>

            <div class="feed-container">
                <img id="feedImg" style="display:none;" alt="Video Feed">
                <span class="feed-placeholder" id="feedPlaceholder">Click Play to start</span>
            </div>

            <div class="stats-panel">
                <div class="stat">
                    <div class="value green" id="eggCount">0</div>
                    <div class="label">Eggs Counted</div>
                </div>
                <div class="stat">
                    <div class="value accent" id="frameNum">0</div>
                    <div class="label">Frame</div>
                </div>
                <div class="stat">
                    <div class="value amber" id="fpsVal">0</div>
                    <div class="label">FPS</div>
                </div>
            </div>

            <div class="controls" style="margin-top: 20px;">
                <a class="btn btn-primary" id="downloadBtn" style="display:none;" download>Download Output (H.264)</a>
            </div>
        </div>
    </div>

    <script src="/js/video.js"></script>
</body>
</html>
```

- [ ] **Step 2: Create app/static/js/video.js**

```javascript
const uploadZone = document.getElementById("uploadZone");
const fileInput = document.getElementById("fileInput");
const uploadCard = document.getElementById("uploadCard");
const playerSection = document.getElementById("playerSection");

const btnPlay = document.getElementById("btnPlay");
const btnStop = document.getElementById("btnStop");
const btnCountStart = document.getElementById("btnCountStart");
const btnCountStop = document.getElementById("btnCountStop");
const feedImg = document.getElementById("feedImg");
const feedPlaceholder = document.getElementById("feedPlaceholder");
const statusDot = document.getElementById("statusDot");
const statusLabel = document.getElementById("statusLabel");
const eggCount = document.getElementById("eggCount");
const frameNum = document.getElementById("frameNum");
const fpsVal = document.getElementById("fpsVal");
const downloadBtn = document.getElementById("downloadBtn");

let sessionId = null;
let pollInterval = null;

// Upload handlers
uploadZone.addEventListener("click", () => fileInput.click());
uploadZone.addEventListener("dragover", (e) => { e.preventDefault(); uploadZone.classList.add("dragover"); });
uploadZone.addEventListener("dragleave", () => uploadZone.classList.remove("dragover"));
uploadZone.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadZone.classList.remove("dragover");
    if (e.dataTransfer.files.length) handleUpload(e.dataTransfer.files[0]);
});
fileInput.addEventListener("change", () => { if (fileInput.files.length) handleUpload(fileInput.files[0]); });

async function handleUpload(file) {
    uploadZone.querySelector(".label").textContent = "Uploading...";
    const formData = new FormData();
    formData.append("file", file);

    try {
        const resp = await fetch("/api/video/upload", { method: "POST", body: formData });
        const data = await resp.json();
        sessionId = data.session_id;
        uploadCard.style.display = "none";
        playerSection.style.display = "block";
    } catch (err) {
        uploadZone.querySelector(".label").textContent = "Upload failed — try again";
        console.error(err);
    }
}

// Controls
btnPlay.addEventListener("click", async () => {
    await fetch(`/api/video/${sessionId}/start`, { method: "POST" });
    feedImg.src = `/api/video/${sessionId}/feed?t=${Date.now()}`;
    feedImg.style.display = "block";
    feedPlaceholder.style.display = "none";
    btnPlay.disabled = true;
    btnStop.disabled = false;
    btnCountStart.disabled = false;
    setStatus(true, "Playing");
    startPolling();
});

btnStop.addEventListener("click", async () => {
    await fetch(`/api/video/${sessionId}/stop`, { method: "POST" });
    btnPlay.disabled = false;
    btnStop.disabled = true;
    btnCountStart.disabled = true;
    btnCountStop.disabled = true;
    setStatus(false, "Stopped");
    stopPolling();
    showDownload();
});

btnCountStart.addEventListener("click", async () => {
    await fetch(`/api/video/${sessionId}/counting/start`, { method: "POST" });
    btnCountStart.disabled = true;
    btnCountStop.disabled = false;
});

btnCountStop.addEventListener("click", async () => {
    await fetch(`/api/video/${sessionId}/counting/stop`, { method: "POST" });
    btnCountStart.disabled = false;
    btnCountStop.disabled = true;
});

function setStatus(active, text) {
    statusDot.className = `status-dot ${active ? "active" : "inactive"}`;
    statusLabel.textContent = text;
}

function startPolling() {
    pollInterval = setInterval(async () => {
        try {
            const resp = await fetch(`/api/video/${sessionId}/status`);
            const s = resp.ok ? await resp.json() : null;
            if (!s) return;

            eggCount.textContent = s.egg_count;
            frameNum.textContent = s.total_frames > 0
                ? `${s.frame_num}/${s.total_frames}`
                : s.frame_num;
            fpsVal.textContent = s.fps;

            if (s.is_complete) {
                setStatus(false, "Complete");
                btnPlay.disabled = false;
                btnStop.disabled = true;
                btnCountStart.disabled = true;
                btnCountStop.disabled = true;
                stopPolling();
                showDownload();
            }
        } catch (_) {}
    }, 500);
}

function stopPolling() {
    if (pollInterval) { clearInterval(pollInterval); pollInterval = null; }
}

function showDownload() {
    downloadBtn.href = `/api/video/${sessionId}/download`;
    downloadBtn.style.display = "inline-flex";
}
```

- [ ] **Step 3: Commit**

```bash
git add app/static/video.html app/static/js/video.js
git commit -m "feat: add video processing page"
```

---

## Task 11: Frontend — Stream page

**Files:**
- Create: `app/static/stream.html`
- Create: `app/static/js/stream.js`

- [ ] **Step 1: Create app/static/stream.html**

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Egg Counter — Live Stream</title>
    <link rel="stylesheet" href="/css/style.css">
</head>
<body>
    <nav class="navbar">
        <span class="logo">EggCounter</span>
        <a href="/">Image</a>
        <a href="/video.html">Video</a>
        <a href="/stream.html" class="active">Stream</a>
    </nav>

    <div class="container">
        <h1 class="page-title">Live Stream</h1>
        <p class="page-subtitle">Connect to RTSP camera and count eggs in real time</p>

        <div class="card">
            <div style="display:flex; gap:12px; margin-bottom:20px;">
                <input class="input" id="rtspUrl" type="text" placeholder="rtsp://user:pass@camera-ip:554/stream">
                <button class="btn btn-success" id="btnConnect">Connect</button>
                <button class="btn btn-danger" id="btnDisconnect" disabled>Disconnect</button>
            </div>

            <div class="controls">
                <button class="btn btn-primary" id="btnCountStart" disabled>Start Counting</button>
                <button class="btn btn-outline" id="btnCountStop" disabled>Stop Counting</button>
                <span style="flex:1;"></span>
                <span class="status-dot inactive" id="statusDot"></span>
                <span style="font-size:13px; color:var(--text-dim);" id="statusLabel">Disconnected</span>
            </div>
        </div>

        <div class="feed-container">
            <img id="feedImg" style="display:none;" alt="Live Feed">
            <span class="feed-placeholder" id="feedPlaceholder">Connect to stream to begin</span>
        </div>

        <div class="stats-panel">
            <div class="stat">
                <div class="value green" id="eggCount">0</div>
                <div class="label">Eggs Counted</div>
            </div>
            <div class="stat">
                <div class="value amber" id="fpsVal">0</div>
                <div class="label">FPS</div>
            </div>
        </div>

        <div class="card" style="margin-top:24px;">
            <h3 style="margin-bottom:16px; font-size:16px;">Settings</h3>
            <div class="setting-row">
                <label>ROI Position</label>
                <input type="range" id="roiSlider" min="0.1" max="0.9" step="0.05" value="0.7">
                <span class="range-value" id="roiValue">0.70</span>
            </div>
            <div class="setting-row">
                <label>Confidence</label>
                <input type="range" id="confSlider" min="0.1" max="0.9" step="0.05" value="0.25">
                <span class="range-value" id="confValue">0.25</span>
            </div>
        </div>
    </div>

    <script src="/js/stream.js"></script>
</body>
</html>
```

- [ ] **Step 2: Create app/static/js/stream.js**

```javascript
const rtspUrl = document.getElementById("rtspUrl");
const btnConnect = document.getElementById("btnConnect");
const btnDisconnect = document.getElementById("btnDisconnect");
const btnCountStart = document.getElementById("btnCountStart");
const btnCountStop = document.getElementById("btnCountStop");
const feedImg = document.getElementById("feedImg");
const feedPlaceholder = document.getElementById("feedPlaceholder");
const statusDot = document.getElementById("statusDot");
const statusLabel = document.getElementById("statusLabel");
const eggCount = document.getElementById("eggCount");
const fpsVal = document.getElementById("fpsVal");
const roiSlider = document.getElementById("roiSlider");
const roiValue = document.getElementById("roiValue");
const confSlider = document.getElementById("confSlider");
const confValue = document.getElementById("confValue");

let pollInterval = null;

// Load config
fetch("/api/config").then(r => r.json()).then(cfg => {
    if (cfg.rtsp_url) rtspUrl.value = cfg.rtsp_url;
    roiSlider.value = cfg.roi_position;
    roiValue.textContent = cfg.roi_position.toFixed(2);
    confSlider.value = cfg.confidence;
    confValue.textContent = cfg.confidence.toFixed(2);
});

// Stream controls
btnConnect.addEventListener("click", async () => {
    const url = rtspUrl.value.trim();
    const body = url ? { url } : {};
    try {
        const resp = await fetch("/api/stream/start", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
        });
        if (!resp.ok) {
            const err = await resp.json();
            alert(err.detail || "Failed to connect");
            return;
        }
        feedImg.src = `/api/stream/feed?t=${Date.now()}`;
        feedImg.style.display = "block";
        feedPlaceholder.style.display = "none";
        btnConnect.disabled = true;
        btnDisconnect.disabled = false;
        btnCountStart.disabled = false;
        setStatus(true, "Connected");
        startPolling();
    } catch (err) {
        console.error(err);
    }
});

btnDisconnect.addEventListener("click", async () => {
    await fetch("/api/stream/stop", { method: "POST" });
    feedImg.style.display = "none";
    feedPlaceholder.style.display = "block";
    btnConnect.disabled = false;
    btnDisconnect.disabled = true;
    btnCountStart.disabled = true;
    btnCountStop.disabled = true;
    setStatus(false, "Disconnected");
    stopPolling();
});

btnCountStart.addEventListener("click", async () => {
    await fetch("/api/stream/counting/start", { method: "POST" });
    btnCountStart.disabled = true;
    btnCountStop.disabled = false;
});

btnCountStop.addEventListener("click", async () => {
    await fetch("/api/stream/counting/stop", { method: "POST" });
    btnCountStart.disabled = false;
    btnCountStop.disabled = true;
});

// Settings sliders
roiSlider.addEventListener("input", () => {
    roiValue.textContent = parseFloat(roiSlider.value).toFixed(2);
});
roiSlider.addEventListener("change", () => {
    fetch("/api/config", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ roi_position: parseFloat(roiSlider.value) }),
    });
});

confSlider.addEventListener("input", () => {
    confValue.textContent = parseFloat(confSlider.value).toFixed(2);
});
confSlider.addEventListener("change", () => {
    fetch("/api/config", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ confidence: parseFloat(confSlider.value) }),
    });
});

function setStatus(active, text) {
    statusDot.className = `status-dot ${active ? "active" : "inactive"}`;
    statusLabel.textContent = text;
}

function startPolling() {
    pollInterval = setInterval(async () => {
        try {
            const resp = await fetch("/api/stream/status");
            const s = await resp.json();
            eggCount.textContent = s.egg_count;
            fpsVal.textContent = s.fps;
            if (!s.is_connected) {
                setStatus(false, s.error || "Disconnected");
                btnConnect.disabled = false;
                btnDisconnect.disabled = true;
                stopPolling();
            }
        } catch (_) {}
    }, 500);
}

function stopPolling() {
    if (pollInterval) { clearInterval(pollInterval); pollInterval = null; }
}
```

- [ ] **Step 3: Commit**

```bash
git add app/static/stream.html app/static/js/stream.js
git commit -m "feat: add live stream page"
```

---

## Task 12: Dockerfile and docker-compose

**Files:**
- Create: `Dockerfile`
- Create: `docker-compose.yml`
- Create: `.dockerignore`

- [ ] **Step 1: Create Dockerfile**

```dockerfile
FROM python:3.11-slim

# System deps: OpenCV needs libgl, ffmpeg for re-encoding
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY app/ app/
COPY best.pt .
COPY .env .

# Create directories
RUN mkdir -p app/uploads app/outputs

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 2: Create docker-compose.yml**

```yaml
services:
  egg-counter:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./app/uploads:/app/app/uploads
      - ./app/outputs:/app/app/outputs
    environment:
      - RTSP_URL=${RTSP_URL:-}
      - MODEL_PATH=best.pt
      - ROI_POSITION=${ROI_POSITION:-0.7}
      - CONFIDENCE=${CONFIDENCE:-0.25}
      - MAX_DISTANCE=${MAX_DISTANCE:-40}
      - MAX_DISAPPEARED=${MAX_DISAPPEARED:-50}
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped
```

- [ ] **Step 3: Create .dockerignore**

```
.venv
.git
__pycache__
*.pyc
output.mp4
output.mkv
*.egg-info
.claude
docs
app/uploads/*
app/outputs/*
```

- [ ] **Step 4: Commit**

```bash
git add Dockerfile docker-compose.yml .dockerignore
git commit -m "feat: add Docker and docker-compose with GPU support"
```

---

## Task 13: Integration test — run the app locally

- [ ] **Step 1: Start the server**

Run: `cd app && uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload`

Expected: Server starts, model loads on startup, no errors.

- [ ] **Step 2: Test image endpoint**

Run in another terminal:
```bash
curl -X POST http://localhost:8000/api/image/detect -F "file=@test_image.jpg" -o result.jpg -D -
```

Expected: Returns JPEG image with `X-Egg-Count` header.

- [ ] **Step 3: Test web pages**

Open in browser:
- `http://localhost:8000/` — image page loads
- `http://localhost:8000/video.html` — video page loads
- `http://localhost:8000/stream.html` — stream page loads
- `http://localhost:8000/docs` — Swagger UI loads

Expected: All pages render with dark theme, navigation works.

- [ ] **Step 4: Test video upload and processing**

Upload a video through the web UI, click Play, then Start Counting.
Expected: MJPEG feed shows annotated frames, egg count updates live.

- [ ] **Step 5: Test video download**

After stopping, click Download. Open the downloaded file.
Expected: Plays correctly in VLC and shares on WhatsApp without corruption.

- [ ] **Step 6: Final commit**

```bash
git add -A
git commit -m "feat: complete egg counter web app with API, frontend, and Docker"
```
