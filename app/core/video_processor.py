import cv2
import time
import threading
import subprocess
import os
from typing import Literal

from app.core.detector import detect_frame
from app.core.counter import EggCounter
from app.core.annotator import annotate_detections


class VideoProcessor:
    """Background video/stream processor with independent play/count controls.

    direction:    "tb" | "lr"
    roi_position: fraction in [0,1] along the direction-of-travel axis
    """

    def __init__(self, source: str, model,
                 direction: Literal["tb", "lr"] = "tb",
                 roi_position: float = 0.7,
                 confidence: float = 0.25, nms_iou: float = 0.45,
                 imgsz: int = 640, max_disappeared: int = 15,
                 max_distance: int = 50, save_raw_path: str = None,
                 is_stream: bool = False):
        self.source = source
        self.model = model
        self.direction = direction
        self.roi_position = roi_position
        self.confidence = confidence
        self.nms_iou = nms_iou
        self.imgsz = imgsz
        self.is_stream = is_stream
        self.save_raw_path = save_raw_path
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

        self.counter = EggCounter(direction=direction, roi_position=roi_position,
                                  max_disappeared=max_disappeared,
                                  max_distance=max_distance)

        self.is_playing = False
        self.is_counting = False
        self.frame_num = 0
        self.total_frames = 0
        self.fps_source = 30.0
        self.fps_display = 0.0
        self.is_complete = False
        self.error = None

        self._latest_frame = None
        self._frame_lock = threading.Lock()

        self._thread = None
        self._stop_event = threading.Event()

        self._writer = None

    @property
    def egg_count(self) -> int:
        return self.counter.total_count

    @property
    def latest_frame(self) -> bytes:
        with self._frame_lock:
            return self._latest_frame

    def start(self):
        if self.is_playing:
            return
        self._stop_event.clear()
        self.is_playing = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        self.is_playing = False

    def start_counting(self):
        self.is_counting = True

    def stop_counting(self):
        self.is_counting = False

    def update_config(self, direction: str, roi_position: float) -> None:
        """Reconfigure direction and ROI. Rebuilds the counter (resets count).

        Safe to call any time. If the capture loop has already learned the
        frame size, the new counter inherits it so the pixel ROI is correct
        on the next frame.
        """
        if direction not in ("tb", "lr"):
            raise ValueError(f"direction must be 'tb' or 'lr', got {direction!r}")
        if not (0.0 <= roi_position <= 1.0):
            raise ValueError(f"roi_position must be in [0,1], got {roi_position}")

        prior_w = self.counter.frame_width
        prior_h = self.counter.frame_height

        self.direction = direction
        self.roi_position = roi_position
        self.counter = EggCounter(direction=direction, roi_position=roi_position,
                                  max_disappeared=self.max_disappeared,
                                  max_distance=self.max_distance)
        if prior_w is not None and prior_h is not None:
            self.counter.set_frame_size(width=prior_w, height=prior_h)

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
            "direction": self.direction,
            "roi_position": self.roi_position,
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

        self.counter.set_frame_size(width=width, height=height)

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

            det_info = detect_frame(self.model, frame, self.confidence, self.nms_iou, self.imgsz)

            if self.is_counting:
                objects = self.counter.update(det_info)
            else:
                tracker_input = [
                    ((d["x1"]+d["x2"])//2, (d["y1"]+d["y2"])//2,
                     d["x1"], d["y1"], d["x2"], d["y2"])
                    for d in det_info
                ]
                objects = self.counter.tracker.update(tracker_input)

            flash_with_frame = [(fx, fy, self.frame_num - i)
                                for i, (fx, fy) in enumerate(
                                    reversed(self.counter.flash_events[-12:]))]

            roi_pos_px = None
            direction_for_draw = None
            if self.is_counting:
                roi_pos_px = (self.counter.roi_y if self.direction == "tb"
                              else self.counter.roi_x)
                direction_for_draw = self.direction

            annotated = annotate_detections(
                frame=frame,
                detections=det_info,
                objects=objects,
                counted_ids=self.counter.counted_ids if self.is_counting else set(),
                trails=self.counter.trails,
                flash_events=flash_with_frame,
                direction=direction_for_draw,
                roi_pos_px=roi_pos_px,
                frame_num=self.frame_num,
                total_count=self.counter.total_count if self.is_counting else 0,
                total_frames=self.total_frames,
                is_stream=self.is_stream,
                fps_display=self.fps_display,
            )

            if self._writer:
                self._writer.write(annotated)

            _, jpeg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
            with self._frame_lock:
                self._latest_frame = jpeg.tobytes()

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
        """Re-encode video to H.264 with faststart for compatibility."""
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
        if os.path.exists(input_path):
            os.remove(input_path)
