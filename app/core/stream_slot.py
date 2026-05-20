import threading
import time
from typing import Optional

import cv2
import numpy as np

from app.core.stream_manager import SlotConfig
from app.core.counter import EggCounter


class StreamSlot:
    """One of 10 fixed slots. Owns the capture thread, the per-slot counter,
    and the latest-frame buffers.

    The model is NOT touched here. A separate BatchScheduler thread reads
    `pending_frame`, runs inference, and writes back `latest_annotated_jpeg`.
    """

    def __init__(self, slot: int, max_disappeared: int = 15,
                 max_distance: int = 50,
                 reconnect_backoff_s: float = 5.0):
        self.slot = slot
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.reconnect_backoff_s = reconnect_backoff_s

        self.config: Optional[SlotConfig] = None
        self.counter: Optional[EggCounter] = None

        self.is_playing = False
        self.is_counting = False
        self.is_complete = False
        self.is_stream = False
        self.frame_num = 0
        self.total_frames = 0
        self.fps_display = 0.0
        self.error: Optional[str] = None

        self._frame_lock = threading.Lock()
        self._pending_frame: Optional[np.ndarray] = None
        self._pending_frame_id: int = 0
        self._latest_annotated_jpeg: Optional[bytes] = None

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._fps_timer = time.time()
        self._fps_frame_count = 0

    @property
    def latest_frame(self) -> Optional[bytes]:
        with self._frame_lock:
            return self._latest_annotated_jpeg

    def take_pending_frame(self) -> Optional[tuple[int, np.ndarray]]:
        """Atomically grab the latest captured frame (drops stale ones).
        Returns (frame_id, frame_bgr) or None."""
        with self._frame_lock:
            if self._pending_frame is None:
                return None
            f = self._pending_frame
            fid = self._pending_frame_id
            self._pending_frame = None
            return fid, f

    def set_annotated_jpeg(self, jpeg_bytes: bytes) -> None:
        with self._frame_lock:
            self._latest_annotated_jpeg = jpeg_bytes

    def configure(self, cfg: SlotConfig) -> None:
        self.config = cfg
        self.counter = EggCounter(direction=cfg.direction,
                                  roi_position=cfg.roi_position,
                                  max_disappeared=self.max_disappeared,
                                  max_distance=self.max_distance)
        if cfg.count_on_start:
            self.is_counting = True

    def start(self) -> None:
        if self.is_playing:
            return
        if self.config is None or self.config.source is None:
            self.error = "Slot has no source configured"
            return
        self.error = None
        self.is_complete = False
        self.frame_num = 0
        self._stop_event.clear()
        self.is_playing = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        self.is_playing = False

    def start_counting(self) -> None:
        self.is_counting = True

    def stop_counting(self) -> None:
        self.is_counting = False

    def reset_counter(self) -> None:
        if self.counter is not None:
            self.counter.reset()

    def get_status(self) -> dict:
        cfg = self.config
        return {
            "is_connected": self.is_playing,
            "is_counting": self.is_counting,
            "egg_count": self.counter.total_count if self.counter else 0,
            "frame_num": self.frame_num,
            "total_frames": self.total_frames,
            "fps": round(self.fps_display, 1),
            "is_complete": self.is_complete,
            "is_stream": self.is_stream,
            "direction": cfg.direction if cfg else None,
            "roi_position": cfg.roi_position if cfg else None,
            "error": self.error,
        }

    def _open_source(self) -> Optional[cv2.VideoCapture]:
        if self.config is None or self.config.source is None:
            self.error = "No source"
            return None
        src = self.config.source
        target = src.url if src.type == "rtsp" else src.path
        cap = cv2.VideoCapture(target)
        if not cap.isOpened():
            self.error = f"Could not open source: {target}"
            return None
        return cap

    def _capture_loop(self) -> None:
        while not self._stop_event.is_set():
            cap = self._open_source()
            if cap is None:
                if self.config and self.config.source and self.config.source.type == "rtsp":
                    time.sleep(self.reconnect_backoff_s)
                    continue
                self.is_playing = False
                return

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.is_stream = (self.total_frames <= 0
                              or (self.config and self.config.source
                                  and self.config.source.type == "rtsp"))
            if self.counter is not None:
                self.counter.set_frame_size(width=width, height=height)

            self._fps_timer = time.time()
            self._fps_frame_count = 0

            while not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    if self.is_stream:
                        self.error = "Stream connection lost"
                        break  # outer loop reconnects
                    else:
                        self.is_complete = True
                        cap.release()
                        self.is_playing = False
                        return

                self.frame_num += 1
                self._fps_frame_count += 1
                elapsed = time.time() - self._fps_timer
                if elapsed >= 0.5:
                    self.fps_display = self._fps_frame_count / elapsed
                    self._fps_frame_count = 0
                    self._fps_timer = time.time()

                with self._frame_lock:
                    self._pending_frame = frame
                    self._pending_frame_id += 1

            cap.release()
            if self._stop_event.is_set():
                break
