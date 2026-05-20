import logging
import threading
import time
from typing import Callable

import cv2
import numpy as np

from app.core.stream_slot import StreamSlot
from app.core.annotator import annotate_detections

logger = logging.getLogger(__name__)


class BatchScheduler:
    """Single thread that batches frames from all active slots.

    On each tick:
      1. Collect (frame_id, frame) from every active slot with pending data.
      2. Run model(list_of_frames) once.
      3. Split Results back per slot; filter to that slot's confidence.
      4. Update the slot's counter and store an annotated JPEG.
    """

    def __init__(self, slots: dict[int, StreamSlot],
                 model_provider: Callable, batch_interval_ms: int = 33,
                 imgsz: int = 640, nms_iou: float = 0.45):
        self.slots = slots
        self.model_provider = model_provider
        self.batch_interval_s = batch_interval_ms / 1000.0
        self.imgsz = imgsz
        self.nms_iou = nms_iou

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            tick_start = time.time()
            try:
                self._tick_once()
            except Exception:
                logger.exception("BatchScheduler tick failed; continuing")
            elapsed = time.time() - tick_start
            wait = self.batch_interval_s - elapsed
            if wait > 0:
                self._stop_event.wait(wait)

    def _collect_batch(self) -> dict[int, tuple[int, np.ndarray]]:
        batch: dict[int, tuple[int, np.ndarray]] = {}
        for slot_num, slot in self.slots.items():
            if not slot.is_playing:
                continue
            grabbed = slot.take_pending_frame()
            if grabbed is None:
                continue
            batch[slot_num] = grabbed
        return batch

    def _tick_once(self) -> None:
        batch = self._collect_batch()
        if not batch:
            return

        slot_nums = list(batch.keys())
        frames = [batch[n][1] for n in slot_nums]

        confidences = []
        for n in slot_nums:
            cfg = self.slots[n].config
            if cfg is not None:
                confidences.append(cfg.confidence)
        if not confidences:
            return
        # NOTE: model runs at min(confidence) across the batch; per-slot
        # filtering happens later in _process_result. A few extra low-conf
        # detections per frame are cheaper than running multiple inference
        # passes, but be aware: dropping a slot's confidence aggressively
        # affects NMS load for the whole batch.
        min_conf = min(confidences)
        model = self.model_provider()
        results = model(frames, imgsz=self.imgsz, conf=min_conf,
                        iou=self.nms_iou, verbose=False)

        for slot_num, result in zip(slot_nums, results):
            slot = self.slots[slot_num]
            frame_id, frame = batch[slot_num]
            self._process_result(slot, frame_id, frame, result)

    def _process_result(self, slot: StreamSlot, frame_id: int,
                        frame: np.ndarray, result) -> None:
        counter = slot.counter
        cfg = slot.config
        if counter is None or cfg is None:
            return
        det_info = self._extract_detections(result, cfg.confidence)

        if slot.is_counting:
            objects = counter.update(det_info)
            counted_ids = counter.counted_ids
            trails = counter.trails
            total_count = counter.total_count
            direction = cfg.direction
            roi_pos_px = (counter.roi_y if direction == "tb"
                          else counter.roi_x)
        else:
            tracker_input = [
                ((d["x1"]+d["x2"])//2, (d["y1"]+d["y2"])//2,
                 d["x1"], d["y1"], d["x2"], d["y2"])
                for d in det_info
            ]
            objects = counter.tracker.update(tracker_input)
            counted_ids = set()
            trails = counter.trails
            total_count = 0
            direction = None
            roi_pos_px = None

        flash_with_frame = [(fx, fy, slot.frame_num - i)
                            for i, (fx, fy) in enumerate(
                                reversed(counter.flash_events[-12:]))]

        annotated = annotate_detections(
            frame=frame,
            detections=det_info,
            objects=objects,
            counted_ids=counted_ids,
            trails=trails,
            flash_events=flash_with_frame,
            direction=direction,
            roi_pos_px=roi_pos_px,
            frame_num=slot.frame_num,
            total_count=total_count,
            total_frames=slot.total_frames,
            is_stream=slot.is_stream,
            fps_display=slot.fps_display,
        )

        _, jpeg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
        slot.set_annotated_jpeg(jpeg.tobytes())

    @staticmethod
    def _extract_detections(result, conf_threshold: float) -> list[dict]:
        """Convert an Ultralytics Results object to list[dict] format."""
        out = []
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return out
        xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else boxes.xyxy
        confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else boxes.conf
        for (x1, y1, x2, y2), c in zip(xyxy, confs):
            if float(c) < conf_threshold:
                continue
            out.append({
                "x1": int(x1), "y1": int(y1),
                "x2": int(x2), "y2": int(y2),
                "conf": float(c),
            })
        return out
