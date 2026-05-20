from collections import deque
from typing import Literal

from app.core.tracker import CentroidTracker


def _det_to_tuple(d):
    cx = (d["x1"] + d["x2"]) // 2
    cy = (d["y1"] + d["y2"]) // 2
    return (cx, cy, d["x1"], d["y1"], d["x2"], d["y2"])


class EggCounter:
    """Direction-aware tracker-based counter.

    direction:
        "tb" — horizontal counting line at roi_position * height; counts
               when a tracked centroid crosses from above to below.
        "lr" — vertical counting line at roi_position * width; counts
               when a tracked centroid crosses from left to right.

    set_frame_size(width, height) must be called once (with first frame
    dims) before update() does anything useful; the actual pixel-space
    line is resolved there.
    """

    def __init__(self, direction: Literal["tb", "lr"], roi_position: float,
                 max_disappeared: int = 15, max_distance: int = 50,
                 trail_length: int = 18):
        if direction not in ("tb", "lr"):
            raise ValueError(f"direction must be 'tb' or 'lr', got {direction!r}")
        if not (0.0 <= roi_position <= 1.0):
            raise ValueError(f"roi_position must be in [0,1], got {roi_position}")

        self.direction = direction
        self.roi_position = roi_position
        self.tracker = CentroidTracker(max_disappeared=max_disappeared,
                                       max_distance=max_distance)
        self.trail_length = trail_length
        self.total_count = 0
        self.counted_ids = set()
        self.last_pos = {}    # obj_id -> last projected coord on direction axis
        self.trails = {}
        self.flash_events = []

        self.frame_width = None
        self.frame_height = None
        self.roi_x = None
        self.roi_y = None

    def set_frame_size(self, width: int, height: int) -> None:
        self.frame_width = width
        self.frame_height = height
        if self.direction == "tb":
            self.roi_y = int(height * self.roi_position)
            self.roi_x = None
        else:
            self.roi_x = int(width * self.roi_position)
            self.roi_y = None

    def _project(self, cx: int, cy: int) -> int:
        return cy if self.direction == "tb" else cx

    def _roi_value(self) -> int:
        return self.roi_y if self.direction == "tb" else self.roi_x

    def update(self, det_info: list[dict]) -> dict:
        detections = [_det_to_tuple(d) for d in det_info]
        objects = self.tracker.update(detections)
        active_ids = set(objects.keys())

        for obj_id, (cx, cy) in objects.items():
            if obj_id not in self.trails:
                self.trails[obj_id] = deque(maxlen=self.trail_length)
            self.trails[obj_id].append((int(cx), int(cy)))

        for old_id in list(self.trails.keys()):
            if old_id not in active_ids:
                del self.trails[old_id]

        roi_val = self._roi_value()
        if roi_val is not None:
            for obj_id, (cx, cy) in objects.items():
                if obj_id in self.counted_ids:
                    continue
                cur = self._project(cx, cy)
                prev = self.last_pos.get(obj_id)
                self.last_pos[obj_id] = cur

                if prev is None:
                    if cur >= roi_val:
                        self.total_count += 1
                        self.counted_ids.add(obj_id)
                        self.flash_events.append((int(cx), int(cy)))
                elif prev < roi_val <= cur:
                    self.total_count += 1
                    self.counted_ids.add(obj_id)
                    self.flash_events.append((int(cx), int(cy)))

        for old_id in list(self.last_pos.keys()):
            if old_id not in active_ids:
                del self.last_pos[old_id]

        for cid in list(self.counted_ids):
            if cid in self.tracker.disappeared and self.tracker.disappeared[cid] > 0:
                self.tracker._deregister(cid)

        return objects

    def reset(self):
        self.tracker.reset()
        self.total_count = 0
        self.counted_ids = set()
        self.last_pos = {}
        self.trails = {}
        self.flash_events = []
