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
