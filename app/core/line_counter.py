"""Trackerless line counter — direction-aware.

direction:
    "tb" — horizontal ROI line at roi_position * height. A detection
           counts if its bbox straddles the line and no detection was
           counted at a nearby x within `dedup_frames` frames.
    "lr" — vertical ROI line at roi_position * width. Same logic, axes
           swapped: bbox straddles x = roi_x, dedup window is on y.
"""
from collections import deque
from typing import Literal


class LineCounter:
    def __init__(self, direction: Literal["tb", "lr"], roi_position: float,
                 perp_dedup_window: int = 30, dedup_frames: int = 8,
                 trail_length: int = 18):
        if direction not in ("tb", "lr"):
            raise ValueError(f"direction must be 'tb' or 'lr', got {direction!r}")
        if not (0.0 <= roi_position <= 1.0):
            raise ValueError(f"roi_position must be in [0,1], got {roi_position}")

        self.direction = direction
        self.roi_position = roi_position
        self.perp_dedup_window = perp_dedup_window
        self.dedup_frames = dedup_frames
        self.trail_length = trail_length
        self.total_count = 0
        self.frame_num = 0
        self.recent = deque()  # (frame_num, perpendicular_coord)
        self.counted_ids = set()
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

    def update(self, det_info: list[dict]) -> dict:
        self.frame_num += 1

        cutoff = self.frame_num - self.dedup_frames
        while self.recent and self.recent[0][0] < cutoff:
            self.recent.popleft()

        objects = {}
        for i, d in enumerate(det_info):
            cx = (d["x1"] + d["x2"]) // 2
            cy = (d["y1"] + d["y2"]) // 2
            objects[self.frame_num * 10000 + i] = (cx, cy)

            if self.direction == "tb":
                if self.roi_y is None:
                    continue
                crosses = d["y1"] <= self.roi_y <= d["y2"]
                perp = cx
            else:
                if self.roi_x is None:
                    continue
                crosses = d["x1"] <= self.roi_x <= d["x2"]
                perp = cy

            if not crosses:
                continue

            recently_counted = any(
                abs(r_perp - perp) < self.perp_dedup_window
                for _, r_perp in self.recent
            )
            if recently_counted:
                continue

            self.total_count += 1
            self.recent.append((self.frame_num, perp))
            self.flash_events.append((cx, cy))

        return objects

    def reset(self):
        self.total_count = 0
        self.frame_num = 0
        self.recent.clear()
        self.counted_ids.clear()
        self.trails.clear()
        self.flash_events.clear()
