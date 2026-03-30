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
