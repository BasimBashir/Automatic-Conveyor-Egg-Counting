import numpy as np
from app.core.annotator import annotate_boxes, annotate_image_detections, REGION_COLOR


def test_annotate_boxes_draws_without_mutating_input():
    frame = np.zeros((200, 320, 3), dtype=np.uint8)
    boxes = [{"x1": 10, "y1": 10, "x2": 80, "y2": 120,
              "class_name": "egg", "conf": 0.9, "obj_id": 3}]
    out = annotate_boxes(frame, boxes)
    assert out.shape == frame.shape
    assert out.sum() > 0          # something was drawn
    assert frame.sum() == 0       # original untouched


def test_annotate_boxes_handles_missing_optional_fields():
    frame = np.zeros((200, 320, 3), dtype=np.uint8)
    boxes = [{"x1": 5, "y1": 5, "x2": 50, "y2": 60}]  # no class_name/conf/obj_id
    out = annotate_boxes(frame, boxes)
    assert out.shape == frame.shape


def test_annotate_boxes_draws_horizontal_region_line_for_tb():
    frame = np.zeros((200, 320, 3), dtype=np.uint8)
    region = [(0, 100), (320, 100)]   # tb: horizontal line
    out = annotate_boxes(frame, [], region_pts=region, region_thickness=4)
    row = out[98:103, :, :]
    assert (row == np.array(REGION_COLOR, dtype=np.uint8)).all(axis=2).any()
    # No region -> no line drawn.
    assert annotate_boxes(frame, []).sum() == 0


def test_annotate_boxes_draws_vertical_region_line_for_lr():
    frame = np.zeros((200, 320, 3), dtype=np.uint8)
    region = [(160, 0), (160, 200)]   # lr: vertical line
    out = annotate_boxes(frame, [], region_pts=region, region_thickness=4)
    col = out[:, 158:163, :]
    assert (col == np.array(REGION_COLOR, dtype=np.uint8)).all(axis=2).any()


def test_annotate_image_detections_returns_total_count():
    frame = np.zeros((200, 320, 3), dtype=np.uint8)
    det = [{"x1": 5, "y1": 5, "x2": 50, "y2": 60, "conf": 0.8, "class_name": "egg"},
           {"x1": 70, "y1": 70, "x2": 110, "y2": 120, "conf": 0.7, "class_name": "egg"}]
    out, total = annotate_image_detections(frame, det)
    assert out.shape == frame.shape
    assert total == 2
