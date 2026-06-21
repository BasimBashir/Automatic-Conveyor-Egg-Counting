import cv2
import numpy as np

# Default bbox color (BGR, green). Single-class egg model.
EGG_COLOR = (0, 230, 118)

# BGR color ultralytics ObjectCounter uses for the counting region/line
# (solutions.SolutionAnnotator.draw_region default). Replicated here so the line
# drawn on our frames is identical to ObjectCounter's own region line.
REGION_COLOR = (104, 0, 123)


def draw_bbox(img, x1, y1, x2, y2, conf, obj_id=None):
    """Corner-accented bbox + small label (no class name — single-class)."""
    color = EGG_COLOR
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
    id_tag = f"#{obj_id} " if obj_id is not None else ""
    label = f"{id_tag}{conf:.0%}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
    cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 6, y1), color, -1)
    cv2.putText(img, label, (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1, cv2.LINE_AA)


def annotate_boxes(frame, boxes, region_pts=None, region_thickness=4):
    """Bbox-only annotation plus the counting line. `boxes` is a list of dicts
    with keys x1,y1,x2,y2 and optional conf,obj_id. `region_pts`, if given, is the
    ObjectCounter region (e.g. [(x,0),(x,h)] or [(0,y),(w,y)]) drawn as the
    counting line in ObjectCounter's own color.

    No on-frame HUD: counts / FPS / tracked totals are returned by the API, so
    they are never burned into the frame."""
    annotated = frame.copy()
    for b in boxes:
        draw_bbox(annotated, int(b["x1"]), int(b["y1"]), int(b["x2"]), int(b["y2"]),
                  conf=float(b.get("conf", 0.0)), obj_id=b.get("obj_id"))
    if region_pts is not None and len(region_pts) >= 2:
        pts = np.array(region_pts, dtype=np.int32)
        cv2.polylines(annotated, [pts], isClosed=False, color=REGION_COLOR,
                      thickness=int(region_thickness), lineType=cv2.LINE_AA)
    return annotated


def annotate_image_detections(frame, det_info):
    """Annotate a single still with bbox-only overlay. Returns (annotated,
    total_count). The count is returned to the caller (and surfaced via the API
    response header) — it is not drawn on the frame."""
    annotated = frame.copy()
    for info in det_info:
        draw_bbox(annotated, info["x1"], info["y1"], info["x2"], info["y2"],
                  conf=info["conf"])
        cx = (info["x1"] + info["x2"]) // 2
        cy = (info["y1"] + info["y2"]) // 2
        cv2.circle(annotated, (cx, cy), 4, EGG_COLOR, -1, cv2.LINE_AA)
    return annotated, len(det_info)
