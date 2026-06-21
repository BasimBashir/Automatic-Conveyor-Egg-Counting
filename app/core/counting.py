"""Helpers for ultralytics ``solutions.ObjectCounter``-based counting.

Counting is delegated entirely to ObjectCounter. These helpers only:
  * translate the egg app's conveyor ``direction`` into the ObjectCounter
    center-line region, and
  * read the per-object tracked boxes ObjectCounter extracted this frame into
    annotate_boxes-compatible dicts.

The counting line is always the frame's center line (like the chicken app),
oriented by the conveyor direction:
  * "tb" (Top -> Bottom): a horizontal line across the width at y = height // 2.
    ObjectCounter treats a horizontal region as "count by vertical motion" —
    downward crossings are counted (IN).
  * "lr" (Left -> Right): a vertical line down the height at x = width // 2.
    ObjectCounter treats a vertical region as "count by horizontal motion" —
    rightward crossings are counted (IN).
"""
from __future__ import annotations


def build_region(direction: str, width: int, height: int) -> list[tuple[int, int]]:
    """Return the ObjectCounter center-line region [(x1,y1),(x2,y2)] for the
    conveyor direction: "tb" (top->bottom) or "lr" (left->right)."""
    if direction not in ("tb", "lr"):
        raise ValueError(f"direction must be 'tb' or 'lr', got {direction!r}")

    if direction == "tb":
        cy = height // 2
        return [(0, cy), (width, cy)]
    cx = width // 2
    return [(cx, 0), (cx, height)]


def extract_solution_boxes(counter) -> list[dict]:
    """Read the per-object tracked boxes the ObjectCounter extracted this frame
    into annotate_boxes-compatible dicts. Defensive across ultralytics versions:
    missing track_ids/confs degrade gracefully."""
    boxes = getattr(counter, "boxes", None)
    if boxes is None:
        return []
    clss = list(getattr(counter, "clss", []) or [])
    track_ids = list(getattr(counter, "track_ids", []) or [])
    confs = list(getattr(counter, "confs", []) or [])
    names = getattr(counter, "names", {}) or {}
    out = []
    for i, xyxy in enumerate(boxes):
        try:
            x1, y1, x2, y2 = (int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]))
        except Exception:
            continue
        ci = int(clss[i]) if i < len(clss) else -1
        cls_name = names.get(ci, "egg") if isinstance(names, dict) else "egg"
        out.append({
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "class_name": cls_name,
            "conf": float(confs[i]) if i < len(confs) else 0.0,
            "obj_id": int(track_ids[i]) if i < len(track_ids) else None,
        })
    return out
