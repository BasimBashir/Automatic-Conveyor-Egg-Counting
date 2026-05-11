import platform
import pathlib
import warnings
import numpy as np

from app.core.model_cache import get_model, preload_model  # noqa: F401 — re-exported for back-compat

warnings.filterwarnings("ignore", category=FutureWarning)
if platform.system() != "Windows":
    pathlib.WindowsPath = pathlib.PosixPath


def load_model(model_path: str):
    """Backward-compatible shim — delegates to the shared model cache."""
    return get_model(model_path)


def detect_frame(model, frame: np.ndarray, conf: float = 0.25,
                 iou: float = 0.45, imgsz: int = 640) -> list[dict]:
    """Run YOLOv5 detection on a single frame. Returns list of detection dicts."""
    model.conf = conf
    model.iou = iou
    results = model(frame, size=imgsz)
    detections = results.pandas().xyxy[0]
    det_info = []
    for _, det in detections.iterrows():
        det_info.append({
            "x1": int(det["xmin"]),
            "y1": int(det["ymin"]),
            "x2": int(det["xmax"]),
            "y2": int(det["ymax"]),
            "conf": float(det["confidence"]),
        })
    return det_info
