import platform
import torch
import pathlib
import warnings
import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)
if platform.system() != "Windows":
    pathlib.WindowsPath = pathlib.PosixPath

_model = None


def load_model(model_path: str) -> object:
    """Load YOLOv5 model. Cached as singleton."""
    global _model
    if _model is None:
        _model = torch.hub.load("ultralytics/yolov5", "custom", path=model_path, trust_repo=True)
    return _model


def detect_frame(model, frame: np.ndarray, conf: float = 0.25) -> list[dict]:
    """Run detection on a single frame. Returns list of detection dicts."""
    model.conf = conf
    results = model(frame)
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
