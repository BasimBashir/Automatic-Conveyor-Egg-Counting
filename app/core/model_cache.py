import logging
import os
import threading

from ultralytics import YOLO

logger = logging.getLogger("model_cache")

_cache: dict[str, YOLO] = {}
_lock = threading.Lock()


def get_model(path: str) -> YOLO:
    """Return a cached ultralytics YOLO model for *path*, loading it on first call.

    Used for the single-image endpoint and the live pre-counting preview. Counting
    itself goes through ``ultralytics.solutions.ObjectCounter`` (which loads its own
    model from the same path).
    """
    if path in _cache:
        return _cache[path]
    with _lock:
        if path not in _cache:
            model = YOLO(path)
            # .pt is a torch module and needs explicit device placement; exported
            # formats (.engine/.onnx) bake the device in at export time and raise
            # if you call .to() on them.
            if path.endswith(".pt"):
                try:
                    model.to("cuda:0")
                except Exception:
                    # No CUDA device available — fall back to CPU silently.
                    pass
            _cache[path] = model
        return _cache[path]


def preload_model(path: str) -> None:
    """Eagerly load *path* into the cache (used at startup and on model switch).

    Tolerant by design: a missing or currently-unloadable weights file logs a
    warning and returns instead of raising, so the API still boots. Inference
    requests made before a valid model is in place will surface the load error
    at request time.
    """
    if not os.path.exists(path):
        logger.warning(
            "Model file '%s' not found — skipping preload. "
            "Place the trained model (e.g. best.pt) at this path before "
            "making inference requests.", path,
        )
        return
    try:
        get_model(path)
    except Exception as exc:  # pragma: no cover - depends on the weights file
        logger.warning(
            "Model '%s' could not be loaded at startup (%s). The API will boot; "
            "inference requests will fail until a compatible model is in place.",
            path, exc,
        )
