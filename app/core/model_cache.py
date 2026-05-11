import platform
import pathlib
import threading
import torch

# Cross-platform pickle compatibility for YOLOv5 .pt files that were trained on
# the opposite OS. Without this, Windows trying to load a Linux-trained
# checkpoint (or vice versa) hits "cannot instantiate 'PosixPath'" or
# "cannot instantiate 'WindowsPath'" inside torch.load's unpickler.
if platform.system() == "Windows":
    pathlib.PosixPath = pathlib.WindowsPath
else:
    pathlib.WindowsPath = pathlib.PosixPath

_cache: dict = {}
_lock = threading.Lock()


def get_model(path: str):
    """Return cached YOLOv5 model for *path*, loading it on first call.

    Uses double-checked locking: cache hits never acquire the lock, so
    ongoing inference is not blocked while a new model is being loaded.
    """
    if path in _cache:
        return _cache[path]
    with _lock:
        if path not in _cache:
            _cache[path] = torch.hub.load(
                "ultralytics/yolov5", "custom", path=path, trust_repo=True
            )
        return _cache[path]


def preload_model(path: str) -> None:
    """Eagerly load *path* into the cache (used at startup and on model switch)."""
    get_model(path)
