import threading
from app.config import Settings


class RuntimeConfig:
    """Thread-safe live configuration.

    Boots from .env via pydantic-settings. Fields can be updated at runtime
    through PATCH /api/config without restarting the container.

    Attribute access (rc.model_path) is supported via __getattr__ so call
    sites read naturally. Writes must go through update().
    """

    def __init__(self) -> None:
        boot = Settings()
        object.__setattr__(self, "_lock", threading.RLock())
        object.__setattr__(self, "_data", {
            "rtsp_url":       boot.rtsp_url,
            "model_path":     boot.model_path,
            "upload_dir":     boot.upload_dir,
            "output_dir":     boot.output_dir,
            "data_dir":       boot.data_dir,
            "stream_reconnect_backoff_s": boot.stream_reconnect_backoff_s,
        })

    def __getattr__(self, name: str):
        data = object.__getattribute__(self, "_data")
        if name in data:
            lock = object.__getattribute__(self, "_lock")
            with lock:
                return data[name]
        raise AttributeError(f"RuntimeConfig has no field '{name}'")

    def snapshot(self) -> dict:
        """Return a copy of the full config (safe for JSON serialisation)."""
        lock = object.__getattribute__(self, "_lock")
        data = object.__getattribute__(self, "_data")
        with lock:
            return dict(data)

    def update(self, patch: dict) -> dict:
        """Merge *patch* into the live config and return the full new state."""
        lock = object.__getattribute__(self, "_lock")
        data = object.__getattribute__(self, "_data")
        with lock:
            for key, value in patch.items():
                if key in data:
                    data[key] = value
            return dict(data)


runtime_config = RuntimeConfig()
