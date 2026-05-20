import json
import os
import tempfile
import threading
from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class Source:
    type: Literal["rtsp", "file"]
    url: Optional[str] = None
    path: Optional[str] = None
    filename: Optional[str] = None

    def to_dict(self) -> dict:
        out = {"type": self.type}
        if self.url is not None:
            out["url"] = self.url
        if self.path is not None:
            out["path"] = self.path
        if self.filename is not None:
            out["filename"] = self.filename
        return out

    @classmethod
    def from_dict(cls, d: dict) -> "Source":
        return cls(type=d["type"],
                   url=d.get("url"),
                   path=d.get("path"),
                   filename=d.get("filename"))


@dataclass
class SlotConfig:
    source: Optional[Source]
    direction: Literal["tb", "lr"] = "tb"
    roi_position: float = 0.7
    confidence: float = 0.25
    enabled: bool = False
    count_on_start: bool = False

    def to_dict(self) -> dict:
        return {
            "source": self.source.to_dict() if self.source else None,
            "direction": self.direction,
            "roi_position": self.roi_position,
            "confidence": self.confidence,
            "enabled": self.enabled,
            "count_on_start": self.count_on_start,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SlotConfig":
        src = d.get("source")
        return cls(
            source=Source.from_dict(src) if src else None,
            direction=d.get("direction", "tb"),
            roi_position=d.get("roi_position", 0.7),
            confidence=d.get("confidence", 0.25),
            enabled=d.get("enabled", False),
            count_on_start=d.get("count_on_start", False),
        )


SLOT_RANGE = range(1, 11)


class StreamManager:
    """Owns the 10 fixed slot configs and persistence to streams.json."""

    def __init__(self, streams_path: str):
        self.streams_path = streams_path
        self._lock = threading.RLock()
        self._configs: dict[int, Optional[SlotConfig]] = {i: None for i in SLOT_RANGE}

    def bootstrap(self, rtsp_url_env: Optional[str]) -> None:
        with self._lock:
            if os.path.exists(self.streams_path):
                self._load_from_disk()
                return

            if rtsp_url_env:
                self._configs[1] = SlotConfig(
                    source=Source(type="rtsp", url=rtsp_url_env),
                    direction="tb", roi_position=0.7, confidence=0.25,
                    enabled=True, count_on_start=False,
                )
            self._save_to_disk()

    def get_config(self, slot: int) -> Optional[SlotConfig]:
        self._validate_slot(slot)
        with self._lock:
            return self._configs[slot]

    def set_config(self, slot: int, cfg: SlotConfig) -> SlotConfig:
        self._validate_slot(slot)
        with self._lock:
            self._configs[slot] = cfg
            self._save_to_disk()
            return cfg

    def patch_config(self, slot: int, patch: dict) -> SlotConfig:
        self._validate_slot(slot)
        with self._lock:
            current = self._configs[slot]
            if current is None:
                current = SlotConfig(source=None)
            merged = current.to_dict()
            for k, v in patch.items():
                if k == "source":
                    merged["source"] = v
                else:
                    merged[k] = v
            new_cfg = SlotConfig.from_dict(merged)
            self._configs[slot] = new_cfg
            self._save_to_disk()
            return new_cfg

    def all_configs(self) -> dict[int, Optional[SlotConfig]]:
        with self._lock:
            return dict(self._configs)

    def _validate_slot(self, slot: int) -> None:
        if slot not in SLOT_RANGE:
            raise ValueError(f"slot must be 1..10, got {slot}")

    def _load_from_disk(self) -> None:
        with open(self.streams_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for i in SLOT_RANGE:
            entry = data.get(str(i))
            self._configs[i] = SlotConfig.from_dict(entry) if entry else None

    def _save_to_disk(self) -> None:
        os.makedirs(os.path.dirname(self.streams_path) or ".", exist_ok=True)
        payload = {str(i): (c.to_dict() if c else None)
                   for i, c in self._configs.items()}
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=os.path.dirname(self.streams_path) or ".", suffix=".tmp"
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            os.replace(tmp_path, self.streams_path)
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise
