import logging
from typing import Optional

from app.core.stream_manager import SlotConfig
from app.core.runtime_config import runtime_config
from app.core.video_processor import VideoProcessor

logger = logging.getLogger(__name__)


class StreamSlot:
    """One of 10 fixed slots. Owns a per-slot VideoProcessor which runs its own
    capture + ultralytics ObjectCounter (detection + tracking + counting) loop.

    Counting is 100% delegated to ObjectCounter — there is no shared inference
    scheduler. Conveyor direction ("tb"/"lr") and ROI position from the slot's
    config drive the ObjectCounter region line.
    """

    def __init__(self, slot: int, reconnect_backoff_s: float = 5.0):
        self.slot = slot
        self.reconnect_backoff_s = reconnect_backoff_s

        self.config: Optional[SlotConfig] = None
        self._proc: Optional[VideoProcessor] = None
        self.error: Optional[str] = None

    # ── lifecycle ──────────────────────────────────────────────────────────
    def configure(self, cfg: SlotConfig) -> None:
        """Store config. If the slot is already running, restart the processor
        so the new source/direction take effect."""
        self.config = cfg
        if self._proc is not None and self._proc.is_playing:
            self._restart_proc()

    def start(self) -> None:
        if self.config is None or self.config.source is None:
            self.error = "Slot has no source configured"
            return
        if self._proc is not None and self._proc.is_playing:
            return
        self.error = None
        self._proc = self._build_proc()
        self._proc.start()
        if self.config.count_on_start:
            self._proc.start_counting()

    def stop(self) -> None:
        if self._proc is not None:
            self._proc.stop()

    def start_counting(self) -> None:
        if self._proc is not None:
            self._proc.start_counting()

    def stop_counting(self) -> None:
        if self._proc is not None:
            self._proc.stop_counting()

    def reset_counter(self) -> None:
        if self._proc is not None:
            self._proc.reset_counts()

    # ── proxies / status ───────────────────────────────────────────────────
    @property
    def is_playing(self) -> bool:
        return self._proc is not None and self._proc.is_playing

    @property
    def is_counting(self) -> bool:
        return self._proc is not None and self._proc.is_counting

    @property
    def latest_frame(self) -> Optional[bytes]:
        return self._proc.latest_frame if self._proc is not None else None

    def get_status(self) -> dict:
        cfg = self.config
        if self._proc is None:
            return {
                "is_connected": False,
                "is_playing": False,
                "is_counting": False,
                "egg_count": 0,
                "frame_num": 0,
                "total_frames": 0,
                "fps": 0.0,
                "is_complete": False,
                "is_stream": False,
                "direction": cfg.direction if cfg else None,
                "error": self.error,
            }
        st = self._proc.get_status()
        # Preserve the legacy "is_connected" key for the dashboard.
        st["is_connected"] = st["is_playing"]
        if st.get("error") is None and self.error is not None:
            st["error"] = self.error
        return st

    # ── internals ──────────────────────────────────────────────────────────
    def _build_proc(self) -> VideoProcessor:
        src = self.config.source
        target = src.url if src.type == "rtsp" else src.path
        is_stream = src.type == "rtsp"
        snap = runtime_config.snapshot()
        return VideoProcessor(
            source=target,
            model_path=snap["model_path"],
            direction=self.config.direction,
            is_stream=is_stream,
            reconnect_backoff_s=self.reconnect_backoff_s,
        )

    def _restart_proc(self) -> None:
        was_counting = self._proc.is_counting if self._proc else False
        if self._proc is not None:
            self._proc.stop()
        self._proc = self._build_proc()
        self._proc.start()
        if was_counting or (self.config and self.config.count_on_start):
            self._proc.start_counting()
