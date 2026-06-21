"""Regression tests for review fixes (post ObjectCounter migration)."""
from fastapi.testclient import TestClient


class _FakeProc:
    """Minimal stand-in for a running VideoProcessor inside a StreamSlot."""
    def __init__(self):
        self.is_playing = True
        self.is_counting = True
        self.egg_count = 42
        self.stopped = False

    def stop(self):
        self.stopped = True

    def get_status(self):
        return {
            "is_playing": self.is_playing,
            "is_counting": self.is_counting,
            "egg_count": self.egg_count,
            "frame_num": 0, "total_frames": 0, "fps": 0.0,
            "is_complete": False, "is_stream": True,
            "direction": "tb", "error": None,
        }


def test_start_endpoint_idempotent_when_already_running(tmp_path):
    """I2 regression: calling /start while already running must not rebuild the
    slot's processor (which would reset the count)."""
    from app.core.runtime_config import runtime_config
    original_data_dir = runtime_config.snapshot()["data_dir"]
    runtime_config.update({"data_dir": str(tmp_path), "rtsp_url": ""})

    try:
        from app.main import app
        with TestClient(app) as client:
            r = client.put("/api/streams/1/config", json={
                "source": {"type": "rtsp", "url": "rtsp://test/feed"},
                "direction": "tb", "enabled": False, "count_on_start": False,
            })
            assert r.status_code == 200

            # Inject a fake running processor (no real capture needed).
            from app.routers.streams import _slots
            fake = _FakeProc()
            _slots[1]._proc = fake

            # Second /start should be a no-op while already running.
            r = client.post("/api/streams/1/start")
            assert r.status_code == 200
            assert r.json() == {"status": "already_running"}

            # The running processor was neither replaced nor stopped.
            assert _slots[1]._proc is fake
            assert fake.stopped is False
            assert _slots[1].get_status()["egg_count"] == 42
    finally:
        runtime_config.update({"data_dir": original_data_dir})


def test_patch_source_null_clears_slot_source(tmp_path):
    """M2 regression: explicit source: null clears the slot's source."""
    from app.core.runtime_config import runtime_config
    original_data_dir = runtime_config.snapshot()["data_dir"]
    runtime_config.update({"data_dir": str(tmp_path), "rtsp_url": ""})

    try:
        from app.main import app
        with TestClient(app) as client:
            client.put("/api/streams/2/config", json={
                "source": {"type": "rtsp", "url": "rtsp://orig"},
                "direction": "tb", "enabled": False, "count_on_start": False,
            })
            r = client.patch("/api/streams/2/config", json={"source": None})
            assert r.status_code == 200
            body = r.json()
            assert body["source"] is None
    finally:
        runtime_config.update({"data_dir": original_data_dir})
