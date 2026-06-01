"""Reproduce the empty-body MJPEG bug reported by client.

The /api/streams/{slot}/feed endpoint reportedly returns HTTP 200 with the
correct Content-Type but zero bytes of body even when inference is healthy
(counter incrementing, frame_num ticking).
"""
import threading
import time

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path):
    from app.core.runtime_config import runtime_config
    orig_data_dir = runtime_config.snapshot()["data_dir"]
    orig_rtsp = runtime_config.snapshot()["rtsp_url"]
    runtime_config.update({"data_dir": str(tmp_path), "rtsp_url": ""})
    try:
        from app.main import app
        with TestClient(app) as c:
            yield c
    finally:
        runtime_config.update({"data_dir": orig_data_dir, "rtsp_url": orig_rtsp})


def test_feed_yields_jpeg_when_buffer_populated(client):
    """If _latest_annotated_jpeg is populated, the /feed endpoint must
    return multipart MJPEG bytes within a reasonable window."""
    # configure slot 1
    r = client.put("/api/streams/1/config", json={
        "source": {"type": "rtsp", "url": "rtsp://test/feed"},
        "direction": "lr", "roi_position": 0.5, "confidence": 0.25,
        "enabled": False, "count_on_start": False,
    })
    assert r.status_code == 200

    from app.routers.streams import _slots
    slot = _slots[1]
    slot.is_playing = True

    # Simulate inference having produced an annotated frame
    fake_jpeg = b"\xff\xd8\xff\xe0FAKEJPEGBYTES\xff\xd9"
    slot.set_annotated_jpeg(fake_jpeg)

    # Stop the slot after a short delay so generator exits
    def stop_later():
        time.sleep(0.3)
        slot.is_playing = False
    threading.Thread(target=stop_later, daemon=True).start()

    with client.stream("GET", "/api/streams/1/feed") as resp:
        assert resp.status_code == 200
        assert "multipart/x-mixed-replace" in resp.headers["content-type"]
        body = b""
        for chunk in resp.iter_bytes():
            body += chunk
            if len(body) > 100:
                break

    assert len(body) > 0, "FEED RETURNED ZERO BYTES — bug reproduced"
    assert b"--frame" in body, "MJPEG boundary marker missing"
    assert fake_jpeg in body, "JPEG payload missing"
