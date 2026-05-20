import json
import os
import tempfile

import pytest

from app.core.stream_manager import StreamManager, SlotConfig, Source


@pytest.fixture()
def tmp_streams_file(tmp_path):
    return tmp_path / "streams.json"


def test_bootstrap_creates_ten_null_slots_when_no_file(tmp_streams_file):
    mgr = StreamManager(streams_path=str(tmp_streams_file))
    mgr.bootstrap(rtsp_url_env=None)

    assert tmp_streams_file.exists()
    data = json.loads(tmp_streams_file.read_text())
    assert set(data.keys()) == {str(i) for i in range(1, 11)}
    assert all(v is None for v in data.values())
    assert all(mgr.get_config(i) is None for i in range(1, 11))


def test_bootstrap_seeds_slot1_from_env_when_fresh(tmp_streams_file):
    mgr = StreamManager(streams_path=str(tmp_streams_file))
    mgr.bootstrap(rtsp_url_env="rtsp://cam.example/feed")

    cfg = mgr.get_config(1)
    assert cfg is not None
    assert cfg.source.type == "rtsp"
    assert cfg.source.url == "rtsp://cam.example/feed"
    assert cfg.enabled is True
    assert cfg.direction == "tb"


def test_bootstrap_does_not_overwrite_existing_file(tmp_streams_file):
    initial = {str(i): None for i in range(1, 11)}
    initial["1"] = {
        "source": {"type": "rtsp", "url": "rtsp://existing/feed"},
        "direction": "lr", "roi_position": 0.4, "confidence": 0.3,
        "enabled": False, "count_on_start": True,
    }
    tmp_streams_file.write_text(json.dumps(initial))

    mgr = StreamManager(streams_path=str(tmp_streams_file))
    mgr.bootstrap(rtsp_url_env="rtsp://should-be-ignored")

    cfg = mgr.get_config(1)
    assert cfg.source.url == "rtsp://existing/feed"
    assert cfg.direction == "lr"
    assert cfg.enabled is False


def test_set_config_persists(tmp_streams_file):
    mgr = StreamManager(streams_path=str(tmp_streams_file))
    mgr.bootstrap(rtsp_url_env=None)

    new_cfg = SlotConfig(
        source=Source(type="file", path="app/uploads/slot3.mp4", filename="x.mp4"),
        direction="lr", roi_position=0.6, confidence=0.4,
        enabled=True, count_on_start=False,
    )
    mgr.set_config(3, new_cfg)

    data = json.loads(tmp_streams_file.read_text())
    assert data["3"]["direction"] == "lr"
    assert data["3"]["source"]["type"] == "file"


def test_patch_config_partial_update(tmp_streams_file):
    mgr = StreamManager(streams_path=str(tmp_streams_file))
    mgr.bootstrap(rtsp_url_env="rtsp://orig/feed")

    mgr.patch_config(1, {"direction": "lr", "roi_position": 0.2})
    cfg = mgr.get_config(1)
    assert cfg.direction == "lr"
    assert cfg.roi_position == 0.2
    assert cfg.source.url == "rtsp://orig/feed"  # untouched


def test_invalid_slot_raises(tmp_streams_file):
    mgr = StreamManager(streams_path=str(tmp_streams_file))
    mgr.bootstrap(rtsp_url_env=None)
    with pytest.raises(ValueError):
        mgr.get_config(0)
    with pytest.raises(ValueError):
        mgr.get_config(11)
