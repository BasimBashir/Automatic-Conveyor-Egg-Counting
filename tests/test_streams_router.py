import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path):
    """Hermetic test client: streams.json lives under tmp_path for each test.

    We patch runtime_config in-process BEFORE entering the TestClient context
    manager so the FastAPI lifespan picks up the temp data_dir when it
    constructs the StreamManager. The streams router's module-level
    _manager/_slots/_scheduler globals are re-attached on every lifespan
    entry, so state is fresh per test.
    """
    from app.core.runtime_config import runtime_config
    original_data_dir = runtime_config.snapshot()["data_dir"]
    original_rtsp_url = runtime_config.snapshot()["rtsp_url"]
    runtime_config.update({
        "data_dir": str(tmp_path),
        "rtsp_url": "",
    })

    try:
        from app.main import app
        with TestClient(app) as c:
            yield c
    finally:
        runtime_config.update({
            "data_dir": original_data_dir,
            "rtsp_url": original_rtsp_url,
        })


def test_list_returns_ten_slots(client):
    resp = client.get("/api/streams")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 10
    assert all(item["config"] is None for item in body)


def test_get_single_slot(client):
    resp = client.get("/api/streams/1")
    assert resp.status_code == 200
    assert resp.json()["slot"] == 1
    assert resp.json()["config"] is None


def test_put_config_creates_slot(client):
    new_cfg = {
        "source": {"type": "rtsp", "url": "rtsp://test/feed"},
        "direction": "lr",
        "roi_position": 0.4,
        "confidence": 0.3,
        "enabled": False,
        "count_on_start": False,
    }
    resp = client.put("/api/streams/3/config", json=new_cfg)
    assert resp.status_code == 200
    assert resp.json()["direction"] == "lr"

    resp2 = client.get("/api/streams/3")
    assert resp2.json()["config"]["source"]["url"] == "rtsp://test/feed"


def test_patch_config_partial(client):
    client.put("/api/streams/4/config", json={
        "source": {"type": "rtsp", "url": "rtsp://orig"},
        "direction": "tb", "roi_position": 0.7, "confidence": 0.25,
        "enabled": False, "count_on_start": False,
    })
    resp = client.patch("/api/streams/4/config",
                        json={"roi_position": 0.2})
    assert resp.status_code == 200
    assert resp.json()["roi_position"] == 0.2
    assert resp.json()["source"]["url"] == "rtsp://orig"


def test_invalid_slot_returns_404(client):
    assert client.get("/api/streams/0").status_code == 404
    assert client.get("/api/streams/11").status_code == 404
