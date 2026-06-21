"""Regression tests for runtime direction updates on video sessions.

VideoProcessor delegates counting to ultralytics ObjectCounter at model defaults;
update_config records the new conveyor direction and requests an ObjectCounter
rebuild (which resets the count) on the next processed frame. The counting line
is always the frame's center line, oriented by direction — there is no ROI knob.
"""
import pytest

from app.core.video_processor import VideoProcessor


def _proc():
    return VideoProcessor(source="dummy.mp4", model_path="best.pt", direction="tb")


def test_update_config_switches_direction_and_requests_reset():
    proc = _proc()
    proc._egg_count = 9  # simulate accumulated count

    proc.update_config(direction="lr")

    assert proc.direction == "lr"
    assert proc._egg_count == 0             # count reset
    assert proc._rebuild_requested is True  # ObjectCounter rebuilt next frame


def test_reset_counts_zeroes_and_requests_rebuild():
    proc = _proc()
    proc._egg_count = 42
    proc.reset_counts()
    assert proc._egg_count == 0
    assert proc._rebuild_requested is True


def test_update_config_rejects_invalid_direction():
    proc = _proc()
    with pytest.raises(ValueError):
        proc.update_config(direction="diagonal")


def test_constructor_rejects_invalid_direction():
    with pytest.raises(ValueError):
        VideoProcessor(source="x.mp4", model_path="best.pt", direction="diagonal")
