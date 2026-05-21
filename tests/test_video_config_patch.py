"""Regression tests for runtime direction/ROI updates on video sessions."""
from unittest.mock import MagicMock

import pytest

from app.core.video_processor import VideoProcessor


def _proc():
    return VideoProcessor(
        source="dummy.mp4", model=MagicMock(),
        direction="tb", roi_position=0.7,
        max_disappeared=15, max_distance=50,
    )


def test_update_config_switches_direction_and_resets_counter():
    proc = _proc()
    proc.counter.set_frame_size(width=200, height=200)
    assert proc.counter.roi_y == 140  # 200 * 0.7
    proc.counter.total_count = 9  # simulate accumulated count

    proc.update_config(direction="lr", roi_position=0.3)

    assert proc.direction == "lr"
    assert proc.roi_position == 0.3
    assert proc.counter.roi_x == 60  # 200 * 0.3
    assert proc.counter.roi_y is None
    assert proc.counter.total_count == 0  # counter rebuilt → reset


def test_update_config_preserves_frame_size_when_known():
    proc = _proc()
    proc.counter.set_frame_size(width=320, height=240)

    proc.update_config(direction="tb", roi_position=0.5)

    assert proc.counter.frame_width == 320
    assert proc.counter.frame_height == 240
    assert proc.counter.roi_y == 120  # 240 * 0.5


def test_update_config_before_frame_size_leaves_pixel_coords_unresolved():
    proc = _proc()
    assert proc.counter.frame_width is None

    proc.update_config(direction="lr", roi_position=0.4)

    assert proc.direction == "lr"
    assert proc.counter.roi_x is None  # not resolved yet


def test_update_config_rejects_invalid_direction():
    proc = _proc()
    with pytest.raises(ValueError):
        proc.update_config(direction="diagonal", roi_position=0.5)


def test_update_config_rejects_out_of_range_roi():
    proc = _proc()
    with pytest.raises(ValueError):
        proc.update_config(direction="tb", roi_position=1.5)
    with pytest.raises(ValueError):
        proc.update_config(direction="tb", roi_position=-0.1)
