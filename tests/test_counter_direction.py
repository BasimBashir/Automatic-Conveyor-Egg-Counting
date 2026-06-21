"""Conveyor-direction tests for the ObjectCounter region builder.

Counting itself is delegated to ultralytics ``solutions.ObjectCounter`` at model
defaults. The only egg-app-specific counting concern is translating the conveyor
``direction`` ("tb"/"lr") into the ObjectCounter region line — always the frame's
center line, oriented by direction, which fixes which way objects must travel to
be counted:

  * "tb" (Top -> Bottom): horizontal center line at y = height // 2.
  * "lr" (Left -> Right): vertical center line at x = width // 2.
"""
import pytest

from app.core.counting import build_region


def test_tb_builds_horizontal_center_line():
    region = build_region("tb", width=200, height=200)
    assert region == [(0, 100), (200, 100)]
    # Horizontal: endpoints share a y, span the full width.
    assert region[0][1] == region[1][1]
    assert {region[0][0], region[1][0]} == {0, 200}


def test_lr_builds_vertical_center_line():
    region = build_region("lr", width=200, height=200)
    assert region == [(100, 0), (100, 200)]
    # Vertical: endpoints share an x, span the full height.
    assert region[0][0] == region[1][0]
    assert {region[0][1], region[1][1]} == {0, 200}


def test_center_line_scales_with_frame_size():
    assert build_region("tb", width=320, height=240) == [(0, 120), (320, 120)]
    assert build_region("lr", width=320, height=240) == [(160, 0), (160, 240)]


def test_rejects_invalid_direction():
    with pytest.raises(ValueError):
        build_region("diagonal", width=200, height=200)
