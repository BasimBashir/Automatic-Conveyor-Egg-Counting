from app.core.line_counter import LineCounter


def _det(x1, y1, x2, y2, conf=0.9):
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "conf": conf}


def test_tb_counts_when_bbox_straddles_horizontal_roi():
    lc = LineCounter(direction="tb", roi_position=0.5,
                     perp_dedup_window=30, dedup_frames=8)
    lc.set_frame_size(width=200, height=200)  # roi_y = 100

    lc.update([_det(40, 90, 60, 110)])
    assert lc.total_count == 1


def test_lr_counts_when_bbox_straddles_vertical_roi():
    lc = LineCounter(direction="lr", roi_position=0.5,
                     perp_dedup_window=30, dedup_frames=8)
    lc.set_frame_size(width=200, height=200)  # roi_x = 100

    lc.update([_det(90, 40, 110, 60)])
    assert lc.total_count == 1


def test_dedup_lr_uses_y_window():
    lc = LineCounter(direction="lr", roi_position=0.5,
                     perp_dedup_window=30, dedup_frames=8)
    lc.set_frame_size(width=200, height=200)

    lc.update([_det(90, 40, 110, 60)])   # cy=50
    lc.update([_det(90, 45, 110, 65)])   # cy=55, within window -> dedup
    assert lc.total_count == 1

    lc.update([_det(90, 120, 110, 140)]) # cy=130, outside window -> new
    assert lc.total_count == 2


def test_reset_clears_state():
    lc = LineCounter(direction="lr", roi_position=0.5,
                     perp_dedup_window=30, dedup_frames=8)
    lc.set_frame_size(width=200, height=200)
    lc.update([_det(90, 40, 110, 60)])
    assert lc.total_count == 1

    lc.reset()
    assert lc.total_count == 0
    assert lc.frame_num == 0
    assert len(lc.recent) == 0
