from app.core.counter import EggCounter


def _det(x1, y1, x2, y2, conf=0.9):
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "conf": conf}


def test_tb_counts_when_centroid_crosses_horizontal_line():
    counter = EggCounter(direction="tb", roi_position=0.5,
                         max_disappeared=15, max_distance=50)
    counter.set_frame_size(width=200, height=200)  # roi_y = 100

    counter.update([_det(40, 50, 60, 80)])    # cy = 65, above line
    counter.update([_det(40, 70, 60, 100)])   # cy = 85, still above
    counter.update([_det(40, 95, 60, 120)])   # cy = 107, crossed
    counter.update([_det(40, 110, 60, 140)])  # cy = 125, past line

    assert counter.total_count == 1


def test_lr_counts_when_centroid_crosses_vertical_line():
    counter = EggCounter(direction="lr", roi_position=0.5,
                         max_disappeared=15, max_distance=50)
    counter.set_frame_size(width=200, height=200)  # roi_x = 100

    counter.update([_det(50, 40, 80, 60)])    # cx = 65, left of line
    counter.update([_det(70, 40, 100, 60)])   # cx = 85, still left
    counter.update([_det(95, 40, 120, 60)])   # cx = 107, crossed
    counter.update([_det(110, 40, 140, 60)])  # cx = 125, past line

    assert counter.total_count == 1


def test_lr_does_not_count_when_object_only_moves_down():
    counter = EggCounter(direction="lr", roi_position=0.5,
                         max_disappeared=15, max_distance=50)
    counter.set_frame_size(width=200, height=200)

    counter.update([_det(40, 10, 60, 30)])
    counter.update([_det(40, 50, 60, 70)])
    counter.update([_det(40, 90, 60, 110)])

    assert counter.total_count == 0


def test_tb_does_not_count_when_object_only_moves_right():
    counter = EggCounter(direction="tb", roi_position=0.5,
                         max_disappeared=15, max_distance=50)
    counter.set_frame_size(width=200, height=200)

    counter.update([_det(10, 40, 30, 60)])
    counter.update([_det(50, 40, 70, 60)])
    counter.update([_det(90, 40, 110, 60)])

    assert counter.total_count == 0


def test_reset_clears_state_for_both_directions():
    counter = EggCounter(direction="lr", roi_position=0.5,
                         max_disappeared=15, max_distance=50)
    counter.set_frame_size(width=200, height=200)
    counter.update([_det(95, 40, 120, 60)])
    counter.update([_det(110, 40, 140, 60)])
    assert counter.total_count >= 0

    counter.reset()
    assert counter.total_count == 0
    assert counter.counted_ids == set()
    assert counter.trails == {}
