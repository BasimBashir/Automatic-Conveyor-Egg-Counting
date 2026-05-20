import time
from unittest.mock import MagicMock

import numpy as np

from app.core.batch_scheduler import BatchScheduler
from app.core.stream_slot import StreamSlot
from app.core.stream_manager import SlotConfig, Source


def _make_slot(slot_num: int, frame_shape=(120, 160, 3)) -> StreamSlot:
    slot = StreamSlot(slot=slot_num)
    slot.configure(SlotConfig(
        source=Source(type="rtsp", url="rtsp://test"),
        direction="tb", roi_position=0.5,
        confidence=0.25, enabled=True, count_on_start=False,
    ))
    slot.counter.set_frame_size(width=frame_shape[1], height=frame_shape[0])
    slot.is_playing = True
    slot._pending_frame = np.zeros(frame_shape, dtype=np.uint8)
    slot._pending_frame_id = 1
    return slot


def test_collect_active_slots_skips_inactive():
    s1 = _make_slot(1)
    s2 = _make_slot(2)
    s2.is_playing = False  # inactive

    fake_model = MagicMock()
    scheduler = BatchScheduler(slots={1: s1, 2: s2}, model_provider=lambda: fake_model)

    batch = scheduler._collect_batch()
    assert set(batch.keys()) == {1}


def test_collect_skips_slot_with_no_pending_frame():
    s1 = _make_slot(1)
    s1._pending_frame = None  # nothing to process

    fake_model = MagicMock()
    scheduler = BatchScheduler(slots={1: s1}, model_provider=lambda: fake_model)

    batch = scheduler._collect_batch()
    assert batch == {}


def test_tick_with_empty_batch_does_not_call_model():
    s1 = _make_slot(1)
    s1.is_playing = False

    fake_model = MagicMock()
    scheduler = BatchScheduler(slots={1: s1}, model_provider=lambda: fake_model)

    scheduler._tick_once()
    fake_model.assert_not_called()
