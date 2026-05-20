import os
import shutil
import time
from typing import Optional, Literal

from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.core.stream_manager import StreamManager, SlotConfig, Source, SLOT_RANGE
from app.core.stream_slot import StreamSlot
from app.core.batch_scheduler import BatchScheduler
from app.core.runtime_config import runtime_config


router = APIRouter(prefix="/api/streams", tags=["streams"])


_manager: Optional[StreamManager] = None
_slots: dict[int, StreamSlot] = {}
_scheduler: Optional[BatchScheduler] = None


def attach_runtime(manager: StreamManager,
                   slots: dict[int, StreamSlot],
                   scheduler: BatchScheduler) -> None:
    global _manager, _slots, _scheduler
    _manager = manager
    _slots = slots
    _scheduler = scheduler


class SourceBody(BaseModel):
    type: Literal["rtsp", "file"]
    url: Optional[str] = None
    path: Optional[str] = None
    filename: Optional[str] = None


class SlotConfigBody(BaseModel):
    source: Optional[SourceBody]
    direction: Literal["tb", "lr"] = "tb"
    roi_position: float = Field(0.7, ge=0.0, le=1.0)
    confidence: float = Field(0.25, ge=0.0, le=1.0)
    enabled: bool = False
    count_on_start: bool = False


class SlotConfigPatch(BaseModel):
    source: Optional[SourceBody] = None
    direction: Optional[Literal["tb", "lr"]] = None
    roi_position: Optional[float] = Field(None, ge=0.0, le=1.0)
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    enabled: Optional[bool] = None
    count_on_start: Optional[bool] = None


def _validate_slot(slot: int) -> None:
    if slot not in SLOT_RANGE:
        raise HTTPException(status_code=404, detail=f"slot must be 1..10, got {slot}")


def _require_manager() -> StreamManager:
    if _manager is None:
        raise HTTPException(status_code=503, detail="StreamManager not ready")
    return _manager


def _slot_payload(slot: int) -> dict:
    mgr = _require_manager()
    cfg = mgr.get_config(slot)
    runtime = _slots[slot].get_status() if slot in _slots else {}
    return {
        "slot": slot,
        "config": cfg.to_dict() if cfg else None,
        "runtime": runtime,
    }


@router.get("")
def list_streams():
    return [_slot_payload(i) for i in SLOT_RANGE]


@router.get("/status")
def aggregate_status():
    return {str(i): _slots[i].get_status() if i in _slots else {}
            for i in SLOT_RANGE}


@router.get("/{slot}")
def get_stream(slot: int):
    _validate_slot(slot)
    return _slot_payload(slot)


@router.get("/{slot}/status")
def get_stream_status(slot: int):
    _validate_slot(slot)
    return _slots[slot].get_status() if slot in _slots else {}


@router.put("/{slot}/config")
def put_config(slot: int, body: SlotConfigBody):
    _validate_slot(slot)
    mgr = _require_manager()

    source = (Source(type=body.source.type,
                     url=body.source.url,
                     path=body.source.path,
                     filename=body.source.filename)
              if body.source else None)
    cfg = SlotConfig(
        source=source,
        direction=body.direction,
        roi_position=body.roi_position,
        confidence=body.confidence,
        enabled=body.enabled,
        count_on_start=body.count_on_start,
    )
    mgr.set_config(slot, cfg)
    if slot in _slots:
        _slots[slot].configure(cfg)
    return cfg.to_dict()


@router.patch("/{slot}/config")
def patch_config(slot: int, body: SlotConfigPatch):
    _validate_slot(slot)
    mgr = _require_manager()

    patch = body.model_dump(exclude_unset=True)
    if "source" in patch:
        if patch["source"] is None:
            # explicit null clears the slot's source
            pass  # leave patch["source"] = None for the manager
        else:
            patch["source"] = body.source.model_dump(exclude_none=True)
    new_cfg = mgr.patch_config(slot, patch)
    if slot in _slots:
        _slots[slot].configure(new_cfg)
    return new_cfg.to_dict()


@router.post("/{slot}/upload")
async def upload_to_slot(slot: int, file: UploadFile = File(...)):
    _validate_slot(slot)
    mgr = _require_manager()
    snap = runtime_config.snapshot()

    upload_dir = snap["upload_dir"]
    os.makedirs(upload_dir, exist_ok=True)

    target = os.path.join(upload_dir, f"slot{slot}_{file.filename}")

    # Delete previous slot-owned file if present
    current = mgr.get_config(slot)
    if current and current.source and current.source.type == "file":
        prev = current.source.path
        if prev and os.path.exists(prev) and prev != target:
            try:
                os.remove(prev)
            except OSError:
                pass

    with open(target, "wb") as f:
        shutil.copyfileobj(file.file, f)

    new_source = Source(type="file", path=target, filename=file.filename)
    if current is not None:
        merged = current.to_dict()
        merged["source"] = new_source.to_dict()
        new_cfg = SlotConfig.from_dict(merged)
    else:
        new_cfg = SlotConfig(source=new_source)

    mgr.set_config(slot, new_cfg)
    if slot in _slots:
        _slots[slot].configure(new_cfg)
    return new_cfg.to_dict()


@router.post("/{slot}/start")
def start_stream(slot: int):
    _validate_slot(slot)
    mgr = _require_manager()
    if slot in _slots and _slots[slot].is_playing:
        return {"status": "already_running"}
    cfg = mgr.get_config(slot)
    if cfg is None or cfg.source is None:
        raise HTTPException(status_code=400, detail="Slot has no source configured")
    _slots[slot].configure(cfg)
    _slots[slot].start()
    return {"status": "started"}


@router.post("/{slot}/stop")
def stop_stream(slot: int):
    _validate_slot(slot)
    _slots[slot].stop()
    return {"status": "stopped"}


@router.post("/{slot}/counting/start")
def start_counting(slot: int):
    _validate_slot(slot)
    if not _slots[slot].is_playing:
        raise HTTPException(status_code=400, detail="Stream not active")
    _slots[slot].start_counting()
    return {"status": "counting"}


@router.post("/{slot}/counting/stop")
def stop_counting(slot: int):
    _validate_slot(slot)
    _slots[slot].stop_counting()
    return {"status": "not_counting"}


@router.post("/{slot}/reset")
def reset_counter(slot: int):
    _validate_slot(slot)
    _slots[slot].reset_counter()
    return {"status": "reset"}


def _mjpeg_generator_for(slot: int):
    while True:
        s = _slots.get(slot)
        if s is None or not s.is_playing:
            break
        frame_bytes = s.latest_frame
        if frame_bytes:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame_bytes
                + b"\r\n"
            )
        time.sleep(0.03)


@router.get("/{slot}/feed")
def stream_feed(slot: int):
    _validate_slot(slot)
    s = _slots.get(slot)
    if s is None or not s.is_playing:
        raise HTTPException(status_code=400, detail="Stream not active")
    return StreamingResponse(
        _mjpeg_generator_for(slot),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
