import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.core.runtime_config import runtime_config
from app.core.model_cache import preload_model, get_model
from app.core.stream_manager import StreamManager, SLOT_RANGE
from app.core.stream_slot import StreamSlot
from app.core.batch_scheduler import BatchScheduler
from app.routers import image, video
from app.routers.streams import router as streams_router, attach_runtime
from app.routers.config_router import router as config_router
from app.routers.export_router import router as export_router
from app.routers.health_router import router as health_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    snap = runtime_config.snapshot()
    os.makedirs(snap["upload_dir"], exist_ok=True)
    os.makedirs(snap["output_dir"], exist_ok=True)
    os.makedirs(snap["data_dir"], exist_ok=True)

    preload_model(snap["model_path"])

    mgr = StreamManager(streams_path=os.path.join(snap["data_dir"], "streams.json"))
    mgr.bootstrap(rtsp_url_env=snap.get("rtsp_url") or None)

    slots = {
        i: StreamSlot(
            slot=i,
            max_disappeared=snap["max_disappeared"],
            max_distance=snap["max_distance"],
            reconnect_backoff_s=snap["stream_reconnect_backoff_s"],
        )
        for i in SLOT_RANGE
    }
    for i in SLOT_RANGE:
        cfg = mgr.get_config(i)
        if cfg is not None:
            slots[i].configure(cfg)
            if cfg.enabled and cfg.source is not None:
                slots[i].start()

    scheduler = BatchScheduler(
        slots=slots,
        model_provider=lambda: get_model(runtime_config.snapshot()["model_path"]),
        batch_interval_ms=snap["stream_batch_interval_ms"],
        imgsz=snap["imgsz"],
        nms_iou=snap["nms_iou"],
    )
    scheduler.start()

    attach_runtime(manager=mgr, slots=slots, scheduler=scheduler)

    try:
        yield
    finally:
        scheduler.stop()
        for slot in slots.values():
            slot.stop()


app = FastAPI(
    title="Egg Counter",
    version="2.1.0",
    description=(
        "Production-grade egg counting API. "
        "Supports up to 10 concurrent RTSP/video streams batched through a "
        "single shared YOLOv8 model, plus image and one-shot video uploads. "
        "All inference parameters are tunable at runtime via PATCH /api/config."
    ),
    lifespan=lifespan,
)

app.include_router(health_router)
app.include_router(config_router)
app.include_router(export_router)
app.include_router(image.router)
app.include_router(video.router)
app.include_router(streams_router)

static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
