from fastapi import APIRouter
from pydantic import BaseModel
from app.config import settings

router = APIRouter(prefix="/api/config", tags=["config"])


class ConfigUpdate(BaseModel):
    roi_position: float | None = None
    confidence: float | None = None
    max_distance: int | None = None
    max_disappeared: int | None = None


@router.get("")
def get_config():
    return {
        "roi_position": settings.roi_position,
        "confidence": settings.confidence,
        "max_distance": settings.max_distance,
        "max_disappeared": settings.max_disappeared,
        "rtsp_url": settings.rtsp_url,
    }


@router.put("")
def update_config(update: ConfigUpdate):
    if update.roi_position is not None:
        settings.roi_position = update.roi_position
    if update.confidence is not None:
        settings.confidence = update.confidence
    if update.max_distance is not None:
        settings.max_distance = update.max_distance
    if update.max_disappeared is not None:
        settings.max_disappeared = update.max_disappeared
    return {"status": "ok", "config": get_config()}
