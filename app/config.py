from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    rtsp_url: str = ""
    model_path: str = "best.pt"
    roi_position: float = 0.7
    confidence: float = 0.25
    nms_iou: float = 0.45
    imgsz: int = 640
    max_distance: int = 40
    max_disappeared: int = 50
    upload_dir: str = "app/uploads"
    output_dir: str = "app/outputs"
    data_dir: str = "app/data"
    stream_batch_interval_ms: int = 33
    stream_reconnect_backoff_s: float = 5.0

    class Config:
        env_file = ".env"


settings = Settings()
