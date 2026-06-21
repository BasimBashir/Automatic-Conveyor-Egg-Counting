from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Runtime configuration. Counting is fixed to ultralytics ObjectCounter at
    model defaults (conf/imgsz), so no detection/counting tuning knobs are
    exposed. The only counting geometry is the conveyor ``direction`` (top->bottom
    or left->right), set per stream slot / per video upload — the counting line
    is always the frame's center line, oriented by that direction.

    Inference runs at ObjectCounter's default imgsz of 640, which matches a
    640x480 @ 30fps camera sub-stream.
    """

    # Source / model
    rtsp_url: str = ""
    model_path: str = "best.pt"

    # Filesystem
    upload_dir: str = "app/uploads"
    output_dir: str = "app/outputs"
    data_dir: str = "app/data"

    # RTSP stream robustness
    stream_reconnect_backoff_s: float = 5.0

    class Config:
        env_file = ".env"
        # Ignore unknown env vars so retired knobs left in a deployment's .env
        # (e.g. CONFIDENCE, ROI_POSITION, MAX_DISTANCE) never break startup.
        extra = "ignore"


settings = Settings()
