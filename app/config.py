from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    rtsp_url: str = ""
    model_path: str = "best.pt"
    roi_position: float = 0.7
    confidence: float = 0.25
    max_distance: int = 40
    max_disappeared: int = 50
    upload_dir: str = "app/uploads"
    output_dir: str = "app/outputs"

    class Config:
        env_file = ".env"


settings = Settings()
