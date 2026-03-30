import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.core.detector import load_model
from app.routers import image, video, stream, config_router

app = FastAPI(title="Egg Counter API", version="1.0.0")

# Ensure directories exist
os.makedirs(settings.upload_dir, exist_ok=True)
os.makedirs(settings.output_dir, exist_ok=True)


@app.on_event("startup")
def startup():
    load_model(settings.model_path)


# Routers
app.include_router(image.router)
app.include_router(video.router)
app.include_router(stream.router)
app.include_router(config_router.router)

# Static files (HTML/CSS/JS) — mounted last so API routes take priority
static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
