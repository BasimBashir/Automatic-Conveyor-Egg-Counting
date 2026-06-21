from fastapi import APIRouter
from app.core.runtime_config import runtime_config

router = APIRouter(tags=["system"])


@router.get("/health", summary="System health check")
def health() -> dict:
    """Returns GPU availability and current model info.

    Used by Docker HEALTHCHECK and monitoring tools.
    """
    import torch
    gpu = torch.cuda.is_available()
    snap = runtime_config.snapshot()
    return {
        "status": "ok",
        "gpu_available": gpu,
        "device_name": torch.cuda.get_device_name(0) if gpu else "cpu",
        "model_path": snap["model_path"],
    }
