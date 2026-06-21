from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, model_validator

from app.core.runtime_config import runtime_config
from app.core.model_cache import preload_model

router = APIRouter(prefix="/api/config", tags=["config"])


class ConfigResponse(BaseModel):
    rtsp_url:        str
    model_path:      str
    upload_dir:      str
    output_dir:      str


class ConfigPatch(BaseModel):
    rtsp_url:        Optional[str]   = None
    model_path:      Optional[str]   = None
    upload_dir:      Optional[str]   = None
    output_dir:      Optional[str]   = None

    @model_validator(mode="after")
    def at_least_one_field(self) -> "ConfigPatch":
        if not any(v is not None for v in self.model_dump().values()):
            raise ValueError("Provide at least one field to update")
        return self


@router.get("", response_model=ConfigResponse, summary="Get current live config")
def get_config() -> dict:
    """Returns the currently active configuration."""
    return runtime_config.snapshot()


@router.patch("", response_model=ConfigResponse, summary="Update config without restart")
def patch_config(body: ConfigPatch) -> dict:
    """Partially update the running configuration.

    All changes take effect immediately for new requests/sessions.
    Running video sessions continue with the config they were started with.

    If **model_path** is changed the new model is loaded synchronously before
    this request returns — the caller knows the engine is ready when they get
    HTTP 200. If the new model fails to load the update is rolled back and
    HTTP 422 is returned.
    """
    patch = {k: v for k, v in body.model_dump().items() if v is not None}

    old_model_path = runtime_config.model_path
    new_model_path = patch.get("model_path")

    updated = runtime_config.update(patch)

    if new_model_path and new_model_path != old_model_path:
        try:
            preload_model(new_model_path)
        except Exception as exc:
            runtime_config.update({"model_path": old_model_path})
            raise HTTPException(
                status_code=422,
                detail=f"Model load failed — config rolled back: {exc}",
            )

    return updated
