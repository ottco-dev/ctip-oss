"""
backend.api.v1.models — Model registry CRUD endpoints.

GET    /models              — list registered models
GET    /models/{id}         — get model detail
POST   /models              — register a new model
PUT    /models/{id}/activate — activate model (set as default for inference)
DELETE /models/{id}         — remove from registry (does not delete file)
POST   /models/{id}/download — queue download job (background task)
"""

from __future__ import annotations

import json
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, Query
from pydantic import BaseModel
from sqlmodel import Session, select

from backend.database import get_session
from backend.models.model_registry import RegisteredModel

router = APIRouter(prefix="/models", tags=["models"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class ModelCreateRequest(BaseModel):
    name: str
    model_type: str
    framework: str = "pytorch"
    variant: str = ""
    file_path: str | None = None
    vram_required_gb: float | None = None
    metrics: dict = {}
    description: str | None = None
    source_url: str | None = None


class ModelResponse(BaseModel):
    id: int
    name: str
    model_type: str
    framework: str
    variant: str
    file_path: str | None
    vram_required_gb: float | None
    metrics: dict
    description: str | None = None
    source_url: str | None = None
    is_downloaded: bool
    is_active: bool
    created_at: str | None

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_response(m: RegisteredModel) -> ModelResponse:
    metrics = {}
    if m.metrics_json:
        try:
            metrics = json.loads(m.metrics_json)
        except Exception:
            pass

    return ModelResponse(
        id=m.id or 0,
        name=m.name,
        model_type=m.model_type,
        framework=m.framework or "pytorch",
        variant=m.variant or "",
        file_path=m.file_path,
        vram_required_gb=m.vram_required_gb,
        metrics=metrics,
        description=getattr(m, "description", None),
        source_url=getattr(m, "source_url", None),
        is_downloaded=bool(m.file_path),
        is_active=bool(getattr(m, "is_active", False)),
        created_at=str(m.created_at) if hasattr(m, "created_at") else None,
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("", response_model=list[ModelResponse])
def list_models(
    model_type: Annotated[str | None, Query()] = None,
    framework: Annotated[str | None, Query()] = None,
    session: Session = Depends(get_session),
) -> list[ModelResponse]:
    """List all registered models with optional type/framework filter."""
    stmt = select(RegisteredModel)
    if model_type:
        stmt = stmt.where(RegisteredModel.model_type == model_type)
    if framework:
        stmt = stmt.where(RegisteredModel.framework == framework)

    models = session.exec(stmt).all()
    return [_to_response(m) for m in models]


@router.get("/{model_id}", response_model=ModelResponse)
def get_model(
    model_id: Annotated[int, Path(gt=0)],
    session: Session = Depends(get_session),
) -> ModelResponse:
    m = session.get(RegisteredModel, model_id)
    if not m:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    return _to_response(m)


@router.post("", response_model=ModelResponse, status_code=201)
def register_model(
    request: ModelCreateRequest,
    session: Session = Depends(get_session),
) -> ModelResponse:
    """Register a new model in the registry."""
    model = RegisteredModel(
        name=request.name,
        model_type=request.model_type,
        framework=request.framework,
        variant=request.variant,
        file_path=request.file_path,
        vram_required_gb=request.vram_required_gb,
        metrics_json=json.dumps(request.metrics),
    )
    session.add(model)
    session.commit()
    session.refresh(model)
    return _to_response(model)


@router.put("/{model_id}/activate")
def activate_model(
    model_id: Annotated[int, Path(gt=0)],
    session: Session = Depends(get_session),
) -> dict:
    """
    Set a model as active (default for inference).
    Only one model per type can be active at a time.
    """
    model = session.get(RegisteredModel, model_id)
    if not model:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

    if not model.file_path:
        raise HTTPException(
            status_code=409,
            detail="Cannot activate model that has not been downloaded yet",
        )

    # Deactivate other models of the same type
    other_models = session.exec(
        select(RegisteredModel).where(
            RegisteredModel.model_type == model.model_type,
            RegisteredModel.id != model_id,
        )
    ).all()

    for m in other_models:
        if hasattr(m, "is_active"):
            m.is_active = False  # type: ignore[assignment]
            session.add(m)

    if hasattr(model, "is_active"):
        model.is_active = True  # type: ignore[assignment]
        session.add(model)

    session.commit()
    return {"message": f"Model {model.name} activated", "model_id": model_id}


@router.delete("/{model_id}")
def remove_model(
    model_id: Annotated[int, Path(gt=0)],
    session: Session = Depends(get_session),
) -> dict:
    """
    Remove model from registry.
    Does NOT delete the weights file from disk.
    """
    model = session.get(RegisteredModel, model_id)
    if not model:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

    session.delete(model)
    session.commit()
    return {"message": f"Model {model_id} removed from registry"}


@router.post("/{model_id}/download")
async def download_model(
    model_id: Annotated[int, Path(gt=0)],
    session: Session = Depends(get_session),
) -> dict:
    """
    Queue a background download job for a model.

    For HuggingFace models: downloads via HF Hub.
    For Ultralytics models: downloads via ultralytics.YOLO().
    """
    from backend.tasks.task_router import task_router
    from backend.models.job import BackgroundJob

    model = session.get(RegisteredModel, model_id)
    if not model:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

    if model.file_path and __import__("pathlib").Path(model.file_path).exists():
        return {"message": "Model already downloaded", "file_path": model.file_path}

    # Create job record
    job = BackgroundJob(
        job_type="model_download",
        status="pending",
        progress=0,
        total_items=1,
        processed_items=0,
    )
    job.set_params({"model_id": model_id, "model_name": model.name})
    session.add(job)
    session.commit()
    session.refresh(job)

    async def _download_task() -> dict:
        """Background download task."""
        import logging
        log = logging.getLogger(__name__)

        try:
            if model.framework == "pytorch" and model.name.lower().startswith("yolo"):
                from ultralytics import YOLO
                yolo = YOLO(f"{model.variant}.pt")  # auto-downloads
                path = str(yolo.ckpt_path)
                log.info("Downloaded %s to %s", model.name, path)
                return {"file_path": path}

            elif model.framework == "transformers":
                from huggingface_hub import snapshot_download
                path = snapshot_download(repo_id=model.name)
                log.info("Downloaded HF model %s to %s", model.name, path)
                return {"file_path": path}

            else:
                raise ValueError(f"Unknown framework: {model.framework}")

        except Exception as exc:
            log.error("Download failed for model %d: %s", model_id, exc)
            raise

    task_router.submit_cpu_task(job.job_uuid or "", _download_task)

    return {
        "message": "Download queued",
        "job_uuid": job.job_uuid,
        "model_id": model_id,
    }
