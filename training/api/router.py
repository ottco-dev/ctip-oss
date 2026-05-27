"""
training/api/router.py — Training API endpoints.

Routes:
  POST /training/start     — submit a training job
  POST /training/stop/{id} — cancel a running/queued job
  GET  /training/jobs      — list all jobs
  GET  /training/jobs/{id} — job detail + progress
  GET  /training/status    — overall training system status
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/training", tags=["training"])


def _get_orchestrator():
    from training.application.training_orchestrator import get_orchestrator
    return get_orchestrator()


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class TrainingStartRequest(BaseModel):
    model: str = Field(default="yolo11s.pt", description="Base model checkpoint")
    dataset_yaml: str = Field(default="", description="Path to dataset YAML")
    epochs: int = Field(default=150, ge=1, le=1000)
    batch_size: int = Field(default=4, ge=1, le=64)
    imgsz: int = Field(default=1280, ge=320, le=4096)
    device: str = Field(default="0", description="CUDA device index or 'cpu'")
    experiment_name: str = Field(default="", description="MLflow experiment name")
    patience: int = Field(default=20, ge=1)
    accumulate: int = Field(default=4, ge=1)
    extra_config: dict = Field(default_factory=dict)


class JobResponse(BaseModel):
    job_id: str
    status: str
    model: str
    dataset_yaml: str
    epochs: int
    current_epoch: int
    progress_pct: float
    best_map50: float
    last_map50: float
    last_loss: float
    experiment_name: str
    mlflow_run_id: str | None
    best_checkpoint: str | None
    queued_at: str
    started_at: str | None
    finished_at: str | None
    error_message: str | None


def _job_to_response(job) -> dict:
    return {
        "job_id": job.job_id,
        "status": job.status.value if hasattr(job.status, "value") else str(job.status),
        "model": job.model,
        "dataset_yaml": job.dataset_yaml,
        "epochs": job.epochs,
        "current_epoch": job.current_epoch,
        "progress_pct": job.progress_pct,
        "best_map50": job.best_map50,
        "last_map50": job.last_map50,
        "last_loss": job.last_loss,
        "experiment_name": job.experiment_name,
        "mlflow_run_id": job.mlflow_run_id,
        "best_checkpoint": job.best_checkpoint,
        "queued_at": job.queued_at.isoformat(),
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "finished_at": job.finished_at.isoformat() if job.finished_at else None,
        "error_message": job.error_message,
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/start", status_code=202)
async def start_training(request: TrainingStartRequest):
    """
    Submit a training job. Returns immediately with job_id.

    Only one GPU training job runs at a time (asyncio.Semaphore(1)).
    Additional submissions are queued in order.
    """
    orchestrator = _get_orchestrator()

    # Ensure worker is running
    import asyncio
    if orchestrator._worker_task is None or orchestrator._worker_task.done():
        asyncio.create_task(orchestrator.start())

    job_id = await orchestrator.submit(
        model=request.model,
        dataset_yaml=request.dataset_yaml,
        epochs=request.epochs,
        batch_size=request.batch_size,
        imgsz=request.imgsz,
        device=request.device,
        experiment_name=request.experiment_name,
        patience=request.patience,
        accumulate=request.accumulate,
        **request.extra_config,
    )

    return {
        "job_id": job_id,
        "status": "queued",
        "message": f"Training job {job_id[:8]} queued. Single-GPU system: only one job runs at a time.",
    }


@router.post("/stop/{job_id}", status_code=200)
async def stop_training(job_id: str):
    """Cancel a queued or running training job."""
    orchestrator = _get_orchestrator()
    ok = await orchestrator.cancel(job_id)
    if not ok:
        raise HTTPException(404, f"Job {job_id} not found")
    return {"job_id": job_id, "cancelled": True}


@router.get("/jobs")
def list_training_jobs(limit: int = 20):
    """List recent training jobs."""
    orchestrator = _get_orchestrator()
    jobs = orchestrator.list_jobs(limit=limit)
    return [_job_to_response(j) for j in jobs]


@router.get("/jobs/{job_id}")
def get_training_job(job_id: str):
    """Get training job detail including live progress."""
    orchestrator = _get_orchestrator()
    job = orchestrator.get_job(job_id)
    if job is None:
        raise HTTPException(404, f"Job {job_id} not found")
    return _job_to_response(job)


@router.get("/status")
def get_training_status():
    """Return overall training system status."""
    orchestrator = _get_orchestrator()
    active = [j for j in orchestrator.list_jobs() if j.status.value == "running"]
    queued = [j for j in orchestrator.list_jobs() if j.status.value == "queued"]

    return {
        "active_jobs": len(active),
        "queued_jobs": len(queued),
        "gpu_semaphore_available": orchestrator._gpu_semaphore._value > 0,
        "active_job": _job_to_response(active[0]) if active else None,
        "queue": [_job_to_response(j) for j in queued[:5]],
    }
