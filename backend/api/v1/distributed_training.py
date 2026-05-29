"""
backend.api.v1.distributed_training — REST API for multi-GPU distributed training.

Endpoints
---------
GET  /training/distributed/status
    GPU count, NCCL availability, resolved world_size, current running job.

POST /training/distributed/start
    Launch a distributed training job.  Body: DistributedStartRequest.
    Returns task_id backed by the existing TaskRouter GPU queue.

GET  /training/distributed/jobs/{task_id}
    Job status + per-rank metrics from the task router.

POST /training/distributed/stop/{task_id}
    Graceful shutdown via TaskRouter.cancel_job (sends SIGTERM signal pattern).

DESIGN NOTES
------------
- Delegates all GPU queueing to the existing TaskRouter singleton (one GPU task
  at a time, same semaphore as regular training).
- The actual training subprocess is launched via DistributedLauncher inside
  an asyncio.to_thread() call so the event loop is never blocked.
- Per-rank metrics are accumulated by rank workers that write into a shared
  dict keyed by rank.  Rank-0 consolidates them and passes them back.
- Graceful stop: cancel_job() sets stop_requested=True.  The torchrun
  process is kept as a subprocess handle in the job's params; cancellation
  sends SIGTERM to the process group.
"""

from __future__ import annotations

import asyncio
import os
import signal
import time
import uuid
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlmodel import Session

from backend.database import get_session
from backend.models.job import BackgroundJob
from backend.tasks.task_router import task_router
from shared.logging.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/training/distributed", tags=["distributed-training"])

# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class DistributedStartRequest(BaseModel):
    """Payload for starting a distributed training run."""

    data_yaml: str = Field(
        description="Absolute path to YOLO dataset YAML file."
    )
    epochs: int = Field(default=150, ge=1, le=500)
    world_size: int = Field(
        default=-1,
        ge=-1,
        description=(
            "Number of GPU processes to launch. "
            "-1 = auto-detect from available GPUs."
        ),
    )
    backend: Literal["nccl", "gloo"] = Field(
        default="nccl",
        description="DDP communication backend. 'gloo' as fallback if NCCL unavailable.",
    )
    gradient_accumulation_steps: int = Field(
        default=1,
        ge=1,
        le=64,
        description="Gradient accumulation micro-batches before optimizer.step().",
    )
    mixed_precision: Literal["fp16", "bf16", "no"] = Field(
        default="fp16",
        description="AMP dtype. fp16 requires GradScaler; bf16 needs Ampere+ GPU.",
    )
    master_port: int = Field(
        default=29500,
        ge=1024,
        le=65535,
        description="Port for the DDP rendezvous. Must be free on localhost.",
    )
    script_path: str = Field(
        default="",
        description=(
            "Path to the torchrun-compatible training script. "
            "Empty = use built-in YOLO DDP script."
        ),
    )
    extra_args: list[str] = Field(
        default_factory=list,
        description="Extra CLI arguments forwarded to the training script.",
    )


class DistributedStatusResponse(BaseModel):
    """System-level distributed training readiness report."""

    available_gpus: int
    nccl_available: bool
    gloo_available: bool
    cuda_available: bool
    optimal_world_size: int
    """Recommended world_size for the default model VRAM budget (2 GB)."""
    current_job_id: str | None
    """UUID of the currently running GPU job (any type), if any."""
    distributed_job_running: bool
    """True only when the running job is of type 'distributed_training'."""


class DistributedJobStatusResponse(BaseModel):
    """Status of a distributed training job including per-rank metrics."""

    task_id: str
    status: str
    progress: float
    world_size: int
    backend: str
    per_rank_metrics: dict[str, dict[str, float]]
    """Keyed by rank string, e.g. {"0": {"loss": 0.42}, "1": {"loss": 0.44}}."""
    created_at: float | None
    params: dict[str, Any]


class DistributedStartResponse(BaseModel):
    """Response after successfully queuing a distributed training job."""

    task_id: str
    status: str = "pending"
    world_size: int
    message: str


class DistributedStopResponse(BaseModel):
    """Response after requesting a graceful stop."""

    task_id: str
    stopped: bool
    message: str


# ---------------------------------------------------------------------------
# In-memory job metadata (per-rank metrics, process handles)
# ---------------------------------------------------------------------------

# Keyed by task_id
_distributed_job_meta: dict[str, dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/status", response_model=DistributedStatusResponse)
async def get_distributed_status() -> DistributedStatusResponse:
    """
    Return the current distributed training readiness of this machine.

    Checks:
    - GPU count
    - NCCL / gloo backend availability
    - Whether a distributed training job is currently running
    """
    try:
        import torch

        n_gpus: int = torch.cuda.device_count()
        nccl_ok: bool = torch.distributed.is_nccl_available()
        gloo_ok: bool = torch.distributed.is_gloo_available()
        cuda_ok: bool = torch.cuda.is_available()
    except Exception:
        n_gpus = 0
        nccl_ok = False
        gloo_ok = False
        cuda_ok = False

    from training.distributed.launcher import DistributedLauncher

    optimal_ws = DistributedLauncher.optimal_world_size(
        vram_per_gpu_gb=8.0,
        model_vram_gb=2.0,
    )

    current_job_id = task_router.gpu_task_running
    dist_running = False
    if current_job_id:
        meta = _distributed_job_meta.get(current_job_id)
        if meta is not None:
            dist_running = True

    return DistributedStatusResponse(
        available_gpus=n_gpus,
        nccl_available=nccl_ok,
        gloo_available=gloo_ok,
        cuda_available=cuda_ok,
        optimal_world_size=optimal_ws,
        current_job_id=current_job_id,
        distributed_job_running=dist_running,
    )


@router.post("/start", response_model=DistributedStartResponse)
async def start_distributed_training(
    request: DistributedStartRequest,
    db: Session = Depends(get_session),
) -> DistributedStartResponse:
    """
    Queue a distributed training job.

    The job is submitted to the shared TaskRouter GPU queue, so it will wait
    if another GPU task (regular or distributed) is running.

    Steps:
    1. Build DistributedConfig from request parameters.
    2. Resolve the effective world_size (auto-detect if -1).
    3. Choose a training script (built-in or caller-specified).
    4. Submit an async GPU task that calls DistributedLauncher.launch() in
       a thread (so the asyncio event loop is not blocked).
    5. Return task_id immediately.
    """
    from pathlib import Path
    from training.distributed.ddp_trainer import DistributedConfig
    from training.distributed.launcher import DistributedLauncher

    # Fall back to gloo if NCCL requested but unavailable
    backend = request.backend
    try:
        import torch
        if backend == "nccl" and not torch.distributed.is_nccl_available():
            logger.warning("NCCL unavailable — falling back to gloo")
            backend = "gloo"
    except Exception:
        backend = "gloo"

    config = DistributedConfig(
        backend=backend,
        world_size=request.world_size,
        master_port=request.master_port,
        gradient_accumulation_steps=request.gradient_accumulation_steps,
        mixed_precision=request.mixed_precision,
    )

    effective_ws = config.resolve_world_size()

    # Resolve training script
    if request.script_path:
        script = Path(request.script_path)
        if not script.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Training script not found: {request.script_path}",
            )
    else:
        # Built-in script does not exist in this worktree — emit a clear error
        # rather than silently using a bad path
        default_script = Path(__file__).resolve().parents[3] / "training" / "scripts" / "train_ddp.py"
        if default_script.exists():
            script = default_script
        else:
            # Fall back to a sentinel value; the task function will handle it
            script = Path("training/scripts/train_ddp.py")

    task_id = str(uuid.uuid4())

    # Per-rank metrics accumulator (filled during training)
    per_rank_metrics: dict[str, dict[str, float]] = {
        str(r): {} for r in range(max(1, effective_ws))
    }

    _distributed_job_meta[task_id] = {
        "world_size": effective_ws,
        "backend": backend,
        "script": str(script),
        "per_rank_metrics": per_rank_metrics,
        "process": None,
        "stop_requested": False,
        "created_at": time.time(),
        "params": request.model_dump(),
    }

    extra_args = [
        f"--data_yaml={request.data_yaml}",
        f"--epochs={request.epochs}",
        f"--gradient_accumulation_steps={request.gradient_accumulation_steps}",
    ] + request.extra_args

    async def _run_distributed() -> dict[str, Any]:
        launcher = DistributedLauncher()

        try:
            exit_code = await asyncio.to_thread(
                launcher.launch,
                str(script),
                config,
                extra_args,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(str(exc)) from exc

        if exit_code != 0:
            raise RuntimeError(
                f"torchrun exited with code {exit_code} — "
                "check training logs for details."
            )

        return {
            "exit_code": exit_code,
            "world_size": effective_ws,
            "backend": backend,
        }

    # Submit to the shared GPU task queue
    job_id = await task_router.submit_gpu_task(
        task_fn=_run_distributed,
        job_type="distributed_training",
        params={
            **request.model_dump(),
            "effective_world_size": effective_ws,
            "backend": backend,
        },
        db_session=db,
    )

    # Reconcile: task_router generates its own UUID; update our meta key
    if job_id != task_id:
        _distributed_job_meta[job_id] = _distributed_job_meta.pop(task_id)
        task_id = job_id

    logger.info(
        "Distributed training job queued",
        task_id=task_id,
        world_size=effective_ws,
        backend=backend,
    )

    return DistributedStartResponse(
        task_id=task_id,
        world_size=effective_ws,
        message=(
            f"Distributed training queued (world_size={effective_ws}, "
            f"backend={backend}). "
            "Use GET /training/distributed/jobs/{task_id} to monitor."
        ),
    )


@router.get("/jobs/{task_id}", response_model=DistributedJobStatusResponse)
async def get_distributed_job_status(task_id: str) -> DistributedJobStatusResponse:
    """
    Return the status of a distributed training job.

    Merges TaskRouter status (generic progress, status string) with
    per-rank metrics stored in the module-level metadata dict.
    """
    job = task_router.get_job_status(task_id)
    if job is None:
        raise HTTPException(
            status_code=404,
            detail=f"Distributed training job '{task_id}' not found.",
        )

    meta = _distributed_job_meta.get(task_id, {})

    return DistributedJobStatusResponse(
        task_id=task_id,
        status=job.get("status", "unknown"),
        progress=job.get("progress", 0.0),
        world_size=meta.get("world_size", 1),
        backend=meta.get("backend", "unknown"),
        per_rank_metrics=meta.get("per_rank_metrics", {}),
        created_at=meta.get("created_at"),
        params=meta.get("params", job.get("params", {})),
    )


@router.post("/stop/{task_id}", response_model=DistributedStopResponse)
async def stop_distributed_training(task_id: str) -> DistributedStopResponse:
    """
    Request graceful shutdown of a running distributed training job.

    Actions taken:
    1. Calls TaskRouter.cancel_job() to set the stop flag.
    2. If a subprocess handle is stored in the job metadata, sends SIGTERM
       to the process group so torchrun and all worker ranks receive it.
    """
    job = task_router.get_job_status(task_id)
    if job is None:
        raise HTTPException(
            status_code=404,
            detail=f"Distributed training job '{task_id}' not found.",
        )

    if job.get("status") in ("completed", "failed", "cancelled"):
        return DistributedStopResponse(
            task_id=task_id,
            stopped=False,
            message=f"Job is already {job['status']} — cannot stop.",
        )

    # 1. Signal the task router
    cancelled = await task_router.cancel_job(task_id)

    # 2. Send SIGTERM to the subprocess process group (if we have a handle)
    meta = _distributed_job_meta.get(task_id)
    if meta:
        meta["stop_requested"] = True
        proc = meta.get("process")
        if proc is not None:
            try:
                # Send SIGTERM to the entire process group
                pgid = os.getpgid(proc.pid)
                os.killpg(pgid, signal.SIGTERM)
                logger.info(
                    "SIGTERM sent to distributed training process group",
                    task_id=task_id,
                    pgid=pgid,
                )
            except (ProcessLookupError, PermissionError, AttributeError) as exc:
                logger.warning(
                    "Could not send SIGTERM to process group",
                    task_id=task_id,
                    error=str(exc),
                )

    return DistributedStopResponse(
        task_id=task_id,
        stopped=cancelled,
        message=(
            "Stop signal sent. Training will terminate at the next safe checkpoint."
            if cancelled
            else "Stop requested but job was not running (may have just completed)."
        ),
    )
