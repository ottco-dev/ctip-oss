"""
backend.api.v1.dataset_streaming — Dataset streaming and conversion endpoints.

Endpoints:
    POST /datasets/convert                    — Start a YOLO→zarr or YOLO→hdf5 conversion job
    GET  /datasets/streaming/{path:path}/stats — Stats for a zarr or HDF5 store
    GET  /datasets/streaming/jobs/{task_id}   — Conversion job status

Conversion jobs are executed in a background thread to avoid blocking the
event loop.  Progress is tracked via the BackgroundJob model (persisted in
the application SQLite database).

Job types:
    "dataset_convert_zarr"  — YOLO → zarr
    "dataset_convert_hdf5"  — YOLO → HDF5
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, field_validator
from sqlmodel import Session, select

from backend.database import get_session
from backend.models.job import BackgroundJob, JobStatus
from shared.logging.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/datasets", tags=["dataset-streaming"])

# Single-thread executor for conversion jobs (CPU/I-O bound, one at a time
# to avoid saturating disk and RAM simultaneously).
_conversion_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ds-convert")


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class ConversionRequest(BaseModel):
    """Request body for POST /datasets/convert."""

    source_path: str = Field(description="Path to YOLO dataset root directory")
    output_path: str = Field(description="Destination path for the zarr/HDF5 output")
    format: Literal["zarr", "hdf5"] = Field(description="Target format: 'zarr' or 'hdf5'")
    image_size: int = Field(default=640, ge=0, le=4096, description="Resize target (0=keep original)")
    val_split: float = Field(default=0.15, gt=0.0, lt=1.0)
    test_split: float = Field(default=0.10, gt=0.0, lt=1.0)
    seed: int = Field(default=42)
    compression: str = Field(
        default="gzip",
        description="HDF5 compression filter: gzip|lzf|none (ignored for zarr)",
    )

    @field_validator("source_path")
    @classmethod
    def source_must_exist(cls, v: str) -> str:
        p = Path(v)
        if not p.exists():
            raise ValueError(f"source_path does not exist: {v}")
        return str(v)

    @field_validator("val_split", "test_split")
    @classmethod
    def split_range(cls, v: float) -> float:
        if not (0 < v < 1):
            raise ValueError("Split fractions must be in (0, 1)")
        return v


class ConversionResponse(BaseModel):
    task_id: str
    status: str
    message: str
    output_path: str


class JobStatusResponse(BaseModel):
    task_id: str
    job_type: str
    status: str
    progress: float
    progress_pct: float
    error_message: str | None
    created_at: float
    started_at: float | None
    finished_at: float | None
    duration_s: float | None
    result: dict[str, Any]


class StoreStatsResponse(BaseModel):
    format: str
    path: str
    stats: dict[str, Any]


# ---------------------------------------------------------------------------
# Background conversion worker
# ---------------------------------------------------------------------------


def _run_conversion_job(
    job_uuid: str,
    db_url: str,
    request_dict: dict[str, Any],
) -> None:
    """Execute a dataset conversion job in a background thread.

    Opens its own DB session (cannot share the FastAPI session across
    thread boundaries).
    """
    # Import here to avoid circular imports at module load time
    from sqlalchemy import create_engine
    from sqlmodel import Session as _Session, select as _select

    engine = create_engine(db_url, connect_args={"check_same_thread": False})

    def _get_job() -> BackgroundJob | None:
        with _Session(engine) as s:
            return s.exec(_select(BackgroundJob).where(BackgroundJob.job_uuid == job_uuid)).first()

    def _update_job(**kwargs: Any) -> None:
        with _Session(engine) as s:
            job = s.exec(_select(BackgroundJob).where(BackgroundJob.job_uuid == job_uuid)).first()
            if job is None:
                return
            for k, v in kwargs.items():
                setattr(job, k, v)
            s.add(job)
            s.commit()

    _update_job(status=JobStatus.RUNNING, started_at=time.time())

    try:
        from shared.datasets.streaming.dataset_converter import DatasetConverter

        fmt = request_dict["format"]
        source = request_dict["source_path"]
        output = request_dict["output_path"]
        image_size = request_dict["image_size"]
        val_split = request_dict["val_split"]
        test_split = request_dict["test_split"]
        seed = request_dict["seed"]

        if fmt == "zarr":
            splits = DatasetConverter.yolo_to_zarr(
                yolo_root=source,
                output_path=output,
                val_split=val_split,
                test_split=test_split,
                image_size=image_size,
                seed=seed,
            )
            result = {
                "format": "zarr",
                "output_path": output,
                "splits": {
                    k: v.stats for k, v in splits.items()
                },
            }
        elif fmt == "hdf5":
            splits = DatasetConverter.yolo_to_hdf5(
                yolo_root=source,
                output_path=output,
                val_split=val_split,
                test_split=test_split,
                image_size=image_size,
                seed=seed,
                compression=request_dict.get("compression", "gzip"),
            )
            result = {
                "format": "hdf5",
                "output_path": output,
                "splits": {
                    k: v.stats for k, v in splits.items()
                },
            }
        else:
            raise ValueError(f"Unknown format: {fmt}")

        _update_job(
            status=JobStatus.COMPLETED,
            progress=1.0,
            finished_at=time.time(),
            result_json=json.dumps(result, default=str),
        )
        logger.info("Conversion job completed", job_uuid=job_uuid, format=fmt)

    except Exception as exc:
        logger.exception("Conversion job failed", job_uuid=job_uuid, error=str(exc))
        _update_job(
            status=JobStatus.FAILED,
            finished_at=time.time(),
            error_message=str(exc),
        )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/convert",
    response_model=ConversionResponse,
    summary="Start a YOLO→zarr or YOLO→HDF5 conversion job",
    status_code=202,
)
async def start_conversion(
    request: ConversionRequest,
    session: Session = Depends(get_session),
) -> ConversionResponse:
    """
    Submit a background conversion job.

    The conversion runs in a background thread.  Poll
    ``GET /datasets/streaming/jobs/{task_id}`` to check progress.

    - **source_path**: absolute path to YOLO dataset root (must contain images/ and labels/)
    - **output_path**: absolute path for the output store
    - **format**: `zarr` or `hdf5`
    - **image_size**: resize target (0 = keep original resolution)
    """
    task_id = str(uuid.uuid4())

    job_type = f"dataset_convert_{request.format}"
    job = BackgroundJob(
        job_uuid=task_id,
        job_type=job_type,
        status=JobStatus.PENDING,
        params_json=request.model_dump_json(),
    )
    session.add(job)
    session.commit()
    session.refresh(job)

    # Get DB URL from engine for background thread
    from backend.database import engine as _engine
    db_url = str(_engine.url)

    # Submit to thread pool
    loop = asyncio.get_event_loop()
    loop.run_in_executor(
        _conversion_executor,
        _run_conversion_job,
        task_id,
        db_url,
        request.model_dump(),
    )

    logger.info(
        "Conversion job submitted",
        task_id=task_id,
        format=request.format,
        source=request.source_path,
    )

    return ConversionResponse(
        task_id=task_id,
        status="pending",
        message=f"Conversion job submitted. Format: {request.format}",
        output_path=request.output_path,
    )


@router.get(
    "/streaming/jobs/{task_id}",
    response_model=JobStatusResponse,
    summary="Get conversion job status",
)
async def get_job_status(
    task_id: str,
    session: Session = Depends(get_session),
) -> JobStatusResponse:
    """
    Retrieve the status and progress of a dataset conversion job.

    When ``status`` is ``completed``, the ``result`` field contains per-split
    statistics (total images, store size, etc.).
    """
    job = session.exec(
        select(BackgroundJob).where(BackgroundJob.job_uuid == task_id)
    ).first()

    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {task_id}")

    try:
        result = json.loads(job.result_json) if job.result_json else {}
    except json.JSONDecodeError:
        result = {}

    return JobStatusResponse(
        task_id=job.job_uuid,
        job_type=job.job_type,
        status=job.status,
        progress=job.progress,
        progress_pct=job.get_progress_pct(),
        error_message=job.error_message,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        duration_s=job.get_duration_s(),
        result=result,
    )


@router.get(
    "/streaming/{path:path}/stats",
    response_model=StoreStatsResponse,
    summary="Get stats for a zarr or HDF5 store",
)
async def get_store_stats(path: str) -> StoreStatsResponse:
    """
    Return statistics for a zarr or HDF5 dataset store.

    ``path`` is the filesystem path to the store.  The format is
    auto-detected:
    - If the path ends with ``.h5`` or ``.hdf5`` → HDF5Dataset
    - Otherwise → ZarrDataset

    Returns a dict with at minimum:
        ``total_images``, ``store_size_mb``
    """
    resolved = Path(path)
    if not resolved.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Store path does not exist: {path}",
        )

    # Auto-detect format
    suffix = resolved.suffix.lower()
    is_hdf5 = suffix in {".h5", ".hdf5", ".hdf"}

    try:
        if is_hdf5:
            from shared.datasets.streaming.hdf5_dataset import HDF5Dataset, HDF5DatasetConfig
            ds = HDF5Dataset(HDF5DatasetConfig(hdf5_path=str(resolved)))
            fmt = "hdf5"
            store_stats = ds.stats
        else:
            from shared.datasets.streaming.zarr_dataset import ZarrDataset, ZarrDatasetConfig
            ds = ZarrDataset(ZarrDatasetConfig(zarr_path=str(resolved)))
            fmt = "zarr"
            store_stats = ds.stats

    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Cannot open store at {path}: {exc}",
        ) from exc

    return StoreStatsResponse(
        format=fmt,
        path=str(resolved),
        stats=store_stats,
    )
