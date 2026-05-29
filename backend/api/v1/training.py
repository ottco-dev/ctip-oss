"""
backend.api.v1.training — Training control endpoints.

Endpoints:
    POST /training/start                  — Start YOLO training job
    POST /training/stop/{run_id}          — Request training stop
    GET  /training/runs                   — List all runs
    GET  /training/runs/{run_id}          — Get run details + metrics
    GET  /training/runs/{run_id}/metrics  — Get per-epoch metrics
    GET  /training/ls-datasets            — List Label Studio projects for dataset selection
    POST /training/prepare-ls-dataset     — Export LS project → YOLO dataset.yaml
"""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlmodel import Session, select

from backend.database import get_session
from backend.models.experiment import Experiment, Run, Metric
from backend.models.job import BackgroundJob
from backend.models.model_registry import RegisteredModel
from backend.tasks.task_router import task_router
from backend.websocket.manager import ws_manager
from shared.logging.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/training", tags=["training"])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SCHEMAS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TrainingStartRequest(BaseModel):
    # ── Core identity ──────────────────────────────────────────────────────
    experiment_name: str = Field(description="Experiment name or ID to log run under")
    model_variant: str = Field(default="yolo11s", description="yolo11n, yolo11s, yolo11m")
    data_yaml: str = Field(description="Path to YOLO dataset YAML file")

    # ── Core hyperparameters ───────────────────────────────────────────────
    epochs: int = Field(default=150, ge=1, le=500)
    batch_size: int = Field(default=4, ge=1, le=32)
    imgsz: int = Field(default=1280, ge=320, le=2048)
    amp: bool = Field(default=True, description="FP16 mixed precision")
    seed: int = Field(default=42)
    notes: str = Field(default="")

    # ── Learning rate schedule ─────────────────────────────────────────────
    lr0: float = Field(default=0.01, gt=0, description="Initial learning rate")
    lrf: float = Field(
        default=0.01, gt=0, le=1.0,
        description="Final learning rate as a fraction of lr0 (cosine schedule endpoint).",
    )
    warmup_epochs: float = Field(
        default=3.0, ge=0.0, le=10.0,
        description="Number of warmup epochs (can be fractional).",
    )
    cos_lr: bool = Field(
        default=True,
        description="Use cosine learning rate schedule. Recommended for longer runs.",
    )

    # ── Regularisation & optimiser ─────────────────────────────────────────
    weight_decay: float = Field(
        default=0.0005, ge=0.0, le=0.1,
        description="L2 weight decay for AdamW / SGD.",
    )
    momentum: float = Field(
        default=0.937, ge=0.0, le=1.0,
        description="SGD momentum / Adam beta1.",
    )

    # ── Early stopping ─────────────────────────────────────────────────────
    patience: int = Field(
        default=50, ge=1, le=500,
        description=(
            "Early-stopping patience (epochs without mAP50 improvement). "
            "Set to `epochs` to disable early stopping."
        ),
    )

    # ── Augmentation ──────────────────────────────────────────────────────
    augment: bool = Field(
        default=True,
        description="Enable YOLO built-in augmentation pipeline (HSV, flip, scale, mosaic).",
    )
    mosaic: float = Field(
        default=1.0, ge=0.0, le=1.0,
        description=(
            "Mosaic augmentation probability [0, 1]. "
            "0 = disabled, 1 = always on. Reduces to 0 in final `close_mosaic` epochs."
        ),
    )
    close_mosaic: int = Field(
        default=10, ge=0, le=50,
        description="Disable mosaic for the final N epochs (stabilises predictions).",
    )
    hsv_h: float = Field(
        default=0.015, ge=0.0, le=0.1,
        description="HSV hue augmentation range (fraction of 180°).",
    )
    hsv_s: float = Field(
        default=0.7, ge=0.0, le=1.0,
        description="HSV saturation augmentation range.",
    )
    hsv_v: float = Field(
        default=0.4, ge=0.0, le=1.0,
        description="HSV value (brightness) augmentation range.",
    )
    degrees: float = Field(
        default=0.0, ge=0.0, le=180.0,
        description="Random rotation range (±degrees). 0 = disabled.",
    )
    scale: float = Field(
        default=0.5, ge=0.0, le=0.9,
        description="Scale augmentation gain (random resize ±fraction).",
    )
    flipud: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Vertical flip probability.",
    )
    fliplr: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Horizontal flip probability.",
    )


class TrainingStartResponse(BaseModel):
    job_uuid: str
    run_uuid: str
    status: str = "pending"
    message: str


class RunSummary(BaseModel):
    id: int
    run_uuid: str
    experiment_id: int
    model_variant: str
    status: str
    best_map50: float
    best_map50_95: float
    best_precision: float
    best_recall: float
    best_epoch: int
    total_epochs: int
    started_at: float | None
    finished_at: float | None
    duration_s: float | None


class MetricPoint(BaseModel):
    epoch: int
    key: str
    value: float


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ENDPOINTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@router.post("/start", response_model=TrainingStartResponse)
async def start_training(
    request: TrainingStartRequest,
    db: Session = Depends(get_session),
) -> TrainingStartResponse:
    """
    Start a YOLO training job.

    Submits to the GPU task queue. If another GPU task is running,
    this job will wait until the GPU is free.
    """
    # Resolve dataset name → absolute YAML path when caller passes a bare name
    from pathlib import Path as _PPath
    from backend.config import get_settings as _get_settings
    _datasets_root = _PPath(_get_settings().data_root) / "datasets"
    if "/" not in request.data_yaml and not request.data_yaml.endswith(".yaml"):
        candidate = _datasets_root / request.data_yaml / "dataset.yaml"
        if candidate.exists():
            request.data_yaml = str(candidate)
    elif not _PPath(request.data_yaml).is_absolute():
        candidate = _datasets_root / request.data_yaml
        if candidate.exists():
            request.data_yaml = str(candidate)

    # Get or create experiment
    experiment = db.exec(
        select(Experiment).where(Experiment.name == request.experiment_name)
    ).first()

    if experiment is None:
        experiment = Experiment(name=request.experiment_name)
        db.add(experiment)
        db.commit()
        db.refresh(experiment)

    # Create run record
    run_uuid = str(uuid.uuid4())
    import json
    run = Run(
        run_uuid=run_uuid,
        experiment_id=experiment.id,
        model_variant=request.model_variant,
        status="pending",
        config_json=json.dumps(request.model_dump()),
    )
    db.add(run)
    db.commit()
    db.refresh(run)

    # Define training coroutine
    async def run_training():
        from training.pipelines.yolo_trainer import YOLOTrainer, TrainingConfig
        from backend.config import get_settings

        config = TrainingConfig(
            mlflow_tracking_uri=get_settings().mlflow_tracking_uri,
            model_variant=request.model_variant,
            data_yaml=request.data_yaml,
            epochs=request.epochs,
            batch_size=request.batch_size,
            imgsz=request.imgsz,
            lr0=request.lr0,
            lrf=request.lrf,
            warmup_epochs=request.warmup_epochs,
            momentum=request.momentum,
            weight_decay=request.weight_decay,
            patience=request.patience,
            cos_lr=request.cos_lr,
            augment=request.augment,
            mosaic=request.mosaic,
            close_mosaic=request.close_mosaic,
            hsv_h=request.hsv_h,
            hsv_s=request.hsv_s,
            hsv_v=request.hsv_v,
            degrees=request.degrees,
            scale=request.scale,
            flipud=request.flipud,
            fliplr=request.fliplr,
            amp=request.amp,
            seed=request.seed,
        )

        async def epoch_callback(epoch: int, metrics: dict) -> None:
            # Broadcast metrics to WebSocket subscribers
            await ws_manager.send_training_update(epoch, metrics, run_uuid)

            # Broadcast human-readable log line for the live log panel
            map50 = metrics.get("metrics/mAP50(B)", metrics.get("val_map50", 0.0))
            box_loss = metrics.get("train/box_loss", metrics.get("train_loss", 0.0))
            cls_loss = metrics.get("train/cls_loss", 0.0)
            precision = metrics.get("metrics/precision(B)", 0.0)
            recall = metrics.get("metrics/recall(B)", 0.0)
            total = request.epochs
            line = (
                f"[{epoch:>4}/{total}]  "
                f"box={box_loss:.4f}  cls={cls_loss:.4f}  "
                f"mAP50={map50:.4f}  P={precision:.3f}  R={recall:.3f}"
            )
            level = "success" if map50 > 0.5 else "info"
            await ws_manager.send_training_log(run_uuid, line, level)

            # Log metrics to DB + update live epoch/mAP50 on the Run row
            map50_val = metrics.get("metrics/mAP50(B)", metrics.get("val_map50", 0.0))
            with Session(db.bind) as new_session:
                for key, value in metrics.items():
                    metric = Metric(
                        run_id=run.id,
                        epoch=epoch,
                        key=key,
                        value=float(value),
                    )
                    new_session.add(metric)
                # Keep the run row fresh so the dashboard table shows live progress
                db_run = new_session.get(Run, run.id)
                if db_run:
                    db_run.best_epoch = epoch
                    if map50_val > db_run.best_map50:
                        db_run.best_map50 = map50_val
                    new_session.add(db_run)
                new_session.commit()

        # Capture the running event loop before the thread starts so sync_callback
        # can schedule coroutines back onto it from the worker thread.
        import asyncio
        _loop = asyncio.get_event_loop()

        def sync_callback(epoch: int, metrics: dict) -> None:
            try:
                asyncio.run_coroutine_threadsafe(epoch_callback(epoch, metrics), _loop)
            except Exception:
                pass

        # Mark run as running in DB (request session is closed — use fresh session)
        with Session(db.bind) as s:
            db_run = s.get(Run, run.id)
            if db_run:
                db_run.status = "running"
                db_run.started_at = time.time()
                db_run.total_epochs = request.epochs
                s.add(db_run)
                s.commit()

        # Emit start log
        await ws_manager.send_training_log(
            run_uuid,
            f"Training started  model={request.model_variant}  epochs={request.epochs}"
            f"  imgsz={request.imgsz}  batch={request.batch_size}",
            "info",
        )
        await ws_manager.send_training_log(
            run_uuid,
            f"Dataset: {request.data_yaml}",
            "info",
        )
        await ws_manager.send_training_log(
            run_uuid,
            f"{'─' * 72}",
            "dim",
        )
        await ws_manager.send_training_log(
            run_uuid,
            f"{'Epoch':>10}  {'box_loss':>10}  {'cls_loss':>10}  "
            f"{'mAP50':>8}  {'P':>6}  {'R':>6}",
            "header",
        )

        trainer = YOLOTrainer(config, on_epoch_end=sync_callback)
        try:
            result = await asyncio.to_thread(trainer.train)
        except Exception as exc:
            await ws_manager.send_training_log(run_uuid, f"ERROR: {exc}", "error")
            with Session(db.bind) as s:
                db_run = s.get(Run, run.id)
                if db_run:
                    db_run.status = "failed"
                    db_run.finished_at = time.time()
                    s.add(db_run)
                    s.commit()
            raise

        await ws_manager.send_training_log(run_uuid, f"{'─' * 72}", "dim")
        await ws_manager.send_training_log(
            run_uuid,
            f"Training complete — best mAP50={result.best_map50:.4f} @ epoch {result.best_epoch}",
            "success",
        )
        await ws_manager.send_training_log(
            run_uuid,
            f"Best model: {result.best_model_path}",
            "info",
        )

        # Update run record with results and auto-register in model registry
        import json as _json
        from pathlib import Path as _Path
        with Session(db.bind) as new_session:
            db_run = new_session.get(Run, run.id)
            if db_run:
                db_run.status = "completed"
                db_run.best_map50 = result.best_map50
                db_run.best_map50_95 = result.best_map50_95
                db_run.best_precision = result.best_precision
                db_run.best_recall = result.best_recall
                db_run.best_epoch = result.best_epoch
                db_run.total_epochs = result.total_epochs_completed
                db_run.best_model_path = result.best_model_path
                db_run.run_dir = result.run_dir
                db_run.mlflow_run_id = result.mlflow_run_id
                db_run.finished_at = time.time()
                new_session.add(db_run)

                # Auto-register best model in the model registry if it exists
                if result.best_model_path and _Path(result.best_model_path).exists():
                    existing = new_session.exec(
                        select(RegisteredModel).where(
                            RegisteredModel.training_run_uuid == run_uuid
                        )
                    ).first()
                    if not existing:
                        file_size_mb = _Path(result.best_model_path).stat().st_size / (1024 * 1024)
                        model_name = f"{request.model_variant} — {request.experiment_name} (mAP50={result.best_map50:.3f})"
                        reg = RegisteredModel(
                            name=model_name,
                            model_type="detection",
                            framework="ultralytics",
                            variant=request.model_variant,
                            file_path=result.best_model_path,
                            file_size_mb=round(file_size_mb, 2),
                            metrics_json=_json.dumps({
                                "map50": result.best_map50,
                                "map50_95": result.best_map50_95,
                                "precision": result.best_precision,
                                "recall": result.best_recall,
                                "epochs": result.total_epochs_completed,
                            }),
                            vram_required_gb=1.5 if "n" in request.model_variant else (2.5 if "s" in request.model_variant else 4.0),
                            training_run_uuid=run_uuid,
                            is_active=True,
                            description=f"Trained on {request.data_yaml.split('/')[-2] if '/' in request.data_yaml else request.data_yaml}",
                        )
                        new_session.add(reg)

                new_session.commit()

        return result.to_dict()

    job_uuid = await task_router.submit_gpu_task(
        task_fn=run_training,
        job_type="training",
        params=request.model_dump(),
        db_session=db,
    )

    return TrainingStartResponse(
        job_uuid=job_uuid,
        run_uuid=run_uuid,
        message=f"Training job queued. GPU task ID: {job_uuid}",
    )


@router.post("/stop/{run_uuid}")
async def stop_training(
    run_uuid: str,
    db: Session = Depends(get_session),
) -> dict[str, Any]:
    """Request graceful training stop (stops after current epoch)."""
    job = task_router.get_job_status_by_run_uuid(run_uuid) if hasattr(task_router, 'get_job_status_by_run_uuid') else None
    cancelled = await task_router.cancel_job(run_uuid)

    # Update DB status
    run = db.exec(select(Run).where(Run.run_uuid == run_uuid)).first()
    if run:
        run.status = "stopped"
        run.finished_at = time.time()
        db.add(run)
        db.commit()

    return {
        "run_uuid": run_uuid,
        "cancelled": cancelled,
        "message": "Stop requested. Training will finish current epoch." if cancelled else "Run not found or already stopped.",
    }


@router.get("/runs", response_model=list[RunSummary])
async def list_runs(
    experiment_id: int | None = None,
    status: str | None = None,
    limit: int = 50,
    db: Session = Depends(get_session),
) -> list[RunSummary]:
    """List training runs, optionally filtered by experiment or status."""
    query = select(Run)
    if experiment_id:
        query = query.where(Run.experiment_id == experiment_id)
    if status:
        query = query.where(Run.status == status)
    query = query.limit(limit).order_by(Run.id.desc())  # type: ignore

    runs = db.exec(query).all()
    return [
        RunSummary(
            id=r.id,
            run_uuid=r.run_uuid,
            experiment_id=r.experiment_id,
            model_variant=r.model_variant,
            status=r.status,
            best_map50=r.best_map50,
            best_map50_95=r.best_map50_95,
            best_precision=r.best_precision,
            best_recall=r.best_recall,
            best_epoch=r.best_epoch,
            total_epochs=r.total_epochs,
            started_at=r.started_at,
            finished_at=r.finished_at,
            duration_s=r.get_duration_s(),
        )
        for r in runs
    ]


@router.get("/runs/{run_uuid}", response_model=dict)
async def get_run(
    run_uuid: str,
    db: Session = Depends(get_session),
) -> dict[str, Any]:
    """Get full run details including config."""
    run = db.exec(select(Run).where(Run.run_uuid == run_uuid)).first()
    if not run:
        raise HTTPException(status_code=404, detail=f"Run '{run_uuid}' not found")
    return run.to_dict()


@router.get("/runs/{run_uuid}/metrics", response_model=list[MetricPoint])
async def get_run_metrics(
    run_uuid: str,
    keys: str | None = None,
    db: Session = Depends(get_session),
) -> list[MetricPoint]:
    """
    Get per-epoch metrics for a run.

    Args:
        keys: Comma-separated metric keys to filter. None = all metrics.
    """
    run = db.exec(select(Run).where(Run.run_uuid == run_uuid)).first()
    if not run:
        raise HTTPException(status_code=404, detail=f"Run '{run_uuid}' not found")

    query = select(Metric).where(Metric.run_id == run.id)
    if keys:
        key_list = [k.strip() for k in keys.split(",")]
        query = query.where(Metric.key.in_(key_list))

    metrics = db.exec(query.order_by(Metric.epoch)).all()  # type: ignore
    return [MetricPoint(epoch=m.epoch, key=m.key, value=m.value) for m in metrics]


@router.get("/status")
async def training_status(db: Session = Depends(get_session)) -> dict[str, Any]:
    """Get overall training system status: active runs, queue depth, GPU lock."""
    active_runs = db.exec(select(Run).where(Run.status.in_(["running", "pending"]))).all()  # type: ignore
    queue_info = task_router.get_queue_status() if hasattr(task_router, "get_queue_status") else {}
    return {
        "active_runs": len(active_runs),
        "queued": len([r for r in active_runs if r.status == "pending"]),
        "running": len([r for r in active_runs if r.status == "running"]),
        "gpu_locked": queue_info.get("gpu_locked", False),
        "queue_depth": queue_info.get("queue_depth", 0),
    }


@router.get("/jobs")
async def list_jobs(
    status: str | None = None,
    limit: int = 50,
    db: Session = Depends(get_session),
) -> list[dict[str, Any]]:
    """List background training jobs from the DB."""
    query = select(BackgroundJob).where(BackgroundJob.job_type == "training")
    if status:
        query = query.where(BackgroundJob.status == status)
    jobs = db.exec(query.order_by(BackgroundJob.id.desc()).limit(limit)).all()  # type: ignore
    return [j.to_dict() for j in jobs]


@router.get("/jobs/{job_id}")
async def get_job(
    job_id: str,
    db: Session = Depends(get_session),
) -> dict[str, Any]:
    """Get a specific background job by ID or UUID."""
    job = db.exec(select(BackgroundJob).where(BackgroundJob.job_uuid == job_id)).first()
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return job.to_dict()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LABEL STUDIO DATASET INTEGRATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _ls_host() -> str:
    from backend.config import get_settings
    return get_settings().label_studio_url.rstrip("/")


def _ls_token() -> str:
    from backend.config import get_settings
    return get_settings().label_studio_api_key


class LSDataset(BaseModel):
    project_id: int
    title: str
    task_count: int
    annotation_count: int
    prediction_count: int
    description: str = ""
    label_config: str = ""


class PrepareDatasetRequest(BaseModel):
    project_id: int
    use_predictions: bool = Field(
        default=False,
        description=(
            "If True, export YOLO pre-annotations as training data. "
            "If False (default), only human-confirmed annotations are used."
        ),
    )
    train_ratio: float = Field(default=0.70, ge=0.3, le=0.9)
    val_ratio: float = Field(default=0.15, ge=0.05, le=0.4)
    seed: int = Field(default=42)


class PrepareDatasetResponse(BaseModel):
    dataset_yaml: str
    dataset_dir: str
    total_tasks: int
    exported_tasks: int
    skipped_tasks: int
    train_count: int
    val_count: int
    test_count: int
    classes: list[str]
    warnings: list[str]


class PrepareDatasetStartedResponse(BaseModel):
    prepare_id: str
    status: str = "started"
    message: str


@router.get("/ls-datasets", response_model=list[LSDataset])
async def list_ls_datasets() -> list[LSDataset]:
    """
    List all Label Studio projects available as training datasets.

    Returns project metadata including task count and annotation count.
    """
    import requests as _requests

    token = _ls_token()
    host = _ls_host()

    if not token:
        raise HTTPException(
            status_code=503,
            detail="LABEL_STUDIO_API_KEY not configured in .env",
        )

    try:
        r = _requests.get(
            f"{host}/api/projects/",
            headers={"Authorization": f"Token {token}"},
            timeout=10,
        )
        r.raise_for_status()
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Cannot reach Label Studio at {host}: {exc}",
        ) from exc

    datasets = []
    for p in r.json().get("results", []):
        if p.get("task_number", 0) == 0:
            continue
        datasets.append(
            LSDataset(
                project_id=p["id"],
                title=p.get("title", ""),
                task_count=p.get("task_number", 0),
                annotation_count=p.get("num_tasks_with_annotations", 0),
                prediction_count=p.get("total_predictions_number", 0),
                description=p.get("description", ""),
                label_config=p.get("label_config", ""),
            )
        )
    return datasets


@router.post("/prepare-ls-dataset", response_model=PrepareDatasetStartedResponse)
async def prepare_ls_dataset(req: PrepareDatasetRequest) -> PrepareDatasetStartedResponse:
    """
    Start a Label Studio → YOLO dataset export as a background task.

    Returns immediately with a `prepare_id`. Progress is streamed to all
    WebSocket /ws/training subscribers as `training_log` messages.
    When complete, a `dataset_ready` WS event is broadcast with the full result.

    Set `use_predictions=True` to use YOLO pre-annotations (no human review required).
    """
    from training.pipelines.ls_dataset_exporter import ExportConfig, export_ls_project

    prepare_id = str(uuid.uuid4())
    loop = asyncio.get_event_loop()

    def _sync_log(line: str, level: str = "info") -> None:
        asyncio.run_coroutine_threadsafe(
            ws_manager.send_training_log(prepare_id, line, level),
            loop,
        )

    config = ExportConfig(
        project_id=req.project_id,
        ls_host=_ls_host(),
        ls_token=_ls_token(),
        use_predictions=req.use_predictions,
        train_ratio=req.train_ratio,
        val_ratio=req.val_ratio,
        seed=req.seed,
        progress_callback=_sync_log,
    )

    async def _export_task() -> None:
        await ws_manager.send_training_log(prepare_id, "═" * 60, "dim")
        await ws_manager.send_training_log(
            prepare_id,
            f"Dataset Preparation  ·  Project {req.project_id}",
            "header",
        )
        await ws_manager.send_training_log(
            prepare_id,
            f"Split: {req.train_ratio:.0%} train / {req.val_ratio:.0%} val / "
            f"{1 - req.train_ratio - req.val_ratio:.0%} test  (seed={req.seed})",
            "dim",
        )
        try:
            result = await asyncio.to_thread(export_ls_project, config)
        except (ValueError, RuntimeError) as exc:
            await ws_manager.send_training_log(prepare_id, f"ERROR: {exc}", "error")
            await ws_manager.broadcast_to_topic("training", {
                "type": "dataset_ready",
                "prepare_id": prepare_id,
                "success": False,
                "error": str(exc),
            })
            return
        except Exception as exc:
            logger.exception("LS dataset export failed: %s", exc)
            await ws_manager.send_training_log(prepare_id, f"ERROR: {exc}", "error")
            await ws_manager.broadcast_to_topic("training", {
                "type": "dataset_ready",
                "prepare_id": prepare_id,
                "success": False,
                "error": str(exc),
            })
            return

        await ws_manager.send_training_log(prepare_id, "─" * 60, "dim")
        await ws_manager.send_training_log(
            prepare_id,
            f"Export complete — {result.exported_tasks} tasks exported, "
            f"{result.skipped_tasks} skipped",
            "success",
        )
        await ws_manager.send_training_log(
            prepare_id,
            f"Train: {result.train_count}  Val: {result.val_count}  Test: {result.test_count}",
            "info",
        )
        await ws_manager.send_training_log(
            prepare_id,
            f"Classes: {', '.join(result.classes)}",
            "info",
        )
        await ws_manager.send_training_log(
            prepare_id,
            f"YAML: {result.dataset_yaml}",
            "dim",
        )

        await ws_manager.broadcast_to_topic("training", {
            "type": "dataset_ready",
            "prepare_id": prepare_id,
            "success": True,
            **result.to_dict(),
        })

    asyncio.create_task(_export_task())

    return PrepareDatasetStartedResponse(
        prepare_id=prepare_id,
        message="Export started. Watch the Training Log for live progress.",
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EVALUATE — Post-training calibration analysis
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class EvaluateRequest(BaseModel):
    """Request payload for post-training model evaluation."""

    model_path: str = Field(
        description="Absolute path to the trained .pt model (best.pt or last.pt)."
    )
    data_yaml: str = Field(
        description="Absolute path to the YOLO dataset YAML with 'val' split."
    )
    iou_threshold: float = Field(
        default=0.50,
        ge=0.1,
        le=1.0,
        description="IoU threshold for TP matching (COCO default = 0.50).",
    )
    conf_threshold: float = Field(
        default=0.001,
        gt=0,
        le=1.0,
        description=(
            "Confidence threshold during evaluation. Keep low (0.001) to "
            "capture all predictions for calibration analysis."
        ),
    )
    imgsz: int = Field(default=1280, ge=320, le=2048)
    num_bins: int = Field(
        default=15,
        ge=5,
        le=50,
        description="ECE bin count (Guo et al. recommend 15).",
    )
    max_images: int | None = Field(
        default=None,
        description="Cap evaluation at N images. None = full validation set.",
    )
    mlflow_run_id: str | None = Field(
        default=None,
        description=(
            "MLflow run ID to log calibration artifacts against. "
            "If set, logs confidence_scores.npy + is_correct.npy + calibration.json."
        ),
    )
    device: str = Field(default="cuda", description="'cuda', 'cuda:0', or 'cpu'.")


class EvaluateResponse(BaseModel):
    """Post-training evaluation result."""

    run_id: str
    model_path: str
    images_evaluated: int
    total_predictions: int
    true_positives: int
    false_positives: int
    total_ground_truths: int
    false_negatives: int
    # Detection metrics
    map50: float
    map50_95: float
    precision: float
    recall: float
    # Calibration metrics
    ece: float
    mce: float
    mean_confidence: float
    # Artifacts
    mlflow_run_id: str | None = None
    confidence_scores_path: str | None = None
    is_correct_path: str | None = None
    calibration_json_path: str | None = None
    eval_time_s: float
    # Calibration interpretation
    calibration_quality: str
    message: str


def _ece_quality(ece: float) -> str:
    if ece < 0.02:
        return "excellent"
    if ece < 0.05:
        return "good"
    if ece < 0.10:
        return "moderate"
    return "poor"


@router.post(
    "/evaluate",
    response_model=EvaluateResponse,
    summary="Run post-training model evaluation",
    description="""
Evaluate a trained model against its validation split.

**What this does:**
1. Runs YOLO `.val()` to get standard mAP50/precision/recall metrics
2. Runs per-image prediction + IoU matching to ground truth
3. Computes Expected Calibration Error (ECE) and reliability diagram data
4. Logs `predictions/confidence_scores.npy` + `predictions/is_correct.npy`
   as MLflow artifacts (if `mlflow_run_id` provided)

**Why ECE matters:**
Confidence calibration determines whether the model's stated probabilities
are trustworthy. ECE < 0.05 = good; ECE > 0.10 = apply post-hoc calibration
before reporting confidence to end users.

**Scientific basis:** Guo et al. (2017). On Calibration of Modern Neural Networks.
ICML 2017. arXiv:1706.04599
""",
)
async def evaluate_model(request: EvaluateRequest) -> EvaluateResponse:
    """Run post-training evaluation with calibration analysis."""
    from pathlib import Path as _Path

    if not _Path(request.model_path).exists():
        raise HTTPException(
            status_code=404,
            detail=f"Model not found: {request.model_path}",
        )
    if not _Path(request.data_yaml).exists():
        raise HTTPException(
            status_code=404,
            detail=f"Dataset YAML not found: {request.data_yaml}",
        )

    from training.evaluation.evaluator import EvaluationConfig, ModelEvaluator

    config = EvaluationConfig(
        model_path=request.model_path,
        data_yaml=request.data_yaml,
        iou_threshold=request.iou_threshold,
        conf_threshold=request.conf_threshold,
        imgsz=request.imgsz,
        num_bins=request.num_bins,
        max_images=request.max_images,
        mlflow_run_id=request.mlflow_run_id,
        device=request.device,
    )

    try:
        evaluator = ModelEvaluator(config)
        result = evaluator.evaluate()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.exception("Evaluation failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Evaluation failed: {exc}",
        )

    quality = _ece_quality(result.ece)
    return EvaluateResponse(
        run_id=result.run_id,
        model_path=result.model_path,
        images_evaluated=result.images_evaluated,
        total_predictions=result.total_predictions,
        true_positives=result.true_positives,
        false_positives=result.false_positives,
        total_ground_truths=result.total_ground_truths,
        false_negatives=result.false_negatives,
        map50=result.map50,
        map50_95=result.map50_95,
        precision=result.precision,
        recall=result.recall,
        ece=result.ece,
        mce=result.mce,
        mean_confidence=result.mean_confidence,
        mlflow_run_id=result.mlflow_run_id,
        confidence_scores_path=result.confidence_scores_path,
        is_correct_path=result.is_correct_path,
        calibration_json_path=result.calibration_json_path,
        eval_time_s=result.eval_time_s,
        calibration_quality=quality,
        message=(
            f"Evaluation complete. ECE={result.ece:.4f} ({quality}), "
            f"mAP50={result.map50:.4f}, n={result.total_predictions} predictions."
        ),
    )
