"""
backend.api.v1.training — Training control endpoints.

Endpoints:
    POST /training/start         — Start YOLO training job
    POST /training/stop/{run_id} — Request training stop
    GET  /training/runs          — List all runs
    GET  /training/runs/{run_id} — Get run details + metrics
    GET  /training/runs/{run_id}/metrics — Get per-epoch metrics
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlmodel import Session, select

from backend.database import get_session
from backend.models.experiment import Experiment, Run, Metric
from backend.models.job import BackgroundJob
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

        config = TrainingConfig(
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
            # Broadcast to WebSocket subscribers
            await ws_manager.send_training_update(epoch, metrics, run_uuid)

            # Log metrics to DB
            with Session(db.bind) as new_session:
                for key, value in metrics.items():
                    metric = Metric(
                        run_id=run.id,
                        epoch=epoch,
                        key=key,
                        value=float(value),
                    )
                    new_session.add(metric)
                new_session.commit()

        # Sync callback adapter (trainer calls sync callback)
        import asyncio

        def sync_callback(epoch: int, metrics: dict) -> None:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.ensure_future(epoch_callback(epoch, metrics))
            except Exception:
                pass

        trainer = YOLOTrainer(config, on_epoch_end=sync_callback)
        result = trainer.train()

        # Update run record with results
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
            duration_s=r.duration_s,
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
