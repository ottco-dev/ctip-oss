"""
training/application/training_orchestrator.py — GPU-aware training coordinator.

Orchestrates YOLO training runs with:
  - Single GPU semaphore enforcement (only 1 training job at a time)
  - MLflow experiment tracking
  - WebSocket progress broadcasts
  - Hardware-aware config selection (RTX 4060 defaults)
  - Checkpoint management
  - Early stopping and failure recovery

This is the application-layer coordinator — it does NOT implement the training
loop itself. Training is delegated to training/pipelines/yolo_trainer.py.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger("trichome.training_orchestrator")

# ---------------------------------------------------------------------------
# Training job state
# ---------------------------------------------------------------------------


class TrainingStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


@dataclass
class TrainingJob:
    """A training job tracked by the orchestrator."""

    job_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: TrainingStatus = TrainingStatus.QUEUED

    # Config
    model: str = "yolo11s.pt"
    dataset_yaml: str = ""
    epochs: int = 150
    batch_size: int = 4
    imgsz: int = 1280
    device: str = "0"
    amp: bool = True
    workers: int = 4
    accumulate: int = 4  # gradient accumulation
    patience: int = 20
    extra_config: dict[str, Any] = field(default_factory=dict)

    # Experiment tracking
    experiment_name: str = ""
    mlflow_run_id: Optional[str] = None
    db_job_id: Optional[str] = None

    # Progress
    current_epoch: int = 0
    total_epochs: int = 0
    best_map50: float = 0.0
    last_map50: float = 0.0
    last_loss: float = 0.0
    progress_pct: float = 0.0

    # Timing
    queued_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    estimated_finish: Optional[datetime] = None

    # Error info
    error_message: Optional[str] = None
    cancelled: bool = False

    # Output
    best_checkpoint: Optional[str] = None
    output_dir: Optional[str] = None


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class TrainingOrchestrator:
    """
    GPU-aware training orchestrator for trichome detection.

    Ensures only one training job runs at a time via asyncio.Semaphore(1).
    Maintains a queue of pending jobs.

    Usage:
        orchestrator = TrainingOrchestrator()

        # Submit a job (returns immediately)
        job_id = await orchestrator.submit(config)

        # Poll status
        status = orchestrator.get_job(job_id)

        # Cancel
        await orchestrator.cancel(job_id)
    """

    def __init__(self) -> None:
        self._gpu_semaphore = asyncio.Semaphore(1)
        self._jobs: dict[str, TrainingJob] = {}
        self._queue: asyncio.Queue[TrainingJob] = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task] = None
        self._on_progress: list[Callable[[TrainingJob], None]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the background worker that processes the training queue."""
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._worker_loop())
            logger.info("Training orchestrator worker started")

    async def submit(
        self,
        model: str = "yolo11s.pt",
        dataset_yaml: str = "",
        epochs: int = 150,
        batch_size: int = 4,
        imgsz: int = 1280,
        device: str = "0",
        experiment_name: str = "",
        **extra_config,
    ) -> str:
        """
        Submit a training job to the queue.

        Returns job_id immediately (non-blocking).
        """
        job = TrainingJob(
            model=model,
            dataset_yaml=dataset_yaml,
            epochs=epochs,
            total_epochs=epochs,
            batch_size=batch_size,
            imgsz=imgsz,
            device=device,
            experiment_name=experiment_name or f"run-{datetime.utcnow().strftime('%Y%m%d-%H%M')}",
            extra_config=extra_config,
        )
        self._jobs[job.job_id] = job
        await self._queue.put(job)

        logger.info("Training job %s queued: %s %d epochs", job.job_id[:8], model, epochs)
        return job.job_id

    async def cancel(self, job_id: str) -> bool:
        """Cancel a queued or running job."""
        job = self._jobs.get(job_id)
        if job is None:
            return False

        job.cancelled = True
        if job.status == TrainingStatus.QUEUED:
            job.status = TrainingStatus.CANCELLED
            job.finished_at = datetime.utcnow()
        # If RUNNING: the training loop checks job.cancelled and stops

        logger.info("Training job %s cancelled", job_id[:8])
        return True

    def get_job(self, job_id: str) -> Optional[TrainingJob]:
        """Return job state by ID."""
        return self._jobs.get(job_id)

    def list_jobs(self, limit: int = 20) -> list[TrainingJob]:
        """Return recent jobs, newest first."""
        jobs = list(self._jobs.values())
        jobs.sort(key=lambda j: j.queued_at, reverse=True)
        return jobs[:limit]

    def add_progress_callback(self, fn: Callable[[TrainingJob], None]) -> None:
        """Register a callback for job progress updates."""
        self._on_progress.append(fn)

    # ------------------------------------------------------------------
    # Worker loop
    # ------------------------------------------------------------------

    async def _worker_loop(self) -> None:
        """Background coroutine: pick jobs from queue and run them."""
        logger.info("Training worker loop started")
        while True:
            try:
                job = await self._queue.get()
                if job.cancelled:
                    self._queue.task_done()
                    continue

                async with self._gpu_semaphore:
                    await self._run_job(job)

                self._queue.task_done()

            except asyncio.CancelledError:
                logger.info("Training worker cancelled")
                break
            except Exception as exc:  # noqa: BLE001
                logger.error("Training worker error: %s", exc, exc_info=True)

    async def _run_job(self, job: TrainingJob) -> None:
        """Execute one training job."""
        job.status = TrainingStatus.RUNNING
        job.started_at = datetime.utcnow()

        logger.info(
            "Starting training job %s: model=%s dataset=%s epochs=%d",
            job.job_id[:8],
            job.model,
            job.dataset_yaml,
            job.epochs,
        )

        try:
            # Run training in an executor so it doesn't block the event loop
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._train_sync, job)

            if not job.cancelled:
                job.status = TrainingStatus.COMPLETED
                logger.info(
                    "Training job %s completed. best_map50=%.4f",
                    job.job_id[:8],
                    job.best_map50,
                )
            else:
                job.status = TrainingStatus.CANCELLED

        except Exception as exc:  # noqa: BLE001
            job.status = TrainingStatus.FAILED
            job.error_message = str(exc)
            logger.error("Training job %s failed: %s", job.job_id[:8], exc, exc_info=True)
        finally:
            job.finished_at = datetime.utcnow()
            self._notify_progress(job)

    def _train_sync(self, job: TrainingJob) -> None:
        """
        Synchronous training execution (runs in executor thread).

        Delegates to YOLOTrainer from training/pipelines/yolo_trainer.py.
        """
        from training.pipelines.yolo_trainer import YOLOTrainer, YOLOTrainingConfig

        config = YOLOTrainingConfig(
            model=job.model,
            data=job.dataset_yaml,
            epochs=job.epochs,
            batch=job.batch_size,
            imgsz=job.imgsz,
            device=job.device,
            amp=job.amp,
            workers=job.workers,
            accumulate=job.accumulate,
            patience=job.patience,
            project=f"runs/detect",
            name=job.experiment_name,
            **job.extra_config,
        )

        trainer = YOLOTrainer(config)

        # Install progress hook
        def _epoch_hook(epoch: int, metrics: dict) -> None:
            if job.cancelled:
                raise RuntimeError("Job cancelled by user")

            job.current_epoch = epoch
            map50 = metrics.get("map50", 0.0)
            loss = metrics.get("box_loss", 0.0)

            job.last_map50 = map50
            job.last_loss = loss
            if map50 > job.best_map50:
                job.best_map50 = map50

            job.progress_pct = round((epoch / max(1, job.total_epochs)) * 100, 1)
            self._notify_progress(job)

        trainer.set_epoch_callback(_epoch_hook)

        # Set MLflow experiment
        if job.experiment_name:
            try:
                import mlflow
                mlflow.set_experiment(job.experiment_name)
                with mlflow.start_run(run_name=job.experiment_name) as run:
                    job.mlflow_run_id = run.info.run_id
                    trainer.train()
                    best_path = trainer.get_best_checkpoint()
                    if best_path:
                        job.best_checkpoint = str(best_path)
                        mlflow.log_artifact(str(best_path), artifact_path="checkpoints")
            except Exception:  # noqa: BLE001
                trainer.train()
                best_path = trainer.get_best_checkpoint()
                if best_path:
                    job.best_checkpoint = str(best_path)
        else:
            trainer.train()
            best_path = trainer.get_best_checkpoint()
            if best_path:
                job.best_checkpoint = str(best_path)

    def _notify_progress(self, job: TrainingJob) -> None:
        """Call all registered progress callbacks."""
        for fn in self._on_progress:
            try:
                fn(job)
            except Exception:  # noqa: BLE001
                pass


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_orchestrator: Optional[TrainingOrchestrator] = None


def get_orchestrator() -> TrainingOrchestrator:
    """Return process-singleton training orchestrator."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = TrainingOrchestrator()
    return _orchestrator
