"""
backend.tasks.task_router — GPU-aware background task dispatcher.

VRAM MANAGEMENT:
The RTX 4060 has 8 GB VRAM. Running training + inference simultaneously
would exceed VRAM and cause OOM errors or degraded performance.

Solution: asyncio.Semaphore(1) for GPU tasks.
Only ONE GPU task runs at a time. Other GPU requests wait in queue.

TASK TYPES:
- GPU tasks: training, batch inference, VLM auto-labeling, SAM segmentation
- CPU tasks: report generation, CSV export, video extraction, quality scoring
- CPU tasks can run concurrently with GPU tasks.

TASK LIFECYCLE:
  submit() → BackgroundJob(pending) → [GPU semaphore acquired] →
  → job running → [GPU semaphore released] → job completed/failed
"""

from __future__ import annotations

import asyncio
import traceback
import uuid
from typing import Any, Callable, Coroutine

from shared.logging.logger import get_logger
from backend.websocket.manager import ws_manager

logger = get_logger(__name__)


class TaskRouter:
    """
    GPU-aware task dispatcher with semaphore-based GPU exclusion.

    Usage:
        router = TaskRouter()

        # Submit a GPU task
        job_id = await router.submit_gpu_task(
            task_fn=run_training,
            job_type="training",
            params=config.dict(),
        )

        # Submit a CPU task (no GPU semaphore)
        job_id = await router.submit_cpu_task(
            task_fn=generate_report,
            job_type="export_report",
            params={"format": "pdf"},
        )
    """

    def __init__(self, max_gpu_tasks: int = 1) -> None:
        self._gpu_semaphore = asyncio.Semaphore(max_gpu_tasks)
        self._active_jobs: dict[str, dict[str, Any]] = {}
        self._gpu_task_running: str | None = None
        """UUID of currently running GPU task (for status display)."""

    async def restore_from_db(self, db_session: Any) -> int:
        """
        Load recent jobs from SQLite into the in-memory cache on startup.

        Any job that was pending/running when the process died is marked failed.
        Returns the number of restored jobs.
        """
        import time as _time
        try:
            from sqlmodel import select
            from backend.models.job import BackgroundJob

            cutoff = _time.time() - 24 * 3600  # last 24 h
            jobs = db_session.exec(
                select(BackgroundJob)
                .where(BackgroundJob.created_at > cutoff)  # type: ignore[arg-type]
                .order_by(BackgroundJob.id.desc())  # type: ignore[attr-defined]
                .limit(200)
            ).all()

            restored = 0
            for job in jobs:
                self._active_jobs[job.job_uuid] = {
                    "job_id": job.job_uuid,
                    "job_type": job.job_type,
                    "status": job.status,
                    "progress": job.progress,
                    "params": {},
                    "stop_requested": False,
                    "created_at": job.created_at,
                }
                # Mark in-flight jobs as failed — they can't recover after restart
                if job.status in ("pending", "running"):
                    self._active_jobs[job.job_uuid]["status"] = "failed"
                    job.status = "failed"
                    job.error_message = "Backend restarted — job interrupted"
                    db_session.add(job)
                    restored += 1

            db_session.commit()
            logger.info("TaskRouter: restored jobs from DB", total=len(jobs), marked_failed=restored)
            return len(jobs)
        except Exception as e:
            logger.warning("TaskRouter: DB restore failed", error=str(e))
            return 0

    @property
    def gpu_task_running(self) -> str | None:
        """UUID of the currently running GPU task, if any."""
        return self._gpu_task_running

    @property
    def gpu_queue_depth(self) -> int:
        """Number of GPU tasks waiting for the semaphore."""
        return max(0, self._gpu_semaphore._value - 1)

    async def submit_gpu_task(
        self,
        task_fn: Callable[..., Coroutine[Any, Any, Any]],
        job_type: str,
        params: dict[str, Any] | None = None,
        on_progress: Callable[[str, float, str], None] | None = None,
        db_session: Any | None = None,
    ) -> str:
        """
        Submit a GPU task for execution.

        The task will wait for the GPU semaphore before running.
        Returns immediately with a job UUID.

        Args:
            task_fn: Async function to run.
            job_type: Job type string for tracking.
            params: Task parameters (for DB record).
            on_progress: Optional progress callback (job_id, 0-1, message).
            db_session: Optional DB session for job persistence.

        Returns:
            Job UUID string.
        """
        job_id = str(uuid.uuid4())

        self._active_jobs[job_id] = {
            "job_id": job_id,
            "job_type": job_type,
            "status": "pending",
            "progress": 0.0,
            "is_gpu": True,
            "params": params or {},
        }

        # Create persisted job record
        if db_session is not None:
            from backend.models.job import BackgroundJob
            job = BackgroundJob(
                job_uuid=job_id,
                job_type=job_type,
                status="pending",
            )
            job.set_params(params or {})
            db_session.add(job)
            db_session.commit()

        # Launch as background task
        asyncio.create_task(
            self._run_gpu_task(job_id, task_fn, on_progress, db_session)
        )

        logger.info("GPU task submitted", job_id=job_id, job_type=job_type)
        await ws_manager.send_job_update(job_id, "pending", 0.0, "Waiting for GPU")
        return job_id

    async def submit_cpu_task(
        self,
        task_fn: Callable[..., Coroutine[Any, Any, Any]],
        job_type: str,
        params: dict[str, Any] | None = None,
        db_session: Any | None = None,
    ) -> str:
        """
        Submit a CPU-only task.

        No GPU semaphore — runs immediately as concurrent asyncio task.
        """
        job_id = str(uuid.uuid4())

        self._active_jobs[job_id] = {
            "job_id": job_id,
            "job_type": job_type,
            "status": "running",
            "progress": 0.0,
            "is_gpu": False,
        }

        asyncio.create_task(self._run_cpu_task(job_id, task_fn, db_session))
        logger.info("CPU task submitted", job_id=job_id, job_type=job_type)
        return job_id

    async def _run_gpu_task(
        self,
        job_id: str,
        task_fn: Callable,
        on_progress: Callable | None,
        db_session: Any | None,
    ) -> None:
        """Acquire GPU semaphore and run task."""
        import time

        logger.info("GPU task waiting for semaphore", job_id=job_id)

        async with self._gpu_semaphore:
            self._gpu_task_running = job_id
            self._active_jobs[job_id]["status"] = "running"

            await ws_manager.send_job_update(job_id, "running", 0.0, "GPU task started")

            if db_session is not None:
                self._update_db_job(db_session, job_id, status="running", started_at=time.time())

            logger.info("GPU task started", job_id=job_id)

            try:
                result = await task_fn()
                self._active_jobs[job_id]["status"] = "completed"
                self._active_jobs[job_id]["progress"] = 1.0

                await ws_manager.send_job_update(job_id, "completed", 1.0, "Done")

                if db_session is not None:
                    self._update_db_job(
                        db_session, job_id,
                        status="completed",
                        progress=1.0,
                        finished_at=time.time(),
                        result=result or {},
                    )

                logger.info("GPU task completed", job_id=job_id)

            except asyncio.CancelledError:
                self._active_jobs[job_id]["status"] = "cancelled"
                await ws_manager.send_job_update(job_id, "cancelled", 0.0)
                logger.info("GPU task cancelled", job_id=job_id)

            except Exception as e:
                error_msg = f"{type(e).__name__}: {e}"
                tb = traceback.format_exc()
                self._active_jobs[job_id]["status"] = "failed"

                await ws_manager.send_job_update(job_id, "failed", 0.0, error_msg)
                logger.error("GPU task failed", job_id=job_id, error=error_msg)

                if db_session is not None:
                    self._update_db_job(
                        db_session, job_id,
                        status="failed",
                        finished_at=time.time(),
                        error_message=error_msg[:1000],
                    )

            finally:
                self._gpu_task_running = None

    async def _run_cpu_task(
        self,
        job_id: str,
        task_fn: Callable,
        db_session: Any | None,
    ) -> None:
        """Run CPU task without GPU semaphore."""
        import time

        try:
            await task_fn()
            self._active_jobs[job_id]["status"] = "completed"
            await ws_manager.send_job_update(job_id, "completed", 1.0)

        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}"
            self._active_jobs[job_id]["status"] = "failed"
            await ws_manager.send_job_update(job_id, "failed", 0.0, error_msg)
            logger.error("CPU task failed", job_id=job_id, error=error_msg)

    def get_job_status(self, job_id: str) -> dict[str, Any] | None:
        """Get in-memory job status."""
        return self._active_jobs.get(job_id)

    def get_all_jobs(self) -> list[dict[str, Any]]:
        """Get all tracked jobs."""
        return list(self._active_jobs.values())

    def _update_db_job(
        self,
        db_session: Any,
        job_id: str,
        **kwargs: Any,
    ) -> None:
        """Update job record in database."""
        try:
            from sqlmodel import select
            from backend.models.job import BackgroundJob

            job = db_session.exec(
                select(BackgroundJob).where(BackgroundJob.job_uuid == job_id)
            ).first()

            if job:
                for key, value in kwargs.items():
                    if key == "result":
                        job.set_result(value)
                    elif hasattr(job, key):
                        setattr(job, key, value)
                db_session.add(job)
                db_session.commit()

        except Exception as e:
            logger.warning("DB job update failed", job_id=job_id, error=str(e))

    async def cancel_job(self, job_id: str) -> bool:
        """Request cancellation of a running job."""
        job = self._active_jobs.get(job_id)
        if not job:
            return False

        if job["status"] in ("completed", "failed", "cancelled"):
            return False

        # For GPU tasks: set stop flag (trainer checks this)
        job["stop_requested"] = True
        logger.info("Job cancellation requested", job_id=job_id)
        return True


# Global task router instance
task_router = TaskRouter(max_gpu_tasks=1)
