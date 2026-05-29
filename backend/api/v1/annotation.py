"""
backend/api/v1/annotation.py — Annotation queue and job management endpoints.

Routes:
  GET    /annotation/queue          — list review queue
  POST   /annotation/queue          — add item to review queue
  GET    /annotation/queue/{id}     — get queue item detail
  PUT    /annotation/queue/{id}     — approve / reject / edit item
  DELETE /annotation/queue/{id}     — remove from queue
  POST   /annotation/auto-label     — trigger VLM auto-labeling job
  GET    /annotation/jobs           — list annotation jobs
  GET    /annotation/jobs/{id}      — job detail
  POST   /annotation/stats          — queue statistics
  GET    /annotation/agreement      — inter-annotator agreement summary
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlmodel import Session, select

from backend.database import get_session
from backend.models.job import BackgroundJob, JobStatus

router = APIRouter(prefix="/annotation", tags=["annotation"])

# ---------------------------------------------------------------------------
# In-memory queue (replace with DB table in production)
# ---------------------------------------------------------------------------
_QUEUE: dict[str, dict[str, Any]] = {}
_STATS: dict[str, Any] = {
    "total_submitted": 0,
    "approved": 0,
    "rejected": 0,
    "pending": 0,
}


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class QueueItem(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    sample_id: str = ""
    dataset_id: str = ""
    image_path: str = ""
    vlm_labels: list[dict[str, Any]] = Field(default_factory=list)
    vlm_backend: str = "moondream"
    priority: int = Field(default=1, ge=0, le=3)  # 0=Low 1=Med 2=High 3=Crit
    status: str = "pending"  # pending | approved | rejected | edited
    confidence: float = 0.0
    submitted_at: datetime = Field(default_factory=datetime.utcnow)
    reviewed_at: datetime | None = None
    reviewer_note: str = ""
    maturity_fractions: dict[str, float] | None = None
    detection_boxes: list[dict[str, Any]] = Field(default_factory=list)
    scientific_caveat: str = (
        "Visual maturity analysis does NOT allow quantitative THC/CBD determination."
    )
    # VLM-specific fields (populated by auto-label pipeline)
    maturity_stage: str | None = None
    clear_fraction: float = 0.0
    cloudy_fraction: float = 0.0
    amber_fraction: float = 0.0
    hallucination_flags: list[str] = Field(default_factory=list)
    review_priority: int = 1
    filename: str = ""
    vlm_confidence: float = 0.0
    queued_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class QueueItemCreate(BaseModel):
    sample_id: str
    dataset_id: str
    image_path: str
    vlm_labels: list[dict[str, Any]] = Field(default_factory=list)
    vlm_backend: str = "moondream"
    priority: int = 1
    confidence: float = 0.0
    maturity_fractions: dict[str, float] | None = None
    detection_boxes: list[dict[str, Any]] = Field(default_factory=list)


class QueueItemUpdate(BaseModel):
    status: str  # approved | rejected | edited
    reviewer_note: str = ""
    edited_labels: list[dict[str, Any]] | None = None
    edited_maturity_fractions: dict[str, float] | None = None


class AutoLabelRequest(BaseModel):
    dataset_id: str
    sample_ids: list[str] | None = None  # None = all unlabeled
    vlm_backend: str = "moondream"  # moondream | florence2 | qwen2vl
    max_samples: int = 100
    batch_size: int = 50  # alias accepted from frontend
    confidence_threshold: float = 0.40
    # VLM Configuration panel fields (Aufgabe 2)
    provider_id: str | None = None         # remote VLM provider (overrides vlm_backend when set)
    model_id: str | None = None            # specific model within the provider
    prompt_name: str | None = None         # named prompt preset
    custom_system_prompt: str | None = None
    custom_user_prompt: str | None = None
    ensemble_mode: bool = False
    ensemble_providers: list[str] | None = None  # provider IDs for ensemble


class QueueStats(BaseModel):
    total_submitted: int
    pending: int
    approved: int
    rejected: int
    approval_rate: float
    average_confidence: float
    by_priority: dict[str, int]
    by_backend: dict[str, int]


class AgreementSummary(BaseModel):
    total_reviewed: int
    cohens_kappa: float | None
    average_confidence: float
    note: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/queue", response_model=list[QueueItem])
def list_queue(
    status: str | None = Query(None, description="Filter by status"),
    priority: int | None = Query(None, description="Filter by priority"),
    dataset_id: str | None = Query(None),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """Return annotation review queue, ordered by priority desc then submitted_at asc."""
    items = list(_QUEUE.values())

    if status:
        items = [i for i in items if i["status"] == status]
    if priority is not None:
        items = [i for i in items if i["priority"] == priority]
    if dataset_id:
        items = [i for i in items if i["dataset_id"] == dataset_id]

    # Sort: priority desc, then submitted_at asc
    items.sort(key=lambda x: (-x["priority"], x["submitted_at"]))
    page = items[offset : offset + limit]
    return [QueueItem(**i) for i in page]


@router.post("/queue", response_model=QueueItem, status_code=201)
def add_to_queue(payload: QueueItemCreate):
    """Add a VLM-generated label to the human review queue."""
    item = QueueItem(
        sample_id=payload.sample_id,
        dataset_id=payload.dataset_id,
        image_path=payload.image_path,
        vlm_labels=payload.vlm_labels,
        vlm_backend=payload.vlm_backend,
        priority=payload.priority,
        confidence=payload.confidence,
        maturity_fractions=payload.maturity_fractions,
        detection_boxes=payload.detection_boxes,
    )
    _QUEUE[item.id] = item.model_dump()
    _STATS["total_submitted"] += 1
    _STATS["pending"] += 1
    return item


@router.get("/queue/{item_id}", response_model=QueueItem)
def get_queue_item(item_id: str):
    if item_id not in _QUEUE:
        raise HTTPException(404, "Queue item not found")
    return QueueItem(**_QUEUE[item_id])


@router.put("/queue/{item_id}", response_model=QueueItem)
def update_queue_item(item_id: str, payload: QueueItemUpdate):
    """Approve, reject, or submit edited labels for a queue item."""
    if item_id not in _QUEUE:
        raise HTTPException(404, "Queue item not found")

    valid_statuses = {"approved", "rejected", "edited"}
    if payload.status not in valid_statuses:
        raise HTTPException(422, f"status must be one of {valid_statuses}")

    item = _QUEUE[item_id]
    old_status = item["status"]

    if old_status == "pending":
        _STATS["pending"] = max(0, _STATS["pending"] - 1)

    item["status"] = payload.status
    item["reviewer_note"] = payload.reviewer_note
    item["reviewed_at"] = datetime.utcnow().isoformat()

    if payload.edited_labels is not None:
        item["vlm_labels"] = payload.edited_labels
    if payload.edited_maturity_fractions is not None:
        item["maturity_fractions"] = payload.edited_maturity_fractions

    if payload.status == "approved":
        _STATS["approved"] += 1
    elif payload.status == "rejected":
        _STATS["rejected"] += 1
    # "edited" counts as approved for training purposes but tracked separately

    _QUEUE[item_id] = item
    return QueueItem(**item)


@router.delete("/queue/{item_id}", status_code=204)
def delete_queue_item(item_id: str):
    if item_id not in _QUEUE:
        raise HTTPException(404, "Queue item not found")
    item = _QUEUE.pop(item_id)
    if item["status"] == "pending":
        _STATS["pending"] = max(0, _STATS["pending"] - 1)


# ---------------------------------------------------------------------------
# Auto-label job
# ---------------------------------------------------------------------------


async def _run_auto_label(job_uuid: str, request: AutoLabelRequest) -> None:
    """
    Async GPU background task: VLM auto-labeling pipeline.

    Finds dataset images → runs AutoLabelPipeline → pushes to review queue.
    GPU semaphore is acquired inside so it queues behind running training/inference.
    """
    import asyncio
    from pathlib import Path

    from shared.logging.logger import get_logger as _get_logger
    _log = _get_logger(__name__)

    def _db_update(status: str, progress: float = 0.0, error: str | None = None, result: dict | None = None) -> None:
        try:
            from backend.database import get_session as _gs
            from backend.models.job import BackgroundJob as _BJ
            from sqlmodel import select as _sel
            with next(_gs()) as _db:
                j = _db.exec(_sel(_BJ).where(_BJ.job_uuid == job_uuid)).first()
                if j:
                    j.status = status
                    j.progress = progress
                    if error:
                        j.error_message = error[:1000]
                    if result:
                        j.set_result(result)
                    _db.add(j)
                    _db.commit()
        except Exception:
            pass

    try:
        from backend.config import get_settings
        from backend.tasks.task_router import task_router
        from vlm_labeling.application.auto_label_pipeline import (
            AutoLabelPipeline,
            AutoLabelPipelineConfig,
        )

        settings = get_settings()
        data_root = Path("./data/datasets")

        # Resolve dataset directory by name or numeric DB id
        candidate_dirs: list[Path] = []
        ds_id = request.dataset_id.strip()
        candidate_dirs.append(data_root / ds_id)

        if ds_id.isdigit():
            try:
                from backend.database import get_session as _gs
                from backend.models.dataset import Dataset as _DS
                with next(_gs()) as _db:
                    ds = _db.get(_DS, int(ds_id))
                    if ds:
                        if ds.root_path:
                            candidate_dirs.insert(0, Path(ds.root_path))
                        candidate_dirs.insert(0, data_root / ds.name)
            except Exception:
                pass

        image_paths: list[Path] = []
        for d in candidate_dirs:
            for split in ("train", "val", ""):
                img_dir = d / "images" / split if split else d / "images"
                if img_dir.exists():
                    found = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
                    image_paths.extend(found)
            if image_paths:
                break

        if not image_paths:
            _db_update("failed", error=f"No images found for dataset '{ds_id}'. "
                       f"Checked: {[str(p) for p in candidate_dirs]}")
            return

        max_imgs = min(request.max_samples or request.batch_size, len(image_paths))
        image_paths = image_paths[:max_imgs]
        _log.info("Auto-label: images found", count=len(image_paths), dataset=ds_id)

        _db_update("running", 0.0)

        async with task_router._gpu_semaphore:
            config = AutoLabelPipelineConfig(
                vlm_backend=request.vlm_backend,
                min_vlm_confidence=request.confidence_threshold,
                enable_hallucination_filter=True,
                max_images=max_imgs,
            )
            pipeline = AutoLabelPipeline(config)

            def _run_sync() -> tuple:
                pipeline.load()
                try:
                    return pipeline.run(image_paths)
                finally:
                    pipeline.unload()

            loop = asyncio.get_event_loop()
            labels, stats = await loop.run_in_executor(None, _run_sync)

        pushed = 0
        for label in labels:
            item_id = label.label_id
            now_iso = datetime.utcnow().isoformat()
            _QUEUE[item_id] = {
                "id": item_id,
                "sample_id": label.image_id,
                "dataset_id": request.dataset_id,
                "image_path": label.image_path,
                "filename": Path(label.image_path).name,
                "vlm_labels": [{"maturity_stage": label.maturity_stage}] if label.maturity_stage else [],
                "vlm_backend": request.vlm_backend,
                "vlm_confidence": label.vlm_confidence,
                "confidence": label.vlm_confidence,
                "priority": (label.filter_result.review_priority if label.filter_result else 1),
                "review_priority": (label.filter_result.review_priority if label.filter_result else 1),
                "status": "pending",
                "submitted_at": now_iso,
                "queued_at": now_iso,
                "maturity_stage": label.maturity_stage,
                "clear_fraction": label.clear_fraction or 0.0,
                "cloudy_fraction": label.cloudy_fraction or 0.0,
                "amber_fraction": label.amber_fraction or 0.0,
                "hallucination_flags": label.hallucination_flags,
                "maturity_fractions": {
                    "clear": label.clear_fraction or 0.0,
                    "cloudy": label.cloudy_fraction or 0.0,
                    "amber": label.amber_fraction or 0.0,
                },
                "detection_boxes": [],
                "reviewer_note": "",
                "scientific_caveat": (
                    "Visual maturity analysis does NOT allow quantitative THC/CBD determination."
                ),
            }
            _STATS["total_submitted"] = _STATS.get("total_submitted", 0) + 1
            _STATS["pending"] = _STATS.get("pending", 0) + 1
            pushed += 1

        result_summary = {**stats.to_dict(), "pushed_to_queue": pushed}
        _db_update("completed", progress=1.0, result=result_summary)
        _log.info("Auto-label complete", job_uuid=job_uuid, pushed=pushed)

    except Exception as exc:
        import traceback
        err = f"{type(exc).__name__}: {exc}"
        _db_update("failed", error=err)
        _log.error("Auto-label failed", job_uuid=job_uuid, error=err)


@router.post("/auto-label", status_code=202)
def trigger_auto_label(
    request: AutoLabelRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_session),
):
    """
    Start a background VLM auto-labeling job.

    Results are pushed to the review queue and require human approval before
    they are eligible for model training (human-in-loop invariant).
    """
    job_uuid_str = str(uuid.uuid4())
    job = BackgroundJob(
        job_uuid=job_uuid_str,
        job_type="auto_label",
        status=JobStatus.PENDING,
        params_json=request.model_dump_json(),
    )
    db.add(job)
    db.commit()

    background_tasks.add_task(_run_auto_label, job_uuid_str, request)

    return {
        "job_id": job_uuid_str,
        "status": "queued",
        "message": (
            f"Auto-label job queued for dataset {request.dataset_id} "
            f"using {request.vlm_backend}. Results require human review before "
            "entering training data."
        ),
        "human_in_loop": True,
    }


# ---------------------------------------------------------------------------
# Job list
# ---------------------------------------------------------------------------


@router.get("/jobs")
def list_annotation_jobs(
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_session),
):
    """List recent annotation-related background jobs."""
    stmt = (
        select(BackgroundJob)
        .where(BackgroundJob.job_type.in_(["auto_label", "sam_assist", "annotation_sync"]))
        .order_by(BackgroundJob.created_at.desc())
        .limit(limit)
    )
    jobs = db.exec(stmt).all()
    def _fmt_ts(ts: object) -> str | None:
        if ts is None:
            return None
        if hasattr(ts, "isoformat"):
            return ts.isoformat()
        return datetime.utcfromtimestamp(float(ts)).isoformat()

    return [
        {
            "id": j.job_uuid,
            "job_uuid": j.job_uuid,
            "type": j.job_type,
            "status": j.status.value if hasattr(j.status, "value") else j.status,
            "progress": j.progress,
            "created_at": _fmt_ts(j.created_at),
            "params": json.loads(j.params_json) if j.params_json else {},
        }
        for j in jobs
    ]


@router.get("/jobs/{job_id}")
def get_annotation_job(job_id: str, db: Session = Depends(get_session)):
    job = db.exec(select(BackgroundJob).where(BackgroundJob.job_uuid == job_id)).first()
    if not job:
        raise HTTPException(404, "Job not found")

    def _fmt(ts: object) -> str | None:
        if ts is None:
            return None
        if hasattr(ts, "isoformat"):
            return ts.isoformat()
        return datetime.utcfromtimestamp(float(ts)).isoformat()

    return {
        "id": job.job_uuid,
        "job_uuid": job.job_uuid,
        "type": job.job_type,
        "status": job.status.value if hasattr(job.status, "value") else job.status,
        "progress": job.progress,
        "error_message": job.error_message,
        "created_at": _fmt(job.created_at),
        "finished_at": _fmt(job.finished_at),
        "result": json.loads(job.result_json) if job.result_json else None,
        "params": json.loads(job.params_json) if job.params_json else {},
    }


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


@router.get("/stats", response_model=QueueStats)
def get_queue_stats():
    """Return aggregated annotation queue statistics."""
    items = list(_QUEUE.values())
    total = len(items)

    by_priority: dict[str, int] = {"0": 0, "1": 0, "2": 0, "3": 0}
    by_backend: dict[str, int] = {}
    confidence_sum = 0.0

    for item in items:
        p = str(item.get("priority", 1))
        by_priority[p] = by_priority.get(p, 0) + 1
        backend = item.get("vlm_backend", "unknown")
        by_backend[backend] = by_backend.get(backend, 0) + 1
        confidence_sum += item.get("confidence", 0.0)

    approved = _STATS["approved"]
    rejected = _STATS["rejected"]
    total_reviewed = approved + rejected
    approval_rate = approved / total_reviewed if total_reviewed > 0 else 0.0

    return QueueStats(
        total_submitted=_STATS["total_submitted"],
        pending=sum(1 for i in items if i["status"] == "pending"),
        approved=approved,
        rejected=rejected,
        approval_rate=round(approval_rate, 3),
        average_confidence=round(confidence_sum / total if total > 0 else 0.0, 3),
        by_priority=by_priority,
        by_backend=by_backend,
    )


# ---------------------------------------------------------------------------
# Inter-annotator agreement
# ---------------------------------------------------------------------------


@router.get("/agreement", response_model=AgreementSummary)
def get_agreement_summary():
    """
    Return inter-annotator agreement summary for reviewed items.

    Cohen's kappa computation requires at least 2 annotators reviewing
    the same items — not yet available without a multi-annotator workflow.
    """
    reviewed = [i for i in _QUEUE.values() if i["status"] in {"approved", "rejected", "edited"}]
    total = len(reviewed)
    avg_conf = (
        sum(i.get("confidence", 0.0) for i in reviewed) / total if total > 0 else 0.0
    )

    return AgreementSummary(
        total_reviewed=total,
        cohens_kappa=None,  # Requires multi-annotator data
        average_confidence=round(avg_conf, 3),
        note=(
            "Cohen's kappa requires multiple annotators reviewing the same images. "
            "Connect CVAT or Label Studio for multi-annotator workflows."
        ),
    )
