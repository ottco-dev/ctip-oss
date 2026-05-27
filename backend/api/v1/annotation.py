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
    sample_id: str
    dataset_id: str
    image_path: str
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
    confidence_threshold: float = 0.70


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


def _run_auto_label(job_id: str, request: AutoLabelRequest):
    """Background task: invoke VLM auto-labeling and push results to queue."""
    try:
        # Lazy import to avoid startup overhead
        from vlm_labeling.application.auto_label_pipeline import AutoLabelPipeline  # noqa: F401

        # In production: iterate samples, call VLM, push each result to queue
        # For now: mock progress updates via job status
        time.sleep(1)  # Simulate some processing time placeholder

    except Exception as exc:  # noqa: BLE001
        # Update job status to failed (best-effort)
        pass


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
    job = BackgroundJob(
        job_type="auto_label",
        status=JobStatus.PENDING,
        params_json=request.model_dump_json(),
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    background_tasks.add_task(_run_auto_label, str(job.id), request)

    return {
        "job_id": str(job.id),
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
    return [
        {
            "id": str(j.id),
            "type": j.job_type,
            "status": j.status.value if hasattr(j.status, "value") else j.status,
            "progress": j.progress,
            "created_at": j.created_at.isoformat() if j.created_at else None,
            "params": json.loads(j.params_json) if j.params_json else {},
        }
        for j in jobs
    ]


@router.get("/jobs/{job_id}")
def get_annotation_job(job_id: str, db: Session = Depends(get_session)):
    job = db.get(BackgroundJob, job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return {
        "id": str(job.id),
        "type": job.job_type,
        "status": job.status.value if hasattr(job.status, "value") else job.status,
        "progress": job.progress,
        "error_message": job.error_message,
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "finished_at": job.finished_at.isoformat() if job.finished_at else None,
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
