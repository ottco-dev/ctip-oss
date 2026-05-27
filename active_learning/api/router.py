"""
active_learning/api/router.py — Active learning API endpoints.

Routes:
  GET  /al/status          — pipeline status + trigger state
  POST /al/cycle           — trigger an AL scoring cycle
  GET  /al/queue           — annotation priority queue (top-k)
  POST /al/queue/boost     — manually boost a queued item's priority
  POST /al/trigger         — force retraining trigger evaluation
  POST /al/annotated       — report N new annotations (updates trigger state)
  GET  /al/drift           — latest drift report
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/al", tags=["active-learning"])

# Lazy pipeline singleton
_pipeline = None


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        from active_learning.application.al_pipeline import (
            ActiveLearningPipeline,
            ALPipelineConfig,
        )
        _pipeline = ActiveLearningPipeline(config=ALPipelineConfig())
    return _pipeline


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class CycleRequest(BaseModel):
    unlabeled_samples: list[dict] = Field(
        default_factory=list,
        description="List of {sample_id, dataset_id, image_path} dicts",
    )
    model_predictions: list[dict] | None = Field(
        default=None,
        description="Pre-computed {sample_id, probabilities, confidence, predicted_class}",
    )


class BoostRequest(BaseModel):
    item_id: str
    boost_amount: float = Field(default=0.5, ge=0.0, le=2.0)


class AnnotatedReport(BaseModel):
    count: int = Field(default=1, ge=1)
    map50: float | None = None


class ManualTriggerRequest(BaseModel):
    reason: str = "manual"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/status")
def get_status():
    """Return active learning pipeline status."""
    return _get_pipeline().get_status()


@router.post("/cycle", status_code=202)
def run_cycle(request: CycleRequest):
    """
    Trigger an active learning scoring cycle.

    Scores the provided unlabeled pool by uncertainty, checks for drift,
    and pushes high-value samples to the annotation priority queue.
    """
    pipeline = _get_pipeline()
    result = pipeline.run_cycle(
        unlabeled_pool=request.unlabeled_samples,
        model_predictions=request.model_predictions,
    )
    return result.__dict__ | {
        "timestamp": result.timestamp.isoformat(),
    }


@router.get("/queue")
def get_queue(k: int = 20):
    """Return top-k items from the annotation priority queue."""
    from active_learning.queuing.priority_queue import get_global_queue

    q = get_global_queue()
    items = q.peek_top_k(k=k)
    return {
        "total_pending": len(q),
        "stats": q.stats().__dict__,
        "top_items": [
            {
                "item_id": e.item_id,
                "sample_id": e.sample_id,
                "dataset_id": e.dataset_id,
                "image_path": e.image_path,
                "priority": round(-e.neg_priority, 4),
                "uncertainty_score": e.uncertainty_score,
                "entropy_score": e.entropy_score,
                "predicted_class": e.predicted_class,
                "predicted_confidence": e.predicted_confidence,
                "status": e.status,
            }
            for e in items
        ],
    }


@router.post("/queue/boost")
def boost_queue_item(request: BoostRequest):
    """Manually increase priority of a queued item."""
    from active_learning.queuing.priority_queue import get_global_queue

    q = get_global_queue()
    ok = q.boost(request.item_id, request.boost_amount)
    if not ok:
        raise HTTPException(404, f"Item {request.item_id} not found or not pending")
    return {"item_id": request.item_id, "boosted": True}


@router.post("/annotated")
def report_annotated(payload: AnnotatedReport):
    """
    Report that N annotations have been approved.

    Updates trigger state and evaluates trigger conditions.
    Returns trigger decision.
    """
    pipeline = _get_pipeline()
    pipeline.on_annotation_approved(payload.count)
    if payload.map50 is not None:
        pipeline.update_model_metrics(payload.map50)

    # Evaluate trigger (non-firing — just returns decision)
    from active_learning.retraining.trigger import RetrainingTrigger
    trigger = pipeline._trigger
    if trigger is None:
        return {"trigger": None}

    decision = trigger.evaluate()
    return {
        "trigger": decision.to_dict(),
        "trigger_status": trigger.get_status(),
    }


@router.post("/trigger")
def force_trigger(request: ManualTriggerRequest):
    """Manually force a retraining trigger evaluation."""
    pipeline = _get_pipeline()
    trigger = pipeline._trigger
    if trigger is None:
        pipeline._init_components()
        trigger = pipeline._trigger

    decision = trigger.manual_trigger(reason=request.reason)
    return decision.to_dict()


@router.get("/drift")
def get_drift_status():
    """Return the last drift detection result."""
    pipeline = _get_pipeline()
    trigger = pipeline._trigger
    if trigger is None:
        return {"drift_score": 0.0, "last_drift_score": None}
    return {
        "last_drift_score": trigger.state.last_drift_score,
        "drift_threshold": trigger.config.drift_score_threshold,
        "drift_detected": trigger.state.last_drift_score >= trigger.config.drift_score_threshold,
    }
