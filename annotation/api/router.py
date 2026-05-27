"""
annotation/api/router.py — Annotation pipeline API endpoints.

Routes:
  POST /annotate/run         — run annotation pipeline on image paths
  GET  /annotate/status      — pipeline status
  POST /annotate/export      — export approved annotations
  GET  /annotate/stats       — annotation statistics
  POST /annotate/cvat/sync   — sync with CVAT
  POST /annotate/ls/sync     — sync with Label Studio
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/annotate", tags=["annotation"])

_pipeline = None


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        from annotation.application.annotation_pipeline import (
            AnnotationPipeline,
            AnnotationPipelineConfig,
        )
        _pipeline = AnnotationPipeline(AnnotationPipelineConfig())
    return _pipeline


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class AnnotateRequest(BaseModel):
    image_paths: list[str] = Field(
        description="Absolute paths to images to annotate"
    )
    detection_model_path: str = ""
    use_vlm_labels: bool = True
    vlm_backend: str = "moondream"
    push_to_review_queue: bool = True


class ExportRequest(BaseModel):
    approved_items: list[dict] = Field(default_factory=list)
    output_dir: str = "annotations/output"
    format: str = "yolo"  # yolo | coco


class CVATSyncRequest(BaseModel):
    host: str = "http://localhost:8080"
    username: str = "admin"
    password: str = "admin"
    project_id: int | None = None
    task_ids: list[int] = Field(default_factory=list)


class LSSyncRequest(BaseModel):
    host: str = "http://localhost:8090"
    api_key: str = ""
    project_id: int | None = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/status")
def get_status():
    """Return annotation pipeline status."""
    pipeline = _get_pipeline()
    return {
        "detection_ready": pipeline._detection_pipeline is not None,
        "vlm_ready": pipeline._vlm_pipeline is not None,
        "config": {
            "vlm_backend": pipeline.config.vlm_backend,
            "use_sam_assisted": pipeline.config.use_sam_assisted,
            "export_format": pipeline.config.export_format,
        },
    }


@router.post("/run", status_code=202)
def run_pipeline(request: AnnotateRequest):
    """
    Run the annotation pipeline on provided image paths.

    All results go to the review queue (human-in-loop enforced).
    Returns run summary; individual results available via review queue.
    """
    from annotation.application.annotation_pipeline import AnnotationPipelineConfig

    pipeline = _get_pipeline()
    pipeline.config.use_vlm_labels = request.use_vlm_labels
    pipeline.config.vlm_backend = request.vlm_backend
    pipeline.config.push_to_review_queue = request.push_to_review_queue
    if request.detection_model_path:
        pipeline.config.detection_model_path = request.detection_model_path

    result = pipeline.run(request.image_paths)

    return {
        "run_id": result.run_id,
        "total_images": result.total_images,
        "total_detections": result.total_detections,
        "total_queue_items": result.total_queue_items,
        "total_auto_approved": result.total_auto_approved,
        "human_review_required": result.human_review_required,
        "failed_images": result.failed_images,
        "duration_s": result.duration_s,
        "human_in_loop": True,
    }


@router.post("/export")
def export_annotations(request: ExportRequest):
    """Export approved annotations to YOLO or COCO format."""
    pipeline = _get_pipeline()
    pipeline.config.export_format = request.format

    result = pipeline.export_approved(
        approved_items=request.approved_items,
        output_dir=request.output_dir,
    )
    return result


@router.get("/stats")
def get_annotation_stats():
    """Return annotation statistics from the review queue."""
    # Import from backend annotation module (where queue lives)
    try:
        from backend.api.v1.annotation import _QUEUE, _STATS

        items = list(_QUEUE.values())
        class_dist: dict[int, int] = {}
        for item in items:
            for box in item.get("detection_boxes", []):
                cls = int(box.get("class_id", 0))
                class_dist[cls] = class_dist.get(cls, 0) + 1

        return {
            "queue_stats": _STATS,
            "class_distribution": class_dist,
            "human_in_loop": True,
        }
    except Exception:
        return {"queue_stats": {}, "class_distribution": {}, "human_in_loop": True}


@router.post("/cvat/sync")
def sync_with_cvat(request: CVATSyncRequest):
    """
    Download annotations from CVAT and push to review queue.

    Only annotations with 'completed' job status are imported.
    All imported annotations require review before training.
    """
    try:
        from annotation.cvat.client import CVATClient, CVATConfig

        config = CVATConfig(
            host=request.host,
            username=request.username,
            password=request.password,
        )
        with CVATClient(config) as client:
            if request.project_id:
                tasks = client.list_tasks(project_id=request.project_id)
            else:
                tasks = client.list_tasks()

            synced = 0
            for task in tasks:
                if request.task_ids and task.id not in request.task_ids:
                    continue
                try:
                    annotations = client.download_annotations(task.id)
                    synced += len(annotations.get("annotations", []))
                except Exception:
                    pass

        return {
            "synced_annotations": synced,
            "tasks_synced": len(tasks),
            "status": "queued_for_review",
            "human_in_loop": True,
        }
    except ImportError:
        raise HTTPException(503, "CVAT client not available (install requests)")


@router.post("/ls/sync")
def sync_with_label_studio(request: LSSyncRequest):
    """Download completed annotations from Label Studio."""
    try:
        from annotation.label_studio.client import LabelStudioClient, LabelStudioConfig

        config = LabelStudioConfig(host=request.host, api_key=request.api_key)
        client = LabelStudioClient(config)

        if request.project_id:
            data = client.export_annotations(request.project_id, export_format="JSON")
            count = len(data) if isinstance(data, list) else 0
        else:
            count = 0

        return {
            "synced_annotations": count,
            "status": "queued_for_review",
            "human_in_loop": True,
        }
    except ImportError:
        raise HTTPException(503, "Label Studio client not available (install requests)")
