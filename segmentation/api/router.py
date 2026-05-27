"""
segmentation/api/router.py — Segmentation API endpoints.

Routes:
  POST /segment           — segment trichomes in an uploaded image
  POST /segment/batch     — batch segmentation (multiple images)
  GET  /segment/status    — loaded backend info
"""

from __future__ import annotations

import io
import time

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from segmentation.schemas.schemas import (
    BatchSegmentRequest,
    MaskData,
    SegmentRequest,
    SegmentResponse,
)

router = APIRouter(prefix="/segment", tags=["segmentation"])

_pipeline = None


def _get_pipeline(config=None):
    global _pipeline
    if _pipeline is None:
        from segmentation.application.segment_pipeline import SegmentPipeline, SegmentPipelineConfig
        _pipeline = SegmentPipeline(config or SegmentPipelineConfig())
    return _pipeline


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/status")
def get_status():
    """Return segmentation backend status."""
    pipeline = _get_pipeline()
    backend = pipeline._backend
    if backend is None:
        return {"loaded": False, "backend": None, "vram_required_gb": None}
    return {
        "loaded": True,
        "backend": type(backend).__name__,
        "vram_required_gb": getattr(backend, "vram_required_gb", None),
    }


@router.post("/", response_model=SegmentResponse)
async def segment_image(
    file: UploadFile = File(...),
    backend: str = "auto",
    score_threshold: float = 0.50,
    max_instances: int = 50,
    refine_masks: bool = True,
    export_polygons: bool = True,
):
    """
    Segment trichomes in an uploaded image.

    Detections from the detection pipeline are used as box prompts for SAM2.
    Returns per-instance masks, polygons, and geometric measurements.
    """
    contents = await file.read()
    arr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(400, "Could not decode image")

    h, w = image.shape[:2]

    from segmentation.application.segment_pipeline import SegmentPipelineConfig
    config = SegmentPipelineConfig(
        backend=backend,
        score_threshold=score_threshold,
        max_instances=max_instances,
        refine_masks=refine_masks,
        export_polygons=export_polygons,
    )

    t0 = time.perf_counter()
    pipeline = _get_pipeline(config)
    results = pipeline.run(image)
    inference_ms = (time.perf_counter() - t0) * 1000

    instances = []
    for i, seg in enumerate(results):
        instances.append(
            MaskData(
                instance_id=i,
                score=round(seg.score, 4),
                area_px=seg.area_px,
                centroid_x=round(seg.centroid[0], 2),
                centroid_y=round(seg.centroid[1], 2),
                bbox=seg.bbox,
                polygon=seg.polygon,
                circularity=round(seg.circularity, 4),
                diameter_um=round(seg.diameter_um, 2) if seg.diameter_um else None,
            )
        )

    backend_name = type(pipeline._backend).__name__ if pipeline._backend else "unknown"

    return SegmentResponse(
        instances=instances,
        total_instances=len(instances),
        backend_used=backend_name,
        inference_ms=round(inference_ms, 2),
        image_width=w,
        image_height=h,
    )


@router.post("/batch")
async def segment_batch(
    files: list[UploadFile] = File(...),
    backend: str = "auto",
    score_threshold: float = 0.50,
    max_instances_per_image: int = 50,
):
    """Segment multiple images in one request."""
    results = []
    for file in files[:20]:  # Cap at 20 images per batch
        contents = await file.read()
        arr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None:
            results.append({"filename": file.filename, "error": "decode_failed"})
            continue

        from segmentation.application.segment_pipeline import SegmentPipeline, SegmentPipelineConfig
        config = SegmentPipelineConfig(
            backend=backend,
            score_threshold=score_threshold,
            max_instances=max_instances_per_image,
        )
        pipeline = _get_pipeline(config)
        segs = pipeline.run(image)

        results.append(
            {
                "filename": file.filename,
                "instance_count": len(segs),
                "instances": [
                    {
                        "score": round(s.score, 4),
                        "area_px": s.area_px,
                        "circularity": round(s.circularity, 4),
                    }
                    for s in segs
                ],
            }
        )

    return {"results": results, "total_images": len(results)}
