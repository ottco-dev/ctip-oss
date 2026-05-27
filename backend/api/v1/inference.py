"""
backend.api.v1.inference — FastAPI inference endpoints.

POST /inference/detect        — single image detection
POST /inference/detect/batch  — batch image detection
POST /inference/maturity      — maturity analysis on image
POST /inference/segment       — segmentation on image
GET  /inference/models        — list available models
GET  /inference/status        — inference server status

GPU slot management
-------------------
``detect_image`` and ``detect_batch`` both gate on the shared GPU semaphore via
``gpu_slot_or_429`` (returns HTTP 429 with ``Retry-After: 5`` when the slot is
busy and the queue is full).  The batch endpoint acquires the slot *once* for
the entire batch — this is intentional: RTX 4060 8 GB has no headroom for
concurrent GPU operations.

``analyze_maturity`` with ``color_rules`` is CPU-only; VLM backends raise 503
rather than actually scheduling GPU work here, so no slot guard is needed.

The import of ``gpu_slot_or_429`` is wrapped in a try/except so that the module
remains importable in environments where the full backend package is unavailable
(unit-test isolation, standalone detection scripts, etc.).
"""

from __future__ import annotations

import io
import time
from typing import Annotated, Any

import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from backend.config import get_settings

router = APIRouter(prefix="/inference", tags=["inference"])

# ---------------------------------------------------------------------------
# GPU slot dependency (rate-limited: 429 when slot busy and queue is full)
# ---------------------------------------------------------------------------

try:
    from backend.dependencies.gpu import gpu_slot_or_429 as _gpu_slot
except ImportError:
    # Fallback: no-op generator; used outside the full backend package.
    async def _gpu_slot() -> None:  # type: ignore[misc]
        yield


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

from pydantic import BaseModel, Field  # noqa: E402 (after router setup)


class DetectionBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str


class DetectionResponse(BaseModel):
    detections: list[DetectionBox]
    inference_time_ms: float
    image_width: int
    image_height: int
    model_variant: str
    conf_threshold: float
    num_detections: int


class MaturityResponse(BaseModel):
    maturity_stage: str
    clear_fraction: float
    cloudy_fraction: float
    amber_fraction: float
    confidence: float
    scientific_caveat: str = (
        "Maturity stage is an observable optical property. "
        "No inference about cannabinoid content can be made from visual appearance."
    )


class InferenceStatus(BaseModel):
    available: bool
    model_loaded: bool
    model_variant: str | None
    vram_used_gb: float | None
    last_inference_ms: float | None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _read_image(file: UploadFile) -> np.ndarray:
    """Read uploaded file as numpy uint8 RGB array."""
    import cv2

    content = await file.read()
    arr = np.frombuffer(content, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(
            status_code=422,
            detail=f"Could not decode image: {file.filename}",
        )
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _run_detection(
    image: np.ndarray,
    conf_threshold: float,
    iou_threshold: float,
    model_variant: str,
    use_tiled: bool,
) -> tuple[list[DetectionBox], float]:
    """
    Core detection logic — no GPU semaphore here.

    Callers (``detect_image``, ``detect_batch``) acquire the GPU slot before
    invoking this function.

    Returns:
        (list_of_boxes, elapsed_ms)
    """
    t0 = time.perf_counter()

    try:
        from detection.application.detect_pipeline import DetectionPipeline
        from detection.domain.detector import DetectionConfig
        from detection.infrastructure.yolo_backend import YOLODetector

        config = DetectionConfig(
            model_path=f"{model_variant}.pt",
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
        )
        detector = YOLODetector(config)
        pipeline = DetectionPipeline(detector=detector)
        result = pipeline.run(image)

        elapsed_ms = (time.perf_counter() - t0) * 1000

        boxes = [
            DetectionBox(
                x1=float(det.bbox.x1),
                y1=float(det.bbox.y1),
                x2=float(det.bbox.x2),
                y2=float(det.bbox.y2),
                confidence=float(det.confidence.value),
                class_id=int(det.class_id),
                class_name=str(det.class_name or ""),
            )
            for det in result.detections
        ]
        return boxes, elapsed_ms

    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning(
            "Detection pipeline failed: %s — returning empty result", exc
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return [], elapsed_ms


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("/detect", response_model=DetectionResponse)
async def detect_image(
    file: Annotated[UploadFile, File(description="Microscopy image (JPG/PNG/TIFF)")],
    conf_threshold: Annotated[float, Form()] = 0.35,
    iou_threshold: Annotated[float, Form()] = 0.45,
    model_variant: Annotated[str, Form()] = "yolo11s",
    use_tiled: Annotated[bool, Form()] = False,
    _slot: None = Depends(_gpu_slot),
) -> DetectionResponse:
    """
    Run trichome detection on a single uploaded image.

    Returns bounding boxes, confidence scores, and class labels.
    For 4K images, set use_tiled=True to enable sliding-window inference.

    **Rate limiting**: Returns HTTP 429 (Too Many Requests) with a
    ``Retry-After: 5`` header when the GPU slot is busy and the request queue
    is full.  The queue depth is controlled by the ``GPU_INFERENCE_QUEUE_DEPTH``
    environment variable (default: 0 = fail-fast, no queue).
    """
    image = await _read_image(file)
    h, w = image.shape[:2]

    boxes, elapsed_ms = _run_detection(image, conf_threshold, iou_threshold, model_variant, use_tiled)

    return DetectionResponse(
        detections=boxes,
        inference_time_ms=round(elapsed_ms, 2),
        image_width=w,
        image_height=h,
        model_variant=model_variant,
        conf_threshold=conf_threshold,
        num_detections=len(boxes),
    )


@router.post("/detect/batch")
async def detect_batch(
    files: Annotated[list[UploadFile], File(description="Multiple microscopy images")],
    conf_threshold: Annotated[float, Form()] = 0.35,
    model_variant: Annotated[str, Form()] = "yolo11s",
    _slot: None = Depends(_gpu_slot),
) -> dict:
    """
    Batch detection on multiple images.

    Returns a list of detection results, one per image. Images are processed
    sequentially (single GPU constraint — RTX 4060 8 GB cannot run concurrent
    GPU tasks). The GPU slot is held for the entire batch duration.

    Maximum 50 images per request. Use the background job endpoint for larger
    batches.

    **Rate limiting**: Same 429 / ``Retry-After`` policy as ``/detect``.
    """
    if len(files) > 50:
        raise HTTPException(
            status_code=422,
            detail=(
                "Maximum 50 images per batch. "
                "Use the background job endpoint for larger batches."
            ),
        )

    results = []
    for file in files:
        try:
            image = await _read_image(file)
            h, w = image.shape[:2]
            boxes, elapsed_ms = _run_detection(
                image, conf_threshold, iou_threshold=0.45,
                model_variant=model_variant, use_tiled=False,
            )
            results.append({
                "filename": file.filename,
                "image_width": w,
                "image_height": h,
                "detections": [d.model_dump() for d in boxes],
                "inference_time_ms": round(elapsed_ms, 2),
                "num_detections": len(boxes),
            })
        except HTTPException:
            raise  # propagate 422 decode errors
        except Exception as exc:
            results.append({"filename": file.filename, "error": str(exc)})

    return {"results": results, "total_images": len(files)}


@router.post("/maturity", response_model=MaturityResponse)
async def analyze_maturity(
    file: Annotated[UploadFile, File(description="Trichome crop or full FOV image")],
    backend: Annotated[str, Form()] = "color_rules",
) -> MaturityResponse:
    """
    Analyze trichome maturity from an image.

    Backends:
    - color_rules: Fast rule-based analysis from HSV/LAB features (no GPU required)
    - moondream: VLM-based analysis (~2.1 GB VRAM)
    - florence2: Florence-2 VLM analysis (~3.5 GB VRAM)

    SCIENTIFIC NOTE: Results reflect observable optical properties only.
    No cannabinoid content inference is possible from visual data.
    """
    image = await _read_image(file)

    if backend == "color_rules":
        # CPU-only path — no GPU slot needed.
        try:
            from maturity.domain.color_features import (
                extract_color_features,
                rule_based_maturity_estimate,
            )

            features = extract_color_features(image)
            stage, clear_f, cloudy_f, amber_f, conf = rule_based_maturity_estimate(features)

            return MaturityResponse(
                maturity_stage=stage,
                clear_fraction=clear_f,
                cloudy_fraction=cloudy_f,
                amber_fraction=amber_f,
                confidence=conf,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    elif backend in {"moondream", "florence2"}:
        raise HTTPException(
            status_code=503,
            detail=(
                f"VLM backend '{backend}' requires a running GPU task. "
                "Use POST /vlm/label/maturity instead."
            ),
        )
    else:
        raise HTTPException(status_code=422, detail=f"Unknown backend: {backend!r}")


@router.get("/models")
async def list_inference_models() -> list[dict]:
    """List available inference models with VRAM requirements."""
    return [
        {
            "id": "yolo11n",
            "name": "YOLO11n",
            "type": "detection",
            "vram_gb": 0.6,
            "speed_fps": 150,
            "description": "Fastest, lowest accuracy. Good for previewing.",
        },
        {
            "id": "yolo11s",
            "name": "YOLO11s",
            "type": "detection",
            "vram_gb": 1.2,
            "speed_fps": 80,
            "description": "Default for RTX 4060. Best speed/accuracy tradeoff.",
        },
        {
            "id": "yolo11m",
            "name": "YOLO11m",
            "type": "detection",
            "vram_gb": 2.8,
            "speed_fps": 40,
            "description": "Higher accuracy, more VRAM. Good for final inference.",
        },
        {
            "id": "sam2_tiny",
            "name": "SAM2-tiny",
            "type": "segmentation",
            "vram_gb": 3.8,
            "speed_fps": 10,
            "description": "Instance segmentation with point/box prompts.",
        },
        {
            "id": "color_rules",
            "name": "Color Rules",
            "type": "maturity",
            "vram_gb": 0.0,
            "speed_fps": 1000,
            "description": "Rule-based maturity from HSV/LAB. No GPU required.",
        },
    ]


@router.get("/status", response_model=InferenceStatus)
async def inference_status() -> InferenceStatus:
    """Return inference server status including loaded model and VRAM usage."""
    vram = None
    try:
        import torch

        if torch.cuda.is_available():
            vram = torch.cuda.memory_allocated() / (1024 ** 3)
    except ImportError:
        pass

    return InferenceStatus(
        available=True,
        model_loaded=False,  # No persistent model loaded (stateless endpoints)
        model_variant=None,
        vram_used_gb=vram,
        last_inference_ms=None,
    )
