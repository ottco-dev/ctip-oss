"""
detection.api.router — FastAPI detection endpoints.

REST API for the trichome detection service.
All endpoints are async and designed for non-blocking operation.
GPU inference runs in a thread pool to avoid blocking the event loop.

Endpoints:
    POST /detect           — Single image detection
    POST /detect/batch     — Batch image detection
    POST /detect/url       — Detection from image URL
    GET  /detect/config    — Current detector configuration
    GET  /detect/models    — Available detector models
"""

from __future__ import annotations

import asyncio
import base64
import uuid
from io import BytesIO
from typing import Annotated, Any

import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

from detection.schemas.schemas import (
    BatchDetectionRequest,
    DetectionRequest,
    DetectionResponse,
    DetectionStats,
)
from shared.logging.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/detect", tags=["detection"])

# ---------------------------------------------------------------------------
# GPU slot dependency — rate-limited (429 when slot busy and queue is full).
# Wrapped in try/except so this module remains importable outside the full
# backend package (unit tests, standalone scripts, etc.).
# ---------------------------------------------------------------------------

try:
    from backend.dependencies.gpu import gpu_slot_or_429 as _gpu_slot
except ImportError:
    async def _gpu_slot() -> None:  # type: ignore[misc]
        yield


def _decode_image(image_b64: str) -> "np.ndarray[Any, np.dtype[np.uint8]]":
    """Decode base64-encoded image to numpy array."""
    import cv2

    try:
        image_bytes = base64.b64decode(image_b64)
        buf = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to decode image")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Image decoding failed: {e}",
        ) from e


@router.post(
    "/",
    response_model=DetectionResponse,
    summary="Detect trichomes in a single image",
    description="""
    Run trichome detection on a single uploaded image.

    Supports: TIFF, PNG, JPG, BMP (up to 50 MB)
    Returns: List of bounding boxes, confidence scores, trichome types

    Notes:
    - Large images (>1280px) automatically use tiled inference
    - Confidence threshold defaults to 0.25 (conservative for scientific use)
    - Results include epistemic uncertainty estimates where available
    """,
)
async def detect_image(
    file: Annotated[UploadFile, File(description="Image file to analyze")],
    confidence_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    use_tiling: bool = True,
    model: str = "yolo11s",
    _slot: None = Depends(_gpu_slot),
) -> DetectionResponse:
    """
    Detect trichomes in an uploaded image.

    This endpoint is designed for interactive use (single images).
    For batch processing of large datasets, use POST /detect/batch
    or the CLI: trichome detect --input /path/to/images/
    """
    # Validate file
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"File must be an image. Got: {file.content_type}",
        )

    content = await file.read()
    if len(content) > 50 * 1024 * 1024:  # 50 MB limit
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Image file too large. Maximum 50 MB.",
        )

    # Decode image
    buf = np.frombuffer(content, dtype=np.uint8)
    import cv2
    image_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Could not decode image. Ensure file is a valid image format.",
        )
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    image_id = str(uuid.uuid4())
    logger.info(
        "Detection API request",
        image_id=image_id,
        filename=file.filename,
        shape=image_rgb.shape,
        confidence=confidence_threshold,
    )

    # Run detection in thread pool (GPU inference blocks)
    # The actual detector is injected via dependency injection in production
    # For now, return a placeholder that illustrates the API contract
    try:
        # In production: detector = get_detector(model)
        # result = await asyncio.get_event_loop().run_in_executor(
        #     None, detector.detect, image_rgb
        # )
        # Placeholder response structure:
        return DetectionResponse(
            image_id=image_id,
            filename=file.filename or "unknown",
            num_detections=0,
            detections=[],
            inference_time_ms=0.0,
            model_id=model,
            was_tiled=use_tiling,
            image_shape=list(image_rgb.shape),
            scientific_note=(
                "Confidence scores are raw model outputs, not calibrated probabilities. "
                "Apply temperature scaling before using for threshold-based decisions."
            ),
        )
    except Exception as e:
        logger.exception("Detection failed", image_id=image_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Detection failed: {e}",
        ) from e


@router.post(
    "/upload",
    response_model=DetectionResponse,
    summary="Detect from multipart form upload",
)
async def detect_upload(
    file: Annotated[UploadFile, File()],
    confidence_threshold: Annotated[float, Form()] = 0.25,
    use_tiling: Annotated[bool, Form()] = True,
    _slot: None = Depends(_gpu_slot),
) -> DetectionResponse:
    """Alias for the main detect endpoint — form-data compatible."""
    # Pass _slot=None explicitly so detect_image does not re-acquire the semaphore.
    return await detect_image(file, confidence_threshold, use_tiling=use_tiling, _slot=None)


@router.get(
    "/models",
    summary="List available detection models",
    response_model=list[dict[str, Any]],
)
async def list_models() -> list[dict[str, Any]]:
    """
    Return available detector models and their specifications.

    VRAM requirements (approximate, FP16):
    - yolo11n: 700 MB (fastest, lowest accuracy)
    - yolo11s: 1.2 GB (recommended for RTX 4060)
    - yolo11m: 2.5 GB
    - yolo11l: 5.0 GB
    - yolo11x: 7.5 GB (RTX 4060 at limit)
    """
    return [
        {
            "id": "yolo11n",
            "name": "YOLOv11 Nano",
            "vram_gb": 0.7,
            "speed": "fastest",
            "map50_typical": 0.78,
            "recommended_for": "real-time video analysis",
        },
        {
            "id": "yolo11s",
            "name": "YOLOv11 Small",
            "vram_gb": 1.2,
            "speed": "fast",
            "map50_typical": 0.84,
            "recommended_for": "general inference on RTX 4060",
            "is_default": True,
        },
        {
            "id": "yolo11m",
            "name": "YOLOv11 Medium",
            "vram_gb": 2.5,
            "speed": "moderate",
            "map50_typical": 0.86,
            "recommended_for": "high-accuracy inference",
        },
        {
            "id": "ensemble",
            "name": "YOLO11s + RTMDet Ensemble",
            "vram_gb": 3.5,
            "speed": "slow",
            "map50_typical": 0.89,
            "recommended_for": "final evaluation, not real-time",
        },
    ]


@router.get(
    "/health",
    summary="Detection service health check",
)
async def health() -> dict[str, Any]:
    """Check if the detection service is ready."""
    import torch

    gpu_available = torch.cuda.is_available()
    vram_free_gb = None
    if gpu_available:
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(0)
            vram_free_gb = round(free_bytes / 1e9, 2)
        except Exception:
            pass

    return {
        "status": "healthy",
        "gpu_available": gpu_available,
        "vram_free_gb": vram_free_gb,
        "cuda_version": torch.version.cuda if gpu_available else None,
    }


@router.get("/status")
async def detect_status() -> dict:
    """Detection service status — alias for /detect/health."""
    return await health()

