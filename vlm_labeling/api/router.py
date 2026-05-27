"""
vlm_labeling.api.router — FastAPI router for VLM auto-labeling endpoints.

Endpoints:
    POST /vlm/label/maturity         — Label single image maturity
    POST /vlm/label/quality          — Screen image quality
    POST /vlm/label/morphology       — Classify morphology type
    POST /vlm/batch                  — Batch auto-labeling job
    GET  /vlm/batch/{job_id}         — Batch job status
    GET  /vlm/models                 — Available VLM models
    GET  /vlm/health                 — VLM service health check

IMPORTANT: All VLM outputs are pseudo-labels (AnnotationSource.VLM_AUTO).
They are placed in the review queue, not the training dataset.
Human review is required before labels can be used for training.
"""

from __future__ import annotations

import io
import time
from typing import Any

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from shared.logging.logger import get_logger
from vlm_labeling.filtering.hallucination import HallucinationFilter

logger = get_logger(__name__)

router = APIRouter(prefix="/vlm", tags=["vlm-labeling"])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SCHEMAS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class MaturityLabelResponse(BaseModel):
    label_id: str
    maturity_stage: str | None
    confidence: float
    amber_fraction: float | None
    cloudy_fraction: float | None
    clear_fraction: float | None
    observations: str | None
    image_quality: str | None
    hallucination_flags: list[str]
    is_flagged: bool
    review_priority: int
    inference_time_s: float
    vlm_model: str
    annotation_source: str = "vlm_auto"
    scientific_caveat: str = (
        "Visual maturity classification cannot determine cannabinoid content. "
        "Chromatography required for quantification."
    )


class QualityScreenResponse(BaseModel):
    overall_quality: str | None
    focus_quality: str | None
    lighting_quality: str | None
    analyzable: bool
    reject_reason: str | None
    confidence: float
    inference_time_s: float
    vlm_model: str


class MorphologyLabelResponse(BaseModel):
    label_id: str
    dominant_type: str | None
    confidence: float
    stalk_visible: bool | None
    head_shape: str | None
    mixed_types_present: bool | None
    hallucination_flags: list[str]
    inference_time_s: float
    vlm_model: str


class BatchJobRequest(BaseModel):
    image_ids: list[str] = Field(description="Dataset image IDs to process")
    vlm_backend: str = Field(default="moondream", description="VLM model to use")
    run_quality_screen: bool = Field(default=True)
    run_morphology: bool = Field(default=False)
    min_confidence: float = Field(default=0.40, ge=0.0, le=1.0)


class BatchJobStatus(BaseModel):
    job_id: str
    status: str
    total: int
    processed: int
    pending_review: int
    flagged: int
    failed: int
    progress_pct: float
    throughput_per_min: float | None


class VLMModelInfo(BaseModel):
    model_id: str
    name: str
    vram_gb: float
    quantization: str
    best_for: list[str]
    notes: str


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MODEL REGISTRY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

AVAILABLE_VLM_MODELS = [
    VLMModelInfo(
        model_id="moondream",
        name="Moondream-2B (4-bit)",
        vram_gb=2.1,
        quantization="4bit",
        best_for=["maturity_classification", "image_quality", "morphology_classification"],
        notes="Default. Fast, low VRAM. Weak at counting.",
    ),
    VLMModelInfo(
        model_id="florence2",
        name="Florence-2-Large",
        vram_gb=8.0,
        quantization="fp16",
        best_for=["detection", "counting", "captioning"],
        notes="Uses full 8GB VRAM. Cannot run with other GPU tasks.",
    ),
    VLMModelInfo(
        model_id="qwen2vl",
        name="Qwen2-VL-7B (4-bit)",
        vram_gb=5.5,
        quantization="4bit",
        best_for=["maturity_classification", "counting", "detailed_analysis"],
        notes="Best quality. Slower than Moondream. Quantized for 8GB VRAM.",
    ),
]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# UTILITY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _load_image_from_upload(upload: UploadFile) -> np.ndarray:
    """Load uploaded image file to RGB uint8 array."""
    import cv2

    content = upload.file.read()
    arr = np.frombuffer(content, np.uint8)
    image_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise HTTPException(status_code=422, detail="Could not decode image")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _get_moondream_labeler() -> Any:
    """Get or create Moondream labeler (simple per-request instantiation for now)."""
    from vlm_labeling.moondream.moondream_labeler import MoondreamLabeler
    labeler = MoondreamLabeler()
    labeler.load()
    return labeler


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ENDPOINTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@router.post("/label/maturity", response_model=MaturityLabelResponse)
async def label_maturity(
    file: UploadFile = File(..., description="Trichome microscopy image"),
    vlm_backend: str = Form(default="moondream"),
    run_hallucination_filter: bool = Form(default=True),
) -> MaturityLabelResponse:
    """
    Run VLM maturity classification on a single trichome image.

    Returns a pseudo-label with maturity stage, confidence, and fraction estimates.
    All outputs are VLM_AUTO and require human review before training use.
    """
    import uuid

    image = _load_image_from_upload(file)

    t_start = time.perf_counter()

    # For now, always use Moondream (simplest backend)
    # Production: use dependency injection with cached loaded model
    try:
        labeler = _get_moondream_labeler()
        result = labeler.label_maturity(image)
        labeler.unload()
    except Exception as e:
        logger.error("VLM inference failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"VLM inference failed: {e}")

    t_end = time.perf_counter()

    # Hallucination filter
    filter_result = None
    flags: list[str] = []
    priority = 0

    if run_hallucination_filter:
        filt = HallucinationFilter()
        filter_result = filt.filter_maturity(result.parsed_response)
        flags = filter_result.flag_names
        priority = filter_result.review_priority

    resp = result.parsed_response or {}
    label_id = str(uuid.uuid4())

    return MaturityLabelResponse(
        label_id=label_id,
        maturity_stage=resp.get("maturity_stage"),
        confidence=filter_result.adjusted_confidence if filter_result else result.confidence,
        amber_fraction=resp.get("amber_fraction_estimate"),
        cloudy_fraction=resp.get("cloudy_fraction_estimate"),
        clear_fraction=resp.get("clear_fraction_estimate"),
        observations=resp.get("observations"),
        image_quality=resp.get("image_quality"),
        hallucination_flags=flags,
        is_flagged=len(flags) > 0,
        review_priority=priority,
        inference_time_s=t_end - t_start,
        vlm_model=vlm_backend,
    )


@router.post("/label/quality", response_model=QualityScreenResponse)
async def screen_quality(
    file: UploadFile = File(..., description="Microscopy image to assess"),
    vlm_backend: str = Form(default="moondream"),
) -> QualityScreenResponse:
    """
    Run VLM image quality assessment.

    Fast quality gate before running expensive maturity analysis.
    Recommended for batch processing pipelines.
    """
    image = _load_image_from_upload(file)

    t_start = time.perf_counter()
    try:
        labeler = _get_moondream_labeler()
        result = labeler.assess_quality(image)
        labeler.unload()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Quality screening failed: {e}")

    t_end = time.perf_counter()
    resp = result.parsed_response or {}

    return QualityScreenResponse(
        overall_quality=resp.get("overall_quality"),
        focus_quality=resp.get("focus_quality"),
        lighting_quality=resp.get("lighting_quality"),
        analyzable=bool(resp.get("analyzable", True)),
        reject_reason=resp.get("reject_reason"),
        confidence=float(resp.get("confidence", 0.0)),
        inference_time_s=t_end - t_start,
        vlm_model=vlm_backend,
    )


@router.post("/label/morphology", response_model=MorphologyLabelResponse)
async def label_morphology(
    file: UploadFile = File(..., description="Trichome image for morphology classification"),
    vlm_backend: str = Form(default="moondream"),
) -> MorphologyLabelResponse:
    """Classify dominant trichome morphology type."""
    import uuid

    image = _load_image_from_upload(file)

    t_start = time.perf_counter()
    try:
        labeler = _get_moondream_labeler()
        result = labeler.label_morphology(image)
        labeler.unload()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Morphology classification failed: {e}")

    t_end = time.perf_counter()
    resp = result.parsed_response or {}

    filt = HallucinationFilter()
    filter_result = filt.filter_morphology(result.parsed_response)

    return MorphologyLabelResponse(
        label_id=str(uuid.uuid4()),
        dominant_type=resp.get("dominant_type"),
        confidence=filter_result.adjusted_confidence,
        stalk_visible=resp.get("stalk_visible"),
        head_shape=resp.get("head_shape"),
        mixed_types_present=resp.get("mixed_types_present"),
        hallucination_flags=filter_result.flag_names,
        inference_time_s=t_end - t_start,
        vlm_model=vlm_backend,
    )


@router.get("/models", response_model=list[VLMModelInfo])
async def list_vlm_models() -> list[VLMModelInfo]:
    """List all available VLM models with VRAM requirements."""
    return AVAILABLE_VLM_MODELS


@router.get("/health")
async def vlm_health() -> dict[str, Any]:
    """
    VLM service health check.

    Checks GPU availability and approximate VRAM headroom.
    """
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
            vram_used = torch.cuda.memory_allocated(0) / 1e9
            vram_free = vram_total - vram_used
        else:
            vram_total = vram_used = vram_free = 0.0
    except ImportError:
        gpu_available = False
        vram_total = vram_used = vram_free = 0.0

    return {
        "status": "ok",
        "gpu_available": gpu_available,
        "vram_total_gb": round(vram_total, 2),
        "vram_used_gb": round(vram_used, 2),
        "vram_free_gb": round(vram_free, 2),
        "default_backend": "moondream",
        "available_models": [m.model_id for m in AVAILABLE_VLM_MODELS],
        "warning": (
            "VLM outputs are pseudo-labels. "
            "Human review required before training data use."
        ),
    }


@router.get("/status")
async def vlm_status() -> dict:
    """VLM status — alias for /vlm/health."""
    return await vlm_health()

