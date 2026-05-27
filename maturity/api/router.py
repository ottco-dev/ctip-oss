"""
maturity.api.router — FastAPI endpoints for trichome maturity analysis.

Endpoints:
  POST /maturity/analyze/crop      — analyze single crop image
  POST /maturity/analyze/batch     — batch analyze multiple crops (form upload)
  GET  /maturity/health            — service health check

SCIENTIFIC NOTE:
All endpoints include the scientific caveat that outputs are optical
observations only and do not imply cannabinoid content measurement.
"""

from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from maturity.application.maturity_pipeline import MaturityPipeline, MaturityPipelineConfig
from maturity.schemas.schemas import (
    MaturityClassificationSchema,
    MaturityAnalysisResponse,
    MaturityFeatureSchema,
    MaturityUncertaintySchema,
    MaturityStageDistributionSchema,
    BatchMaturityResponse,
)
from shared.core.entities import MaturityLabel

# GPU semaphore dependency: acquire before model inference.
# Currently MaturityPipeline uses CPU-only HSV/LAB/LBP analysis;
# the dependency is added as future-proofing for CNN/VLM backends.
# Import lazily so the router stays importable without the full backend package.
try:
    from backend.dependencies.gpu import gpu_slot as _gpu_slot
except ImportError:  # running outside full backend (e.g. testing)
    async def _gpu_slot() -> None:  # type: ignore[misc]
        yield

router = APIRouter(prefix="/maturity", tags=["Maturity Analysis"])

_pipeline = MaturityPipeline()


SCIENTIFIC_CAVEAT = (
    "Color-based maturity classification is an optical proxy only. "
    "Does not measure cannabinoid content (requires GC-MS/HPLC). "
    "No direct THC correlation implied. High biological variability exists."
)


async def _decode_image(file: UploadFile) -> np.ndarray:
    data = await file.read()
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(422, "Cannot decode image")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _label_to_schema(
    label: MaturityLabel,
    instance_id: str = "unknown",
) -> MaturityAnalysisResponse:
    """Convert a MaturityLabel domain object to API response schema."""
    class_probs = {
        k.value if hasattr(k, "value") else str(k): float(v)
        for k, v in label.class_probabilities.items()
    }

    features = MaturityFeatureSchema(
        mean_hue=label.mean_hue,
        mean_saturation=label.mean_saturation,
        mean_value=label.mean_value,
        translucency_score=label.translucency_score,
        amber_ratio=label.amber_ratio,
        shannon_entropy=label.texture_entropy,
    )

    uncertainty = MaturityUncertaintySchema(
        epistemic=label.epistemic_uncertainty,
        aleatoric=label.aleatoric_uncertainty,
    )

    classification = MaturityClassificationSchema(
        stage=label.stage.value,
        confidence=float(label.confidence),
        class_probabilities=class_probs,
        uncertainty=uncertainty,
        features=features,
        scientific_caveat=SCIENTIFIC_CAVEAT,
    )

    return MaturityAnalysisResponse(
        instance_id=instance_id,
        classification=classification,
    )


@router.get("/health")
async def health() -> dict:
    return {"status": "ok", "module": "maturity"}


@router.post(
    "/analyze/crop",
    response_model=MaturityAnalysisResponse,
    summary="Analyze maturity of a single trichome crop image",
)
async def analyze_crop(
    file: UploadFile = File(..., description="RGB trichome crop image"),
    instance_id: str = Form(default="unknown"),
    use_texture: bool = Form(default=True),
    use_translucency: bool = Form(default=True),
    _slot: None = Depends(_gpu_slot),
) -> MaturityAnalysisResponse:
    """
    Classify the maturity stage of a single trichome crop.

    Accepts any image format supported by OpenCV.
    Returns stage classification (clear/cloudy/amber/degraded),
    confidence, per-class probabilities, and extracted features.

    **Important scientific note**: This is an optical color classification only.
    It does NOT measure or predict cannabinoid concentrations.
    """
    crop_rgb = await _decode_image(file)

    if use_texture != _pipeline.config.use_texture or use_translucency != _pipeline.config.use_translucency:
        cfg = MaturityPipelineConfig(
            use_texture=use_texture,
            use_translucency=use_translucency,
        )
        pipeline = MaturityPipeline(cfg)
    else:
        pipeline = _pipeline

    try:
        label = pipeline.analyze_crop(crop_rgb)
    except Exception as e:
        raise HTTPException(500, f"Maturity analysis failed: {e}")

    return _label_to_schema(label, instance_id=instance_id)


@router.post(
    "/analyze/population",
    response_model=MaturityStageDistributionSchema,
    summary="Analyze maturity distribution across multiple crops",
)
async def analyze_population(
    files: List[UploadFile] = File(
        ..., description="Multiple trichome crop images"
    ),
    _slot: None = Depends(_gpu_slot),
) -> MaturityStageDistributionSchema:
    """
    Analyze maturity distribution across a population of trichome crops.

    Returns the fraction of clear, cloudy, amber, and degraded trichomes.
    Useful for characterizing the maturity state of an entire microscopy image.

    **Scientific note**: Population-level statistics. Not a harvest recommendation.
    """
    if not files:
        raise HTTPException(422, "At least one file required")

    if len(files) > 500:
        raise HTTPException(422, "Maximum 500 crops per request")

    from collections import Counter
    import numpy as np

    stage_counts: Counter = Counter()
    confidences = []
    failed = 0

    for f in files:
        try:
            crop = await _decode_image(f)
            label = _pipeline.analyze_crop(crop)
            stage_counts[label.stage.value] += 1
            confidences.append(float(label.confidence))
        except Exception:
            failed += 1

    total = sum(stage_counts.values())
    if total == 0:
        raise HTTPException(422, "All crops failed analysis")

    def _frac(stage: str) -> float:
        return stage_counts.get(stage, 0) / total

    dominant = max(stage_counts, key=stage_counts.get) if stage_counts else "unknown"
    pop_conf = float(np.mean(confidences)) if confidences else 0.0

    return MaturityStageDistributionSchema(
        clear_fraction=_frac("clear"),
        cloudy_fraction=_frac("cloudy"),
        amber_fraction=_frac("amber"),
        degraded_fraction=_frac("degraded"),
        total_analyzed=total,
        dominant_stage=dominant,
        population_confidence=pop_conf,
    )
