"""
morphology.api.router — FastAPI endpoints for morphology analysis.

Endpoints:
  POST /morphology/analyze       — analyze single image (detect + classify all trichomes)
  POST /morphology/instance      — classify morphology from uploaded mask PNG
  POST /morphology/density       — generate density map from centroid list
  GET  /morphology/health        — service health check
"""

from __future__ import annotations

import io
import json
from collections import Counter
from typing import List, Optional

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel, Field

from morphology.domain.geometric import extract_geometric_descriptors
from morphology.domain.stalk_detector import detect_stalk_and_head
from morphology.domain.density_map import TrichomeCentroid, compute_density_map
from morphology.classification.classifier import MorphologyClassifier
from morphology.schemas.schemas import (
    MorphologyTypeSchema,
    GeometricDescriptorsSchema,
    StalkSchema,
    HeadSchema,
    MorphologyAnalysisResponse,
    DensityMapResponse,
)

# GPU semaphore dependency: acquire before model inference.
# MorphologyClassifier may load an ONNX model on GPU (via CUDA EP).
# Geometric-only paths (density map, health) bypass this.
try:
    from backend.dependencies.gpu import gpu_slot as _gpu_slot
except ImportError:
    async def _gpu_slot() -> None:  # type: ignore[misc]
        yield

router = APIRouter(prefix="/morphology", tags=["Morphology Analysis"])
_classifier = MorphologyClassifier()


class CentroidInput(BaseModel):
    x: float
    y: float
    trichome_type: str = "unknown"
    confidence: float = Field(default=1.0, ge=0, le=1)


class DensityRequest(BaseModel):
    centroids: List[CentroidInput]
    image_height: int = Field(gt=0)
    image_width: int = Field(gt=0)
    grid_rows: int = Field(default=8, ge=2, le=32)
    grid_cols: int = Field(default=8, ge=2, le=32)
    um_per_pixel: Optional[float] = None


async def _decode_image(file: UploadFile) -> np.ndarray:
    data = await file.read()
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(422, "Cannot decode image")
    return img


async def _decode_mask(file: UploadFile) -> np.ndarray:
    data = await file.read()
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise HTTPException(422, "Cannot decode mask image")
    return img


@router.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "classifier": "cnn" if _classifier.has_model else "geometric",
    }


@router.post(
    "/instance",
    response_model=MorphologyAnalysisResponse,
    summary="Classify a single trichome from its mask",
)
async def classify_instance(
    mask_file: UploadFile = File(..., description="Grayscale PNG binary mask"),
    instance_id: str = Form(default="unknown"),
    _slot: None = Depends(_gpu_slot),
) -> MorphologyAnalysisResponse:
    """
    Classify trichome morphological type from a binary mask image.

    Computes geometric descriptors, stalk/head segmentation, and returns
    a full morphology classification.

    Returns:
        MorphologyAnalysisResponse with type, confidence, and geometric features.
    """
    mask = await _decode_mask(mask_file)

    # Geometric descriptors
    geo = extract_geometric_descriptors(mask)
    if not geo.is_valid:
        raise HTTPException(422, f"Mask too small (area={geo.area_px:.0f}px). Min 25px².")

    # Stalk + head detection
    stalk, head = detect_stalk_and_head(mask)

    # Classification
    morph = _classifier.predict_geometric(geo=geo, stalk=stalk, head=head)

    geo_schema = GeometricDescriptorsSchema(
        area_px=geo.area_px,
        perimeter_px=geo.perimeter_px,
        circularity=geo.circularity,
        elongation=geo.elongation,
        convexity=geo.convexity,
        solidity=geo.solidity,
        compactness=geo.compactness,
        aspect_ratio=geo.aspect_ratio,
        major_axis_px=geo.major_axis_px,
        minor_axis_px=geo.minor_axis_px,
        orientation_deg=geo.orientation_deg,
        centroid_x=geo.centroid_x,
        centroid_y=geo.centroid_y,
    )

    stalk_schema = StalkSchema(
        stalk_length_px=stalk.stalk_length_px,
        stalk_width_px=stalk.stalk_width_px,
        has_visible_stalk=stalk.has_visible_stalk,
        confidence=stalk.confidence,
    )

    head_schema = (
        HeadSchema(
            head_area_px=head.head_area_px,
            head_diameter_px=head.head_diameter_px,
            head_circularity=head.head_circularity,
            head_centroid_x=head.head_centroid_x,
            head_centroid_y=head.head_centroid_y,
        )
        if head is not None
        else None
    )

    class_probs = {
        k.value if hasattr(k, "value") else str(k): v
        for k, v in morph.class_probabilities.items()
    }

    morph_schema = MorphologyTypeSchema(
        primary_type=morph.primary_type.value,
        confidence=float(morph.confidence),
        secondary_type=morph.secondary_type.value if morph.secondary_type else None,
        secondary_confidence=float(morph.secondary_confidence)
        if morph.secondary_confidence
        else None,
        head_diameter_px=morph.head_diameter_px,
        stalk_length_px=morph.stalk_length_px,
        head_circularity=morph.head_circularity,
        elongation=morph.elongation,
        class_probabilities=class_probs,
        model_id=morph.model_id,
    )

    return MorphologyAnalysisResponse(
        instance_id=instance_id,
        morphology=morph_schema,
        geometric=geo_schema,
        stalk=stalk_schema,
        head=head_schema,
    )


@router.post(
    "/density",
    response_model=DensityMapResponse,
    summary="Compute trichome spatial density from centroid list",
)
async def compute_density(body: DensityRequest) -> DensityMapResponse:
    """
    Generate a trichome density map from a list of centroid coordinates.

    Returns grid-based counts, KDE surface stats, and absolute density
    (if um_per_pixel is provided).
    """
    if not body.centroids:
        raise HTTPException(422, "At least one centroid required")

    centroids = [
        TrichomeCentroid(
            x=c.x, y=c.y,
            trichome_type=c.trichome_type,
            confidence=c.confidence,
        )
        for c in body.centroids
    ]

    result = compute_density_map(
        centroids=centroids,
        image_height=body.image_height,
        image_width=body.image_width,
        grid_rows=body.grid_rows,
        grid_cols=body.grid_cols,
        um_per_pixel=body.um_per_pixel,
    )

    type_counts = dict(Counter(c.trichome_type for c in body.centroids))

    return DensityMapResponse(
        total_count=result.total_count,
        uniformity_index=result.uniformity_index,
        density_per_mm2=result.density_per_mm2,
        peak_density_cell=list(result.peak_density_cell),
        image_shape=list(result.image_shape),
        type_distribution=type_counts,
    )


@router.post(
    "/density/heatmap",
    summary="Generate density heatmap PNG image",
)
async def density_heatmap(body: DensityRequest) -> Response:
    """
    Return a PNG density heatmap image for visualization.

    Blue = low density, Red = high density (JET colormap).
    """
    if not body.centroids:
        raise HTTPException(422, "At least one centroid required")

    centroids = [
        TrichomeCentroid(x=c.x, y=c.y, trichome_type=c.trichome_type)
        for c in body.centroids
    ]

    result = compute_density_map(
        centroids=centroids,
        image_height=body.image_height,
        image_width=body.image_width,
        grid_rows=body.grid_rows,
        grid_cols=body.grid_cols,
    )

    _, buf = cv2.imencode(".png", result.heatmap_bgr)
    return Response(content=buf.tobytes(), media_type="image/png")
