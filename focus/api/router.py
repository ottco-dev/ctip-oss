"""focus.api.router — FastAPI router for focus analysis endpoints."""

from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field

import cv2
import numpy as np

from focus.metrics.composite import compute_focus_score
from focus.guidance.heatmap import generate_focus_heatmap, annotate_focus_regions
from focus.guidance.autofocus import compute_regional_guidance

router = APIRouter(prefix="/focus", tags=["Focus Analysis"])


class FocusScoreResponse(BaseModel):
    composite: float = Field(..., ge=0, le=1, description="Overall focus score [0,1]")
    laplacian_variance: float
    tenengrad: float
    normalized_variance: float
    fft_score: float
    quality_label: str
    is_acceptable: bool
    is_good: bool


class AutofocusGuidanceResponse(BaseModel):
    current_score: float
    direction: str
    magnitude: str
    region_advice: list[str]
    action: str
    confidence: float


async def _read_image(file: UploadFile) -> np.ndarray:
    data = await file.read()
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(422, "Cannot decode image")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


@router.post("/score", response_model=FocusScoreResponse, summary="Score image focus quality")
async def score_focus(file: UploadFile = File(...)) -> FocusScoreResponse:
    """
    Compute composite focus score for an uploaded image.

    Returns all individual metrics (Laplacian variance, Tenengrad,
    normalized variance, FFT score) and a composite [0,1] score.
    """
    img = await _read_image(file)
    result = compute_focus_score(img)

    return FocusScoreResponse(
        composite=result.composite,
        laplacian_variance=result.laplacian_variance,
        tenengrad=result.tenengrad,
        normalized_variance=result.normalized_variance,
        fft_score=result.fft_score,
        quality_label=result.quality_label,
        is_acceptable=result.is_acceptable,
        is_good=result.is_good,
    )


@router.post("/guidance", response_model=AutofocusGuidanceResponse,
             summary="Get autofocus guidance for a live frame")
async def autofocus_guidance(file: UploadFile = File(...)) -> AutofocusGuidanceResponse:
    """
    Analyze a microscope image and provide actionable focus guidance.

    Returns direction/magnitude recommendations for Z-axis adjustment
    and per-region advice for partial-focus images.
    """
    img = await _read_image(file)
    guidance = compute_regional_guidance(img)

    return AutofocusGuidanceResponse(
        current_score=guidance.current_score,
        direction=guidance.direction,
        magnitude=guidance.magnitude,
        region_advice=guidance.region_advice,
        action=guidance.action,
        confidence=guidance.confidence,
    )


@router.post("/heatmap", summary="Generate focus heatmap overlay image")
async def focus_heatmap(
    file: UploadFile = File(...),
    grid_rows: int = 8,
    grid_cols: int = 8,
    annotate: bool = False,
) -> Response:
    """
    Generate a color-coded focus quality heatmap.

    Returns a PNG image where:
    - Green = sharp regions (good for analysis)
    - Yellow = acceptable regions
    - Red = blurry regions (unreliable)

    Set annotate=true to add score labels on the grid cells.
    """
    img = await _read_image(file)
    result = generate_focus_heatmap(img, grid=(grid_rows, grid_cols))

    output_img = result.overlay_rgb
    if annotate:
        output_img = annotate_focus_regions(img, result)

    _, buf = cv2.imencode(".png", cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR))
    return Response(content=buf.tobytes(), media_type="image/png")
