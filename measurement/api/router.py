"""
measurement.api.router — FastAPI endpoints for microscope calibration and measurement.

Endpoints:
  GET  /measurement/profiles                  — list all calibration profiles
  POST /measurement/profiles                  — create a new profile (manual)
  POST /measurement/profiles/calibrate        — create profile from stage micrometer (manual px)
  POST /measurement/profiles/calibrate/auto   — create profile via auto scale-bar detection (TDB-001)
  GET  /measurement/profiles/{id}             — get specific profile
  DELETE /measurement/profiles/{id}           — delete a profile
  PUT  /measurement/profiles/{id}/default     — set default profile
  POST /measurement/measure/mask              — measure single trichome mask
  GET  /measurement/health                    — health check
"""

from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from measurement.domain.profile_manager import MicroscopeProfile, ProfileManager
from measurement.domain.measurer import Measurer
from measurement.schemas.schemas import (
    MicroscopeProfileSchema,
    CreateProfileRequest,
    StageMicrometerRequest,
    TrichomeMeasurementsSchema,
    MeasurementUncertaintySchema,
)
from measurement.calibration.stage_micrometer import detect_scale_bar_px, ScaleBarDetectionResult

router = APIRouter(prefix="/measurement", tags=["Measurement & Calibration"])


def _get_profile_manager() -> ProfileManager:
    """
    Return a singleton ProfileManager backed by a persistent JSON file.

    Storage path resolution (first match wins):
      1. TRICHOME_PROFILES_PATH env var
      2. DATA_ROOT env var → <data_root>/calibration/profiles.json
      3. ./data/calibration/profiles.json (local dev fallback)

    Fixes TDB-004: custom profiles were lost on server restart because
    ProfileManager was instantiated without a storage_path.
    """
    import os
    from pathlib import Path as _Path

    env_path = os.environ.get("TRICHOME_PROFILES_PATH")
    if env_path:
        storage = _Path(env_path)
    else:
        data_root = os.environ.get("DATA_ROOT", "./data")
        storage = _Path(data_root) / "calibration" / "profiles.json"

    try:
        storage.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        storage = None  # type: ignore[assignment]

    return ProfileManager(storage_path=storage)


_profile_manager = _get_profile_manager()


def _profile_to_schema(p: MicroscopeProfile) -> MicroscopeProfileSchema:
    return MicroscopeProfileSchema(
        profile_id=p.profile_id,
        name=p.name,
        um_per_pixel=p.um_per_pixel,
        objective=p.objective,
        camera=p.camera,
        image_width=p.image_width,
        image_height=p.image_height,
        calibration_method=p.calibration_method,
        calibration_date=p.calibration_date,
        uncertainty_um=p.uncertainty_um,
        notes=p.notes,
    )


@router.get("/health")
async def health() -> dict:
    default = _profile_manager.default_profile
    return {
        "status": "ok",
        "profiles": len(_profile_manager.list_profiles()),
        "default_profile": default.profile_id if default else None,
    }


@router.get("/profiles", response_model=List[MicroscopeProfileSchema])
async def list_profiles() -> List[MicroscopeProfileSchema]:
    """List all available microscope calibration profiles."""
    return [_profile_to_schema(p) for p in _profile_manager.list_profiles()]


@router.post("/profiles", response_model=MicroscopeProfileSchema)
async def create_profile(body: CreateProfileRequest) -> MicroscopeProfileSchema:
    """Create a new calibration profile from a manual µm/px value."""
    profile = MicroscopeProfile(
        name=body.name,
        um_per_pixel=body.um_per_pixel,
        objective=body.objective,
        camera=body.camera,
        image_width=body.image_width,
        image_height=body.image_height,
        calibration_method=body.calibration_method,
        notes=body.notes,
    )
    _profile_manager.add_profile(profile, set_default=body.set_default)
    return _profile_to_schema(profile)


@router.post("/profiles/calibrate", response_model=MicroscopeProfileSchema)
async def calibrate_from_stage_micrometer(
    body: StageMicrometerRequest,
) -> MicroscopeProfileSchema:
    """
    Create a calibration profile from a stage micrometer measurement.

    Provide the measured scale bar length in pixels and its known
    physical length in µm. Uncertainty is automatically computed.
    """
    profile = _profile_manager.create_from_stage_micrometer(
        name=body.name,
        scale_bar_px=body.scale_bar_px,
        scale_bar_um=body.scale_bar_um,
        objective=body.objective,
        camera=body.camera,
        image_width=body.image_width,
        image_height=body.image_height,
        notes=body.notes,
        set_default=body.set_default,
    )
    return _profile_to_schema(profile)


class ScaleBarDetectionResponse(BaseModel):
    """Response schema for the auto scale-bar detection endpoint."""

    detected: bool
    scale_bar_px: float
    confidence: float
    num_line_groups: int
    method: str
    message: str
    # Set to the resulting profile when detected=True and save_profile=True
    profile: Optional[MicroscopeProfileSchema] = None


@router.post(
    "/profiles/calibrate/auto",
    response_model=ScaleBarDetectionResponse,
    summary="Auto-detect scale bar from micrometer image (TDB-001)",
)
async def calibrate_auto_detect(
    image_file: UploadFile = File(..., description="Grayscale or RGB stage micrometer image"),
    scale_bar_um: float = Form(
        ...,
        description="Known physical length of the scale bar in µm (e.g. 100.0 for a 100 µm bar)",
    ),
    profile_name: str = Form(default="Auto-calibrated profile"),
    objective: str = Form(default=""),
    camera: str = Form(default=""),
    notes: str = Form(default=""),
    set_default: bool = Form(default=False),
    save_profile: bool = Form(
        default=True,
        description="If True, save the detected calibration as a new profile",
    ),
    min_confidence: float = Form(
        default=0.5,
        description="Minimum detection confidence to accept (0–1). Below this, returns detected=False.",
    ),
) -> ScaleBarDetectionResponse:
    """
    Automatically detect the scale bar length in a stage micrometer image using
    Hough-line detection, then create a calibrated profile.

    **Workflow**

    1. Upload a JPG/PNG/TIFF image of a stage micrometer at the imaging magnification.
    2. Provide the known physical length of the visible scale bar (in µm).
    3. The endpoint auto-detects the scale bar pixel length via:
       - CLAHE contrast enhancement
       - Canny edge detection
       - Probabilistic Hough line transform
       - Horizontal-segment clustering and span measurement
    4. If detected with sufficient confidence and `save_profile=True`, a calibration
       profile is saved and returned.

    **Confidence**

    `confidence` reflects the fraction of image width covered by the detected bar.
    Values below `min_confidence` return `detected=False` — use the manual
    `/profiles/calibrate` endpoint instead.

    **Notes**

    - Best results with bright-field stage micrometer images (dark lines on white).
    - Image should be oriented so scale bar lines are horizontal.
    - Acceptable uncertainty: ±1–2% for bars spanning >30% of the image width.
    """
    data = await image_file.read()
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise HTTPException(422, "Cannot decode image. Upload JPG, PNG or TIFF.")

    if scale_bar_um <= 0:
        raise HTTPException(422, "scale_bar_um must be > 0")

    # Run detection
    detection = detect_scale_bar_px(img)

    # Check confidence threshold
    if not detection.detected or detection.confidence < min_confidence:
        return ScaleBarDetectionResponse(
            detected=False,
            scale_bar_px=detection.scale_bar_px,
            confidence=detection.confidence,
            num_line_groups=detection.num_line_groups,
            method=detection.method,
            message=detection.message,
        )

    # Optionally create and save the profile
    saved_profile_schema: Optional[MicroscopeProfileSchema] = None
    if save_profile:
        profile = _profile_manager.create_from_stage_micrometer(
            name=profile_name,
            scale_bar_px=detection.scale_bar_px,
            scale_bar_um=scale_bar_um,
            objective=objective,
            camera=camera,
            image_width=int(img.shape[1]),
            image_height=int(img.shape[0]),
            notes=(
                f"{notes} [Auto-detected: {detection.message}]".strip()
            ),
            set_default=set_default,
        )
        saved_profile_schema = _profile_to_schema(profile)

    return ScaleBarDetectionResponse(
        detected=True,
        scale_bar_px=detection.scale_bar_px,
        confidence=detection.confidence,
        num_line_groups=detection.num_line_groups,
        method=detection.method,
        message=detection.message,
        profile=saved_profile_schema,
    )


@router.get("/profiles/{profile_id}", response_model=MicroscopeProfileSchema)
async def get_profile(profile_id: str) -> MicroscopeProfileSchema:
    """Get a specific calibration profile by ID."""
    p = _profile_manager.get_profile(profile_id)
    if p is None:
        raise HTTPException(404, f"Profile '{profile_id}' not found")
    return _profile_to_schema(p)


@router.delete("/profiles/{profile_id}")
async def delete_profile(profile_id: str) -> dict:
    """Delete a calibration profile (built-in defaults cannot be deleted)."""
    try:
        existed = _profile_manager.delete_profile(profile_id)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if not existed:
        raise HTTPException(404, f"Profile '{profile_id}' not found")
    return {"deleted": profile_id}


@router.put("/profiles/{profile_id}/default")
async def set_default_profile(profile_id: str) -> dict:
    """Set the default calibration profile."""
    try:
        _profile_manager.set_default(profile_id)
    except KeyError as e:
        raise HTTPException(404, str(e))
    return {"default": profile_id}


@router.post(
    "/measure/mask",
    response_model=TrichomeMeasurementsSchema,
    summary="Measure physical dimensions from a trichome mask",
)
async def measure_from_mask(
    mask_file: UploadFile = File(..., description="Grayscale PNG binary mask"),
    profile_id: Optional[str] = Form(default=None),
    focus_score: Optional[float] = Form(default=None),
) -> TrichomeMeasurementsSchema:
    """
    Measure a trichome's physical dimensions from its binary mask.

    Returns head diameter, stalk length, total height in µm with
    uncertainty estimates based on the selected calibration profile.
    """
    data = await mask_file.read()
    arr = np.frombuffer(data, np.uint8)
    mask = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise HTTPException(422, "Cannot decode mask image")

    # Select profile
    if profile_id:
        profile = _profile_manager.get_profile(profile_id)
        if profile is None:
            raise HTTPException(404, f"Profile '{profile_id}' not found")
    else:
        profile = _profile_manager.default_profile
        if profile is None:
            raise HTTPException(400, "No default profile configured")

    from morphology.domain.geometric import extract_geometric_descriptors
    from morphology.domain.stalk_detector import detect_stalk_and_head

    geo = extract_geometric_descriptors(mask)
    if not geo.is_valid:
        raise HTTPException(422, f"Mask too small (area={geo.area_px:.0f}px). Min 25px².")

    stalk, head = detect_stalk_and_head(mask)
    measurer = Measurer(profile)

    result = measurer.measure(
        head_diameter_px=head.head_diameter_px if head else None,
        head_area_px=head.head_area_px if head else None,
        head_circularity=head.head_circularity if head else geo.circularity,
        stalk_length_px=stalk.stalk_length_px if stalk.has_visible_stalk else None,
        stalk_width_px=stalk.stalk_width_px if stalk.has_visible_stalk else None,
        total_height_px=geo.major_axis_px,
        total_area_px=geo.area_px,
    )

    return TrichomeMeasurementsSchema(
        head_diameter_um=result.head_diameter_um,
        head_area_um2=result.head_area_um2,
        head_circularity=result.head_circularity,
        stalk_length_um=result.stalk_length_um,
        stalk_width_um=result.stalk_width_um,
        total_height_um=result.total_height_um,
        total_area_um2=result.total_area_um2,
        head_stalk_ratio=result.head_stalk_ratio,
        uncertainties=MeasurementUncertaintySchema(
            head_diameter_um=result.head_diameter_uncertainty_um,
            stalk_length_um=result.stalk_length_uncertainty_um,
        ),
        um_per_pixel=result.um_per_pixel,
        calibration_method=result.calibration_method,
        morphology_hint=result.morphology_hint,
    )
