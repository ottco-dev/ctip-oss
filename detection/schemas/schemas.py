"""
detection.schemas.schemas — Pydantic request/response models for detection API.

All schemas use Pydantic v2 for validation and serialization.
These schemas define the public API contract — changes require versioning.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class DetectionBox(BaseModel):
    """Single detection result."""

    id: str
    bbox: list[float] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="[x_min, y_min, x_max, y_max] in pixel coordinates",
    )
    confidence: float = Field(..., ge=0.0, le=1.0)
    calibrated_confidence: float | None = Field(
        None, description="Temperature-scaled confidence (if calibration applied)"
    )
    uncertainty: float | None = Field(
        None, description="Epistemic uncertainty estimate (0=certain, 1=very uncertain)"
    )
    trichome_type: str = Field(
        ...,
        description="One of: capitate_stalked, capitate_sessile, bulbous, non_glandular, unknown",
    )
    is_uncertain: bool = Field(
        default=False,
        description="True if uncertainty > 0.15 — should be flagged for human review",
    )

    @field_validator("trichome_type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        valid = {
            "capitate_stalked", "capitate_sessile",
            "bulbous", "non_glandular", "unknown"
        }
        if v not in valid:
            raise ValueError(f"Invalid trichome type: {v}. Must be one of {valid}")
        return v


class DetectionResponse(BaseModel):
    """Response from single-image detection."""

    image_id: str
    filename: str
    num_detections: int
    detections: list[DetectionBox]
    inference_time_ms: float
    model_id: str
    was_tiled: bool
    image_shape: list[int] = Field(
        ..., description="[height, width, channels]"
    )
    scientific_note: str | None = Field(
        None,
        description="Important scientific caveats about result interpretation",
    )


class DetectionRequest(BaseModel):
    """Request body for JSON-based detection (base64 image)."""

    image_b64: str = Field(..., description="Base64-encoded image bytes")
    confidence_threshold: float = Field(0.25, ge=0.05, le=0.95)
    iou_threshold: float = Field(0.45, ge=0.1, le=0.95)
    use_tiling: bool = True
    model: str = "yolo11s"


class BatchDetectionRequest(BaseModel):
    """Request for batch detection."""

    image_paths: list[str] = Field(
        ..., description="Absolute paths to images on the server"
    )
    confidence_threshold: float = 0.25
    use_tiling: bool = True
    model: str = "yolo11s"
    export_crops: bool = False


class DetectionStats(BaseModel):
    """Aggregated statistics over a detection result set."""

    total_images: int
    total_detections: int
    mean_confidence: float
    mean_detections_per_image: float
    type_distribution: dict[str, float]
    uncertain_fraction: float
    inference_fps: float
