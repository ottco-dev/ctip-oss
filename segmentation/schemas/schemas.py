"""
segmentation/schemas/schemas.py — Pydantic schemas for segmentation API.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class SegmentRequest(BaseModel):
    model_backend: str = Field(default="auto", description="auto | sam2_tiny | mobile_sam")
    score_threshold: float = Field(default=0.50, ge=0.0, le=1.0)
    min_mask_area_px: int = Field(default=25, ge=1)
    max_mask_area_fraction: float = Field(default=0.5, ge=0.0, le=1.0)
    max_instances: int = Field(default=50, ge=1, le=200)
    refine_masks: bool = True
    export_polygons: bool = True


class MaskData(BaseModel):
    instance_id: int
    class_id: int | None = None
    class_name: str | None = None
    score: float
    area_px: int
    centroid_x: float
    centroid_y: float
    bbox: list[float] = Field(default_factory=list, description="[x1, y1, x2, y2]")
    polygon: list[list[float]] | None = None  # [[x,y], ...]
    circularity: float = 0.0
    diameter_um: float | None = None


class SegmentResponse(BaseModel):
    instances: list[MaskData]
    total_instances: int
    backend_used: str
    inference_ms: float
    image_width: int
    image_height: int


class BatchSegmentRequest(BaseModel):
    model_backend: str = "auto"
    score_threshold: float = 0.50
    max_instances_per_image: int = 50
