"""morphology.schemas.schemas — Pydantic schemas for morphology API."""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class GeometricDescriptorsSchema(BaseModel):
    area_px: float
    perimeter_px: float
    circularity: float = Field(ge=0, le=1)
    elongation: float = Field(ge=1)
    convexity: float = Field(ge=0, le=1)
    solidity: float = Field(ge=0, le=1)
    compactness: float
    aspect_ratio: float
    major_axis_px: float
    minor_axis_px: float
    orientation_deg: float
    centroid_x: float
    centroid_y: float


class StalkSchema(BaseModel):
    stalk_length_px: float
    stalk_width_px: float
    has_visible_stalk: bool
    confidence: float = Field(ge=0, le=1)


class HeadSchema(BaseModel):
    head_area_px: float
    head_diameter_px: float
    head_circularity: float = Field(ge=0, le=1)
    head_centroid_x: float
    head_centroid_y: float


class MorphologyTypeSchema(BaseModel):
    primary_type: str
    confidence: float = Field(ge=0, le=1)
    secondary_type: Optional[str] = None
    secondary_confidence: Optional[float] = None
    head_diameter_px: Optional[float] = None
    stalk_length_px: Optional[float] = None
    head_circularity: Optional[float] = None
    elongation: Optional[float] = None
    class_probabilities: Dict[str, float] = Field(default_factory=dict)
    model_id: str = "geometric"


class MorphologyAnalysisResponse(BaseModel):
    instance_id: str
    morphology: MorphologyTypeSchema
    geometric: Optional[GeometricDescriptorsSchema] = None
    stalk: Optional[StalkSchema] = None
    head: Optional[HeadSchema] = None


class DensityMapResponse(BaseModel):
    total_count: int
    uniformity_index: float
    density_per_mm2: Optional[float] = None
    peak_density_cell: List[int]
    image_shape: List[int]
    type_distribution: Dict[str, int] = Field(default_factory=dict)


class BatchMorphologyResponse(BaseModel):
    analyzed: int
    failed: int
    type_distribution: Dict[str, int]
    results: List[MorphologyAnalysisResponse]
    density: Optional[DensityMapResponse] = None
