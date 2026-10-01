"""morphology.schemas.schemas — Pydantic schemas for morphology API."""

from __future__ import annotations

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
    secondary_type: str | None = None
    secondary_confidence: float | None = None
    head_diameter_px: float | None = None
    stalk_length_px: float | None = None
    head_circularity: float | None = None
    elongation: float | None = None
    class_probabilities: dict[str, float] = Field(default_factory=dict)
    model_id: str = "geometric"


class MorphologyAnalysisResponse(BaseModel):
    instance_id: str
    morphology: MorphologyTypeSchema
    geometric: GeometricDescriptorsSchema | None = None
    stalk: StalkSchema | None = None
    head: HeadSchema | None = None


class DensityMapResponse(BaseModel):
    total_count: int
    uniformity_index: float
    density_per_mm2: float | None = None
    peak_density_cell: list[int]
    image_shape: list[int]
    type_distribution: dict[str, int] = Field(default_factory=dict)


class BatchMorphologyResponse(BaseModel):
    analyzed: int
    failed: int
    type_distribution: dict[str, int]
    results: list[MorphologyAnalysisResponse]
    density: DensityMapResponse | None = None
