"""measurement.schemas.schemas — API request/response models."""

from __future__ import annotations

from pydantic import BaseModel, Field


class MicroscopeProfileSchema(BaseModel):
    profile_id: str
    name: str
    um_per_pixel: float = Field(gt=0)
    objective: str = ""
    camera: str = ""
    image_width: int | None = None
    image_height: int | None = None
    calibration_method: str = "manual"
    calibration_date: str | None = None
    uncertainty_um: float | None = None
    notes: str = ""


class CreateProfileRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    um_per_pixel: float = Field(gt=0, description="Micrometers per pixel")
    objective: str = ""
    camera: str = ""
    image_width: int | None = None
    image_height: int | None = None
    calibration_method: str = "manual"
    notes: str = ""
    set_default: bool = False


class StageMicrometerRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    scale_bar_px: float = Field(gt=0, description="Measured scale bar length in pixels")
    scale_bar_um: float = Field(gt=0, description="Known scale bar length in µm")
    objective: str = ""
    camera: str = ""
    image_width: int | None = None
    image_height: int | None = None
    notes: str = ""
    set_default: bool = False


class MeasurementUncertaintySchema(BaseModel):
    head_diameter_um: float | None = None
    stalk_length_um: float | None = None


class TrichomeMeasurementsSchema(BaseModel):
    head_diameter_um: float | None = None
    head_area_um2: float | None = None
    head_circularity: float | None = None
    stalk_length_um: float | None = None
    stalk_width_um: float | None = None
    total_height_um: float | None = None
    total_area_um2: float | None = None
    head_stalk_ratio: float | None = None
    uncertainties: MeasurementUncertaintySchema = Field(
        default_factory=MeasurementUncertaintySchema
    )
    um_per_pixel: float = 1.0
    calibration_method: str = ""
    morphology_hint: str = ""


class PopulationStatsSchema(BaseModel):
    n: int
    head_diameter_um: dict = Field(default_factory=dict)
    stalk_length_um: dict = Field(default_factory=dict)
    total_height_um: dict = Field(default_factory=dict)
    head_area_um2: dict = Field(default_factory=dict)
    head_stalk_ratio: dict = Field(default_factory=dict)
