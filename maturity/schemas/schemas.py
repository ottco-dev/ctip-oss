"""maturity.schemas.schemas — Pydantic schemas for maturity analysis API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class MaturityFeatureSchema(BaseModel):
    """Color and texture features from a single trichome crop."""

    # Color features (HSV)
    mean_hue: float | None = None
    mean_saturation: float | None = None
    mean_value: float | None = None

    # LAB
    mean_l: float | None = None
    mean_a: float | None = None
    mean_b: float | None = None

    # Texture
    lbp_uniformity: float | None = None
    glcm_contrast: float | None = None
    glcm_energy: float | None = None
    shannon_entropy: float | None = None

    # Translucency
    translucency_score: float | None = Field(
        default=None, ge=0, le=1,
        description="Estimated translucency [0=opaque, 1=transparent]"
    )

    # Oxidation / degradation
    amber_ratio: float | None = Field(
        default=None, ge=0, le=1,
        description="Fraction of amber-colored pixels in head region"
    )
    is_degraded: bool | None = None


class MaturityUncertaintySchema(BaseModel):
    epistemic: float | None = None
    aleatoric: float | None = None


class MaturityClassificationSchema(BaseModel):
    stage: str = Field(description="Maturity stage: clear, cloudy, amber, degraded, mixed")
    confidence: float = Field(ge=0, le=1)
    class_probabilities: dict[str, float] = Field(default_factory=dict)
    uncertainty: MaturityUncertaintySchema = Field(default_factory=MaturityUncertaintySchema)
    features: MaturityFeatureSchema = Field(default_factory=MaturityFeatureSchema)
    model_id: str = ""
    scientific_caveat: str = (
        "Color-based maturity classification is an optical proxy only. "
        "Does not measure cannabinoid content. No direct THC correlation implied."
    )


class MaturityAnalysisResponse(BaseModel):
    instance_id: str
    classification: MaturityClassificationSchema
    gradcam_available: bool = False


class BatchMaturityRequest(BaseModel):
    """Request body for batch maturity analysis."""
    min_confidence: float = Field(default=0.3, ge=0, le=1)
    use_texture: bool = True
    use_translucency: bool = True


class BatchMaturityResponse(BaseModel):
    analyzed: int
    failed: int
    stage_distribution: dict[str, int] = Field(default_factory=dict)
    results: list[MaturityAnalysisResponse]
    mean_confidence: float | None = None


class MaturityStageDistributionSchema(BaseModel):
    """Population-level maturity distribution."""
    clear_fraction: float = Field(ge=0, le=1)
    cloudy_fraction: float = Field(ge=0, le=1)
    amber_fraction: float = Field(ge=0, le=1)
    degraded_fraction: float = Field(ge=0, le=1)
    total_analyzed: int
    dominant_stage: str
    population_confidence: float = Field(ge=0, le=1)
    scientific_note: str = (
        "Population-level maturity distribution based on optical color analysis only. "
        "Biological variability is high. Not a harvest recommendation."
    )
