"""
shared.core.entities — Domain entities with identity.

Unlike value objects, entities are defined by their identity (ID),
not their attributes. Two detections with the same bounding box
but different IDs are different entities.

Design:
- Entities are mutable (within constraints)
- Entities have unique IDs
- Entities carry audit trails (who annotated, when, how)
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from shared.core.enums import (
    AnnotationSource,
    MaturityStage,
    TrichomeType,
)
from shared.core.value_objects import (
    BoundingBox,
    CalibrationScale,
    Confidence,
    ImageDimensions,
    Mask,
    Micrometer,
    PolygonPoints,
)


def _generate_id() -> str:
    return str(uuid.uuid4())


@dataclass
class Detection:
    """
    A single trichome detection — bounding box + class + confidence.

    This is the output of the detection pipeline stage.
    Does NOT include segmentation mask (that's Instance).

    Lifecycle:
    1. Raw model output → Detection (uncalibrated)
    2. NMS applied → filtered Detection
    3. Confidence calibration → calibrated Detection
    4. (Optional) uncertainty estimation added
    """

    id: str = field(default_factory=_generate_id)
    bounding_box: BoundingBox = field(default_factory=lambda: BoundingBox(0, 0, 1, 1))
    confidence: Confidence = field(default_factory=lambda: Confidence(0.5))
    trichome_type: TrichomeType = TrichomeType.UNKNOWN
    model_id: str = ""
    image_id: str = ""
    frame_index: int | None = None  # For video analysis

    # Calibrated confidence (post-temperature scaling or Platt scaling)
    calibrated_confidence: Confidence | None = None

    # Epistemic uncertainty from MC Dropout or ensemble variance
    uncertainty: float | None = None

    # Raw logit before sigmoid (for calibration post-processing)
    raw_logit: float | None = None

    # Additional detection metadata
    class_id: int = 0
    is_difficult: bool = False  # Hard example flag for training
    is_crowd: bool = False  # Overlapping/crowd region flag

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def effective_confidence(self) -> Confidence:
        """Returns calibrated confidence if available, else raw."""
        return self.calibrated_confidence or self.confidence

    @property
    def area_pixels(self) -> float:
        return self.bounding_box.area

    @property
    def is_high_confidence(self) -> bool:
        return self.effective_confidence >= 0.75

    @property
    def is_uncertain(self) -> bool:
        """Flag detections with high uncertainty for human review."""
        if self.uncertainty is None:
            return False
        return self.uncertainty > 0.15  # >15% epistemic uncertainty → flag

    def to_dict(self) -> dict[str, Any]:
        """Serialize to plain dict for storage/API."""
        return {
            "id": self.id,
            "bbox": list(self.bounding_box.to_xyxy()),
            "confidence": float(self.confidence),
            "calibrated_confidence": float(self.calibrated_confidence)
            if self.calibrated_confidence
            else None,
            "uncertainty": self.uncertainty,
            "trichome_type": self.trichome_type.value,
            "model_id": self.model_id,
            "image_id": self.image_id,
            "frame_index": self.frame_index,
            "is_high_confidence": self.is_high_confidence,
            "is_uncertain": self.is_uncertain,
        }

    def __repr__(self) -> str:
        return (
            f"Detection(id={self.id[:8]}..., "
            f"conf={self.effective_confidence}, "
            f"type={self.trichome_type.value}, "
            f"box={self.bounding_box})"
        )


@dataclass
class Instance:
    """
    A segmented trichome instance — detection + pixel-level mask.

    This is the output of the full detection + segmentation pipeline.
    Contains everything needed for downstream analysis (maturity, morphology,
    measurement).

    The separation between Detection and Instance is intentional:
    Not all detections will have masks (mask computation is expensive).
    """

    id: str = field(default_factory=_generate_id)
    detection: Detection | None = None
    mask: Mask | None = None
    polygon: PolygonPoints | None = None

    # Physical measurements (populated by measurement service)
    head_diameter_um: Micrometer | None = None
    stalk_length_um: Micrometer | None = None
    total_height_um: Micrometer | None = None
    calibration_scale: CalibrationScale | None = None

    # Analysis results
    maturity_label: MaturityLabel | None = None
    morphology_type: MorphologyType | None = None

    # Source information
    image_id: str = ""
    image_path: Path | None = None
    annotation_source: AnnotationSource = AnnotationSource.MODEL_PSEUDO

    # Crop of the original image for this instance
    crop: NDArray[np.uint8] | None = None

    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def has_mask(self) -> bool:
        return self.mask is not None

    @property
    def has_measurements(self) -> bool:
        return self.head_diameter_um is not None

    @property
    def confidence(self) -> Confidence | None:
        return self.detection.effective_confidence if self.detection else None

    @property
    def bounding_box(self) -> BoundingBox | None:
        return self.detection.bounding_box if self.detection else None

    @property
    def trichome_type(self) -> TrichomeType:
        return self.detection.trichome_type if self.detection else TrichomeType.UNKNOWN

    @property
    def head_stalk_ratio(self) -> float | None:
        """
        Head diameter to stalk length ratio.

        Biologically informative metric:
        - Capitate stalked: head/stalk ratio typically 0.3–0.8
        - High ratio = prominent head relative to stalk
        """
        if self.head_diameter_um is None or self.stalk_length_um is None:
            return None
        if self.stalk_length_um.value == 0:
            return None
        return self.head_diameter_um.value / self.stalk_length_um.value

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "id": self.id,
            "image_id": self.image_id,
            "trichome_type": self.trichome_type.value,
            "annotation_source": self.annotation_source.value,
            "has_mask": self.has_mask,
            "has_measurements": self.has_measurements,
        }
        if self.detection:
            result["detection"] = self.detection.to_dict()
        if self.head_diameter_um:
            result["head_diameter_um"] = self.head_diameter_um.value
        if self.stalk_length_um:
            result["stalk_length_um"] = self.stalk_length_um.value
        if self.maturity_label:
            result["maturity"] = self.maturity_label.to_dict()
        if self.morphology_type:
            result["morphology"] = self.morphology_type.to_dict()
        return result

    def __repr__(self) -> str:
        return (
            f"Instance(id={self.id[:8]}..., "
            f"type={self.trichome_type.value}, "
            f"mask={self.has_mask}, "
            f"measurements={self.has_measurements})"
        )


@dataclass
class MaturityLabel:
    """
    Maturity analysis result for a single trichome instance.

    IMPORTANT SCIENTIFIC CAVEAT:
    Color-based maturity classification is an indirect proxy for
    cannabinoid content. No visual system can directly measure THC, CBD,
    or terpene concentrations. This classification reflects OPTICAL STATE
    only. Use paired chromatography data for validation.

    Reference:
      Fischedick, J.T. et al. (2010). Metabolic fingerprinting of
      Cannabis sativa L. Phytochemistry 71(17-18):2058-2073.
    """

    stage: MaturityStage
    confidence: Confidence

    # Per-class probability distribution (softmax output)
    class_probabilities: dict[MaturityStage, float] = field(default_factory=dict)

    # Feature-level explanations
    mean_hue: float | None = None  # HSV hue (0-180 in OpenCV)
    mean_saturation: float | None = None  # HSV saturation
    mean_value: float | None = None  # HSV value/brightness
    translucency_score: float | None = None  # 0=opaque, 1=transparent
    amber_ratio: float | None = None  # Fraction of amber pixels in head region
    texture_entropy: float | None = None  # Shannon entropy of texture

    # Uncertainty
    epistemic_uncertainty: float | None = None
    aleatoric_uncertainty: float | None = None

    # Analysis metadata
    model_id: str = ""
    analyzed_region: str = "head"  # "head", "full_trichome", "head_only"

    created_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def is_harvest_ready_indicator(self) -> bool:
        """
        Heuristic indicator only — NOT a harvest recommendation.
        User must use domain knowledge and additional data.
        """
        return self.stage in (
            MaturityStage.CLOUDY,
            MaturityStage.CLOUDY_AMBER_MIX,
            MaturityStage.AMBER,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage.value,
            "confidence": float(self.confidence),
            "class_probabilities": {
                k.value: v for k, v in self.class_probabilities.items()
            },
            "features": {
                "mean_hue": self.mean_hue,
                "mean_saturation": self.mean_saturation,
                "mean_value": self.mean_value,
                "translucency_score": self.translucency_score,
                "amber_ratio": self.amber_ratio,
                "texture_entropy": self.texture_entropy,
            },
            "uncertainty": {
                "epistemic": self.epistemic_uncertainty,
                "aleatoric": self.aleatoric_uncertainty,
            },
            "scientific_caveat": (
                "Color-based classification is an optical proxy only. "
                "Does not directly measure cannabinoid content."
            ),
        }

    def __repr__(self) -> str:
        return (
            f"MaturityLabel(stage={self.stage.value}, "
            f"conf={self.confidence}, "
            f"amber_ratio={self.amber_ratio:.2f if self.amber_ratio else 'N/A'})"
        )


@dataclass
class MorphologyType:
    """
    Trichome morphological type classification result.

    Supports multi-label output because some trichomes in mid-development
    may show transitional morphology (especially sessile → stalked transition).
    """

    primary_type: TrichomeType
    confidence: Confidence

    # Secondary type for ambiguous cases
    secondary_type: TrichomeType | None = None
    secondary_confidence: Confidence | None = None

    # Geometric measurements supporting classification
    head_diameter_px: float | None = None
    stalk_length_px: float | None = None
    head_circularity: float | None = None  # 4π·Area/Perimeter² → 1.0 = perfect circle
    elongation: float | None = None  # major/minor axis ratio

    # Model output
    model_id: str = ""
    class_probabilities: dict[TrichomeType, float] = field(default_factory=dict)

    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "primary_type": self.primary_type.value,
            "confidence": float(self.confidence),
            "secondary_type": self.secondary_type.value if self.secondary_type else None,
            "geometric_features": {
                "head_diameter_px": self.head_diameter_px,
                "stalk_length_px": self.stalk_length_px,
                "head_circularity": self.head_circularity,
                "elongation": self.elongation,
            },
            "class_probabilities": {
                k.value: v for k, v in self.class_probabilities.items()
            },
        }


@dataclass
class TrichomeRegion:
    """
    A spatial region of interest containing multiple trichome instances.

    Represents an analyzed image patch or full image's worth of instances.
    Used for population-level analysis (density maps, distribution statistics,
    maturity ratios across the sample).
    """

    id: str = field(default_factory=_generate_id)
    image_id: str = ""
    image_path: Path | None = None
    image_dimensions: ImageDimensions | None = None

    instances: list[Instance] = field(default_factory=list)
    calibration_scale: CalibrationScale | None = None

    # Session metadata
    microscope_profile: str | None = None
    sample_id: str | None = None
    strain_name: str | None = None
    harvest_day: int | None = None  # Days since seed germination

    # Analysis quality
    image_quality_score: float | None = None  # 0-1
    focus_score: float | None = None  # Higher = sharper

    created_at: datetime = field(default_factory=datetime.utcnow)
    analyzed_at: datetime | None = None

    @property
    def total_trichomes(self) -> int:
        return len(self.instances)

    @property
    def trichome_density(self) -> float | None:
        """
        Trichome density in trichomes per mm².

        Requires calibration scale and image dimensions.
        """
        if self.calibration_scale is None or self.image_dimensions is None:
            return None
        # Convert image area from pixels to mm²
        um_per_px = self.calibration_scale.um_per_pixel
        img_area_um2 = (
            self.image_dimensions.width * um_per_px *
            self.image_dimensions.height * um_per_px
        )
        img_area_mm2 = img_area_um2 / (1000 ** 2)
        if img_area_mm2 == 0:
            return None
        return self.total_trichomes / img_area_mm2

    @property
    def maturity_distribution(self) -> dict[MaturityStage, float]:
        """Relative frequency of each maturity stage."""
        from collections import Counter

        labeled = [
            inst.maturity_label.stage
            for inst in self.instances
            if inst.maturity_label is not None
        ]
        if not labeled:
            return {}
        counts = Counter(labeled)
        total = len(labeled)
        return {stage: count / total for stage, count in counts.items()}

    @property
    def morphology_distribution(self) -> dict[TrichomeType, float]:
        """Relative frequency of each morphology type."""
        from collections import Counter

        typed = [
            inst.morphology_type.primary_type
            for inst in self.instances
            if inst.morphology_type is not None
        ]
        if not typed:
            return {}
        counts = Counter(typed)
        total = len(typed)
        return {t: count / total for t, count in counts.items()}

    @property
    def mean_head_diameter_um(self) -> float | None:
        """Mean head diameter across all measured instances."""
        diameters = [
            inst.head_diameter_um.value
            for inst in self.instances
            if inst.head_diameter_um is not None
        ]
        if not diameters:
            return None
        return float(np.mean(diameters))

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "image_id": self.image_id,
            "total_trichomes": self.total_trichomes,
            "density_per_mm2": self.trichome_density,
            "maturity_distribution": {
                k.value: v for k, v in self.maturity_distribution.items()
            },
            "morphology_distribution": {
                k.value: v for k, v in self.morphology_distribution.items()
            },
            "mean_head_diameter_um": self.mean_head_diameter_um,
            "image_quality_score": self.image_quality_score,
            "focus_score": self.focus_score,
            "sample_metadata": {
                "sample_id": self.sample_id,
                "strain_name": self.strain_name,
                "harvest_day": self.harvest_day,
                "microscope_profile": self.microscope_profile,
            },
        }

    def __repr__(self) -> str:
        return (
            f"TrichomeRegion(id={self.id[:8]}..., "
            f"n={self.total_trichomes}, "
            f"density={self.trichome_density:.1f}/mm² if calibrated)"
        )
