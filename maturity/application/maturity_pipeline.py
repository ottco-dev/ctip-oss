"""
maturity.application.maturity_pipeline — End-to-end trichome maturity analysis pipeline.

Orchestrates:
  1. Crop extraction from Instance masks/bounding boxes
  2. Color feature extraction (HSV + LAB)
  3. Texture feature extraction (LBP + GLCM)
  4. Translucency estimation
  5. Degradation/oxidation detection
  6. Rule-based or CNN maturity classification
  7. Uncertainty estimation (epistemic via ensemble, aleatoric from class entropy)
  8. Explainability (GradCAM or feature importance report)

INPUT:  List[Instance] with crops or masks
OUTPUT: Instance objects with maturity_label populated

SCIENTIFIC CONSTRAINT:
  No claims about cannabinoid content are made or implied.
  All outputs are optical observations with calibrated uncertainty.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from numpy.typing import NDArray

from shared.core.entities import Instance, MaturityLabel
from shared.core.enums import MaturityStage
from shared.core.value_objects import Confidence

from maturity.domain.color_features import extract_color_features, rule_based_maturity_estimate
from maturity.domain.texture_features import extract_texture_features
from maturity.domain.translucency import estimate_translucency
from maturity.domain.degradation import assess_degradation
from maturity.domain.analyzer import MaturityAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class MaturityPipelineConfig:
    """Configuration for the maturity analysis pipeline."""

    use_texture: bool = True
    """Include texture features (LBP, GLCM, Gabor)."""

    use_translucency: bool = True
    """Include translucency estimation."""

    use_degradation: bool = True
    """Check for degraded/oxidized trichomes."""

    use_analyzer: bool = True
    """Use the MaturityAnalyzer ensemble (more accurate, slower)."""

    min_crop_size_px: int = 16
    """Minimum crop dimension (width or height) to attempt analysis."""

    crop_size: int = 64
    """Target crop size for analysis (resized if needed)."""

    uncertainty_threshold: float = 0.25
    """Flag predictions with epistemic uncertainty above this value."""


@dataclass
class MaturityPipelineResult:
    """Complete result of the maturity pipeline."""

    instances: List[Instance]
    stage_distribution: Dict[str, float] = field(default_factory=dict)
    mean_confidence: float = 0.0
    total: int = 0
    analyzed: int = 0
    failed: int = 0
    high_uncertainty: int = 0

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "analyzed": self.analyzed,
            "failed": self.failed,
            "high_uncertainty": self.high_uncertainty,
            "mean_confidence": self.mean_confidence,
            "stage_distribution": self.stage_distribution,
        }


class MaturityPipeline:
    """
    Production maturity analysis pipeline.

    Accepts Instance objects with crops or masks and populates
    the maturity_label field with calibrated classification results.
    """

    def __init__(self, config: Optional[MaturityPipelineConfig] = None) -> None:
        self.config = config or MaturityPipelineConfig()
        self._analyzer = MaturityAnalyzer() if self.config.use_analyzer else None
        logger.info(
            f"MaturityPipeline initialized: "
            f"texture={self.config.use_texture}, "
            f"translucency={self.config.use_translucency}, "
            f"degradation={self.config.use_degradation}"
        )

    def analyze(self, instances: List[Instance]) -> MaturityPipelineResult:
        """
        Analyze maturity of a list of trichome instances.

        Args:
            instances: List of Instance objects. Must have crop or mask set.

        Returns:
            MaturityPipelineResult with all instances populated.
        """
        result = MaturityPipelineResult(instances=list(instances), total=len(instances))
        confidences = []

        for inst in instances:
            crop = self._get_crop(inst)
            if crop is None:
                result.failed += 1
                continue

            try:
                label = self._analyze_one(crop, inst.id)
                inst.maturity_label = label
                result.analyzed += 1
                confidences.append(float(label.confidence))

                if label.epistemic_uncertainty and label.epistemic_uncertainty > self.config.uncertainty_threshold:
                    result.high_uncertainty += 1

            except Exception as e:
                logger.warning(f"Maturity analysis failed for {inst.id[:8]}: {e}")
                result.failed += 1

        # Stage distribution
        stage_counts = Counter(
            inst.maturity_label.stage.value
            for inst in instances
            if inst.maturity_label is not None
        )
        total_labeled = sum(stage_counts.values()) or 1
        result.stage_distribution = {
            stage: count / total_labeled
            for stage, count in stage_counts.items()
        }

        result.mean_confidence = float(np.mean(confidences)) if confidences else 0.0

        logger.info(
            f"Maturity complete: {result.analyzed}/{result.total} analyzed, "
            f"dist={result.stage_distribution}"
        )
        return result

    def analyze_crop(self, crop_rgb: NDArray[np.uint8]) -> MaturityLabel:
        """
        Analyze a single trichome crop and return a MaturityLabel.

        Args:
            crop_rgb: RGB crop image, uint8.

        Returns:
            MaturityLabel with stage, confidence, features, uncertainty.

        Raises:
            TypeError: If the internal analyzer returns an unexpected type
                (guards against silent API contract breakage — TDB-003).
        """
        result = self._analyze_one(crop_rgb, instance_id="direct")
        # TDB-003 guard: explicit runtime type assertion to detect analyzer API drift.
        # If the underlying analyzer ever returns something other than MaturityLabel
        # (e.g., a refactor changes the return type), this surfaces the error immediately
        # instead of silently falling through to a rule-based fallback.
        if not isinstance(result, MaturityLabel):
            raise TypeError(
                f"MaturityPipeline._analyze_one() returned {type(result).__name__!r}, "
                f"expected MaturityLabel. Analyzer API contract broken — check "
                f"maturity.domain.analyzer.MaturityAnalyzer.analyze()."
            )
        return result

    def _get_crop(self, inst: Instance) -> Optional[NDArray[np.uint8]]:
        """Extract a usable crop from an Instance."""
        # Direct crop first
        if inst.crop is not None and inst.crop.size > 0:
            h, w = inst.crop.shape[:2]
            if min(h, w) >= self.config.min_crop_size_px:
                return inst.crop

        # Extract from mask if available
        if inst.mask is not None:
            mask_arr = (
                inst.mask.data if hasattr(inst.mask, "data") else np.array(inst.mask)
            )
            # Return mask as grayscale crop (fallback — color analysis will be limited)
            if mask_arr.ndim == 2 and mask_arr.size > 0:
                h, w = mask_arr.shape
                if min(h, w) >= self.config.min_crop_size_px:
                    # Convert grayscale mask to 3-channel for uniform processing
                    import cv2
                    return cv2.cvtColor(mask_arr, cv2.COLOR_GRAY2RGB)

        return None

    def _analyze_one(
        self,
        crop_rgb: NDArray[np.uint8],
        instance_id: str = "",
    ) -> MaturityLabel:
        """Full maturity analysis for a single crop."""
        import cv2

        # Resize to target size
        h, w = crop_rgb.shape[:2]
        if max(h, w) > self.config.crop_size or min(h, w) < self.config.crop_size // 2:
            crop_rgb = cv2.resize(
                crop_rgb,
                (self.config.crop_size, self.config.crop_size),
                interpolation=cv2.INTER_AREA,
            )

        # 1. Color features (always)
        color = extract_color_features(crop_rgb)

        # 2. Texture features (optional)
        texture = extract_texture_features(crop_rgb) if self.config.use_texture else None

        # 3. Translucency (optional)
        translucency_score: Optional[float] = None
        if self.config.use_translucency:
            try:
                trans = estimate_translucency(crop_rgb)
                translucency_score = getattr(trans, "score", None)
            except Exception:
                pass

        # 4. Degradation (optional)
        is_degraded = False
        if self.config.use_degradation:
            try:
                deg = assess_degradation(crop_rgb)
                is_degraded = getattr(deg, "is_degraded", False)
            except Exception:
                pass

        # 5. Classification
        if self._analyzer is not None:
            try:
                result = self._analyzer.analyze(crop_rgb)
                stage = result.stage
                confidence = result.confidence
                class_probs = getattr(result, "class_probabilities", {})
                epistemic = getattr(result, "epistemic_uncertainty", None)
                aleatoric = getattr(result, "aleatoric_uncertainty", None)
            except Exception as e:
                logger.debug(f"Analyzer failed, falling back to rules: {e}")
                stage, confidence, class_probs = self._rule_classify(
                    color, texture, translucency_score, is_degraded
                )
                epistemic = aleatoric = None
        else:
            stage, confidence, class_probs = self._rule_classify(
                color, texture, translucency_score, is_degraded
            )
            epistemic = aleatoric = None

        return MaturityLabel(
            stage=stage,
            confidence=Confidence(confidence),
            class_probabilities=class_probs,
            mean_hue=getattr(color, "mean_hue", None),
            mean_saturation=getattr(color, "mean_saturation", None),
            mean_value=getattr(color, "mean_value", None),
            translucency_score=translucency_score,
            amber_ratio=getattr(color, "amber_ratio", None),
            texture_entropy=getattr(texture, "shannon_entropy", None) if texture else None,
            epistemic_uncertainty=epistemic,
            aleatoric_uncertainty=aleatoric,
        )

    def _rule_classify(
        self,
        color,
        texture,
        translucency: Optional[float],
        is_degraded: bool,
    ) -> tuple:
        """
        Rule-based classification from extracted features.

        Uses `rule_based_maturity_estimate` from color_features which applies
        botanically-grounded hue/saturation thresholds. If degradation flag is
        set, overrides stage to DEGRADED.
        """
        try:
            if is_degraded:
                stage = MaturityStage.DEGRADED
                confidence = 0.65
                class_probs = {MaturityStage.DEGRADED.value: 0.65}
            else:
                stage, confidence = rule_based_maturity_estimate(color)
                class_probs = {stage.value: confidence}
        except Exception as e:
            logger.warning(f"Rule classification failed: {e}")
            stage = MaturityStage.UNKNOWN
            confidence = 0.3
            class_probs = {}

        return stage, confidence, class_probs
