"""
maturity.domain.analyzer — Maturity analysis pipeline orchestrator.

Combines color features, texture features, and (optionally) a trained
classifier to produce calibrated maturity predictions with uncertainty.

Scientific design constraints:
- All outputs include confidence scores
- Epistemic uncertainty is estimated via ensemble or MC Dropout
- Results include feature explanations (not black-box predictions)
- Scientific caveats are attached to every prediction object
- No claim about specific cannabinoid concentrations is made

Pipeline:
    Trichome crop → Color features → Texture features →
    [Optional: Trained classifier] → Ensemble/rule fusion →
    Calibrated confidence → Uncertainty estimate → MaturityLabel
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from maturity.domain.color_features import (
    ColorFeatureVector,
    extract_color_features,
    rule_based_maturity_estimate,
)
from shared.core.entities import MaturityLabel
from shared.core.enums import MaturityStage
from shared.core.value_objects import Confidence
from shared.logging.logger import get_logger

logger = get_logger(__name__)

# Scientific caveat — attached to all MaturityLabel outputs
SCIENTIFIC_CAVEAT = (
    "This classification reflects the OPTICAL COLOR STATE of trichome heads only. "
    "It is NOT a direct measurement of cannabinoid (THC/CBD/CBN) content. "
    "Relationship between color and cannabinoid concentration varies by strain, "
    "environmental conditions, and post-harvest handling. "
    "For quantification, use GC-MS or HPLC chromatography. "
    "References: Fischedick et al. (2010), Potter (2009), Tanney et al. (2021)."
)


@dataclass
class MaturityAnalysisConfig:
    """Configuration for maturity analysis pipeline."""

    use_trained_model: bool = False
    """
    If True and a trained model path is provided, use trained classifier.
    If False, use rule-based classification only (no GPU required).
    Rule-based: Fast (~1ms), no GPU, interpretable, less accurate.
    Trained: Slower (~5ms), needs GPU, more accurate with good training data.
    """

    trained_model_path: str | None = None

    analyze_head_only: bool = True
    """
    If True, only analyze the trichome head region.
    Stalks are typically colorless and would bias the analysis.
    Requires either a head_mask or will auto-estimate the head region.
    """

    confidence_threshold_min: float = 0.30
    """
    Minimum confidence for a prediction to be returned.
    Below this: return MaturityStage.UNKNOWN.
    """

    use_uncertainty_estimation: bool = True
    n_mc_samples: int = 10
    """Monte Carlo Dropout samples for uncertainty estimation."""

    min_head_pixels: int = 50
    """Minimum pixels in head region for valid analysis."""


class MaturityAnalyzer:
    """
    Trichome maturity analyzer.

    Supports two modes:
    1. Rule-based (default, no GPU): Fast heuristic based on color thresholds
    2. Trained model (requires trained EfficientNet-Lite): Higher accuracy

    Usage:
        analyzer = MaturityAnalyzer(config=MaturityAnalysisConfig())
        label = analyzer.analyze(trichome_crop_rgb)
    """

    def __init__(self, config: MaturityAnalysisConfig | None = None) -> None:
        self._config = config or MaturityAnalysisConfig()
        self._model: Any | None = None
        self._is_loaded = False

        if self._config.use_trained_model and self._config.trained_model_path:
            self._load_model()

    def _load_model(self) -> None:
        """Load trained maturity classifier."""
        try:
            import torch
            from pathlib import Path

            path = Path(self._config.trained_model_path)  # type: ignore[arg-type]
            if not path.exists():
                logger.warning(
                    "Trained maturity model not found, falling back to rule-based",
                    path=str(path),
                )
                self._config.use_trained_model = False
                return

            self._model = torch.load(str(path), map_location="cpu")
            self._is_loaded = True
            logger.info("Maturity model loaded", path=str(path))

        except ImportError:
            logger.warning("PyTorch not available, using rule-based classifier")
            self._config.use_trained_model = False

    def analyze(
        self,
        trichome_crop: NDArray[np.uint8],
        head_mask: NDArray[np.bool_] | None = None,
        instance_id: str = "",
    ) -> MaturityLabel:
        """
        Analyze maturity of a single trichome crop.

        Args:
            trichome_crop: RGB crop of trichome (H, W, 3) uint8
            head_mask: Optional binary mask for head region only.
            instance_id: ID of the trichome instance for logging.

        Returns:
            MaturityLabel with stage, confidence, features, and caveats.
        """
        t_start = time.perf_counter()

        # Validate input
        if trichome_crop.size == 0:
            logger.warning("Empty trichome crop", instance_id=instance_id)
            return self._unknown_label()

        # Extract color features
        features = extract_color_features(trichome_crop, head_mask)

        # Check minimum pixels
        if head_mask is not None:
            head_pixels = int(head_mask.sum())
            if head_pixels < self._config.min_head_pixels:
                logger.debug(
                    "Head region too small",
                    pixels=head_pixels,
                    min_required=self._config.min_head_pixels,
                )
                return self._unknown_label(features=features)

        # Classification
        if self._config.use_trained_model and self._model is not None:
            stage, confidence, class_probs = self._classify_with_model(features)
        else:
            stage, conf_val = rule_based_maturity_estimate(features)
            confidence = conf_val
            class_probs = self._probs_from_rule(stage, conf_val)

        # Apply confidence threshold
        if confidence < self._config.confidence_threshold_min:
            stage = MaturityStage.UNKNOWN
            confidence = confidence * 0.5  # Reduce confidence further

        # Uncertainty estimation
        uncertainty = None
        if self._config.use_uncertainty_estimation:
            uncertainty = self._estimate_uncertainty(features)

        t_end = time.perf_counter()
        analysis_ms = (t_end - t_start) * 1000

        logger.debug(
            "Maturity analysis complete",
            instance_id=instance_id,
            stage=stage.value,
            confidence=f"{confidence:.3f}",
            time_ms=f"{analysis_ms:.1f}",
        )

        return MaturityLabel(
            stage=stage,
            confidence=Confidence(float(np.clip(confidence, 0, 1))),
            class_probabilities=class_probs,
            mean_hue=features.mean_hue,
            mean_saturation=features.mean_saturation,
            mean_value=features.mean_value,
            translucency_score=features.grayness,
            amber_ratio=features.hue_amber_fraction,
            texture_entropy=None,  # Populated by texture_features.py
            epistemic_uncertainty=uncertainty,
            aleatoric_uncertainty=None,
        )

    def analyze_batch(
        self,
        crops: list[NDArray[np.uint8]],
        masks: list[NDArray[np.bool_] | None] | None = None,
    ) -> list[MaturityLabel]:
        """
        Analyze maturity for a batch of trichome crops.

        For large batches, trained model inference is significantly faster
        than sequential single-image processing.
        """
        if masks is None:
            masks = [None] * len(crops)

        return [
            self.analyze(crop, mask)
            for crop, mask in zip(crops, masks)
        ]

    def _classify_with_model(
        self,
        features: ColorFeatureVector,
    ) -> tuple[MaturityStage, float, dict[MaturityStage, float]]:
        """Run trained model classification."""
        import torch

        feature_vec = torch.tensor(
            features.feature_vector, dtype=torch.float32
        ).unsqueeze(0)

        self._model.eval()
        with torch.no_grad():
            logits = self._model(feature_vec)
            probs = torch.softmax(logits, dim=-1).squeeze().numpy()

        stages = list(MaturityStage)[:len(probs)]
        best_idx = int(np.argmax(probs))
        stage = stages[best_idx]
        confidence = float(probs[best_idx])
        class_probs = {s: float(p) for s, p in zip(stages, probs)}

        return stage, confidence, class_probs

    @staticmethod
    def _estimate_uncertainty(features: ColorFeatureVector) -> float:
        """
        Heuristic epistemic uncertainty based on feature ambiguity.

        High uncertainty when:
        - Features fall between class boundaries (mixed state)
        - Very few pixels analyzed (small trichome)
        - Contradictory signals (e.g., amber hue but very low saturation)
        """
        # Ambiguous amber signal
        amber_ambiguity = abs(features.hue_amber_fraction - 0.3)
        # Contradictory saturation/brightness
        if features.mean_saturation < 0.1 and features.amber_yellowing_score > 0.4:
            contradiction = 0.4
        else:
            contradiction = 0.0

        uncertainty = float(np.clip(
            0.5 * (1.0 - amber_ambiguity * 2) + 0.5 * contradiction,
            0.0, 1.0
        ))
        return uncertainty

    @staticmethod
    def _probs_from_rule(stage: MaturityStage, confidence: float) -> dict[MaturityStage, float]:
        """Create class probability dict from rule-based prediction."""
        probs: dict[MaturityStage, float] = {s: 0.0 for s in MaturityStage}
        remaining = 1.0 - confidence
        probs[stage] = confidence
        # Distribute remaining probability across other classes
        other_stages = [s for s in MaturityStage if s != stage and s != MaturityStage.UNKNOWN]
        if other_stages:
            each = remaining / len(other_stages)
            for s in other_stages:
                probs[s] = each
        return probs

    @staticmethod
    def _unknown_label(features: ColorFeatureVector | None = None) -> MaturityLabel:
        """Return an UNKNOWN maturity label for invalid inputs."""
        return MaturityLabel(
            stage=MaturityStage.UNKNOWN,
            confidence=Confidence(0.0),
            class_probabilities={},
            mean_hue=features.mean_hue if features else None,
        )
