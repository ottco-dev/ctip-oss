"""
maturity.explainability.feature_report — Per-instance feature breakdown for maturity analysis.

Generates human-readable, machine-readable feature reports for individual
trichome maturity predictions.

The report answers:
1. WHAT was classified and with what confidence?
2. WHY was this classification made? (which features drove the decision)
3. HOW reliable is this classification? (uncertainty, sample quality)
4. WHAT CAN'T we say? (epistemic constraints per scientific_rules.py)

OUTPUT FORMATS:
- Python dataclass (structured, for API responses)
- Markdown (human-readable reports)
- JSON (for frontend display)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

from shared.core.enums import MaturityStage
from maturity.domain.scientific_rules import (
    get_stage_caveat,
    get_report_preamble,
    APPROVED_CLAIMS,
)


@dataclass
class ColorFeatureSummary:
    """Summary of color features driving the classification."""
    dominant_hue_label: str    # "clear_white", "cloudy_cream", "amber", "brown", "dark"
    hsv_saturation: float      # Mean saturation [0,1]
    hsv_value: float           # Mean value [0,1]
    amber_pixel_fraction: float
    clear_pixel_fraction: float
    cloudy_pixel_fraction: float
    key_color_finding: str     # Single most important color observation


@dataclass
class TextureFeatureSummary:
    """Summary of texture features."""
    entropy: float
    entropy_label: str         # "low/uniform", "medium", "high/complex"
    lbp_uniformity: float
    glcm_contrast: float
    contrast_label: str        # "smooth", "textured", "rough"
    key_texture_finding: str


@dataclass
class QualityFlags:
    """Image and crop quality indicators."""
    focus_score: float | None
    focus_label: str | None
    crop_size_px: tuple[int, int]
    exposure_ok: bool
    contains_background: bool  # If crop is too large and contains non-trichome area
    overall_quality: str       # "good", "acceptable", "poor"


@dataclass
class InstanceFeatureReport:
    """
    Complete feature-level explanation for a single trichome maturity prediction.
    """

    # Identity
    trichome_id: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    # Prediction
    predicted_stage: str = "unknown"
    confidence: float = 0.0
    is_reliable: bool = False

    # Feature summaries
    color_summary: ColorFeatureSummary | None = None
    texture_summary: TextureFeatureSummary | None = None
    quality_flags: QualityFlags | None = None

    # Probability distribution
    stage_probabilities: dict[str, float] = field(default_factory=dict)

    # Scientific context
    stage_caveat: str = ""
    prohibited_claims: list[str] = field(default_factory=list)
    preamble: str = field(default_factory=get_report_preamble)

    # Explanation text
    explanation: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        d = {}
        d["trichome_id"] = self.trichome_id
        d["timestamp"] = self.timestamp
        d["prediction"] = {
            "stage": self.predicted_stage,
            "confidence": round(self.confidence, 4),
            "is_reliable": self.is_reliable,
            "probabilities": {k: round(v, 4) for k, v in self.stage_probabilities.items()},
        }
        if self.color_summary:
            d["color_features"] = {
                "dominant_hue": self.color_summary.dominant_hue_label,
                "saturation": round(self.color_summary.hsv_saturation, 3),
                "value": round(self.color_summary.hsv_value, 3),
                "amber_fraction": round(self.color_summary.amber_pixel_fraction, 3),
                "finding": self.color_summary.key_color_finding,
            }
        if self.texture_summary:
            d["texture_features"] = {
                "entropy": round(self.texture_summary.entropy, 3),
                "entropy_label": self.texture_summary.entropy_label,
                "lbp_uniformity": round(self.texture_summary.lbp_uniformity, 3),
                "contrast": round(self.texture_summary.glcm_contrast, 3),
                "finding": self.texture_summary.key_texture_finding,
            }
        if self.quality_flags:
            d["quality"] = {
                "focus_score": self.quality_flags.focus_score,
                "crop_size": list(self.quality_flags.crop_size_px),
                "overall": self.quality_flags.overall_quality,
            }
        d["scientific_context"] = {
            "caveat": self.stage_caveat,
            "preamble": self.preamble,
            "prohibited_claims": self.prohibited_claims,
        }
        d["explanation"] = self.explanation
        return d

    def to_markdown(self) -> str:
        """Generate human-readable Markdown report."""
        lines = [
            f"# Trichome Maturity Analysis Report",
            f"**ID:** `{self.trichome_id}`  |  **Time:** {self.timestamp}",
            "",
            "## Classification Result",
            f"| Property | Value |",
            "| --- | --- |",
            f"| **Predicted Stage** | {self.predicted_stage.upper()} |",
            f"| **Confidence** | {self.confidence:.1%} |",
            f"| **Reliable** | {'✓ Yes' if self.is_reliable else '⚠ Low confidence'} |",
            "",
        ]

        if self.stage_probabilities:
            lines += ["## Probability Distribution", "| Stage | Probability |", "| --- | --- |"]
            for stage, prob in sorted(self.stage_probabilities.items(), key=lambda x: -x[1]):
                bar = "█" * int(prob * 20)
                lines.append(f"| {stage} | {bar} {prob:.1%} |")
            lines.append("")

        if self.color_summary:
            lines += [
                "## Color Evidence",
                f"- **Dominant hue:** {self.color_summary.dominant_hue_label}",
                f"- **Saturation:** {self.color_summary.hsv_saturation:.2f} | "
                f"**Value:** {self.color_summary.hsv_value:.2f}",
                f"- **Amber pixels:** {self.color_summary.amber_pixel_fraction:.1%}",
                f"- **Key finding:** *{self.color_summary.key_color_finding}*",
                "",
            ]

        if self.texture_summary:
            lines += [
                "## Texture Evidence",
                f"- **Entropy:** {self.texture_summary.entropy:.2f} ({self.texture_summary.entropy_label})",
                f"- **LBP Uniformity:** {self.texture_summary.lbp_uniformity:.2f}",
                f"- **Contrast:** {self.texture_summary.contrast_label}",
                f"- **Key finding:** *{self.texture_summary.key_texture_finding}*",
                "",
            ]

        if self.explanation:
            lines += ["## Summary", self.explanation, ""]

        lines += [
            "## ⚠ Scientific Caveats",
            f"> {self.stage_caveat}",
            "",
            "## Prohibited Claims",
            "> The following claims **cannot** be made from this analysis:",
        ]
        for claim in self.prohibited_claims:
            lines.append(f"> - {claim}")

        lines += ["", "---", f"*{self.preamble}*"]
        return "\n".join(lines)


def build_color_summary(color_features) -> ColorFeatureSummary:
    """
    Build color summary from a ColorFeatureVector.

    Args:
        color_features: ColorFeatureVector from color_features.py

    Returns:
        ColorFeatureSummary
    """
    # Determine dominant hue label
    s = color_features.mean_saturation_hsv if hasattr(color_features, 'mean_saturation_hsv') else 0.3
    v = color_features.mean_value_hsv if hasattr(color_features, 'mean_value_hsv') else 0.6
    amber = color_features.amber_fraction if hasattr(color_features, 'amber_fraction') else 0.0
    clear = color_features.clear_fraction if hasattr(color_features, 'clear_fraction') else 0.0
    cloudy = getattr(color_features, 'cloudy_fraction', 0.0)

    if amber > 0.30:
        hue_label = "amber/golden"
        finding = f"Strong amber coloration detected ({amber:.0%} of pixels)"
    elif clear > 0.50 and s < 0.20:
        hue_label = "clear/transparent"
        finding = "Low saturation, high value — consistent with clear/glassy appearance"
    elif v > 0.50 and s < 0.30:
        hue_label = "cloudy/milky"
        finding = "Moderate saturation — consistent with cloudy/opaque appearance"
    elif v < 0.30:
        hue_label = "dark/degraded"
        finding = "Very low value detected — possible degradation or dark coloration"
    else:
        hue_label = "mixed"
        finding = "Mixed color distribution observed"

    return ColorFeatureSummary(
        dominant_hue_label=hue_label,
        hsv_saturation=s,
        hsv_value=v,
        amber_pixel_fraction=amber,
        clear_pixel_fraction=clear,
        cloudy_pixel_fraction=cloudy,
        key_color_finding=finding,
    )


def build_texture_summary(texture_features) -> TextureFeatureSummary:
    """Build texture summary from a TextureFeatureVector."""
    entropy = getattr(texture_features, 'shannon_entropy', 2.0)
    uniformity = getattr(texture_features, 'lbp_uniformity', 0.7)
    contrast = getattr(texture_features, 'glcm_contrast', 0.5)

    # Entropy labels
    if entropy < 1.5:
        entropy_label = "low/uniform"
        texture_finding = "Highly uniform texture — consistent with clear/smooth trichome"
    elif entropy < 3.0:
        entropy_label = "medium"
        texture_finding = "Moderate texture complexity — consistent with cloudy/granular interior"
    else:
        entropy_label = "high/complex"
        texture_finding = "Complex heterogeneous texture — consistent with degraded/mixed state"

    # Contrast labels
    if contrast < 0.1:
        contrast_label = "smooth"
    elif contrast < 0.5:
        contrast_label = "textured"
    else:
        contrast_label = "rough"

    return TextureFeatureSummary(
        entropy=entropy,
        entropy_label=entropy_label,
        lbp_uniformity=uniformity,
        glcm_contrast=contrast,
        contrast_label=contrast_label,
        key_texture_finding=texture_finding,
    )


def generate_explanation_text(
    stage: str,
    confidence: float,
    color_summary: ColorFeatureSummary | None,
    texture_summary: TextureFeatureSummary | None,
) -> str:
    """Generate human-readable explanation paragraph."""
    parts = [f"The trichome was classified as **{stage.upper()}** with {confidence:.0%} confidence."]

    if color_summary:
        parts.append(f"Color analysis found {color_summary.key_color_finding}.")

    if texture_summary:
        parts.append(f"Texture analysis indicated {texture_summary.key_texture_finding}.")

    if confidence < 0.60:
        parts.append(
            "Note: Confidence is below 60%. Classification should be treated as uncertain. "
            "Image quality or atypical morphology may be affecting accuracy."
        )

    return " ".join(parts)


def create_instance_report(
    trichome_id: str,
    stage: MaturityStage,
    confidence: float,
    probabilities: dict[str, float],
    color_features=None,
    texture_features=None,
    focus_score: float | None = None,
    crop_size: tuple[int, int] = (64, 64),
) -> InstanceFeatureReport:
    """
    Create a complete per-instance feature report.

    Args:
        trichome_id: Unique identifier for this trichome detection
        stage: Predicted MaturityStage
        confidence: Prediction confidence [0,1]
        probabilities: Dict of stage_name → probability
        color_features: ColorFeatureVector (optional)
        texture_features: TextureFeatureVector (optional)
        focus_score: Image focus score (optional)
        crop_size: (H, W) of the trichome crop in pixels

    Returns:
        InstanceFeatureReport
    """
    color_summary = build_color_summary(color_features) if color_features else None
    texture_summary = build_texture_summary(texture_features) if texture_features else None

    # Quality flags
    focus_label = None
    if focus_score is not None:
        if focus_score >= 0.70:
            focus_label = "good"
        elif focus_score >= 0.40:
            focus_label = "acceptable"
        else:
            focus_label = "poor"

    crop_ok = crop_size[0] >= 20 and crop_size[1] >= 20
    overall_quality = "good" if (focus_label in (None, "good", "acceptable") and crop_ok) else "poor"

    quality_flags = QualityFlags(
        focus_score=focus_score,
        focus_label=focus_label,
        crop_size_px=crop_size,
        exposure_ok=True,  # Would need exposure check in pipeline
        contains_background=False,
        overall_quality=overall_quality,
    )

    explanation = generate_explanation_text(
        stage.value, confidence, color_summary, texture_summary
    )

    prohibited = [
        "THC/CBD/CBN concentration cannot be determined from trichome color.",
        "Optimal harvest timing cannot be specified from this analysis.",
        "Strain or cultivar cannot be identified from morphology.",
    ]

    return InstanceFeatureReport(
        trichome_id=trichome_id,
        predicted_stage=stage.value,
        confidence=confidence,
        is_reliable=confidence >= 0.55,
        color_summary=color_summary,
        texture_summary=texture_summary,
        quality_flags=quality_flags,
        stage_probabilities=probabilities,
        stage_caveat=get_stage_caveat(stage),
        prohibited_claims=prohibited,
        explanation=explanation,
    )
