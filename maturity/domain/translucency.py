"""
maturity.domain.translucency — Translucency estimation for trichome heads.

SCIENTIFIC BACKGROUND:
━━━━━━━━━━━━━━━━━━━━━
Clear/transparent trichomes transmit light through the secretory head,
making the internal structure and underlying tissue visible.

As trichomes mature from clear → cloudy, the secretory cavity fills
with cannabinoid acids and terpenes. This dense resinous mixture
scatters and reflects light more than it transmits, causing the
characteristic opaque/milky appearance.

OPTICAL PHYSICS:
The transition from clear to cloudy follows Mie scattering theory:
- Clear state: Rayleigh/minimal scattering, high transmittance
- Cloudy state: Strong Mie scattering from resin droplets (~1-10µm)
- Amber state: Absorption dominates (chromophore formation via oxidation)

MEASUREMENT APPROACH (INDIRECT):
Direct transmittance measurement requires:
1. Polarization microscopy (measuring birefringence)
2. Darkfield illumination (separating scattered from transmitted light)

With standard brightfield microscopy (our target platform), we estimate
translucency INDIRECTLY from:
1. Relative luminance vs background (transparent heads appear brighter)
2. Internal detail visibility (can we see internal structure through head?)
3. Background bleed-through (edge/background visible through head center)
4. Hue saturation in head region (transparent = lower saturation)

IMPORTANT CAVEAT:
This is an APPROXIMATION. Without polarized light or darkfield,
translucency estimation from standard brightfield images has
significant uncertainty. Results should be treated as soft signals,
not hard measurements.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray


@dataclass
class TranslucencyResult:
    """
    Translucency estimate for a single trichome head crop.

    All values are approximate estimates from brightfield microscopy.
    """

    translucency_score: float
    """
    Estimated translucency in [0, 1].
    0 = completely opaque (amber/degraded)
    1 = completely transparent (clear phase)
    """

    opacity_score: float
    """
    Complement of translucency: 1 - translucency_score.
    Higher = more cloudy/opaque.
    """

    relative_luminance: float
    """Mean head luminance relative to surrounding background."""

    saturation_in_head: float
    """Mean HSV saturation in the head region."""

    internal_structure_score: float
    """
    Estimate of how much internal structure is visible through the head.
    High = transparent (can see through), low = opaque (solid color).
    """

    edge_visibility: float
    """
    Whether the background is visible through the head edges.
    High = transparent edges (clear phase indicator).
    """

    confidence: float
    """Confidence in the estimate (lower for small crops or uniform images)."""

    caveat: str = (
        "Translucency estimated from brightfield microscopy only. "
        "Requires polarization/darkfield for accurate measurement."
    )


def estimate_translucency(
    head_crop: NDArray[np.uint8],
    background_mask: NDArray[np.bool_] | None = None,
) -> TranslucencyResult:
    """
    Estimate trichome head translucency from a brightfield image crop.

    Args:
        head_crop: RGB image crop of a single trichome head
                   (typically 30-100px square)
        background_mask: Optional boolean mask marking background pixels
                         in the crop (True = background, False = head)

    Returns:
        TranslucencyResult with all translucency estimates
    """
    if head_crop.size == 0 or head_crop.shape[0] < 4 or head_crop.shape[1] < 4:
        return TranslucencyResult(
            translucency_score=0.5, opacity_score=0.5,
            relative_luminance=0.5, saturation_in_head=0.5,
            internal_structure_score=0.5, edge_visibility=0.5,
            confidence=0.0,
        )

    h, w = head_crop.shape[:2]
    gray = cv2.cvtColor(head_crop, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(head_crop, cv2.COLOR_RGB2HSV)

    # Define head region mask (center ellipse as proxy for head area)
    cy, cx = h // 2, w // 2
    ry, rx = max(h // 3, 2), max(w // 3, 2)
    y_idx, x_idx = np.ogrid[:h, :w]
    head_mask = ((x_idx - cx) / rx) ** 2 + ((y_idx - cy) / ry) ** 2 <= 1.0

    if head_mask.sum() < 4:
        head_mask = np.ones((h, w), dtype=bool)

    # 1. Relative luminance vs background
    head_lum = float(gray[head_mask].mean()) / 255.0 if head_mask.any() else 0.5
    bg_mask = ~head_mask
    bg_lum = float(gray[bg_mask].mean()) / 255.0 if bg_mask.any() else head_lum
    relative_luminance = float(np.clip(head_lum / (bg_lum + 0.01), 0, 3))

    # Transparent heads are brighter than background in brightfield
    # (more light transmitted through the head to the camera)
    # Normalize: 1.0 = same as background, >1 = brighter = more transparent
    luminance_translucency = float(np.clip((relative_luminance - 0.5) / 1.5, 0, 1))

    # 2. Saturation in head region (transparent = low saturation)
    saturation = float(hsv[:, :, 1][head_mask].mean()) / 255.0

    # Clear trichomes have low saturation (near-white/transparent appearance)
    # Cloudy: slightly higher saturation
    # Amber: high saturation (golden hue)
    # Inverted: high saturation → low translucency
    saturation_translucency = float(np.clip(1.0 - saturation * 1.5, 0, 1))

    # 3. Internal structure visibility via local variance
    # Transparent heads show internal detail → higher local variance
    head_region = gray[head_mask].reshape(-1) if head_mask.any() else gray.ravel()
    local_variance = float(head_region.std()) / 255.0

    # High variance = visible internal structure = more transparent
    structure_score = float(np.clip(local_variance * 4, 0, 1))

    # 4. Edge visibility through head (edge region transparency)
    # Create edge ring mask
    inner_ry, inner_rx = max(ry - 3, 1), max(rx - 3, 1)
    inner_mask = ((x_idx - cx) / inner_rx) ** 2 + ((y_idx - cy) / inner_ry) ** 2 <= 1.0
    edge_ring = head_mask & ~inner_mask

    if edge_ring.any() and bg_mask.any():
        edge_lum = float(gray[edge_ring].mean()) / 255.0
        edge_visibility = float(np.clip(edge_lum / (bg_lum + 0.01), 0, 2))
        edge_visibility = float(np.clip(edge_visibility - 0.5, 0, 1))
    else:
        edge_visibility = 0.5

    # 5. Composite translucency score
    # Weights based on reliability of each proxy measure
    translucency_score = float(np.clip(
        0.30 * luminance_translucency
        + 0.35 * saturation_translucency
        + 0.20 * structure_score
        + 0.15 * edge_visibility,
        0.0, 1.0
    ))

    # Confidence: based on head crop size and background contrast
    confidence = float(np.clip(
        min(h, w) / 64.0  # Larger crops → more reliable
        * min(abs(head_lum - bg_lum) * 5, 1.0)  # Better contrast → more reliable
        * 0.7,  # Max confidence 0.7 for brightfield-only
        0.05, 0.7
    ))

    return TranslucencyResult(
        translucency_score=translucency_score,
        opacity_score=1.0 - translucency_score,
        relative_luminance=relative_luminance,
        saturation_in_head=saturation,
        internal_structure_score=structure_score,
        edge_visibility=edge_visibility,
        confidence=confidence,
    )


def classify_transparency_state(result: TranslucencyResult) -> str:
    """
    Map translucency score to a categorical transparency label.

    Labels:
    - "transparent":  score ≥ 0.70 (likely clear phase)
    - "semi-opaque":  score 0.40-0.70 (transitional)
    - "opaque":       score < 0.40 (cloudy/amber/degraded)
    """
    if result.translucency_score >= 0.70:
        return "transparent"
    elif result.translucency_score >= 0.40:
        return "semi-opaque"
    else:
        return "opaque"
