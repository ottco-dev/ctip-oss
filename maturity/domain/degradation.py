"""
maturity.domain.degradation — Detection of degraded trichome states.

DEGRADATION MECHANISMS:
━━━━━━━━━━━━━━━━━━━━━━
Trichome degradation occurs through multiple pathways:

1. PHOTO-OXIDATION (UV exposure):
   THC → CBN via decarboxylation + oxidation chain
   Color progression: amber → brown → dark brown
   Structural: head may collapse as volatile terpenes evaporate

2. MECHANICAL DAMAGE:
   Physical pressure (handling, wind) → burst secretory cells
   Visual: ruptured head, spreading resin, irregular shape

3. ENZYMATIC DEGRADATION (post-harvest):
   Plant enzymes continue degrading cannabinoids after harvest
   Visual: progressive browning from inside out

4. MOLD/FUNGAL CONTAMINATION:
   Filamentous fungi colonize resin glands
   Visual: white/grey fuzzy structures, irregular morphology

5. OXIDATIVE BROWNING:
   Similar to fruit browning — polyphenol oxidase activity
   Progressive darkening independent of UV

VISUAL SIGNATURES (detectable by CV):
- Collapsed heads: deformed ellipse, aspect ratio anomalies
- Burst heads: irregular edges, spreading dark spots
- Brown/black coloration: HSV hue shift toward dark browns
- Lost stalks: detached heads (morphology indicator)
- Irregular internal texture: LBP pattern becomes highly non-uniform
- Very low trichome count: batch-level signal

WHAT CANNOT BE DETECTED RELIABLY:
- Early-stage enzymatic degradation (no visible change yet)
- Mold contamination (requires higher magnification or UV)
- Specific cannabinoid degradation products (requires chromatography)
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray


@dataclass
class DegradationResult:
    """Assessment of trichome degradation state."""

    degradation_score: float
    """Overall degradation score [0, 1]. 0 = pristine, 1 = fully degraded."""

    is_degraded: bool
    """True if degradation score exceeds threshold (> 0.55)."""

    collapse_score: float
    """Evidence of structural collapse (shape deformation)."""

    color_degradation_score: float
    """Evidence of oxidative browning (dark brown/black coloration)."""

    texture_degradation_score: float
    """Evidence of texture irregularity (burst/damaged structure)."""

    burst_probability: float
    """Probability that secretory head is ruptured/burst."""

    degradation_type: str
    """Most likely degradation type: 'oxidized', 'collapsed', 'burst', 'pristine'"""

    confidence: float
    """Confidence in degradation assessment."""

    warnings: list[str]
    """List of specific degradation indicators detected."""


def detect_color_degradation(
    image: NDArray[np.uint8],
) -> float:
    """
    Detect oxidative color degradation from image color analysis.

    Oxidized/degraded trichomes exhibit:
    - Dark brown to black hue (HSV hue ~10-30° and low value)
    - Very low value (V < 80 in HSV) for blackened trichomes
    - Reduced saturation in uniformly darkened areas

    Args:
        image: RGB image of trichome head crop

    Returns:
        Color degradation score [0, 1]. Higher = more degraded.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    h = hsv[:, :, 0]  # 0-180 in OpenCV
    s = hsv[:, :, 1] / 255.0
    v = hsv[:, :, 2] / 255.0

    # Brown hue: OpenCV hue 8-20 (corresponds to 16-40° in standard HSV)
    brown_hue_mask = (h >= 8) & (h <= 25) & (s > 0.25) & (v > 0.15) & (v < 0.65)
    brown_fraction = float(brown_hue_mask.mean())

    # Very dark pixels (black/extremely dark brown) — collapsed/fully oxidized
    very_dark_mask = v < 0.12
    dark_fraction = float(very_dark_mask.mean())

    # Combined degradation score
    color_deg = float(np.clip(
        0.60 * brown_fraction * 2.5  # Brown is strongest indicator
        + 0.40 * dark_fraction * 3.0,  # Very dark also indicates degradation
        0.0, 1.0
    ))

    return color_deg


def detect_structural_collapse(
    image: NDArray[np.uint8],
    min_size_px: int = 10,
) -> tuple[float, float]:
    """
    Detect structural collapse and burst probability from image morphology.

    A healthy trichome head is roughly circular.
    Collapsed heads show:
    - Very low aspect ratio (flattened)
    - Irregular boundary (high contour complexity)
    - Dark spreading spots (burst resin)

    Args:
        image: RGB image crop
        min_size_px: Minimum head size in pixels to analyze

    Returns:
        (collapse_score, burst_probability) each in [0, 1]
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape

    if h < min_size_px or w < min_size_px:
        return 0.5, 0.3  # Uncertain for very small crops

    # Threshold to isolate head from background
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return 0.5, 0.3

    # Largest contour = head
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)

    if area < 16:
        return 0.5, 0.3

    # Circularity: 4π·area / perimeter² = 1 for perfect circle
    perimeter = cv2.arcLength(cnt, True)
    circularity = 4 * np.pi * area / (perimeter ** 2 + 1e-6)

    # Convexity: ratio of contour area to convex hull area
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    convexity = area / (hull_area + 1e-6)

    # Ellipse fitting
    if len(cnt) >= 5:
        ellipse = cv2.fitEllipse(cnt)
        major_ax = max(ellipse[1])
        minor_ax = min(ellipse[1])
        aspect_ratio = minor_ax / (major_ax + 1e-6)
    else:
        aspect_ratio = 0.5

    # Healthy head: circularity ~0.7-1.0, convexity ~0.85-1.0, aspect ~0.7-1.0
    # Collapsed: low all three

    collapse_score = float(np.clip(
        (1 - circularity) * 0.4
        + (1 - convexity) * 0.3
        + (1 - aspect_ratio) * 0.3,
        0.0, 1.0
    ))

    # Burst probability: many concavities in contour = burst cell walls
    # Measure via defect analysis
    hull_ints = cv2.convexHull(cnt, returnPoints=False)
    try:
        defects = cv2.convexityDefects(cnt, hull_ints)
        if defects is not None:
            significant_defects = np.sum(defects[:, 0, 3] > 256 * 5)  # >5px depth
            burst_prob = float(np.clip(significant_defects / 10.0, 0, 1))
        else:
            burst_prob = 0.0
    except cv2.error:
        burst_prob = 0.0

    return collapse_score, burst_prob


def detect_texture_irregularity(
    image: NDArray[np.uint8],
) -> float:
    """
    Detect degradation-associated texture irregularities.

    Degraded trichomes exhibit high local texture variance in the
    head region — a mix of dark spots (burst/oxidized) and light
    areas (intact resin) creates heterogeneous texture.

    Args:
        image: RGB trichome crop

    Returns:
        Texture degradation score [0, 1]
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    h, w = gray.shape

    # Local standard deviation via sliding window
    # High local_std → irregular texture (patches of different intensities)
    kernel_size = max(3, min(h, w) // 4) | 1  # Odd number, ~1/4 of image

    # Compute local mean and local std via box filter
    local_mean = cv2.blur(gray, (kernel_size, kernel_size))
    local_sq_mean = cv2.blur(gray ** 2, (kernel_size, kernel_size))
    local_std = np.sqrt(np.maximum(local_sq_mean - local_mean ** 2, 0))

    # High std = heterogeneous texture
    mean_local_std = float(local_std.mean()) / 255.0

    # Normalize: std of 0.1+ (25/255) is suspicious, 0.3+ is likely degraded
    texture_score = float(np.clip(mean_local_std * 4 - 0.3, 0, 1))

    return texture_score


def assess_degradation(
    image: NDArray[np.uint8],
    degradation_threshold: float = 0.55,
) -> DegradationResult:
    """
    Full degradation assessment for a trichome head crop.

    Combines color, structural, and texture analysis to produce
    a comprehensive degradation score with type classification.

    Args:
        image: RGB image crop of a single trichome head
        degradation_threshold: Score above which trichome is flagged as degraded

    Returns:
        DegradationResult with complete degradation assessment
    """
    warnings: list[str] = []

    color_deg = detect_color_degradation(image)
    collapse, burst = detect_structural_collapse(image)
    texture_deg = detect_texture_irregularity(image)

    # Weighted composite degradation score
    degradation_score = float(np.clip(
        0.45 * color_deg
        + 0.30 * collapse
        + 0.15 * burst
        + 0.10 * texture_deg,
        0.0, 1.0
    ))

    is_degraded = degradation_score > degradation_threshold

    # Determine primary degradation type
    if color_deg > 0.5 and collapse < 0.4:
        deg_type = "oxidized"
        warnings.append("Strong brown/dark coloration detected — oxidation suspected")
    elif collapse > 0.5:
        deg_type = "collapsed"
        warnings.append("Deformed head shape detected — structural collapse suspected")
    elif burst > 0.4:
        deg_type = "burst"
        warnings.append("Contour defects detected — ruptured secretory cell suspected")
    elif degradation_score > degradation_threshold:
        deg_type = "oxidized"  # Default
    else:
        deg_type = "pristine"

    if color_deg > 0.7:
        warnings.append("Severe oxidative browning — likely post-harvest degradation")
    if burst > 0.6:
        warnings.append("High burst probability — handle samples carefully")

    # Confidence: higher for images with strong signals
    confidence = float(np.clip(
        max(color_deg, collapse, burst) * 1.5,
        0.1, 0.9
    ))

    return DegradationResult(
        degradation_score=degradation_score,
        is_degraded=is_degraded,
        collapse_score=collapse,
        color_degradation_score=color_deg,
        texture_degradation_score=texture_deg,
        burst_probability=burst,
        degradation_type=deg_type,
        confidence=confidence,
        warnings=warnings,
    )
