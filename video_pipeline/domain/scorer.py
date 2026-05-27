"""
video_pipeline.domain.scorer — Frame quality scoring for microscopy video.

Combines multiple quality dimensions into a single quality score:

1. FOCUS SCORE (weight=0.55):
   Primary quality indicator for microscopy.
   Uses Laplacian variance + Tenengrad + FFT composite.
   Reference: Pertuz et al. (2013). Pattern Recognition 46(5):1415–1432.

2. EXPOSURE SCORE (weight=0.25):
   Histogram-based exposure assessment.
   Penalizes overexposed (>250) and underexposed (<5) images.
   Well-exposed microscopy images use 20–80% of dynamic range.

3. NOISE SCORE (weight=0.20):
   Estimates image noise from high-frequency residuals.
   High noise → low score.
   Uses variance of the Laplacian of the Laplacian (2nd-order noise estimate).

OUTPUT:
- Composite quality score [0, 1]
- Per-dimension subscores
- Quality label: "excellent", "good", "acceptable", "poor", "unusable"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from numpy.typing import NDArray


@dataclass
class FrameQualityScore:
    """Quality assessment for a single video frame."""

    composite: float
    """Overall quality score [0, 1]. Weighted combination of all subscores."""

    focus: float
    """Focus quality [0, 1]. Primary dimension for microscopy."""

    exposure: float
    """Exposure quality [0, 1]. 1.0 = well-exposed for microscopy."""

    noise: float
    """Inverse noise level [0, 1]. 1.0 = low noise."""

    @property
    def quality_label(self) -> str:
        """Human-readable quality label."""
        if self.composite >= 0.80:
            return "excellent"
        elif self.composite >= 0.60:
            return "good"
        elif self.composite >= 0.40:
            return "acceptable"
        elif self.composite >= 0.20:
            return "poor"
        else:
            return "unusable"

    @property
    def is_usable(self) -> bool:
        """True if the frame is worth keeping for analysis."""
        return self.composite >= 0.35 and self.focus >= 0.25

    @property
    def is_excellent(self) -> bool:
        return self.composite >= 0.75


# Weights for composite score computation
_WEIGHT_FOCUS = 0.55
_WEIGHT_EXPOSURE = 0.25
_WEIGHT_NOISE = 0.20


def score_frame(
    frame_rgb: NDArray[np.uint8],
    *,
    use_focus_composite: bool = True,
) -> FrameQualityScore:
    """
    Score a single microscopy video frame on multiple quality dimensions.

    Args:
        frame_rgb:           RGB frame, uint8.
        use_focus_composite: If True, use composite focus score.
                             If False, use fast Laplacian only (faster).

    Returns:
        FrameQualityScore with all subscores and composite.
    """
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)

    # --- Focus score ---
    if use_focus_composite:
        try:
            from focus.metrics.composite import compute_focus_score
            focus_result = compute_focus_score(frame_rgb)
            focus_score = focus_result.composite
        except Exception:
            focus_score = _fast_focus_score(gray)
    else:
        focus_score = _fast_focus_score(gray)

    # --- Exposure score ---
    exposure_score = _score_exposure(gray)

    # --- Noise score ---
    noise_score = _score_noise(gray)

    # --- Composite ---
    composite = (
        _WEIGHT_FOCUS * focus_score
        + _WEIGHT_EXPOSURE * exposure_score
        + _WEIGHT_NOISE * noise_score
    )

    return FrameQualityScore(
        composite=float(np.clip(composite, 0.0, 1.0)),
        focus=float(focus_score),
        exposure=float(exposure_score),
        noise=float(noise_score),
    )


def _fast_focus_score(gray: NDArray[np.uint8]) -> float:
    """Fast Laplacian variance focus metric, normalized to [0,1]."""
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    var = float(lap.var())
    # Empirical normalization: variance > 2000 → good focus for microscopy
    return float(np.clip(var / 2000.0, 0.0, 1.0))


def _score_exposure(gray: NDArray[np.uint8]) -> float:
    """
    Score image exposure from the histogram.

    Penalizes:
    - Overexposed pixels (>250 intensity) → harsh reflections
    - Underexposed pixels (<5 intensity) → black areas
    - Very concentrated histograms (lack of contrast)

    Returns:
        Exposure score [0, 1]. 1.0 = ideal exposure.
    """
    n_pixels = gray.size
    if n_pixels == 0:
        return 0.0

    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist_norm = hist / n_pixels  # Normalized frequency

    # Overexposure penalty: fraction of pixels >250
    overexposed = float(hist[250:].sum() / n_pixels)
    overexposed_penalty = np.clip(overexposed * 5.0, 0.0, 1.0)

    # Underexposure penalty: fraction of pixels <5
    underexposed = float(hist[:5].sum() / n_pixels)
    underexposed_penalty = np.clip(underexposed * 5.0, 0.0, 1.0)

    # Mean intensity: ideal range for microscopy is 80–200
    mean_intensity = float(gray.mean())
    if mean_intensity < 30 or mean_intensity > 230:
        brightness_penalty = 0.4
    elif mean_intensity < 60 or mean_intensity > 210:
        brightness_penalty = 0.15
    else:
        brightness_penalty = 0.0

    exposure = 1.0 - overexposed_penalty - underexposed_penalty - brightness_penalty
    return float(np.clip(exposure, 0.0, 1.0))


def _score_noise(gray: NDArray[np.uint8]) -> float:
    """
    Estimate noise level using the second-order Laplacian of Laplacian.

    High frequency noise contributes high values to the second-order
    Laplacian that are NOT present in genuinely sharp edges.

    Returns:
        Noise score [0, 1]. 1.0 = low noise (good).
    """
    # Method: sigma estimate from median absolute deviation of high-pass residuals
    # Reference: Immerkaer (1996). Fast noise variance estimation. CVIU 64(2):300-302.
    gray_f = gray.astype(np.float64)

    h = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)
    filtered = cv2.filter2D(gray_f, -1, h)

    h_crop, w_crop = filtered.shape
    if h_crop < 3 or w_crop < 3:
        return 1.0  # Too small to assess noise

    # Trim edges to avoid border artifacts
    interior = filtered[2:-2, 2:-2]
    sigma = float(np.std(interior))

    # Empirical normalization: sigma > 25 → very noisy for 8-bit microscopy
    noise_level = float(np.clip(sigma / 25.0, 0.0, 1.0))
    noise_score = 1.0 - noise_level  # Invert: high noise → low score

    return float(np.clip(noise_score, 0.0, 1.0))
