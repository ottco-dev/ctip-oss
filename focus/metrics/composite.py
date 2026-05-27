"""
focus.metrics.composite — Composite focus scoring for microscopy images.

WHY FOCUS SCORING MATTERS:
Out-of-focus trichomes produce unreliable analysis results:
- Detection: reduced recall (missed trichomes) + increased false positives
- Segmentation: blurred boundaries → inaccurate measurements
- Maturity analysis: blurred color regions → misclassification
- Morphology: loss of structural detail → wrong type classification

For video analysis, focus scoring drives frame selection:
select the N sharpest frames from a session video.

METRICS IMPLEMENTED:

1. Laplacian Variance (LVAR)
   - Fastest metric. Good for overall focus assessment.
   - Computes variance of Laplacian-filtered image.
   - Unfocused images have lower high-frequency content → lower variance.
   - Reference: Pertuz, S. et al. (2013). Analysis of focus measure operators
     for shape-from-focus. Pattern Recognition 46(5):1415-1432.

2. Tenengrad (TENG)
   - Sobel gradient magnitude sum. More robust than LVAR for bright-field.
   - Reference: Krotkov, E. (1988). Focusing. IJCV 1(3):223-237.

3. Normalized Variance (NVAR)
   - Variance normalized by mean intensity. Illumination-invariant.

4. Frequency Domain (FFT)
   - High-frequency content in DCT/FFT. Most robust but slower.
   - Best for detecting subtle focus differences in stacks.

COMPOSITE SCORE:
Weighted combination of all metrics, normalized to [0, 1].
Weights determined empirically on microscopy focus stack datasets.

High score (>0.7): Sharp, suitable for analysis
Medium (0.4-0.7):  Acceptable with caveats
Low (<0.4):        Too blurry — reject or flag for review
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np
from numpy.typing import NDArray


@dataclass
class FocusScoreResult:
    """Comprehensive focus assessment for a single image."""

    composite: float
    """Combined focus score in [0, 1]. Primary metric."""

    laplacian_variance: float
    tenengrad: float
    normalized_variance: float
    fft_score: float

    quality_label: str
    """Human-readable quality assessment."""

    region_scores: NDArray[np.float32] | None = None
    """Per-region scores as 2D grid (if regional analysis requested)."""

    @property
    def is_acceptable(self) -> bool:
        return self.composite >= 0.40

    @property
    def is_good(self) -> bool:
        return self.composite >= 0.70

    def to_dict(self) -> dict[str, float | str]:
        return {
            "composite": self.composite,
            "laplacian_variance": self.laplacian_variance,
            "tenengrad": self.tenengrad,
            "normalized_variance": self.normalized_variance,
            "fft_score": self.fft_score,
            "quality_label": self.quality_label,
        }


def compute_laplacian_variance(gray: NDArray[np.uint8]) -> float:
    """
    Laplacian variance focus metric.

    Applies Laplacian filter and measures output variance.
    Sharp images have high-frequency edges → high variance.
    Blurry images have smooth gradients → low variance.

    Computational cost: Very low (single 3×3 convolution).

    Args:
        gray: Grayscale image (H, W) uint8

    Returns:
        Laplacian variance (unbounded positive float).
        Typical range for microscopy: 50-5000.
    """
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return float(laplacian.var())


def compute_tenengrad(
    gray: NDArray[np.uint8],
    threshold: float = 0.0,
) -> float:
    """
    Tenengrad focus measure.

    Uses Sobel operator to compute gradient magnitude.
    Sum of squared gradients above threshold.

    More robust than Laplacian variance for:
    - Images with specular reflections (common in microscopy)
    - High-noise images (Sobel is less noise-sensitive than Laplacian)

    Args:
        gray: Grayscale image
        threshold: Gradient magnitude threshold (default 0 = no thresholding)

    Returns:
        Mean squared gradient magnitude (normalized by pixel count).
    """
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag_sq = gx ** 2 + gy ** 2

    if threshold > 0:
        grad_mag_sq = grad_mag_sq[grad_mag_sq > threshold ** 2]

    return float(grad_mag_sq.mean()) if grad_mag_sq.size > 0 else 0.0


def compute_normalized_variance(gray: NDArray[np.uint8]) -> float:
    """
    Normalized variance focus measure.

    Variance divided by mean intensity — illumination-invariant.
    Useful when comparing images taken under different lighting conditions.

    Returns:
        Normalized variance. Typical range: 0.1-2.0 for microscopy.
    """
    mean_intensity = float(gray.mean())
    if mean_intensity < 1.0:
        return 0.0
    return float(gray.var()) / mean_intensity


def compute_fft_score(
    gray: NDArray[np.uint8],
    high_freq_radius_fraction: float = 0.15,
) -> float:
    """
    Frequency domain focus score.

    Computes ratio of high-frequency energy to total energy via FFT.
    Sharp images have more high-frequency content.

    The `high_freq_radius_fraction` defines the boundary between
    "high" and "low" frequency components as fraction of image size.
    0.15 = outer 15% of frequency spectrum = high frequency.

    Args:
        gray: Grayscale image
        high_freq_radius_fraction: Threshold for HF/LF separation.

    Returns:
        HF energy fraction in [0, 1]. Higher = sharper.
    """
    # Compute FFT and shift zero frequency to center
    fft = np.fft.fft2(gray.astype(np.float64))
    fft_shifted = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shifted) ** 2

    # Create radial frequency mask
    h, w = gray.shape
    cy, cx = h // 2, w // 2
    y_grid, x_grid = np.ogrid[:h, :w]
    dist = np.sqrt((x_grid - cx) ** 2 + (y_grid - cy) ** 2)
    max_dist = np.sqrt(cx ** 2 + cy ** 2)

    # High frequency = outer ring
    high_freq_mask = dist > (high_freq_radius_fraction * max_dist)

    total_energy = magnitude.sum()
    if total_energy == 0:
        return 0.0

    hf_energy = magnitude[high_freq_mask].sum()
    return float(hf_energy / total_energy)


def compute_focus_score(
    image: NDArray[np.uint8],
    compute_regional: bool = False,
    region_grid: tuple[int, int] = (4, 4),
) -> FocusScoreResult:
    """
    Compute composite focus score for an image.

    Weights (empirically determined on microscopy focus stack data):
    - Laplacian variance: 0.35 (sensitive to fine detail)
    - Tenengrad:          0.35 (robust to noise)
    - Normalized variance: 0.15 (illumination invariance)
    - FFT score:           0.15 (global frequency analysis)

    Args:
        image: RGB or grayscale image
        compute_regional: If True, compute per-region scores on a grid.
        region_grid: Grid dimensions (rows, cols) for regional analysis.

    Returns:
        FocusScoreResult with all metrics and composite score.
    """
    # Convert to grayscale
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    # Compute all metrics
    lvar = compute_laplacian_variance(gray)
    teng = compute_tenengrad(gray)
    nvar = compute_normalized_variance(gray)
    fft_s = compute_fft_score(gray)

    # Normalize each metric to [0, 1] using empirical ranges
    # (determined from microscopy focus stack dataset)
    lvar_norm = min(lvar / 3000.0, 1.0)
    teng_norm = min(teng / 50000.0, 1.0)
    nvar_norm = min(nvar / 1.5, 1.0)
    fft_norm = min(fft_s / 0.15, 1.0)

    # Weighted composite
    composite = (
        0.35 * lvar_norm
        + 0.35 * teng_norm
        + 0.15 * nvar_norm
        + 0.15 * fft_norm
    )
    composite = float(np.clip(composite, 0.0, 1.0))

    # Quality label
    if composite >= 0.75:
        quality_label = "excellent"
    elif composite >= 0.55:
        quality_label = "good"
    elif composite >= 0.35:
        quality_label = "acceptable"
    elif composite >= 0.20:
        quality_label = "poor"
    else:
        quality_label = "unusable"

    # Optional regional analysis
    region_scores = None
    if compute_regional:
        region_scores = _compute_regional_scores(gray, region_grid)

    return FocusScoreResult(
        composite=composite,
        laplacian_variance=lvar_norm,
        tenengrad=teng_norm,
        normalized_variance=nvar_norm,
        fft_score=fft_norm,
        quality_label=quality_label,
        region_scores=region_scores,
    )


def _compute_regional_scores(
    gray: NDArray[np.uint8],
    grid: tuple[int, int],
) -> NDArray[np.float32]:
    """
    Compute focus scores on a spatial grid of image regions.

    Returns a 2D array where each cell is the composite focus score
    for that region. Useful for generating focus heatmaps.
    """
    rows, cols = grid
    h, w = gray.shape
    scores = np.zeros((rows, cols), dtype=np.float32)

    row_step = h // rows
    col_step = w // cols

    for r in range(rows):
        for c in range(cols):
            y1 = r * row_step
            y2 = (r + 1) * row_step if r < rows - 1 else h
            x1 = c * col_step
            x2 = (c + 1) * col_step if c < cols - 1 else w
            region = gray[y1:y2, x1:x2]
            if region.size > 0:
                lvar = compute_laplacian_variance(region)
                teng = compute_tenengrad(region)
                scores[r, c] = float(np.clip(
                    0.5 * min(lvar / 3000, 1) + 0.5 * min(teng / 50000, 1),
                    0, 1
                ))

    return scores


def generate_focus_heatmap(
    image: NDArray[np.uint8],
    grid: tuple[int, int] = (8, 8),
) -> NDArray[np.uint8]:
    """
    Generate a color-coded focus heatmap overlay.

    Red = blurry regions, Green = sharp regions.
    Useful for guiding microscope adjustment in live mode.

    Returns:
        Color heatmap as RGB image (H, W, 3)
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    region_scores = _compute_regional_scores(gray, grid)
    h, w = gray.shape

    # Resize score grid to full image size
    score_map = cv2.resize(
        (region_scores * 255).astype(np.uint8),
        (w, h),
        interpolation=cv2.INTER_CUBIC,
    )

    # Apply colormap: use RdYlGn if available (OpenCV ≥ 4.x), else JET
    _cmap = getattr(cv2, "COLORMAP_RdYlGn", cv2.COLORMAP_JET)
    heatmap_bgr = cv2.applyColorMap(score_map, _cmap)
    return cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)


def rank_frames_by_focus(
    frame_scores: list[tuple[int, FocusScoreResult]],
    min_score: float = 0.35,
) -> list[tuple[int, FocusScoreResult]]:
    """
    Rank video frames by focus quality.

    Args:
        frame_scores: List of (frame_index, FocusScoreResult)
        min_score: Minimum composite score to include in ranking

    Returns:
        Filtered and sorted list (best focus first).
    """
    filtered = [(idx, score) for idx, score in frame_scores if score.composite >= min_score]
    return sorted(filtered, key=lambda x: x[1].composite, reverse=True)
