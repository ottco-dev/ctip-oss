"""
focus.metrics.laplacian — Laplacian-based focus metrics.

The Laplacian operator computes the second spatial derivative of image
intensity. Sharp images contain edges with rapid intensity changes →
high second-derivative magnitude → high Laplacian variance.

Blurry images have smooth gradients → low Laplacian values.

VARIANTS IMPLEMENTED:
1. Laplacian Variance (LVAR) — standard metric, fast
2. Modified Laplacian (MLAP) — sum of absolute diagonal Laplacian values
3. Squared Laplacian Gradient (SLG) — more sensitive to fine structures

References:
  Pertuz, S. et al. (2013). Pattern Recognition 46(5):1415-1432.
  Nayar, S.K. & Nakagawa, Y. (1994). PAMI 16(8):824-831.
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray


def laplacian_variance(gray: NDArray[np.uint8]) -> float:
    """
    Standard Laplacian variance focus metric.

    Fastest and most commonly used focus measure.
    Applies a 3×3 Laplacian kernel and returns output variance.

    Typical microscopy ranges:
    - Unusable (<0.3ms blur):  lvar < 50
    - Poor:                    lvar 50-200
    - Acceptable:              lvar 200-800
    - Good:                    lvar 800-2500
    - Excellent (in-focus):    lvar > 2500

    Args:
        gray: Grayscale uint8 image (H, W)

    Returns:
        Laplacian variance as float (higher = sharper)
    """
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def modified_laplacian(gray: NDArray[np.uint8]) -> float:
    """
    Modified Laplacian (Nayar & Nakagawa 1994).

    More robust than standard Laplacian for images with diagonal edges.
    Uses horizontal + vertical Laplacian kernels and sums absolute values.

    ML = Σ|M_x(x,y)| + Σ|M_y(x,y)|

    where M_x = [−1, 2, −1] horizontally,
          M_y = [−1, 2, −1] vertically.

    Args:
        gray: Grayscale uint8 image (H, W)

    Returns:
        Modified Laplacian sum (unnormalized, higher = sharper)
    """
    f = gray.astype(np.float64)

    # Horizontal: M_x = conv with [-1, 2, -1]
    kernel_x = np.array([[0, 0, 0], [-1, 2, -1], [0, 0, 0]], dtype=np.float64)
    # Vertical: M_y = conv with [-1, 2, -1]^T
    kernel_y = np.array([[0, -1, 0], [0, 2, 0], [0, -1, 0]], dtype=np.float64)

    lx = cv2.filter2D(f, cv2.CV_64F, kernel_x)
    ly = cv2.filter2D(f, cv2.CV_64F, kernel_y)

    ml = np.abs(lx) + np.abs(ly)
    return float(ml.mean())


def squared_laplacian_gradient(gray: NDArray[np.uint8]) -> float:
    """
    Squared Laplacian Gradient (SLG).

    Similar to Laplacian variance but computes squared values before summing.
    More sensitive to fine structural detail — useful for detecting subtle
    focus differences in through-focus stacks.

    SLG = Σ(Lap(x,y))² / (H × W)

    Args:
        gray: Grayscale uint8 image (H, W)

    Returns:
        Mean squared Laplacian response (higher = sharper)
    """
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float((lap ** 2).mean())


def laplacian_energy_of_gradient(gray: NDArray[np.uint8]) -> float:
    """
    Energy of Laplacian (EOL).

    Applies Gaussian smoothing first to reduce noise sensitivity,
    then measures Laplacian energy.
    Suitable for noisy microscopy images (camera noise at high ISO/gain).

    Args:
        gray: Grayscale uint8 image (H, W)

    Returns:
        Energy of Laplacian (higher = sharper)
    """
    # Gaussian pre-smoothing to reduce noise
    smoothed = cv2.GaussianBlur(gray, (3, 3), 0)
    lap = cv2.Laplacian(smoothed.astype(np.float64), cv2.CV_64F)
    return float((lap ** 2).mean())


def regional_laplacian_variance(
    gray: NDArray[np.uint8],
    tile_size: int = 64,
    percentile: float = 75.0,
) -> float:
    """
    Regional Laplacian variance — robust to partially blurry images.

    Divides image into tiles, computes lvar per tile, returns high percentile.
    Useful when only part of the image is in focus (depth of field effects).

    Args:
        gray: Grayscale uint8 image (H, W)
        tile_size: Tile dimension in pixels (64 recommended for trichomes)
        percentile: Use this percentile of tile scores (default: 75th)

    Returns:
        Robust focus score — not affected by blurry margins.
    """
    h, w = gray.shape
    tile_scores: list[float] = []

    for y in range(0, h - tile_size + 1, tile_size):
        for x in range(0, w - tile_size + 1, tile_size):
            tile = gray[y:y + tile_size, x:x + tile_size]
            tile_scores.append(laplacian_variance(tile))

    if not tile_scores:
        return laplacian_variance(gray)

    return float(np.percentile(tile_scores, percentile))
