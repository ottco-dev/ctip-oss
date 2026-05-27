"""
focus.metrics.tenengrad — Tenengrad and gradient-based focus metrics.

The Tenengrad metric measures image sharpness via gradient magnitude.
Sharp images have strong, well-defined edges → high gradient magnitudes.
Blurry images have diffuse edges → low gradient magnitudes.

Advantages over Laplacian-based metrics:
- Less sensitive to image noise (Sobel operator inherently smooths noise)
- More robust to specular reflections (common on trichome heads)
- Performs well on bright-field microscopy with variable illumination

References:
  Krotkov, E. (1988). Focusing. IJCV 1(3):223-237.
  Groen, F.C.A. et al. (1985). Cytometry 6(2):81-91.
  Santos, A. et al. (1997). J. Microscopy 188(3):264-272.
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray


def tenengrad(
    gray: NDArray[np.uint8],
    ksize: int = 3,
    threshold: float = 0.0,
) -> float:
    """
    Tenengrad focus measure (Krotkov 1988).

    Computes Sobel gradients in X and Y directions,
    then returns mean of squared gradient magnitudes.

    TENG = Σ(Gx² + Gy²) / N  for pixels where √(Gx²+Gy²) > threshold

    Args:
        gray: Grayscale uint8 image (H, W)
        ksize: Sobel kernel size (3 or 5; 3 is faster, 5 is more accurate)
        threshold: Minimum gradient magnitude to include. 0 = no filter.

    Returns:
        Mean squared gradient magnitude (higher = sharper)
    """
    f = gray.astype(np.float64)
    gx = cv2.Sobel(f, cv2.CV_64F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(f, cv2.CV_64F, 0, 1, ksize=ksize)
    g_sq = gx ** 2 + gy ** 2

    if threshold > 0:
        mask = np.sqrt(g_sq) > threshold
        values = g_sq[mask]
    else:
        values = g_sq.ravel()

    return float(values.mean()) if values.size > 0 else 0.0


def tenengrad_variance(gray: NDArray[np.uint8], ksize: int = 3) -> float:
    """
    Tenengrad Variance (TENV) — Santos et al. 1997.

    Combines gradient magnitude mean AND variance for a more
    discriminative focus measure. Particularly effective when
    comparing images with similar mean gradient (e.g., two slightly
    different focus positions).

    TENV = Σ(G(x,y) - μ_G)² / N
    where G(x,y) = Gx² + Gy² (gradient energy per pixel)

    Args:
        gray: Grayscale uint8 image (H, W)
        ksize: Sobel kernel size

    Returns:
        Variance of gradient energy distribution (higher = sharper)
    """
    f = gray.astype(np.float64)
    gx = cv2.Sobel(f, cv2.CV_64F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(f, cv2.CV_64F, 0, 1, ksize=ksize)
    g_energy = gx ** 2 + gy ** 2
    return float(g_energy.var())


def absolute_gradient_sum(gray: NDArray[np.uint8]) -> float:
    """
    Absolute Gradient Sum — Prewitt operator variant.

    Uses Prewitt operator (unweighted Sobel) and sums absolute values.
    Faster than Tenengrad (no squaring). Less sensitive to large edges
    but better for images with many fine structures.

    Args:
        gray: Grayscale uint8 image (H, W)

    Returns:
        Mean absolute gradient (higher = sharper)
    """
    f = gray.astype(np.float64)
    # Prewitt-like: simple finite differences
    gx = np.abs(np.diff(f, axis=1, append=f[:, -1:]))
    gy = np.abs(np.diff(f, axis=0, append=f[-1:, :]))
    return float((gx + gy).mean())


def steerable_filter_focus(gray: NDArray[np.uint8]) -> float:
    """
    Multi-orientation gradient energy focus metric.

    Computes gradient at 4 orientations (0°, 45°, 90°, 135°) and
    combines to detect edges at all orientations equally.
    More isotropic than standard Sobel (which favors H/V edges).

    Useful for trichome stalk analysis where structures are
    oriented at arbitrary angles.

    Args:
        gray: Grayscale uint8 image (H, W)

    Returns:
        Isotropic gradient energy (higher = sharper)
    """
    f = gray.astype(np.float64)

    # Compute kernels for 4 orientations
    kernels = [
        np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]),   # 0° (Sobel-X)
        np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]]),    # 45°
        np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]),    # 90° (Sobel-Y)
        np.array([[2, 1, 0], [1, 0, -1], [0, -1, -2]]),    # 135°
    ]

    energy = np.zeros_like(f)
    for k in kernels:
        response = cv2.filter2D(f, cv2.CV_64F, k.astype(np.float64))
        energy += response ** 2

    return float(energy.mean() / len(kernels))


def compute_gradient_map(
    gray: NDArray[np.uint8],
    ksize: int = 3,
) -> NDArray[np.float32]:
    """
    Compute per-pixel gradient magnitude map for visualization.

    Returns a float32 map normalized to [0, 1] where bright pixels
    indicate sharp edges (high focus contribution).

    Args:
        gray: Grayscale uint8 image
        ksize: Sobel kernel size

    Returns:
        Gradient magnitude map (H, W) float32 in [0, 1]
    """
    f = gray.astype(np.float64)
    gx = cv2.Sobel(f, cv2.CV_64F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(f, cv2.CV_64F, 0, 1, ksize=ksize)
    mag = np.sqrt(gx ** 2 + gy ** 2)

    max_val = mag.max()
    if max_val > 0:
        mag = mag / max_val

    return mag.astype(np.float32)
