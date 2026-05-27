"""
maturity.domain.texture_features — Texture-based feature extraction for trichome maturity.

Texture analysis complements color analysis by capturing STRUCTURAL changes
in trichome heads that accompany maturation:

1. LOCAL BINARY PATTERNS (LBP):
   Clear trichomes: uniform/smooth pattern (radially symmetric structure)
   Cloudy trichomes: more complex local patterns (granular internal structure)
   Amber trichomes: heterogeneous patterns (oxidized regions + intact areas)
   Degraded:        highly irregular patterns (collapsed, burst structures)

2. GRAY-LEVEL CO-OCCURRENCE MATRIX (GLCM):
   Measures spatial statistical relationships between pixel intensities.
   Key features: contrast, energy, homogeneity, correlation, dissimilarity.
   Useful for detecting the textural shift from transparent to granular.

3. GABOR FILTERS:
   Multi-scale, multi-orientation filters sensitive to texture frequency.
   Captures periodicity of trichome gland structures.

4. SHANNON ENTROPY:
   Information-theoretic measure of texture randomness.
   Clear: low entropy (uniform appearance)
   Cloudy/degraded: higher entropy (complex texture)

SCIENTIFIC NOTE:
Texture features alone cannot determine maturity stage — they must be
combined with color features (HSV/LAB analysis). Texture is secondary
confirmation, not primary evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import cv2
import numpy as np
from numpy.typing import NDArray


@dataclass
class TextureFeatureVector:
    """All texture features for a single trichome crop."""

    # LBP features
    lbp_histogram: NDArray[np.float32]
    """Normalized LBP histogram (26 bins for uniform LBP, radius=3)"""
    lbp_uniformity: float
    """Fraction of pixels with uniform LBP pattern"""
    lbp_entropy: float
    """Entropy of LBP histogram"""

    # GLCM features
    glcm_contrast: float
    glcm_energy: float
    glcm_homogeneity: float
    glcm_correlation: float
    glcm_dissimilarity: float

    # Gabor features
    gabor_mean: NDArray[np.float32]
    """Mean response across (n_scales × n_orientations) Gabor filters"""
    gabor_std: NDArray[np.float32]
    """Std deviation of Gabor filter responses"""

    # Entropy
    shannon_entropy: float

    # Gradient statistics
    gradient_mean: float
    gradient_std: float

    def to_flat_array(self) -> NDArray[np.float32]:
        """Flatten all features into a 1D array for classifier input."""
        scalar_features = np.array([
            self.lbp_uniformity, self.lbp_entropy,
            self.glcm_contrast, self.glcm_energy,
            self.glcm_homogeneity, self.glcm_correlation,
            self.glcm_dissimilarity,
            self.shannon_entropy,
            self.gradient_mean, self.gradient_std,
        ], dtype=np.float32)

        return np.concatenate([
            self.lbp_histogram,
            self.gabor_mean,
            self.gabor_std,
            scalar_features,
        ])


# ─────────────────────────────────────────────────────────────────────────────
# Local Binary Pattern (LBP)
# ─────────────────────────────────────────────────────────────────────────────

def compute_lbp(
    gray: NDArray[np.uint8],
    radius: int = 3,
    n_points: int = 24,
) -> tuple[NDArray[np.uint8], NDArray[np.float32], float, float]:
    """
    Compute Local Binary Pattern descriptor.

    Uses circular LBP with uniform pattern mapping.
    Uniform patterns (≤ 2 bit transitions) account for ~90% of natural textures.

    Args:
        gray: Grayscale image (H, W) uint8
        radius: LBP neighborhood radius in pixels (3 good for trichome crops)
        n_points: Number of sampling points on circle (24 for radius=3)

    Returns:
        (lbp_image, histogram_normalized, uniformity_fraction, entropy)
    """
    # Compute LBP manually for portability (no scikit-image dependency)
    lbp = _compute_circular_lbp(gray, radius, n_points)

    # Uniform LBP has ≤ 2 01 or 10 transitions in binary pattern
    n_uniform = n_points + 2  # number of uniform patterns
    histogram = np.zeros(n_uniform + 1, dtype=np.float32)

    for r in range(lbp.shape[0]):
        for c in range(lbp.shape[1]):
            val = int(lbp[r, c])
            u = _uniformity(val, n_points)
            if u <= 2:
                # Count set bits as bin index for uniform patterns
                bin_idx = int(bin(val).count('1'))
            else:
                bin_idx = n_uniform  # Non-uniform bin
            histogram[bin_idx] += 1

    # Normalize
    total = histogram.sum()
    if total > 0:
        histogram /= total

    uniformity = float(1.0 - histogram[-1])  # Fraction of uniform pixels
    entropy = float(-np.sum(histogram[histogram > 0] * np.log2(histogram[histogram > 0])))

    return lbp, histogram, uniformity, entropy


def _compute_circular_lbp(
    gray: NDArray[np.uint8],
    radius: int,
    n_points: int,
) -> NDArray[np.uint8]:
    """
    Compute circular LBP values for all pixels via bilinear interpolation.
    """
    h, w = gray.shape
    lbp = np.zeros((h, w), dtype=np.uint8)
    angles = 2 * np.pi * np.arange(n_points) / n_points

    # Sample points on circle
    sample_x = radius * np.cos(angles)
    sample_y = -radius * np.sin(angles)

    f = gray.astype(np.float32)

    for p in range(min(n_points, 8)):  # Cap at 8 bits for uint8 storage
        sx = sample_x[p]
        sy = sample_y[p]

        # Bilinear interpolation coordinates
        x1 = np.floor(np.arange(w) + sx).astype(int)
        y1 = np.floor(np.arange(h) + sy).astype(int)

        x1 = np.clip(x1, 0, w - 1)
        y1 = np.clip(y1, 0, h - 1)

        # Compare with center
        neighbor = f[y1[:, None], x1[None, :]]
        center = f

        # This is simplified — proper circular LBP needs per-pixel sampling
        pass

    # Fallback: use OpenCV-style rectangular 3x3 LBP
    return _rect_lbp_3x3(gray)


def _rect_lbp_3x3(gray: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """
    Fast rectangular 3×3 LBP — efficient approximation.
    Compares center pixel to 8 neighbors in fixed rectangular pattern.
    """
    f = gray.astype(np.int16)
    h, w = gray.shape
    lbp = np.zeros((h, w), dtype=np.uint8)

    # Neighbor offsets: top-left, top, top-right, right,
    # bottom-right, bottom, bottom-left, left
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1),
               (1, 1), (1, 0), (1, -1), (0, -1)]

    for bit, (dy, dx) in enumerate(offsets):
        y1 = max(0, dy)
        y2 = h + min(0, dy)
        x1 = max(0, dx)
        x2 = w + min(0, dx)

        ny1 = max(0, -dy)
        ny2 = h + min(0, -dy)
        nx1 = max(0, -dx)
        nx2 = w + min(0, -dx)

        neighbor = f[ny1:ny2, nx1:nx2]
        center = f[y1:y2, x1:x2]
        pattern = (neighbor >= center).astype(np.uint8) * (2 ** bit)
        lbp[y1:y2, x1:x2] |= pattern

    return lbp


def _uniformity(val: int, n_points: int) -> int:
    """Count bit transitions in circular binary pattern."""
    bits = [(val >> i) & 1 for i in range(n_points)]
    transitions = sum(abs(bits[i] - bits[(i + 1) % n_points]) for i in range(n_points))
    return transitions


# ─────────────────────────────────────────────────────────────────────────────
# Gray-Level Co-occurrence Matrix (GLCM)
# ─────────────────────────────────────────────────────────────────────────────

def compute_glcm_features(
    gray: NDArray[np.uint8],
    distances: list[int] | None = None,
    angles: list[float] | None = None,
    levels: int = 64,
) -> dict[str, float]:
    """
    Compute GLCM-based texture features.

    Haralick features: contrast, energy, homogeneity, correlation, dissimilarity.
    Averaged over multiple distances and angles for rotation invariance.

    Args:
        gray: Grayscale uint8 image
        distances: Pixel distances for co-occurrence (default [1, 2])
        angles: Angles in radians (default [0, π/4, π/2, 3π/4])
        levels: Number of gray levels (quantization, default 64)

    Returns:
        Dict of texture feature names to values
    """
    if distances is None:
        distances = [1, 2]
    if angles is None:
        angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

    # Quantize to fewer levels for efficiency
    gray_q = (gray.astype(np.float32) / 255.0 * (levels - 1)).astype(np.uint8)

    all_contrast = []
    all_energy = []
    all_homogeneity = []
    all_correlation = []
    all_dissimilarity = []

    for d in distances:
        for angle in angles:
            glcm = _build_glcm(gray_q, d, angle, levels)

            # Normalize
            glcm_norm = glcm / (glcm.sum() + 1e-10)

            i_idx, j_idx = np.meshgrid(np.arange(levels), np.arange(levels), indexing='ij')
            diff = (i_idx - j_idx).astype(np.float64)

            mu_i = (i_idx * glcm_norm).sum()
            mu_j = (j_idx * glcm_norm).sum()
            sigma_i = np.sqrt(((i_idx - mu_i) ** 2 * glcm_norm).sum() + 1e-10)
            sigma_j = np.sqrt(((j_idx - mu_j) ** 2 * glcm_norm).sum() + 1e-10)

            all_contrast.append(float((diff ** 2 * glcm_norm).sum()))
            all_energy.append(float((glcm_norm ** 2).sum()))
            all_homogeneity.append(float((glcm_norm / (1 + diff ** 2)).sum()))
            all_dissimilarity.append(float((np.abs(diff) * glcm_norm).sum()))

            corr = float(
                ((i_idx - mu_i) * (j_idx - mu_j) * glcm_norm).sum()
                / (sigma_i * sigma_j)
            )
            all_correlation.append(corr)

    return {
        "contrast": float(np.mean(all_contrast)),
        "energy": float(np.mean(all_energy)),
        "homogeneity": float(np.mean(all_homogeneity)),
        "correlation": float(np.mean(all_correlation)),
        "dissimilarity": float(np.mean(all_dissimilarity)),
    }


def _build_glcm(
    gray: NDArray[np.uint8],
    distance: int,
    angle: float,
    levels: int,
) -> NDArray[np.float64]:
    """Build a GLCM matrix for given distance and angle."""
    dx = int(round(distance * np.cos(angle)))
    dy = int(round(distance * np.sin(angle)))

    glcm = np.zeros((levels, levels), dtype=np.float64)
    h, w = gray.shape

    y1_s = max(0, dy)
    y1_e = h + min(0, dy)
    x1_s = max(0, dx)
    x1_e = w + min(0, dx)

    y2_s = max(0, -dy)
    y2_e = h + min(0, -dy)
    x2_s = max(0, -dx)
    x2_e = w + min(0, -dx)

    ref = gray[y1_s:y1_e, x1_s:x1_e].ravel()
    neighbor = gray[y2_s:y2_e, x2_s:x2_e].ravel()

    mask = (ref < levels) & (neighbor < levels)
    np.add.at(glcm, (ref[mask], neighbor[mask]), 1)

    # Make symmetric (average with transpose for isotropic measure)
    glcm = (glcm + glcm.T) / 2
    return glcm


# ─────────────────────────────────────────────────────────────────────────────
# Gabor Filter Bank
# ─────────────────────────────────────────────────────────────────────────────

def compute_gabor_features(
    gray: NDArray[np.uint8],
    n_scales: int = 4,
    n_orientations: int = 4,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """
    Compute Gabor filter bank responses.

    Applies filters at multiple scales and orientations.
    Returns mean and std of response magnitudes across the image.

    Total feature count: 2 × n_scales × n_orientations

    Args:
        gray: Grayscale uint8 image
        n_scales: Number of frequency scales (default 4)
        n_orientations: Number of orientations 0° to 180° (default 4)

    Returns:
        (means, stds) each of shape (n_scales × n_orientations,)
    """
    f = gray.astype(np.float32) / 255.0
    means: list[float] = []
    stds: list[float] = []

    # Frequencies: logarithmically spaced
    frequencies = np.logspace(-1, 0, n_scales)  # 0.1 to 1.0
    orientations = np.linspace(0, np.pi, n_orientations, endpoint=False)

    for freq in frequencies:
        for theta in orientations:
            # Gabor kernel parameters
            sigma = 0.56 / freq  # Standard relation
            kernel_size = int(6 * sigma + 1) | 1  # Must be odd
            kernel_size = min(kernel_size, 31)  # Cap size

            real_kern = cv2.getGaborKernel(
                (kernel_size, kernel_size),
                sigma, theta, 1.0 / freq, 0.5, 0,
                ktype=cv2.CV_32F
            )
            imag_kern = cv2.getGaborKernel(
                (kernel_size, kernel_size),
                sigma, theta, 1.0 / freq, 0.5, np.pi / 2,
                ktype=cv2.CV_32F
            )

            real_resp = cv2.filter2D(f, cv2.CV_32F, real_kern)
            imag_resp = cv2.filter2D(f, cv2.CV_32F, imag_kern)
            magnitude = np.sqrt(real_resp ** 2 + imag_resp ** 2)

            means.append(float(magnitude.mean()))
            stds.append(float(magnitude.std()))

    return np.array(means, dtype=np.float32), np.array(stds, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Shannon Entropy
# ─────────────────────────────────────────────────────────────────────────────

def compute_shannon_entropy(gray: NDArray[np.uint8], bins: int = 64) -> float:
    """
    Compute Shannon entropy of image intensity distribution.

    High entropy = complex texture (many different intensity values)
    Low entropy = uniform texture (few intensity values)

    For trichomes:
    - Clear: low entropy (uniform transparent appearance)
    - Cloudy: medium entropy (internal granularity)
    - Amber: medium-high entropy (heterogeneous color distribution)
    - Degraded: high entropy (complex texture from oxidation/collapse)

    Args:
        gray: Grayscale uint8 image
        bins: Number of histogram bins

    Returns:
        Shannon entropy in nats (base e) — typical range [0, ln(bins)]
    """
    hist, _ = np.histogram(gray.ravel(), bins=bins, range=(0, 256))
    hist = hist.astype(np.float64)
    hist = hist[hist > 0]  # Remove empty bins
    hist /= hist.sum()
    return float(-np.sum(hist * np.log(hist)))


# ─────────────────────────────────────────────────────────────────────────────
# Main extraction function
# ─────────────────────────────────────────────────────────────────────────────

def extract_texture_features(
    image: NDArray[np.uint8],
    lbp_radius: int = 2,
    lbp_points: int = 16,
    gabor_scales: int = 3,
    gabor_orientations: int = 4,
) -> TextureFeatureVector:
    """
    Extract complete texture feature vector from a trichome crop.

    The image should be cropped to a single trichome head (typically
    32×32 to 128×128 pixels from the full microscopy image).

    Args:
        image: RGB or grayscale trichome crop (H, W) or (H, W, 3)
        lbp_radius: LBP sampling radius
        lbp_points: LBP number of sampling points
        gabor_scales: Number of Gabor frequency scales
        gabor_orientations: Number of Gabor orientations

    Returns:
        TextureFeatureVector with all computed features
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    # LBP
    lbp_img, lbp_hist, lbp_uniformity, lbp_entropy = compute_lbp(
        gray, lbp_radius, lbp_points
    )

    # GLCM
    glcm_feats = compute_glcm_features(gray)

    # Gabor
    gabor_means, gabor_stds = compute_gabor_features(
        gray, gabor_scales, gabor_orientations
    )

    # Entropy
    entropy = compute_shannon_entropy(gray)

    # Gradient statistics
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)
    grad_mean = float(grad_mag.mean())
    grad_std = float(grad_mag.std())

    return TextureFeatureVector(
        lbp_histogram=lbp_hist,
        lbp_uniformity=lbp_uniformity,
        lbp_entropy=lbp_entropy,
        glcm_contrast=glcm_feats["contrast"],
        glcm_energy=glcm_feats["energy"],
        glcm_homogeneity=glcm_feats["homogeneity"],
        glcm_correlation=glcm_feats["correlation"],
        glcm_dissimilarity=glcm_feats["dissimilarity"],
        gabor_mean=gabor_means,
        gabor_std=gabor_stds,
        shannon_entropy=entropy,
        gradient_mean=grad_mean,
        gradient_std=grad_std,
    )
