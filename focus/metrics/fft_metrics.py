"""
focus.metrics.fft_metrics — Frequency-domain focus metrics.

Frequency domain analysis reveals image sharpness through the
distribution of spatial frequencies. Sharp images contain significant
high-frequency content (fine edges, textures). Blurry images are
low-pass filtered — high frequencies are attenuated.

METRICS IMPLEMENTED:

1. FFT High-Frequency Energy Ratio
   Ratio of energy in the high-frequency band to total energy.
   Robust, globally captures sharpness.

2. DCT High-Frequency Score
   Same idea using Discrete Cosine Transform.
   Slightly faster due to real-valued output.

3. Power Spectral Slope
   Slope of log power vs log frequency. Sharp images have shallower
   slope (less roll-off toward high frequencies).

4. Brenner's Focus Measure
   Classic measure based on inter-pixel differences.
   Fast approximation to frequency content without full FFT.

References:
  Vollath, D. (1987). Optik 75(2):78-80.
  Brenner, J.F. et al. (1976). J. Histochem. Cytochem. 24:100-111.
  Huang, W. & Bhanu, B. (2007). ICCV Workshop on Focus.
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray


def fft_high_frequency_ratio(
    gray: NDArray[np.uint8],
    high_freq_fraction: float = 0.15,
) -> float:
    """
    Ratio of high-frequency energy to total energy via 2D FFT.

    Computes 2D FFT, shifts zero-frequency to center, then measures
    energy in the outer ring (high-frequency region) vs total.

    high_freq_fraction = fraction of frequency spectrum considered "high".
    0.15 means the outer 15% of the radial frequency range.

    Higher values → sharper image.
    Typical range for in-focus microscopy: 0.05 - 0.25.

    Args:
        gray: Grayscale uint8 image (H, W)
        high_freq_fraction: Radial fraction defining high-frequency band

    Returns:
        HF energy fraction in [0, 1]
    """
    f = gray.astype(np.float64)
    fft = np.fft.fft2(f)
    fft_shifted = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shifted) ** 2

    h, w = gray.shape
    cy, cx = h // 2, w // 2
    y_idx, x_idx = np.ogrid[:h, :w]
    dist = np.sqrt((x_idx - cx) ** 2 + (y_idx - cy) ** 2)
    max_dist = np.sqrt(cx ** 2 + cy ** 2)

    hf_mask = dist > (high_freq_fraction * max_dist)
    total = magnitude.sum()

    if total == 0:
        return 0.0

    return float(magnitude[hf_mask].sum() / total)


def dct_high_frequency_score(
    gray: NDArray[np.uint8],
    high_coeff_fraction: float = 0.15,
) -> float:
    """
    DCT-based focus score using high-frequency coefficient energy.

    Applies 2D Discrete Cosine Transform and measures energy in
    high-frequency DCT coefficients (upper-right quadrant).

    Slightly faster than FFT for same image size (real-valued output).

    Args:
        gray: Grayscale uint8 image (H, W)
        high_coeff_fraction: Fraction of DCT coefficients considered "high"

    Returns:
        HF DCT energy fraction in [0, 1]
    """
    f = gray.astype(np.float32)
    dct = cv2.dct(f)  # OpenCV DCT (2D)
    energy = dct ** 2

    h, w = dct.shape
    h_thresh = int(h * (1.0 - high_coeff_fraction))
    w_thresh = int(w * (1.0 - high_coeff_fraction))

    total_energy = energy.sum()
    if total_energy == 0:
        return 0.0

    # High frequency = large coefficient indices (bottom-right of DCT)
    hf_energy = energy[h_thresh:, :].sum() + energy[:, w_thresh:].sum()
    hf_energy -= energy[h_thresh:, w_thresh:].sum()  # avoid double-counting

    return float(np.clip(hf_energy / total_energy, 0.0, 1.0))


def power_spectral_slope(
    gray: NDArray[np.uint8],
    n_bins: int = 20,
) -> float:
    """
    Power spectral slope — gradient of log power vs log frequency.

    Sharp images have shallower (less negative) slope because high
    frequencies are preserved. Blurry images have steep negative slope
    due to low-pass filter attenuation.

    Returns a value where:
    - Near 0: very sharp (flat spectrum)
    - Very negative (< -4): very blurry

    Normalized to [0, 1] for use in composite scores:
    - 1.0 = sharpest possible (slope ≈ 0)
    - 0.0 = most blurry (slope ≈ -6 or steeper)

    Args:
        gray: Grayscale uint8 image (H, W)
        n_bins: Number of radial frequency bins for averaging

    Returns:
        Normalized sharpness from slope in [0, 1]
    """
    f = gray.astype(np.float64)
    fft = np.fft.fft2(f)
    fft_shifted = np.fft.fftshift(fft)
    power = np.abs(fft_shifted) ** 2 + 1e-10  # avoid log(0)

    h, w = gray.shape
    cy, cx = h // 2, w // 2
    y_idx, x_idx = np.ogrid[:h, :w]
    dist = np.sqrt((x_idx - cx) ** 2 + (y_idx - cy) ** 2)
    max_dist = np.sqrt(cx ** 2 + cy ** 2)

    # Radial bins
    bin_edges = np.linspace(1e-6, 1.0, n_bins + 1)
    log_freq = []
    log_power = []

    for i in range(n_bins):
        r_min = bin_edges[i] * max_dist
        r_max = bin_edges[i + 1] * max_dist
        mask = (dist >= r_min) & (dist < r_max)
        if mask.sum() == 0:
            continue
        avg_power = power[mask].mean()
        log_freq.append(np.log10(bin_edges[i] * max_dist + 1e-6))
        log_power.append(np.log10(avg_power))

    if len(log_freq) < 4:
        return 0.5

    # Linear regression to find slope
    coeffs = np.polyfit(log_freq, log_power, 1)
    slope = coeffs[0]

    # Normalize: slope of 0 = 1.0 (very sharp), slope of -6 = 0.0 (very blurry)
    normalized = float(np.clip((slope + 6.0) / 6.0, 0.0, 1.0))
    return normalized


def brenner_focus(gray: NDArray[np.uint8], step: int = 2) -> float:
    """
    Brenner's focus measure (Brenner et al. 1976).

    Classic, fast focus metric based on squared inter-pixel differences.
    Approximates high-frequency content without FFT.

    B = Σ(f(x+step, y) - f(x, y))²

    Simple but effective. Widely used in automated microscopy.

    Args:
        gray: Grayscale uint8 image (H, W)
        step: Pixel step for difference computation (typically 2)

    Returns:
        Brenner focus measure (higher = sharper, unbounded)
    """
    f = gray.astype(np.float64)
    diff = f[:, step:] - f[:, :-step]
    return float((diff ** 2).mean())


def vollath_f4(gray: NDArray[np.uint8]) -> float:
    """
    Vollath's F4 correlation focus measure (Vollath 1987).

    Based on spatial correlation between adjacent pixels.
    Robust to noise and works well for periodic textures
    (e.g., repeated trichome structures in dense fields).

    F4 = Σf(x,y)·f(x+1,y) - Σf(x,y)·f(x+2,y)

    Args:
        gray: Grayscale uint8 image (H, W)

    Returns:
        Vollath F4 score (higher = sharper)
    """
    f = gray.astype(np.float64)
    # A = Σ f(x,y) * f(x+1,y)
    term1 = (f[:, :-1] * f[:, 1:]).mean()
    # B = Σ f(x,y) * f(x+2,y)
    term2 = (f[:, :-2] * f[:, 2:]).mean()

    return float(term1 - term2)


def compute_frequency_profile(
    gray: NDArray[np.uint8],
    n_bins: int = 32,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute radially-averaged power spectrum profile.

    Returns (frequencies, power_values) for plotting focus profile
    or diagnosing blur characteristics.

    Args:
        gray: Grayscale uint8 image
        n_bins: Number of frequency bins

    Returns:
        Tuple of (normalized_frequencies, average_power_per_bin)
        frequencies: 0 to 1 (normalized spatial frequency)
        power: log10-normalized power in each bin
    """
    f = gray.astype(np.float64)
    fft = np.fft.fft2(f)
    fft_shifted = np.fft.fftshift(fft)
    power = np.abs(fft_shifted) ** 2

    h, w = gray.shape
    cy, cx = h // 2, w // 2
    y_idx, x_idx = np.ogrid[:h, :w]
    dist = np.sqrt((x_idx - cx) ** 2 + (y_idx - cy) ** 2)
    max_dist = np.sqrt(cx ** 2 + cy ** 2)
    norm_dist = dist / max_dist

    bin_edges = np.linspace(0, 1.0, n_bins + 1)
    freq_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    avg_power = np.zeros(n_bins)

    for i in range(n_bins):
        mask = (norm_dist >= bin_edges[i]) & (norm_dist < bin_edges[i + 1])
        if mask.sum() > 0:
            avg_power[i] = power[mask].mean()

    # Log normalize
    avg_power = np.log10(avg_power + 1)
    if avg_power.max() > 0:
        avg_power = avg_power / avg_power.max()

    return freq_centers, avg_power
