"""
morphology.domain.density_map — Trichome spatial density mapping.

Generates kernel density estimates (KDE) and discrete grid density maps
to visualize trichome population distribution over a microscopy image.

APPLICATIONS:
- Identify high-density glandular regions (important for harvest decisions)
- Detect non-uniform sampling in training datasets
- Quality indicator: uniform vs. patchy trichome distribution
- Population-level analysis for scientific reporting

METHODS:
1. Gaussian KDE — smooth continuous density surface
2. Grid-based binning — fast discrete count per cell
3. Normalized density map — [0, 1] normalized for visualization

SCIENTIFIC NOTE:
Trichome density per unit area (trichomes/mm²) is a scientifically
reported metric in cannabis research (Potter, 2009; Tanney et al., 2021).
Requires calibrated pixel→µm conversion for absolute density values.

Reference:
  Potter, D.J. (2009). The propagation, characterisation and optimisation
  of Cannabis sativa L. as a phytopharmaceutical. PhD thesis, King's College London.

  Tanney, C.A.S. et al. (2021). Cannabis glandular trichomes: A cellular
  metabolite factory. Frontiers in Plant Science 12:721986.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
from numpy.typing import NDArray


@dataclass
class TrichomeCentroid:
    """Centroid location of a detected trichome instance."""
    x: float
    y: float
    trichome_type: str = "unknown"
    confidence: float = 1.0


@dataclass
class DensityMapResult:
    """Output of trichome density analysis."""

    density_grid: NDArray[np.float32]
    """
    Grid-based count map, shape (grid_rows, grid_cols).
    Each cell contains the number of trichome centroids in that cell.
    """

    kde_map: NDArray[np.float32]
    """
    Gaussian KDE density surface, shape (H, W) — same as image dimensions.
    Values are probability densities (not counts). Normalized to [0,1] for
    visualization. Requires scipy; falls back to density_grid if unavailable.
    """

    heatmap_bgr: NDArray[np.uint8]
    """
    False-color density heatmap overlay (BGR), shape (H, W, 3).
    Blue=low, Green=medium, Red=high density.
    """

    total_count: int
    """Total number of trichomes contributing to the map."""

    peak_density_cell: Tuple[int, int]
    """(row, col) of the highest-density grid cell."""

    uniformity_index: float
    """
    Coefficient of variation (std/mean) of grid cell counts.
    0 = perfectly uniform, higher = more clustered.
    Useful for quality assessment of coverage.
    """

    density_per_mm2: Optional[float]
    """
    Absolute trichome density in trichomes/mm².
    Only populated if um_per_pixel is provided.
    """

    image_shape: Tuple[int, int]
    """(H, W) of the source image."""


def compute_density_map(
    centroids: List[TrichomeCentroid],
    image_height: int,
    image_width: int,
    *,
    grid_rows: int = 8,
    grid_cols: int = 8,
    kde_bandwidth: float = 30.0,
    um_per_pixel: Optional[float] = None,
) -> DensityMapResult:
    """
    Compute trichome density maps from centroid positions.

    Args:
        centroids:      List of trichome centroid positions.
        image_height:   Image height in pixels.
        image_width:    Image width in pixels.
        grid_rows:      Number of rows in the discrete density grid.
        grid_cols:      Number of columns in the discrete density grid.
        kde_bandwidth:  Gaussian kernel bandwidth in pixels for KDE.
        um_per_pixel:   Calibration factor for converting to physical density.

    Returns:
        DensityMapResult with all density representations.
    """
    if image_height <= 0 or image_width <= 0:
        raise ValueError(f"Invalid image dimensions: {image_height}×{image_width}")

    if not centroids:
        return _empty_result(image_height, image_width, grid_rows, grid_cols)

    xs = np.array([c.x for c in centroids], dtype=np.float32)
    ys = np.array([c.y for c in centroids], dtype=np.float32)

    # --- Grid-based count ---
    grid = np.zeros((grid_rows, grid_cols), dtype=np.float32)
    cell_h = image_height / grid_rows
    cell_w = image_width / grid_cols

    for x, y in zip(xs, ys):
        row = int(np.clip(y / cell_h, 0, grid_rows - 1))
        col = int(np.clip(x / cell_w, 0, grid_cols - 1))
        grid[row, col] += 1.0

    # Peak density cell
    peak_idx = np.unravel_index(np.argmax(grid), grid.shape)
    peak_cell = (int(peak_idx[0]), int(peak_idx[1]))

    # Uniformity index (coefficient of variation)
    nonzero_cells = grid[grid > 0]
    if len(nonzero_cells) > 1:
        uniformity = float(np.std(nonzero_cells) / (np.mean(nonzero_cells) + 1e-8))
    else:
        uniformity = 0.0

    # --- Gaussian KDE ---
    kde_map = _compute_kde(xs, ys, image_height, image_width, bandwidth=kde_bandwidth)

    # --- Heatmap visualization ---
    heatmap = _kde_to_heatmap(kde_map, image_height, image_width)

    # --- Physical density ---
    density_mm2: Optional[float] = None
    if um_per_pixel is not None and um_per_pixel > 0:
        image_area_um2 = (image_height * um_per_pixel) * (image_width * um_per_pixel)
        image_area_mm2 = image_area_um2 / (1000.0 ** 2)
        if image_area_mm2 > 0:
            density_mm2 = len(centroids) / image_area_mm2

    return DensityMapResult(
        density_grid=grid,
        kde_map=kde_map,
        heatmap_bgr=heatmap,
        total_count=len(centroids),
        peak_density_cell=peak_cell,
        uniformity_index=uniformity,
        density_per_mm2=density_mm2,
        image_shape=(image_height, image_width),
    )


def _compute_kde(
    xs: NDArray[np.float32],
    ys: NDArray[np.float32],
    height: int,
    width: int,
    bandwidth: float,
) -> NDArray[np.float32]:
    """Compute Gaussian KDE density map."""
    try:
        from scipy.stats import gaussian_kde

        points = np.vstack([xs / width, ys / height])  # normalize to [0,1]
        kde = gaussian_kde(points, bw_method=bandwidth / max(height, width))

        # Evaluate on a coarse grid then resize (memory efficient)
        grid_h, grid_w = min(64, height), min(64, width)
        yg, xg = np.mgrid[0:1:complex(0, grid_h), 0:1:complex(0, grid_w)]
        grid_coords = np.vstack([xg.ravel(), yg.ravel()])
        density_coarse = kde(grid_coords).reshape(grid_h, grid_w).astype(np.float32)

        # Resize to full image
        kde_full = cv2.resize(density_coarse, (width, height), interpolation=cv2.INTER_LINEAR)
    except Exception:
        # Fallback: Gaussian splat at each centroid
        kde_full = np.zeros((height, width), dtype=np.float32)
        for x, y in zip(xs, ys):
            cx, cy = int(np.clip(x, 0, width - 1)), int(np.clip(y, 0, height - 1))
            kde_full[cy, cx] += 1.0
        sigma = int(max(bandwidth, 5))
        kde_full = cv2.GaussianBlur(kde_full, (0, 0), sigma).astype(np.float32)

    # Normalize to [0, 1]
    max_val = float(kde_full.max())
    if max_val > 0:
        kde_full /= max_val
    return kde_full


def _kde_to_heatmap(
    kde_map: NDArray[np.float32],
    height: int,
    width: int,
) -> NDArray[np.uint8]:
    """Convert KDE map to false-color BGR heatmap."""
    uint8_map = (kde_map * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(uint8_map, cv2.COLORMAP_JET)
    return heatmap  # BGR


def _empty_result(
    h: int, w: int, grid_rows: int, grid_cols: int
) -> DensityMapResult:
    return DensityMapResult(
        density_grid=np.zeros((grid_rows, grid_cols), dtype=np.float32),
        kde_map=np.zeros((h, w), dtype=np.float32),
        heatmap_bgr=np.zeros((h, w, 3), dtype=np.uint8),
        total_count=0,
        peak_density_cell=(0, 0),
        uniformity_index=0.0,
        density_per_mm2=None,
        image_shape=(h, w),
    )


def overlay_density_on_image(
    image_bgr: NDArray[np.uint8],
    density_result: DensityMapResult,
    alpha: float = 0.4,
) -> NDArray[np.uint8]:
    """
    Overlay the density heatmap on top of the source image.

    Args:
        image_bgr:       Source BGR image.
        density_result:  Result from compute_density_map.
        alpha:           Heatmap opacity [0, 1].

    Returns:
        Blended BGR image with density overlay.
    """
    if image_bgr.shape[:2] != density_result.image_shape:
        heatmap = cv2.resize(
            density_result.heatmap_bgr,
            (image_bgr.shape[1], image_bgr.shape[0]),
        )
    else:
        heatmap = density_result.heatmap_bgr

    return cv2.addWeighted(image_bgr, 1.0 - alpha, heatmap, alpha, 0)
