"""
focus.guidance.heatmap — Focus heatmap generation for microscopy guidance.

Generates spatial focus quality maps to guide:
1. Microscope operator positioning (which areas to focus on)
2. Analysis pipeline (which regions to trust for detection/classification)
3. Video frame quality assessment (spatial focus distribution)

HEATMAP TYPES:
- Composite focus heatmap: per-region composite score (best overall metric)
- Laplacian heatmap: per-pixel Laplacian magnitude (finest resolution)
- Gradient magnitude heatmap: edge density map
- Overlay heatmap: color overlay on original image

COLOR CODING (all maps):
  ■ Green  (high score, ≥ 0.7):   Sharp, reliable for analysis
  ■ Yellow (medium, 0.4-0.7):     Acceptable, use with caution
  ■ Red    (low score, < 0.4):    Blurry, exclude or flag
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from focus.metrics.composite import _compute_regional_scores


@dataclass
class FocusHeatmapResult:
    """Complete focus heatmap output."""

    score_map: NDArray[np.float32]
    """Per-region focus scores, shape (rows, cols)"""

    heatmap_rgb: NDArray[np.uint8]
    """Full-resolution color heatmap, shape (H, W, 3)"""

    overlay_rgb: NDArray[np.uint8]
    """Original image with transparent heatmap overlay, shape (H, W, 3)"""

    mask_sharp: NDArray[np.bool_]
    """Boolean mask: pixels in sharp regions"""

    mask_blurry: NDArray[np.bool_]
    """Boolean mask: pixels in blurry regions"""

    mean_score: float
    """Spatial mean focus score"""

    sharp_fraction: float
    """Fraction of image area that is sharp (score >= 0.7)"""

    acceptable_fraction: float
    """Fraction of image area that is acceptable (score >= 0.4)"""


def generate_focus_heatmap(
    image: NDArray[np.uint8],
    grid: tuple[int, int] = (8, 8),
    overlay_alpha: float = 0.45,
) -> FocusHeatmapResult:
    """
    Generate comprehensive focus heatmap for a microscopy image.

    Args:
        image: Input image (RGB or grayscale)
        grid: Grid dimensions (rows, cols) for regional analysis
        overlay_alpha: Opacity of heatmap overlay (0=invisible, 1=opaque)

    Returns:
        FocusHeatmapResult with all heatmap outputs
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        rgb = image.copy()
    else:
        gray = image.copy()
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    h, w = gray.shape

    # Compute regional scores
    score_map = _compute_regional_scores(gray, grid)
    mean_score = float(score_map.mean())

    # Upsample score map to full image resolution
    score_full = cv2.resize(
        score_map,
        (w, h),
        interpolation=cv2.INTER_CUBIC,
    )
    score_full = np.clip(score_full, 0.0, 1.0).astype(np.float32)

    # Generate color heatmap — RdYlGn if available (OpenCV ≥ 4.x), else JET
    score_uint8 = (score_full * 255).astype(np.uint8)
    _cmap = getattr(cv2, "COLORMAP_RdYlGn", cv2.COLORMAP_JET)
    heatmap_bgr = cv2.applyColorMap(score_uint8, _cmap)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    # Blend with original image
    overlay_float = rgb.astype(np.float32) * (1 - overlay_alpha) + \
                    heatmap_rgb.astype(np.float32) * overlay_alpha
    overlay_rgb = np.clip(overlay_float, 0, 255).astype(np.uint8)

    # Quality masks
    mask_sharp = score_full >= 0.70
    mask_blurry = score_full < 0.40

    sharp_fraction = float(mask_sharp.mean())
    acceptable_fraction = float((score_full >= 0.40).mean())

    return FocusHeatmapResult(
        score_map=score_map,
        heatmap_rgb=heatmap_rgb,
        overlay_rgb=overlay_rgb,
        mask_sharp=mask_sharp,
        mask_blurry=mask_blurry,
        mean_score=mean_score,
        sharp_fraction=sharp_fraction,
        acceptable_fraction=acceptable_fraction,
    )


def generate_laplacian_heatmap(
    image: NDArray[np.uint8],
    blur_radius: int = 5,
) -> NDArray[np.uint8]:
    """
    Per-pixel Laplacian magnitude heatmap.

    Higher resolution than grid-based heatmap — shows exact
    pixel locations of edges (sharp areas).

    Applies Gaussian smoothing to the Laplacian magnitude map
    for a smoother visual output.

    Args:
        image: Input image (RGB or grayscale)
        blur_radius: Gaussian smoothing radius for the magnitude map

    Returns:
        RGB heatmap (H, W, 3)
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap_mag = np.abs(lap).astype(np.float32)

    # Smooth for visual output
    if blur_radius > 0:
        ksize = blur_radius * 2 + 1
        lap_mag = cv2.GaussianBlur(lap_mag, (ksize, ksize), 0)

    # Normalize to uint8
    lap_norm = cv2.normalize(lap_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    heatmap_bgr = cv2.applyColorMap(lap_norm, cv2.COLORMAP_HOT)
    return cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)


def generate_gradient_heatmap(
    image: NDArray[np.uint8],
) -> NDArray[np.uint8]:
    """
    Sobel gradient magnitude heatmap.

    Shows edge density — sharp images have brighter, more defined
    edge regions. Useful for identifying stalked trichome boundaries.

    Args:
        image: Input image (RGB or grayscale)

    Returns:
        RGB gradient magnitude heatmap (H, W, 3)
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    f = gray.astype(np.float32)
    gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)

    mag_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(mag_norm, cv2.COLORMAP_JET)
    return cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)


def annotate_focus_regions(
    image: NDArray[np.uint8],
    result: FocusHeatmapResult,
    draw_labels: bool = True,
) -> NDArray[np.uint8]:
    """
    Draw focus quality annotations on image (bounding boxes + score labels).

    Draws colored rectangles on the grid regions:
    - Green border: sharp region
    - Yellow border: acceptable region
    - Red border: blurry region

    Args:
        image: Original RGB image
        result: FocusHeatmapResult from generate_focus_heatmap()
        draw_labels: Whether to draw score text labels

    Returns:
        Annotated RGB image
    """
    annotated = image.copy()
    h, w = annotated.shape[:2]
    rows, cols = result.score_map.shape

    row_step = h // rows
    col_step = w // cols

    for r in range(rows):
        for c in range(cols):
            score = float(result.score_map[r, c])
            x1 = c * col_step
            y1 = r * row_step
            x2 = min(x1 + col_step, w)
            y2 = min(y1 + row_step, h)

            # Color coding
            if score >= 0.70:
                color = (0, 200, 0)      # Green
            elif score >= 0.40:
                color = (255, 200, 0)    # Yellow
            else:
                color = (220, 50, 50)    # Red

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 1)

            if draw_labels:
                label = f"{score:.2f}"
                cx = x1 + col_step // 2 - 15
                cy = y1 + row_step // 2 + 4
                cv2.putText(
                    annotated, label, (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA
                )

    return annotated
