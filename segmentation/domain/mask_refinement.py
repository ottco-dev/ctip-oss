"""
segmentation/domain/mask_refinement.py — Mask post-processing and refinement.

Post-SAM2 mask cleanup:
  1. Morphological operations (close small holes, remove noise)
  2. Watershed-based boundary refinement for touching trichomes
  3. Contour smoothing
  4. Small component removal
  5. Convexity enforcement (optional, for bulbous trichomes)

All functions work on boolean or uint8 masks (H, W).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from scipy import ndimage


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class RefinementConfig:
    """Parameters for mask refinement pipeline."""

    # Morphological cleaning
    close_kernel_size: int = 3       # Close small holes
    open_kernel_size: int = 2        # Remove noise speckles
    dilate_px: int = 0               # Optional boundary dilation

    # Small component removal
    min_component_area_px: int = 25  # Remove components smaller than this

    # Watershed (for touching trichomes)
    use_watershed: bool = False      # Enable watershed separation
    watershed_min_distance: int = 8  # Min distance between markers

    # Contour smoothing
    smooth_epsilon_fraction: float = 0.01  # Fraction of arc length for DP approx

    # Convexity
    enforce_convexity: bool = False  # Wrap mask in convex hull


# ---------------------------------------------------------------------------
# Core operations
# ---------------------------------------------------------------------------


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill enclosed holes in a binary mask."""
    binary = (mask > 0).astype(bool)
    filled = ndimage.binary_fill_holes(binary)
    return filled.astype(np.uint8) * 255


def remove_small_components(
    mask: np.ndarray,
    min_area_px: int = 25,
) -> np.ndarray:
    """Remove connected components smaller than min_area_px."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    cleaned = np.zeros_like(mask)
    for label_id in range(1, num_labels):  # Skip background (0)
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area >= min_area_px:
            cleaned[labels == label_id] = 255
    return cleaned


def morphological_clean(
    mask: np.ndarray,
    close_kernel: int = 3,
    open_kernel: int = 2,
) -> np.ndarray:
    """Apply closing (fill holes) then opening (remove noise)."""
    mask_u8 = (mask > 0).astype(np.uint8) * 255

    if close_kernel > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (close_kernel, close_kernel)
        )
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, k)

    if open_kernel > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (open_kernel, open_kernel)
        )
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, k)

    return mask_u8


def smooth_contour(mask: np.ndarray, epsilon_fraction: float = 0.01) -> np.ndarray:
    """
    Smooth mask boundary using Douglas-Peucker contour approximation.

    Args:
        mask: Binary mask (uint8 or bool).
        epsilon_fraction: DP approximation tolerance as fraction of arc length.

    Returns:
        Smoothed binary mask (uint8, 0 or 255).
    """
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return mask_u8

    smoothed = np.zeros_like(mask_u8)
    for contour in contours:
        arc = cv2.arcLength(contour, closed=True)
        eps = epsilon_fraction * arc
        approx = cv2.approxPolyDP(contour, eps, closed=True)
        cv2.fillPoly(smoothed, [approx], 255)

    return smoothed


def enforce_convex_hull(mask: np.ndarray) -> np.ndarray:
    """Replace mask with its convex hull — useful for near-spherical bulbous trichomes."""
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask_u8

    all_points = np.vstack(contours)
    hull = cv2.convexHull(all_points)
    convex = np.zeros_like(mask_u8)
    cv2.fillPoly(convex, [hull], 255)
    return convex


def watershed_separate(
    mask: np.ndarray,
    image: Optional[np.ndarray] = None,
    min_distance: int = 8,
) -> list[np.ndarray]:
    """
    Separate touching trichomes using distance transform + watershed.

    Args:
        mask: Binary mask potentially containing touching trichomes.
        image: Optional BGR image for marker-controlled watershed.
        min_distance: Minimum distance between local maxima (trichome centers).

    Returns:
        List of separated binary masks (one per trichome).
    """
    mask_u8 = (mask > 0).astype(np.uint8)

    # Distance transform
    dist = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5)
    dist_norm = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)

    # Local maxima as markers
    from scipy.ndimage import label as scipy_label
    from scipy.ndimage import maximum_filter

    local_max = (dist_norm == maximum_filter(dist_norm, size=min_distance * 2 + 1))
    local_max &= dist_norm > 0.3  # Only strong maxima

    markers, n_markers = scipy_label(local_max)

    if n_markers <= 1:
        return [mask_u8 * 255]

    # Watershed
    ws_image = (
        image if image is not None
        else cv2.cvtColor((dist_norm * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    )

    # Mark unknown region
    unknown = (mask_u8 - (dist_norm > 0.3).astype(np.uint8)).clip(0, 1).astype(np.uint8)
    markers[unknown == 1] = 0
    markers = markers.astype(np.int32)

    cv2.watershed(ws_image, markers)

    # Extract individual masks
    separated: list[np.ndarray] = []
    for label_id in range(1, n_markers + 1):
        component = (markers == label_id).astype(np.uint8) * 255
        if component.sum() > 0:
            separated.append(component)

    return separated if separated else [mask_u8 * 255]


# ---------------------------------------------------------------------------
# Full refinement pipeline
# ---------------------------------------------------------------------------


def refine_mask(
    mask: np.ndarray,
    image: Optional[np.ndarray] = None,
    config: Optional[RefinementConfig] = None,
) -> np.ndarray:
    """
    Full mask refinement pipeline.

    Args:
        mask: Raw binary mask from SAM2 (bool or uint8).
        image: Optional BGR image for watershed.
        config: Refinement parameters.

    Returns:
        Refined binary mask (uint8, 0 or 255).
    """
    cfg = config or RefinementConfig()

    # Ensure uint8
    mask_u8 = (mask > 0).astype(np.uint8) * 255

    # Step 1: Morphological clean
    mask_u8 = morphological_clean(mask_u8, cfg.close_kernel_size, cfg.open_kernel_size)

    # Step 2: Fill holes
    if cfg.close_kernel_size > 0:
        mask_u8 = fill_holes(mask_u8)

    # Step 3: Remove small components
    if cfg.min_component_area_px > 0:
        mask_u8 = remove_small_components(mask_u8, cfg.min_component_area_px)

    # Step 4: Optional dilation
    if cfg.dilate_px > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (cfg.dilate_px * 2 + 1, cfg.dilate_px * 2 + 1)
        )
        mask_u8 = cv2.dilate(mask_u8, k, iterations=1)

    # Step 5: Contour smoothing
    if cfg.smooth_epsilon_fraction > 0:
        mask_u8 = smooth_contour(mask_u8, cfg.smooth_epsilon_fraction)

    # Step 6: Convexity
    if cfg.enforce_convexity:
        mask_u8 = enforce_convex_hull(mask_u8)

    return mask_u8


def batch_refine(
    masks: list[np.ndarray],
    image: Optional[np.ndarray] = None,
    config: Optional[RefinementConfig] = None,
) -> list[np.ndarray]:
    """Refine a list of masks from a batch segmentation result."""
    return [refine_mask(m, image, config) for m in masks]
