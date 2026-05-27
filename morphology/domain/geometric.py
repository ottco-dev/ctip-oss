"""
morphology.domain.geometric — Geometric feature extraction from trichome instance masks.

Computes shape descriptors used for morphological type classification:

FEATURES EXTRACTED:
1. Area (px²) and perimeter (px)
2. Circularity  = 4π·Area / Perimeter²  (1.0 = perfect circle)
3. Elongation   = major_axis / minor_axis (from PCA on mask pixels)
4. Convexity    = area / convex_hull_area  (1.0 = fully convex)
5. Solidity     = area / bounding_rect_area
6. Aspect ratio = bbox_width / bbox_height
7. Compactness  = area / (major_axis²)

BIOLOGICAL INTERPRETATION:
- Bulbous:          high circularity (>0.7), low elongation (<1.4), small area
- Capitate-sessile: medium circularity, slightly elongated, medium area
- Capitate-stalked: low circularity (<0.5), high elongation (>2.0), large area
- Non-glandular:    highly elongated (>3.0), irregular, low convexity

Reference:
  Turner, J.C. et al. (1981). Interrelationships of glandular trichomes
  and cannabinoid content I. American Journal of Botany 68(6):853–862.
  DOI: 10.2307/2442850

  Small, E. (1979). The Species Problem in Cannabis: Science & Semantics.
  Corpus Information Services, Toronto, Canada.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from numpy.typing import NDArray


@dataclass
class GeometricDescriptors:
    """
    Geometric shape descriptors for a single trichome mask.

    All pixel-based measurements are in original image pixel units.
    Convert to physical units (µm) using a CalibrationScale.
    """

    # --- Basic measurements ---
    area_px: float
    """Area of the mask in pixels squared."""

    perimeter_px: float
    """Perimeter of the mask contour in pixels."""

    bounding_width_px: float
    """Width of the axis-aligned bounding rectangle in pixels."""

    bounding_height_px: float
    """Height of the axis-aligned bounding rectangle in pixels."""

    # --- Shape indices ---
    circularity: float
    """
    4π·Area/Perimeter² ∈ [0, 1].
    1.0 = perfect circle. 0 = highly non-circular.
    Undefined (→ 0) for degenerate contours with perimeter ≈ 0.
    """

    elongation: float
    """
    major_axis / minor_axis from PCA on mask pixel coordinates.
    1.0 = isotropic. Higher = more elongated.
    Bulbous: ~1.0–1.5. Stalked: >2.5.
    """

    convexity: float
    """
    area / convex_hull_area ∈ (0, 1].
    1.0 = fully convex. Low values indicate concavities.
    """

    solidity: float
    """
    area / bounding_rect_area ∈ (0, 1].
    Measures how densely the mask fills its bounding box.
    """

    compactness: float
    """
    area / (major_axis²). Dimensionless measure of how compact the shape is.
    Equivalent to relative fill along the principal axis.
    """

    aspect_ratio: float
    """bounding_width / bounding_height. >1 → wider, <1 → taller."""

    # --- PCA-derived ---
    major_axis_px: float
    """Length of the principal (longest) axis in pixels."""

    minor_axis_px: float
    """Length of the secondary (shortest) axis in pixels."""

    orientation_deg: float
    """
    Orientation of the major axis relative to horizontal, in degrees.
    Range: [-90, 90]. Used for stalk angle estimation.
    """

    # --- Convex hull ---
    convex_hull_area_px: float
    """Area of the convex hull in pixels."""

    # --- Centroid ---
    centroid_x: float
    """X coordinate of the mask centroid in pixels."""

    centroid_y: float
    """Y coordinate of the mask centroid in pixels."""

    @property
    def is_valid(self) -> bool:
        """True if the mask had enough pixels for reliable measurements."""
        return self.area_px >= 25  # At least 25 pixels (~5×5)

    @property
    def shape_index(self) -> float:
        """Combined shape index = circularity × convexity. ∈ [0,1]."""
        return self.circularity * self.convexity

    def to_feature_vector(self) -> NDArray[np.float32]:
        """
        Return a fixed-length feature vector for ML classifiers.

        Features (7): [circularity, elongation, convexity, solidity,
                       compactness, aspect_ratio, normalized_area]
        All values are clipped to [0, 1] or [0, 10] depending on feature.
        """
        return np.array(
            [
                np.clip(self.circularity, 0.0, 1.0),
                np.clip(self.elongation / 10.0, 0.0, 1.0),  # normalize to [0,1]
                np.clip(self.convexity, 0.0, 1.0),
                np.clip(self.solidity, 0.0, 1.0),
                np.clip(self.compactness, 0.0, 1.0),
                np.clip(self.aspect_ratio / 5.0, 0.0, 1.0),  # normalize
                np.clip(self.area_px / 10000.0, 0.0, 1.0),   # normalize ~100×100
            ],
            dtype=np.float32,
        )


def extract_geometric_descriptors(
    mask: NDArray[np.uint8],
    *,
    contour: Optional[NDArray] = None,
) -> GeometricDescriptors:
    """
    Extract geometric descriptors from a binary trichome mask.

    Args:
        mask:    Binary mask, uint8, 0/255 or 0/1.
                 Shape: (H, W) — single-channel.
        contour: Pre-computed contour array (opencv format) to skip re-finding.

    Returns:
        GeometricDescriptors with all shape measurements.

    Raises:
        ValueError: If mask is empty or has invalid shape.
    """
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape {mask.shape}")

    # Normalize to 0/255
    binary = (mask > 0).astype(np.uint8) * 255

    # Find contour if not provided
    if contour is None:
        cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return _degenerate_descriptors()
        # Use the largest contour
        contour = max(cnts, key=cv2.contourArea)

    area = cv2.contourArea(contour)
    if area < 4:
        return _degenerate_descriptors()

    perimeter = cv2.arcLength(contour, closed=True)

    # Bounding rect
    x, y, w, h = cv2.boundingRect(contour)

    # Circularity
    if perimeter > 0:
        circularity = (4.0 * math.pi * area) / (perimeter ** 2)
        circularity = float(np.clip(circularity, 0.0, 1.0))
    else:
        circularity = 0.0

    # Convex hull
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    convexity = float(area / hull_area) if hull_area > 0 else 1.0

    # Solidity
    rect_area = float(w * h) if w > 0 and h > 0 else 1.0
    solidity = float(area / rect_area)

    # Aspect ratio
    aspect_ratio = float(w) / float(h) if h > 0 else 1.0

    # PCA on pixel coordinates for elongation + orientation
    ys, xs = np.where(binary > 0)
    if len(xs) < 5:
        major_axis = max(w, h)
        minor_axis = min(w, h)
        elongation = major_axis / minor_axis if minor_axis > 0 else 1.0
        orientation = 0.0
        cx, cy = float(x + w / 2), float(y + h / 2)
    else:
        pts = np.stack([xs, ys], axis=1).astype(np.float64)
        mean = pts.mean(axis=0)
        centered = pts - mean
        cov = np.cov(centered.T)
        if cov.ndim == 0:
            cov = np.eye(2) * cov
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        # eigenvalues sorted ascending → largest is major axis variance
        λ1, λ2 = eigenvalues  # λ1 ≤ λ2
        major_axis = 2.0 * math.sqrt(max(λ2, 0.0))
        minor_axis = 2.0 * math.sqrt(max(λ1, 0.0))
        elongation = major_axis / minor_axis if minor_axis > 1e-6 else 1.0

        # Orientation: angle of eigenvector corresponding to λ2
        ev = eigenvectors[:, 1]  # major axis eigenvector
        orientation = float(math.degrees(math.atan2(ev[1], ev[0])))

        cx, cy = float(mean[0]), float(mean[1])

    # Compactness
    compactness = float(area / (major_axis ** 2)) if major_axis > 0 else 0.0

    return GeometricDescriptors(
        area_px=float(area),
        perimeter_px=float(perimeter),
        bounding_width_px=float(w),
        bounding_height_px=float(h),
        circularity=circularity,
        elongation=float(np.clip(elongation, 1.0, 20.0)),
        convexity=float(np.clip(convexity, 0.0, 1.0)),
        solidity=float(np.clip(solidity, 0.0, 1.0)),
        compactness=float(np.clip(compactness, 0.0, 1.0)),
        aspect_ratio=float(aspect_ratio),
        major_axis_px=float(major_axis),
        minor_axis_px=float(minor_axis),
        orientation_deg=float(orientation),
        convex_hull_area_px=float(hull_area),
        centroid_x=cx,
        centroid_y=cy,
    )


def _degenerate_descriptors() -> GeometricDescriptors:
    """Return zero-filled descriptors for empty/invalid masks."""
    return GeometricDescriptors(
        area_px=0.0,
        perimeter_px=0.0,
        bounding_width_px=0.0,
        bounding_height_px=0.0,
        circularity=0.0,
        elongation=1.0,
        convexity=0.0,
        solidity=0.0,
        compactness=0.0,
        aspect_ratio=1.0,
        major_axis_px=0.0,
        minor_axis_px=0.0,
        orientation_deg=0.0,
        convex_hull_area_px=0.0,
        centroid_x=0.0,
        centroid_y=0.0,
    )


def contour_from_mask(mask: NDArray[np.uint8]) -> Optional[NDArray]:
    """
    Extract the largest outer contour from a binary mask.

    Returns None if no valid contour found.
    """
    binary = (mask > 0).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    return max(cnts, key=cv2.contourArea)
