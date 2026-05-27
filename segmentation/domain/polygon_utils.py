"""
segmentation/domain/polygon_utils.py — Mask ↔ polygon conversion utilities.

Provides:
  - mask_to_polygon: binary mask → simplified polygon (list of [x, y] points)
  - polygon_to_mask: polygon → binary mask
  - mask_to_rle: binary mask → COCO RLE encoding
  - rle_to_mask: COCO RLE → binary mask
  - polygon_area: compute polygon area via shoelace formula
  - polygon_centroid: compute centroid of a polygon
  - simplify_polygon: Douglas-Peucker simplification
"""

from __future__ import annotations

import struct
from typing import Optional

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

Polygon = list[list[float]]   # [[x, y], ...]
RLE = dict                    # COCO RLE: {"counts": ..., "size": [H, W]}


# ---------------------------------------------------------------------------
# Mask → Polygon
# ---------------------------------------------------------------------------


def mask_to_polygon(
    mask: np.ndarray,
    simplify_epsilon: float = 1.0,
    min_area: float = 10.0,
) -> list[Polygon]:
    """
    Convert a binary mask to a list of polygons.

    Args:
        mask: Binary mask (uint8 or bool), shape (H, W).
        simplify_epsilon: Douglas-Peucker epsilon in pixels.
        min_area: Minimum polygon area; smaller polygons discarded.

    Returns:
        List of polygons, each as [[x0,y0], [x1,y1], ...].
        Multiple polygons if mask has separate components.
    """
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polygons: list[Polygon] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        if simplify_epsilon > 0:
            arc = cv2.arcLength(contour, closed=True)
            eps = simplify_epsilon / arc * arc  # Absolute eps in pixels
            eps = simplify_epsilon
            approx = cv2.approxPolyDP(contour, eps, closed=True)
        else:
            approx = contour

        # Flatten to [[x, y], ...]
        poly = approx.reshape(-1, 2).tolist()
        if len(poly) >= 3:
            polygons.append([[float(p[0]), float(p[1])] for p in poly])

    return polygons


def mask_to_coco_segmentation(
    mask: np.ndarray,
    simplify_epsilon: float = 1.0,
) -> list[list[float]]:
    """
    Convert mask to COCO segmentation format (flat list per polygon).

    Returns:
        List of [x1, y1, x2, y2, ...] flat lists (COCO segmentation format).
    """
    polygons = mask_to_polygon(mask, simplify_epsilon=simplify_epsilon)
    return [
        [coord for point in poly for coord in point]
        for poly in polygons
    ]


# ---------------------------------------------------------------------------
# Polygon → Mask
# ---------------------------------------------------------------------------


def polygon_to_mask(
    polygon: Polygon,
    height: int,
    width: int,
) -> np.ndarray:
    """
    Convert a polygon to a binary mask.

    Args:
        polygon: List of [x, y] points.
        height: Output mask height.
        width: Output mask width.

    Returns:
        Binary mask, shape (H, W), dtype uint8, values 0 or 255.
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    pts = np.array([[int(p[0]), int(p[1])] for p in polygon], dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def flat_polygon_to_mask(
    flat: list[float],
    height: int,
    width: int,
) -> np.ndarray:
    """Convert COCO flat polygon [x1,y1,x2,y2,...] to binary mask."""
    polygon = [[flat[i], flat[i + 1]] for i in range(0, len(flat) - 1, 2)]
    return polygon_to_mask(polygon, height, width)


# ---------------------------------------------------------------------------
# RLE encoding (COCO format)
# ---------------------------------------------------------------------------


def mask_to_rle(mask: np.ndarray) -> RLE:
    """
    Encode binary mask as COCO RLE.

    COCO RLE: column-major (Fortran order) run-length encoding.

    Args:
        mask: Binary mask (uint8 or bool), shape (H, W).

    Returns:
        Dict {"counts": bytes, "size": [H, W]} (COCO format).
    """
    h, w = mask.shape
    mask_bool = (mask > 0).astype(np.uint8)

    # Fortran-order flatten (column-major)
    flat = mask_bool.ravel(order="F")

    # Run-length encoding
    counts = []
    current_val = 0  # Start with 0 (background)
    run = 0

    for px in flat:
        if px == current_val:
            run += 1
        else:
            counts.append(run)
            run = 1
            current_val = 1 - current_val

    counts.append(run)

    return {"counts": counts, "size": [h, w]}


def rle_to_mask(rle: RLE) -> np.ndarray:
    """
    Decode COCO RLE to binary mask.

    Args:
        rle: Dict with "counts" (list or bytes) and "size" [H, W].

    Returns:
        Binary mask, shape (H, W), dtype uint8 (0 or 255).
    """
    h, w = rle["size"]
    counts = rle["counts"]

    flat = np.zeros(h * w, dtype=np.uint8)
    idx = 0
    val = 0
    for run in counts:
        flat[idx : idx + run] = val * 255
        idx += run
        val = 1 - val

    mask = flat.reshape(h, w, order="F")
    return mask


# ---------------------------------------------------------------------------
# Polygon geometry
# ---------------------------------------------------------------------------


def polygon_area(polygon: Polygon) -> float:
    """Compute polygon area using the shoelace formula."""
    n = len(polygon)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += polygon[i][0] * polygon[j][1]
        area -= polygon[j][0] * polygon[i][1]
    return abs(area) / 2.0


def polygon_centroid(polygon: Polygon) -> tuple[float, float]:
    """Compute centroid (cx, cy) of a polygon."""
    n = len(polygon)
    if n == 0:
        return 0.0, 0.0
    cx = sum(p[0] for p in polygon) / n
    cy = sum(p[1] for p in polygon) / n
    return cx, cy


def polygon_perimeter(polygon: Polygon) -> float:
    """Compute polygon perimeter."""
    n = len(polygon)
    peri = 0.0
    for i in range(n):
        j = (i + 1) % n
        dx = polygon[j][0] - polygon[i][0]
        dy = polygon[j][1] - polygon[i][1]
        peri += (dx * dx + dy * dy) ** 0.5
    return peri


def polygon_circularity(polygon: Polygon) -> float:
    """
    Compute circularity = 4π * area / perimeter².

    1.0 = perfect circle, < 1.0 = elongated/irregular.
    Useful for classifying bulbous (high circularity) vs stalked trichomes.
    """
    area = polygon_area(polygon)
    peri = polygon_perimeter(polygon)
    if peri == 0.0:
        return 0.0
    import math
    return 4 * math.pi * area / (peri * peri)


def simplify_polygon(polygon: Polygon, epsilon: float = 1.0) -> Polygon:
    """
    Simplify polygon using Douglas-Peucker algorithm.

    Args:
        polygon: Input polygon [[x, y], ...].
        epsilon: Tolerance in pixels.

    Returns:
        Simplified polygon.
    """
    pts = np.array([[p[0], p[1]] for p in polygon], dtype=np.float32).reshape(-1, 1, 2)
    approx = cv2.approxPolyDP(pts, epsilon, closed=True)
    return [[float(p[0][0]), float(p[0][1])] for p in approx]


def bbox_from_polygon(polygon: Polygon) -> tuple[float, float, float, float]:
    """Return (x1, y1, x2, y2) bounding box of a polygon."""
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return min(xs), min(ys), max(xs), max(ys)


def polygon_iou(poly_a: Polygon, poly_b: Polygon, h: int, w: int) -> float:
    """Compute IoU between two polygons via mask rasterization."""
    mask_a = polygon_to_mask(poly_a, h, w)
    mask_b = polygon_to_mask(poly_b, h, w)
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return float(intersection) / float(union) if union > 0 else 0.0
