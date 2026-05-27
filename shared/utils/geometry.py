"""
shared.utils.geometry — Polygon and mask geometry operations.

Functions:
    polygon_area(points)              — Shoelace formula area
    polygon_centroid(points)          — Centroid of polygon
    simplify_polygon(points, tol)     — Ramer-Douglas-Peucker simplification
    mask_to_polygon(mask)             — Binary mask → polygon points
    polygon_to_mask(points, h, w)     — Polygon → binary mask
    compute_iou_boxes(box_a, box_b)   — 2D IoU for axis-aligned boxes
    expand_box(box, margin, img_size) — Expand bounding box with margin
    clip_box(box, img_size)           — Clip box to image bounds
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def polygon_area(points: NDArray[np.float64]) -> float:
    """
    Compute polygon area using Shoelace formula.

    Args:
        points: (N, 2) array of (x, y) coordinates.

    Returns:
        Absolute area in square pixels.
    """
    if len(points) < 3:
        return 0.0
    x = points[:, 0]
    y = points[:, 1]
    return float(0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def polygon_centroid(points: NDArray[np.float64]) -> tuple[float, float]:
    """
    Compute centroid of a polygon.

    Args:
        points: (N, 2) array of (x, y) coordinates.

    Returns:
        (cx, cy) centroid coordinates.
    """
    return float(np.mean(points[:, 0])), float(np.mean(points[:, 1]))


def simplify_polygon(
    points: NDArray[np.float64],
    tolerance: float = 2.0,
) -> NDArray[np.float64]:
    """
    Simplify polygon using Ramer-Douglas-Peucker algorithm.

    Reduces number of points while preserving shape.
    Tolerance in pixels.

    Args:
        points: (N, 2) polygon points.
        tolerance: Perpendicular distance threshold.

    Returns:
        Simplified polygon (M, 2) with M <= N.
    """
    if len(points) <= 2:
        return points

    try:
        import cv2
        # cv2.approxPolyDP requires float32 contour format
        contour = points.astype(np.float32).reshape(-1, 1, 2)
        simplified = cv2.approxPolyDP(contour, tolerance, closed=True)
        return simplified.reshape(-1, 2).astype(np.float64)
    except Exception:
        # Fallback: return original
        return points


def mask_to_polygon(
    mask: NDArray[np.bool_],
    simplify_tolerance: float = 2.0,
) -> NDArray[np.float64] | None:
    """
    Convert a binary mask to polygon points.

    Finds the largest external contour and optionally simplifies it.

    Args:
        mask: Boolean 2D array (H, W).
        simplify_tolerance: RDP simplification tolerance (pixels). 0 = no simplification.

    Returns:
        (N, 2) polygon points in (x, y) order, or None if no contour found.
    """
    import cv2

    mask_u8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    points = largest.reshape(-1, 2).astype(np.float64)

    if simplify_tolerance > 0:
        points = simplify_polygon(points, tolerance=simplify_tolerance)

    return points


def polygon_to_mask(
    points: NDArray[np.float64],
    height: int,
    width: int,
) -> NDArray[np.bool_]:
    """
    Rasterize polygon points to a binary mask.

    Args:
        points: (N, 2) polygon in (x, y) order.
        height: Output mask height.
        width: Output mask width.

    Returns:
        Boolean mask (H, W).
    """
    import cv2

    mask = np.zeros((height, width), dtype=np.uint8)
    contour = np.round(points).astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [contour], 255)
    return mask.astype(bool)


def compute_iou_box(
    box_a: tuple[float, float, float, float],
    box_b: tuple[float, float, float, float],
) -> float:
    """
    Compute IoU between two axis-aligned bounding boxes.

    Args:
        box_a: (x1, y1, x2, y2)
        box_b: (x1, y1, x2, y2)

    Returns:
        IoU in [0, 1].
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    if inter_area == 0:
        return 0.0

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union_area = area_a + area_b - inter_area

    return float(inter_area / union_area) if union_area > 0 else 0.0


def expand_box(
    box: tuple[float, float, float, float],
    margin_px: float,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    """
    Expand a bounding box by margin_px on each side, clipped to image bounds.

    Args:
        box: (x1, y1, x2, y2)
        margin_px: Expansion in pixels.
        image_width, image_height: Image dimensions for clipping.

    Returns:
        Expanded (x1, y1, x2, y2).
    """
    x1, y1, x2, y2 = box
    x1 = max(0, x1 - margin_px)
    y1 = max(0, y1 - margin_px)
    x2 = min(image_width, x2 + margin_px)
    y2 = min(image_height, y2 + margin_px)
    return x1, y1, x2, y2


def clip_box(
    box: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    """Clip bounding box to image bounds."""
    x1, y1, x2, y2 = box
    return (
        max(0, x1),
        max(0, y1),
        min(image_width, x2),
        min(image_height, y2),
    )


# Aliases for backward compatibility with tests
def compute_iou(box_a: list, box_b: list) -> float:
    """Alias for compute_iou_box — takes [x1,y1,x2,y2] lists."""
    return compute_iou_box(
        tuple(box_a[:4]),  # type: ignore[arg-type]
        tuple(box_b[:4]),  # type: ignore[arg-type]
    )



def polygon_area(polygon: list) -> float:
    """
    Compute polygon area using the Shoelace formula.
    
    Args:
        polygon: List of (x, y) tuples or [x, y] lists.
    
    Returns:
        Area in square pixels (absolute value).
    """
    pts = [(float(p[0]), float(p[1])) for p in polygon]
    n = len(pts)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += pts[i][0] * pts[j][1]
        area -= pts[j][0] * pts[i][1]
    return abs(area) / 2.0

