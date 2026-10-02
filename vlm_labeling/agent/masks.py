"""Mask shape checks for agent / click labelling: glandular heads are round, stalks and merged heads are not."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

MIN_CIRCULARITY = 0.75
MAX_ASPECT = 1.5


@dataclass(frozen=True)
class Shape:
    area: float
    circularity: float
    aspect: float

    @property
    def round(self) -> bool:
        return self.area > 0 and self.circularity >= MIN_CIRCULARITY and self.aspect <= MAX_ASPECT


def shape_of(mask: NDArray[np.bool_]) -> Shape:
    """Area, circularity (4πA/P²) and min-area-rectangle aspect of the largest contour."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return Shape(0.0, 0.0, 99.0)
    c = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(c))
    perimeter = float(cv2.arcLength(c, True))
    (_, _), (w, h), _ = cv2.minAreaRect(c)
    return Shape(area, 4 * np.pi * area / (perimeter * perimeter + 1e-9), max(w, h) / max(1.0, min(w, h)))


def head_only(mask: NDArray[np.bool_], point: tuple[int, int]) -> tuple[NDArray[np.bool_], str]:
    """
    Keep the head around the clicked point.

    Returns (mask, status): "ok" if the mask is already round, "trimmed" if cutting it to a disk around the click
    (radius from the local mask width) makes it round, "rejected" otherwise or if the click is outside the mask.
    """
    h, w = mask.shape
    x, y = min(max(point[0], 0), w - 1), min(max(point[1], 0), h - 1)
    if not mask.any():
        return mask, "rejected"
    if shape_of(mask).round:
        return mask, "ok"
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    if dist[y, x] <= 0:
        return mask, "rejected"
    r = max(6.0, float(dist[y, x]) * 1.15)
    yy, xx = np.ogrid[:h, :w]
    trimmed = mask & ((yy - y) ** 2 + (xx - x) ** 2 <= r * r)
    return (trimmed, "trimmed") if shape_of(trimmed).round else (mask, "rejected")


def polygon_of(mask: NDArray[np.bool_], epsilon_px: float = 1.5) -> list[list[float]]:
    """Simplified outline of the largest contour as [[x, y], ...] in pixels."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return []
    c = cv2.approxPolyDP(max(contours, key=cv2.contourArea), epsilon_px, True)
    return [[float(p[0][0]), float(p[0][1])] for p in c]
