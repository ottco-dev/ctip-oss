"""
morphology.domain.stalk_detector — Stalk region detection within trichome masks.

Detects and measures the stalk (peduncle) of capitate-stalked trichomes.

ALGORITHM:
1. Skeletonize the binary mask (medial axis transform)
2. Find the longest skeleton path (approximates stalk axis)
3. Detect the narrowing point between stalk and head using width profile
4. Measure stalk length from base to constriction point

BIOLOGICAL CONTEXT:
The stalk (peduncle) connects the secretory gland head to the epidermal
surface. Stalk length is a defining feature distinguishing trichome types:

  Bulbous:             no discernible stalk (<5 µm)
  Capitate-sessile:    very short stalk (5–30 µm)
  Capitate-stalked:    prominent stalk (50–500 µm)

Reference:
  Kim, E.S. & Mahlberg, P.G. (1997). Secretory cavity development in
  glandular trichomes of Cannabis sativa. American Journal of Botany 84(2):220.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
from numpy.typing import NDArray


@dataclass
class StalkMeasurement:
    """Measurements of the trichome stalk region."""

    stalk_length_px: float
    """Estimated stalk length in pixels (base-to-head junction)."""

    stalk_width_px: float
    """Mean width of the stalk region in pixels."""

    stalk_base_y: Optional[float]
    """Y-coordinate of the stalk base (lowest point, closest to epidermis)."""

    head_junction_y: Optional[float]
    """Y-coordinate where stalk meets the gland head."""

    has_visible_stalk: bool
    """True if a distinct stalk region was detected."""

    confidence: float
    """Confidence in the stalk detection [0, 1]."""

    @property
    def stalk_aspect_ratio(self) -> float:
        """Length / Width ratio of the stalk. >3 indicates a well-defined stalk."""
        if self.stalk_width_px < 1:
            return 0.0
        return self.stalk_length_px / self.stalk_width_px


@dataclass
class HeadMeasurement:
    """Measurements of the secretory gland head region."""

    head_area_px: float
    """Area of the head region in pixels."""

    head_diameter_px: float
    """Estimated head diameter (from equivalent circle: 2√(area/π))."""

    head_circularity: float
    """Circularity of the head region [0,1]. Higher = more spherical."""

    head_centroid_x: float
    head_centroid_y: float
    """Centroid of the head region in pixel coordinates."""


def detect_stalk_and_head(
    mask: NDArray[np.uint8],
    *,
    min_stalk_length_px: float = 8.0,
    width_sample_points: int = 20,
) -> Tuple[StalkMeasurement, Optional[HeadMeasurement]]:
    """
    Detect and measure stalk and head regions within a trichome mask.

    Strategy:
    1. Compute a horizontal width profile along the mask's vertical axis.
    2. Find the local minimum of width (constriction = stalk/head junction).
    3. Classify below-constriction as stalk, above as head.

    Args:
        mask:                Binary mask, uint8, shape (H, W).
        min_stalk_length_px: Minimum stalk length to report has_visible_stalk=True.
        width_sample_points: Number of horizontal slices for width profile.

    Returns:
        Tuple of (StalkMeasurement, HeadMeasurement|None).
        HeadMeasurement is None if head cannot be separated from stalk.
    """
    binary = (mask > 0).astype(np.uint8) * 255
    rows, cols = binary.shape

    if rows < 10 or cols < 5:
        return _no_stalk(), None

    # Build vertical width profile: for each row, count non-zero columns
    width_profile = np.zeros(rows, dtype=np.float32)
    for r in range(rows):
        width_profile[r] = float(binary[r].sum() / 255)

    # Find bounding rows (first/last non-zero)
    nonzero_rows = np.where(width_profile > 0)[0]
    if len(nonzero_rows) < 5:
        return _no_stalk(), None

    row_top = int(nonzero_rows[0])
    row_bot = int(nonzero_rows[-1])
    row_span = row_bot - row_top

    if row_span < 5:
        return _no_stalk(), None

    # Sample width profile at equal intervals within bounding rows
    sample_ys = np.linspace(row_top, row_bot, width_sample_points, dtype=int)
    sampled_widths = width_profile[sample_ys]

    # Find constriction: minimum width in the lower 60% of the trichome
    # (stalk is typically at the base/bottom)
    lower_boundary = int(width_sample_points * 0.4)
    search_region = sampled_widths[:lower_boundary]

    if len(search_region) < 3:
        return _no_stalk(), None

    max_width = sampled_widths.max()
    if max_width < 1:
        return _no_stalk(), None

    # Normalized profile
    norm_profile = sampled_widths / max_width

    # Constriction: find point with minimum width ratio in lower portion
    # that is significantly narrower than the maximum (ratio < 0.6)
    constriction_idx = None
    min_val = 1.0
    for i, val in enumerate(norm_profile[:lower_boundary]):
        if val < min_val and val < 0.60:
            min_val = val
            constriction_idx = i

    if constriction_idx is None or constriction_idx == 0:
        # No clear constriction → treat as head-only (bulbous or sessile)
        area = float(binary.sum() / 255)
        diameter = 2.0 * math.sqrt(area / math.pi) if area > 0 else 0.0
        cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        circ = 0.0
        cx, cy = cols / 2.0, rows / 2.0
        if cnts:
            cnt = max(cnts, key=cv2.contourArea)
            area_c = cv2.contourArea(cnt)
            peri = cv2.arcLength(cnt, True)
            circ = (4 * math.pi * area_c / peri ** 2) if peri > 0 else 0.0
            m = cv2.moments(cnt)
            if m["m00"] > 0:
                cx = m["m10"] / m["m00"]
                cy = m["m01"] / m["m00"]

        head = HeadMeasurement(
            head_area_px=area,
            head_diameter_px=diameter,
            head_circularity=float(np.clip(circ, 0.0, 1.0)),
            head_centroid_x=float(cx),
            head_centroid_y=float(cy),
        )
        stalk = StalkMeasurement(
            stalk_length_px=0.0,
            stalk_width_px=0.0,
            stalk_base_y=None,
            head_junction_y=None,
            has_visible_stalk=False,
            confidence=0.7,
        )
        return stalk, head

    # Junction row (in original image coordinates)
    junction_row = int(sample_ys[constriction_idx])

    # Stalk region: below junction
    stalk_mask = np.zeros_like(binary)
    stalk_mask[junction_row:, :] = binary[junction_row:, :]

    # Head region: above junction
    head_mask = np.zeros_like(binary)
    head_mask[:junction_row, :] = binary[:junction_row, :]

    stalk_area = float(stalk_mask.sum() / 255)
    stalk_length = float(row_bot - junction_row)

    if stalk_length > 0 and stalk_area > 0:
        stalk_width = stalk_area / stalk_length
    else:
        stalk_width = 0.0

    has_stalk = stalk_length >= min_stalk_length_px and stalk_width < max_width * 0.7

    # Confidence: based on clarity of constriction
    constriction_depth = 1.0 - min_val  # 0 = no constriction, 1 = full constriction
    confidence = float(np.clip(constriction_depth * 2.0, 0.0, 1.0))

    stalk = StalkMeasurement(
        stalk_length_px=stalk_length,
        stalk_width_px=stalk_width,
        stalk_base_y=float(row_bot),
        head_junction_y=float(junction_row),
        has_visible_stalk=has_stalk,
        confidence=confidence,
    )

    # Head measurements
    head_area = float(head_mask.sum() / 255)
    head_diameter = 2.0 * math.sqrt(head_area / math.pi) if head_area > 0 else 0.0

    head_cnts, _ = cv2.findContours(head_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    head_circ = 0.0
    hcx, hcy = float(cols / 2), float(junction_row / 2)
    if head_cnts:
        hcnt = max(head_cnts, key=cv2.contourArea)
        ha = cv2.contourArea(hcnt)
        hp = cv2.arcLength(hcnt, True)
        head_circ = (4 * math.pi * ha / hp ** 2) if hp > 0 else 0.0
        m = cv2.moments(hcnt)
        if m["m00"] > 0:
            hcx = m["m10"] / m["m00"]
            hcy = m["m01"] / m["m00"]

    head = HeadMeasurement(
        head_area_px=head_area,
        head_diameter_px=head_diameter,
        head_circularity=float(np.clip(head_circ, 0.0, 1.0)),
        head_centroid_x=float(hcx),
        head_centroid_y=float(hcy),
    )

    return stalk, head


def _no_stalk() -> StalkMeasurement:
    return StalkMeasurement(
        stalk_length_px=0.0,
        stalk_width_px=0.0,
        stalk_base_y=None,
        head_junction_y=None,
        has_visible_stalk=False,
        confidence=0.0,
    )
