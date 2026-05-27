"""
shared.core.value_objects — Immutable domain value objects.

Value objects are defined by their attributes, not identity.
They are immutable and comparable by value.

Design principle: These objects carry scientific meaning,
not just data. A Confidence value knows its valid range.
A Micrometer value knows it cannot be negative.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterator, Sequence

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Pixel:
    """
    A pixel coordinate value.

    Pixel coordinates are always non-negative integers in image space.
    Origin (0,0) is top-left corner (OpenCV/NumPy convention).
    """

    x: int
    y: int

    def __post_init__(self) -> None:
        if self.x < 0 or self.y < 0:
            raise ValueError(f"Pixel coordinates must be non-negative, got ({self.x}, {self.y})")

    def distance_to(self, other: "Pixel") -> float:
        """Euclidean distance in pixels."""
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def to_tuple(self) -> tuple[int, int]:
        return (self.x, self.y)

    def __repr__(self) -> str:
        return f"Pixel(x={self.x}, y={self.y})"


@dataclass(frozen=True)
class Micrometer:
    """
    A physical length measurement in micrometers (µm).

    Micrometers are the standard unit for microscopy measurements.
    1 µm = 10⁻⁶ meters.

    Typical trichome size ranges:
    - Bulbous: 10–15 µm head diameter
    - Capitate sessile: 25–100 µm head diameter
    - Capitate stalked: 150–500 µm total height

    Scientific note: All µm values derived from pixel measurements
    carry inherent uncertainty from the calibration process.
    Always pair with a MeasurementUncertainty.
    """

    value: float

    def __post_init__(self) -> None:
        if self.value < 0:
            raise ValueError(f"Physical length cannot be negative, got {self.value} µm")
        if not math.isfinite(self.value):
            raise ValueError(f"Micrometer value must be finite, got {self.value}")

    @classmethod
    def from_pixels(cls, pixels: float, scale: "CalibrationScale") -> "Micrometer":
        """Convert pixel measurement to micrometers using calibration scale."""
        return cls(value=pixels * scale.um_per_pixel)

    def to_mm(self) -> float:
        return self.value / 1000.0

    def __repr__(self) -> str:
        return f"{self.value:.2f} µm"

    def __add__(self, other: "Micrometer") -> "Micrometer":
        return Micrometer(self.value + other.value)

    def __truediv__(self, divisor: float) -> "Micrometer":
        return Micrometer(self.value / divisor)


@dataclass(frozen=True)
class CalibrationScale:
    """
    Pixel-to-physical-unit calibration scale.

    Derived from microscope settings and reference measurements.
    This is the fundamental calibration object that enables
    all physical measurements.

    Uncertainty: All scale factors have measurement uncertainty.
    Use uncertainty_um_per_pixel to propagate errors correctly.
    """

    um_per_pixel: float
    """Micrometers per pixel — the core scale factor."""

    uncertainty_um_per_pixel: float = 0.0
    """Calibration uncertainty in µm/px (±1 standard deviation)."""

    objective_magnification: float | None = None
    """Microscope objective magnification (e.g., 10, 20, 40, 100)."""

    digital_zoom: float = 1.0
    """Additional digital zoom factor."""

    source: str = "unknown"
    """How this calibration was determined (e.g., 'stage_micrometer', 'known_reference')."""

    def __post_init__(self) -> None:
        if self.um_per_pixel <= 0:
            raise ValueError(f"Scale factor must be positive, got {self.um_per_pixel}")
        if self.digital_zoom <= 0:
            raise ValueError(f"Digital zoom must be positive, got {self.digital_zoom}")

    @property
    def relative_uncertainty(self) -> float:
        """Relative uncertainty as fraction (e.g., 0.05 = 5%)."""
        if self.um_per_pixel == 0:
            return float("inf")
        return self.uncertainty_um_per_pixel / self.um_per_pixel

    def pixels_to_um(self, pixels: float) -> tuple[float, float]:
        """
        Convert pixels to micrometers with uncertainty propagation.

        Returns:
            Tuple of (value_um, uncertainty_um)
        """
        value = pixels * self.um_per_pixel
        uncertainty = pixels * self.uncertainty_um_per_pixel
        return value, uncertainty

    def __repr__(self) -> str:
        return (
            f"CalibrationScale({self.um_per_pixel:.4f} µm/px "
            f"±{self.uncertainty_um_per_pixel:.4f}, "
            f"obj={self.objective_magnification}×)"
        )


@dataclass(frozen=True)
class Confidence:
    """
    A prediction confidence score in [0, 1].

    Represents the model's estimated probability that a prediction is correct.
    Note: Raw model scores are NOT calibrated probabilities.
    Use calibrated confidence where possible (Platt scaling, isotonic regression).

    See: Guo et al. (2017). "On Calibration of Modern Neural Networks."
    ICML 2017. https://arxiv.org/abs/1706.04599
    """

    value: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.value <= 1.0):
            raise ValueError(f"Confidence must be in [0,1], got {self.value}")

    @classmethod
    def from_logit(cls, logit: float) -> "Confidence":
        """Convert raw logit to confidence via sigmoid."""
        return cls(value=1.0 / (1.0 + math.exp(-logit)))

    @property
    def is_high(self) -> bool:
        return self.value >= 0.75

    @property
    def is_low(self) -> bool:
        return self.value < 0.35

    def __repr__(self) -> str:
        return f"Confidence({self.value:.3f})"

    def __float__(self) -> float:
        return self.value

    def __eq__(self, other: object) -> bool:  # type: ignore[override]
        if isinstance(other, Confidence):
            return self.value == other.value
        if isinstance(other, (int, float)):
            return self.value == float(other)
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.value)

    def __lt__(self, threshold: object) -> bool:
        if isinstance(threshold, Confidence):
            return self.value < threshold.value
        if isinstance(threshold, (int, float)):
            return self.value < float(threshold)
        return NotImplemented  # type: ignore[return-value]

    def __le__(self, threshold: object) -> bool:
        if isinstance(threshold, Confidence):
            return self.value <= threshold.value
        if isinstance(threshold, (int, float)):
            return self.value <= float(threshold)
        return NotImplemented  # type: ignore[return-value]

    def __ge__(self, threshold: object) -> bool:
        if isinstance(threshold, Confidence):
            return self.value >= threshold.value
        if isinstance(threshold, (int, float)):
            return self.value >= float(threshold)
        return NotImplemented  # type: ignore[return-value]

    def __gt__(self, threshold: object) -> bool:
        if isinstance(threshold, Confidence):
            return self.value > threshold.value
        if isinstance(threshold, (int, float)):
            return self.value > float(threshold)
        return NotImplemented  # type: ignore[return-value]


@dataclass(frozen=True)
class BoundingBox:
    """
    Axis-aligned bounding box in pixel coordinates.

    Convention: (x_min, y_min, x_max, y_max) — top-left to bottom-right.
    This is the XYXY format, preferred for intersection calculations.

    Internally, YOLO uses XYWH (center_x, center_y, width, height).
    Conversion methods are provided.
    """

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def __post_init__(self) -> None:
        if self.x_min >= self.x_max:
            raise ValueError(
                f"x_min ({self.x_min}) must be less than x_max ({self.x_max})"
            )
        if self.y_min >= self.y_max:
            raise ValueError(
                f"y_min ({self.y_min}) must be less than y_max ({self.y_max})"
            )
        if any(v < 0 for v in [self.x_min, self.y_min, self.x_max, self.y_max]):
            raise ValueError("Bounding box coordinates must be non-negative")

    @classmethod
    def from_xywh(cls, x: float, y: float, w: float, h: float) -> "BoundingBox":
        """Create from center (x, y) and width/height format."""
        return cls(
            x_min=x - w / 2,
            y_min=y - h / 2,
            x_max=x + w / 2,
            y_max=y + h / 2,
        )

    @classmethod
    def from_xyxy(cls, x1: float, y1: float, x2: float, y2: float) -> "BoundingBox":
        """Create from top-left / bottom-right format."""
        return cls(x_min=x1, y_min=y1, x_max=x2, y_max=y2)

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x_min + self.x_max) / 2, (self.y_min + self.y_max) / 2)

    @property
    def aspect_ratio(self) -> float:
        """Width / Height ratio."""
        return self.width / self.height if self.height > 0 else 0.0

    def iou(self, other: "BoundingBox") -> float:
        """
        Intersection over Union (IoU) with another bounding box.

        IoU = Area(A ∩ B) / Area(A ∪ B)

        Values: [0, 1]. Higher = more overlap.
        """
        inter_x_min = max(self.x_min, other.x_min)
        inter_y_min = max(self.y_min, other.y_min)
        inter_x_max = min(self.x_max, other.x_max)
        inter_y_max = min(self.y_max, other.y_max)

        inter_w = max(0.0, inter_x_max - inter_x_min)
        inter_h = max(0.0, inter_y_max - inter_y_min)
        intersection = inter_w * inter_h

        union = self.area + other.area - intersection
        return intersection / union if union > 0 else 0.0

    def contains_point(self, x: float, y: float) -> bool:
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max

    def expand(self, margin: float) -> "BoundingBox":
        """Expand box by margin pixels on all sides."""
        return BoundingBox(
            x_min=max(0, self.x_min - margin),
            y_min=max(0, self.y_min - margin),
            x_max=self.x_max + margin,
            y_max=self.y_max + margin,
        )

    def clip_to_image(self, img_w: int, img_h: int) -> "BoundingBox":
        """Clip box to image boundaries."""
        return BoundingBox(
            x_min=max(0.0, self.x_min),
            y_min=max(0.0, self.y_min),
            x_max=min(float(img_w), self.x_max),
            y_max=min(float(img_h), self.y_max),
        )

    def to_xyxy(self) -> tuple[float, float, float, float]:
        return (self.x_min, self.y_min, self.x_max, self.y_max)

    def to_xywh(self) -> tuple[float, float, float, float]:
        """Convert to center format."""
        cx, cy = self.center
        return (cx, cy, self.width, self.height)

    def to_tlwh(self) -> tuple[float, float, float, float]:
        """Top-left x, y, width, height format."""
        return (self.x_min, self.y_min, self.width, self.height)

    def __repr__(self) -> str:
        return (
            f"BoundingBox(x=[{self.x_min:.1f}, {self.x_max:.1f}], "
            f"y=[{self.y_min:.1f}, {self.y_max:.1f}], "
            f"w={self.width:.1f}, h={self.height:.1f})"
        )


@dataclass(frozen=True)
class ImageDimensions:
    """
    Image spatial dimensions.

    Note: In NumPy/OpenCV, arrays are (H, W, C), not (W, H, C).
    This class uses the conventional (width, height) representation
    but provides helpers for both orderings.
    """

    width: int
    height: int
    channels: int = 3

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError(
                f"Image dimensions must be positive, got ({self.width}, {self.height})"
            )
        if self.channels not in (1, 3, 4):
            raise ValueError(f"Channels must be 1, 3, or 4, got {self.channels}")

    @property
    def numpy_shape(self) -> tuple[int, int, int]:
        """Returns (H, W, C) — NumPy array shape convention."""
        return (self.height, self.width, self.channels)

    @property
    def total_pixels(self) -> int:
        return self.width * self.height

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height

    def to_tuple(self) -> tuple[int, int]:
        return (self.width, self.height)

    def __repr__(self) -> str:
        return f"ImageDimensions({self.width}×{self.height}, {self.channels}ch)"


@dataclass(frozen=True)
class PolygonPoints:
    """
    An ordered sequence of (x, y) polygon vertices.

    Used for instance segmentation polygon representations.
    Points are in image pixel coordinates.
    Polygon must be closed (but first/last point may differ —
    closure is handled by rendering code).
    """

    points: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        if len(self.points) < 3:
            raise ValueError(
                f"Polygon requires at least 3 points, got {len(self.points)}"
            )
        for i, (x, y) in enumerate(self.points):
            if x < 0 or y < 0:
                raise ValueError(
                    f"Polygon point {i} has negative coordinates: ({x}, {y})"
                )

    @classmethod
    def from_array(cls, arr: NDArray[np.float32]) -> "PolygonPoints":
        """Create from Nx2 numpy array."""
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(f"Expected Nx2 array, got shape {arr.shape}")
        return cls(points=tuple((float(p[0]), float(p[1])) for p in arr))

    def to_array(self) -> NDArray[np.float32]:
        """Convert to Nx2 numpy array."""
        return np.array(self.points, dtype=np.float32)

    @property
    def num_points(self) -> int:
        return len(self.points)

    @property
    def area(self) -> float:
        """
        Polygon area using the Shoelace formula.
        Reference: https://en.wikipedia.org/wiki/Shoelace_formula
        """
        n = len(self.points)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += self.points[i][0] * self.points[j][1]
            area -= self.points[j][0] * self.points[i][1]
        return abs(area) / 2.0

    @property
    def bounding_box(self) -> BoundingBox:
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        return BoundingBox(
            x_min=min(xs),
            y_min=min(ys),
            x_max=max(xs),
            y_max=max(ys),
        )

    def simplify(self, tolerance: float = 2.0) -> "PolygonPoints":
        """
        Simplify polygon using Ramer-Douglas-Peucker algorithm.

        Reduces vertex count while preserving shape within tolerance pixels.
        """
        from shapely.geometry import Polygon as ShapelyPolygon

        shapely_poly = ShapelyPolygon(self.points)
        simplified = shapely_poly.simplify(tolerance, preserve_topology=True)
        coords = list(simplified.exterior.coords)[:-1]  # Remove closing point
        return PolygonPoints(points=tuple((float(x), float(y)) for x, y in coords))

    def __iter__(self) -> Iterator[tuple[float, float]]:
        return iter(self.points)

    def __len__(self) -> int:
        return len(self.points)

    def __repr__(self) -> str:
        return f"PolygonPoints({len(self.points)} vertices, area={self.area:.1f}px²)"


@dataclass(frozen=True)
class Mask:
    """
    A binary segmentation mask.

    Stored as a boolean numpy array of shape (H, W).
    True = foreground (trichome), False = background.

    Memory note: For large images, consider sparse representations.
    1920×1080 mask = ~2MB in uint8, ~250KB if sparse.
    """

    data: NDArray[np.bool_]
    """Binary mask array, shape (H, W)."""

    def __post_init__(self) -> None:
        if self.data.ndim != 2:
            raise ValueError(f"Mask must be 2D, got shape {self.data.shape}")
        if self.data.dtype != np.bool_:
            object.__setattr__(self, "data", self.data.astype(np.bool_))

    @classmethod
    def from_uint8(cls, arr: NDArray[np.uint8]) -> "Mask":
        """Create mask from uint8 array (non-zero = foreground)."""
        return cls(data=(arr > 0))

    @classmethod
    def from_polygon(
        cls, polygon: PolygonPoints, image_dims: ImageDimensions
    ) -> "Mask":
        """Rasterize polygon to binary mask."""
        import cv2

        mask = np.zeros((image_dims.height, image_dims.width), dtype=np.uint8)
        pts = polygon.to_array().astype(np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], 1)
        return cls(data=mask.astype(np.bool_))

    @property
    def height(self) -> int:
        return self.data.shape[0]

    @property
    def width(self) -> int:
        return self.data.shape[1]

    @property
    def area_pixels(self) -> int:
        """Number of foreground pixels."""
        return int(self.data.sum())

    @property
    def bounding_box(self) -> BoundingBox | None:
        """Tight bounding box around mask foreground, or None if mask is empty."""
        rows = np.any(self.data, axis=1)
        cols = np.any(self.data, axis=0)
        if not rows.any():
            return None
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return BoundingBox(
            x_min=float(cmin),
            y_min=float(rmin),
            x_max=float(cmax + 1),
            y_max=float(rmax + 1),
        )

    def iou(self, other: "Mask") -> float:
        """
        Mask IoU — pixel-level intersection over union.

        More precise than bounding box IoU for irregular shapes.
        """
        if self.data.shape != other.data.shape:
            raise ValueError(
                f"Cannot compute IoU between masks of different shapes: "
                f"{self.data.shape} vs {other.data.shape}"
            )
        intersection = np.logical_and(self.data, other.data).sum()
        union = np.logical_or(self.data, other.data).sum()
        return float(intersection / union) if union > 0 else 0.0

    def dice_score(self, other: "Mask") -> float:
        """
        Dice similarity coefficient.

        Dice = 2|A ∩ B| / (|A| + |B|)

        More robust than IoU for imbalanced foreground/background.
        """
        if self.data.shape != other.data.shape:
            raise ValueError("Masks must have same shape for Dice score")
        intersection = np.logical_and(self.data, other.data).sum()
        total = self.data.sum() + other.data.sum()
        return float(2 * intersection / total) if total > 0 else 1.0

    def to_uint8(self) -> NDArray[np.uint8]:
        return (self.data * 255).astype(np.uint8)

    def to_polygon(self, simplify_tolerance: float = 2.0) -> PolygonPoints | None:
        """
        Convert binary mask to polygon via contour extraction.

        Uses OpenCV findContours → takes largest contour.
        Returns None if no valid contour found.
        """
        import cv2

        contours, _ = cv2.findContours(
            self.to_uint8(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None

        # Take largest contour
        largest = max(contours, key=cv2.contourArea)
        if len(largest) < 3:
            return None

        points = largest.reshape(-1, 2)
        polygon = PolygonPoints(
            points=tuple((float(p[0]), float(p[1])) for p in points)
        )
        if simplify_tolerance > 0:
            polygon = polygon.simplify(simplify_tolerance)
        return polygon

    def __repr__(self) -> str:
        return (
            f"Mask(shape={self.data.shape}, "
            f"area={self.area_pixels}px, "
            f"coverage={100*self.area_pixels/(self.height*self.width):.2f}%)"
        )
