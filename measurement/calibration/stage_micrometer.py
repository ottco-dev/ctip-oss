"""
measurement.calibration.stage_micrometer — Microscope calibration module.

CALIBRATION IS CRITICAL:
Without accurate pixel-to-micrometer calibration, all size measurements
are meaningless. This module implements multiple calibration methods
suited to different lab setups.

CALIBRATION METHODS:

1. Stage Micrometer Calibration (Gold Standard):
   - Use a certified stage micrometer (e.g., 1mm/100 divisions = 10µm/div)
   - Image the stage micrometer at the same magnification as your samples
   - Measure the pixel distance corresponding to a known physical distance
   - Compute µm/pixel = known_distance_µm / measured_pixels

2. Reference Object Calibration:
   - Use an object of known size in the same field of view
   - e.g., a pollen grain (~10µm) or a known reference bead
   - Less accurate than stage micrometer (reference object measurement error)

3. Objective-Based Estimation (Approximate):
   - Use microscope objective magnification + sensor pixel size
   - µm/pixel = (pixel_size_µm / objective_magnification) / digital_zoom
   - Accuracy: ±20-30% (acceptable only for approximate analysis)
   - Requires knowing camera sensor specs

4. Known Trichome Size Reference (Domain-Specific):
   - Use capitate-stalked trichome head as size reference
   - Head diameter typically 150-500µm (mean ~287µm for high-cannabinoid strains)
   - HIGH UNCERTAINTY — only use as last resort

UNCERTAINTY PROPAGATION:
All calibrated measurements carry uncertainty from:
- Measurement error in calibration (typically ±2-5 pixels)
- Sub-pixel resolution limits
- Distortion at image edges
- Temperature effects on optics (minor)

Reference:
  Murphy, D.B. & Davidson, M.W. (2012). Fundamentals of Light Microscopy
  and Electronic Imaging. John Wiley & Sons, 2nd ed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from shared.core.value_objects import CalibrationScale


@dataclass
class MicroscopeProfile:
    """
    Complete microscope calibration profile.

    One profile per objective/magnification/camera combination.
    Profiles should be stored and reused — not recomputed for every session.
    """

    profile_id: str
    name: str
    description: str = ""

    # Hardware specifications
    objective_magnification: float = 40.0
    """e.g., 10, 20, 40, 100"""

    digital_zoom: float = 1.0
    camera_model: str = ""
    sensor_pixel_size_um: float | None = None
    """Physical pixel size in µm (from camera spec sheet)"""

    # Calibration result
    um_per_pixel: float = 0.0
    """Calibrated scale factor. This is the critical value."""

    uncertainty_um_per_pixel: float = 0.0
    """±1 standard deviation in µm/px from calibration measurement."""

    calibration_method: str = "unknown"
    """One of: stage_micrometer, reference_object, objective_estimate, domain_reference"""

    calibration_date: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    calibrated_by: str = ""

    # Validation
    num_measurements: int = 0
    """Number of independent measurements used to compute µm/pixel."""

    measurement_cv: float | None = None
    """Coefficient of variation of calibration measurements (%)."""

    # Notes
    notes: str = ""

    @property
    def is_calibrated(self) -> bool:
        return self.um_per_pixel > 0

    @property
    def relative_uncertainty_percent(self) -> float:
        if self.um_per_pixel == 0:
            return float("inf")
        return 100.0 * self.uncertainty_um_per_pixel / self.um_per_pixel

    def to_calibration_scale(self) -> CalibrationScale:
        """Convert to CalibrationScale value object."""
        return CalibrationScale(
            um_per_pixel=self.um_per_pixel,
            uncertainty_um_per_pixel=self.uncertainty_um_per_pixel,
            objective_magnification=self.objective_magnification,
            digital_zoom=self.digital_zoom,
            source=f"{self.calibration_method}/{self.profile_id}",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "name": self.name,
            "um_per_pixel": self.um_per_pixel,
            "uncertainty_um_per_pixel": self.uncertainty_um_per_pixel,
            "relative_uncertainty_percent": self.relative_uncertainty_percent,
            "calibration_method": self.calibration_method,
            "objective_magnification": self.objective_magnification,
            "num_measurements": self.num_measurements,
            "calibration_date": self.calibration_date,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "MicroscopeProfile":
        with open(path) as f:
            data = json.load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def __repr__(self) -> str:
        return (
            f"MicroscopeProfile('{self.name}', "
            f"{self.um_per_pixel:.4f} µm/px ±{self.relative_uncertainty_percent:.1f}%, "
            f"method={self.calibration_method})"
        )


class StageMicrometerCalibrator:
    """
    Interactive stage micrometer calibration.

    Workflow:
    1. Image a certified stage micrometer at target magnification
    2. User marks two points on the micrometer (or provides pixel distance)
    3. User provides known physical distance (e.g., "10 divisions × 10 µm/div = 100 µm")
    4. Compute µm/px and uncertainty from multiple measurements
    5. Save as MicroscopeProfile

    Uncertainty estimation:
    Repeat measurements with different micrometer lines and average.
    CV < 2% = excellent calibration
    CV 2-5% = acceptable
    CV > 5% = recalibrate

    For automated calibration (no user interaction):
    - Use automatic line detection on stage micrometer image
    - Measure all visible lines
    - Compute µm/px from line spacing statistics
    """

    def __init__(self, known_spacing_um: float = 10.0) -> None:
        """
        Args:
            known_spacing_um: Physical distance per stage micrometer division (µm).
                Common: 10 µm/div (1mm/100div stage micrometer)
        """
        self.known_spacing_um = known_spacing_um
        self._measurements: list[float] = []  # List of measured pixel distances

    def add_measurement(self, pixel_distance: float) -> None:
        """
        Record a single calibration measurement.

        Args:
            pixel_distance: Number of pixels corresponding to known_spacing_um.
        """
        if pixel_distance <= 0:
            raise ValueError(f"Pixel distance must be positive, got {pixel_distance}")
        self._measurements.append(pixel_distance)

    def add_two_point_measurement(
        self,
        point1: tuple[float, float],
        point2: tuple[float, float],
        num_divisions: float = 1.0,
    ) -> None:
        """
        Compute pixel distance from two points and record measurement.

        Args:
            point1: First point (x, y) in pixel coordinates
            point2: Second point (x, y) in pixel coordinates
            num_divisions: Number of stage micrometer divisions between points.
        """
        pixel_dist = np.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)
        # pixel_dist corresponds to (num_divisions × known_spacing_um) µm
        # So per-division pixel count:
        per_division_pixels = pixel_dist / num_divisions
        self._measurements.append(per_division_pixels)

    def compute_calibration(
        self,
        profile_name: str,
        objective_magnification: float = 40.0,
    ) -> MicroscopeProfile:
        """
        Compute calibration from all recorded measurements.

        Returns:
            MicroscopeProfile with computed µm/pixel and uncertainty.

        Raises:
            ValueError: If no measurements recorded.
        """
        if not self._measurements:
            raise ValueError(
                "No measurements recorded. "
                "Call add_measurement() or add_two_point_measurement() first."
            )

        measurements = np.array(self._measurements)
        mean_px = float(measurements.mean())
        std_px = float(measurements.std()) if len(measurements) > 1 else 0.0

        # µm/pixel = known_spacing_um / mean_pixels_per_division
        um_per_pixel = self.known_spacing_um / mean_px

        # Uncertainty propagation: σ(µm/px) = (known_spacing / mean_px²) × σ_px
        uncertainty = (self.known_spacing_um / mean_px ** 2) * std_px if len(measurements) > 1 else um_per_pixel * 0.03

        cv = float(100 * std_px / mean_px) if mean_px > 0 else float("inf")

        return MicroscopeProfile(
            profile_id=f"cal_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            name=profile_name,
            objective_magnification=objective_magnification,
            um_per_pixel=um_per_pixel,
            uncertainty_um_per_pixel=uncertainty,
            calibration_method="stage_micrometer",
            num_measurements=len(measurements),
            measurement_cv=cv,
            notes=f"Calibrated with {self.known_spacing_um} µm/div stage micrometer. "
                  f"n={len(measurements)} measurements, CV={cv:.2f}%",
        )


def estimate_scale_from_objective(
    objective_magnification: float,
    digital_zoom: float = 1.0,
    sensor_pixel_size_um: float = 2.4,  # Sony IMX477 common sensor pixel
    camera_binning: int = 1,
) -> MicroscopeProfile:
    """
    Approximate µm/pixel from objective magnification and sensor specs.

    Formula: µm/pixel = (sensor_pixel_size_µm × camera_binning) / (objective_mag × digital_zoom)

    ACCURACY WARNING:
    This is an APPROXIMATION (±20-30% error possible) due to:
    - Lens aberrations not accounted for
    - Camera adapter magnification may not be 1×
    - Digital zoom accuracy varies
    - C-mount vs. phototube differences

    Use only for rough estimates. Prefer stage_micrometer calibration.

    Args:
        objective_magnification: Objective lens magnification (e.g., 40)
        digital_zoom: Digital magnification factor
        sensor_pixel_size_um: Physical pixel size of camera sensor (µm)
        camera_binning: Pixel binning factor (1=no binning)

    Returns:
        MicroscopeProfile with estimated scale and high uncertainty.
    """
    total_magnification = objective_magnification * digital_zoom
    um_per_pixel = (sensor_pixel_size_um * camera_binning) / total_magnification
    uncertainty = um_per_pixel * 0.25  # 25% uncertainty for objective estimation

    return MicroscopeProfile(
        profile_id=f"obj_est_{int(objective_magnification)}x",
        name=f"{int(objective_magnification)}× objective estimate",
        description="APPROXIMATE — use stage micrometer for accurate measurements",
        objective_magnification=objective_magnification,
        digital_zoom=digital_zoom,
        sensor_pixel_size_um=sensor_pixel_size_um,
        um_per_pixel=um_per_pixel,
        uncertainty_um_per_pixel=uncertainty,
        calibration_method="objective_estimate",
        notes=(
            f"Estimated from: {sensor_pixel_size_um}µm pixel / {total_magnification}× total mag. "
            f"Uncertainty: ±25%. Use stage micrometer calibration for accurate measurements."
        ),
    )


# ---------------------------------------------------------------------------
# Automated scale-bar detection (TDB-001 — wires Hough-line detection)
# ---------------------------------------------------------------------------

from dataclasses import dataclass as _dataclass
from typing import Optional as _Optional


@_dataclass
class ScaleBarDetectionResult:
    """Result of automated scale-bar detection from a micrometer image."""

    detected: bool
    """True if a convincing scale bar was found."""

    scale_bar_px: float
    """Detected scale bar length in pixels (0.0 if not detected)."""

    confidence: float
    """Detection confidence in [0, 1]. < 0.5 = unreliable, use manual input."""

    num_line_groups: int
    """Number of distinct line groups found (for QC)."""

    method: str
    """Which sub-algorithm produced the result."""

    message: str
    """Human-readable summary or error description."""


def detect_scale_bar_px(
    image_gray: "np.ndarray",
    *,
    min_line_length_frac: float = 0.05,
    max_line_gap_frac: float = 0.01,
    canny_low: int = 30,
    canny_high: int = 100,
    hough_threshold: int = 50,
    max_angle_deg: float = 5.0,
) -> ScaleBarDetectionResult:
    """
    Automatically detect a scale bar in a grayscale stage micrometer image.

    Algorithm
    ---------
    1. CLAHE contrast enhancement (deals with faint micrometer lines).
    2. Gaussian blur to suppress noise.
    3. Canny edge detection.
    4. Probabilistic Hough line transform to find line segments.
    5. Filter to near-horizontal lines (|angle| < max_angle_deg).
    6. Cluster co-linear segments and measure the span of the longest cluster.
    7. Report the pixel length of the longest contiguous segment cluster.

    This is well-suited to bright-field stage micrometer images where the
    calibration lines are horizontal dark marks on a uniform background.

    Parameters
    ----------
    image_gray : ndarray
        Grayscale image (H, W) uint8.
    min_line_length_frac : float
        Minimum line segment length as a fraction of image width.
    max_line_gap_frac : float
        Maximum gap between collinear segments (fraction of image width).
    canny_low, canny_high : int
        Canny hysteresis thresholds.
    hough_threshold : int
        Hough accumulator vote threshold.
    max_angle_deg : float
        Maximum deviation from horizontal allowed for detected lines.

    Returns
    -------
    ScaleBarDetectionResult
    """
    try:
        import cv2
    except ImportError:
        return ScaleBarDetectionResult(
            detected=False,
            scale_bar_px=0.0,
            confidence=0.0,
            num_line_groups=0,
            method="none",
            message="cv2 not available — install opencv-python",
        )

    h, w = image_gray.shape[:2]
    min_len = max(10, int(min_line_length_frac * w))
    max_gap = max(2, int(max_line_gap_frac * w))
    max_angle_rad = float(np.deg2rad(max_angle_deg))

    # ── 1. Contrast enhance ────────────────────────────────────────────────
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(image_gray)

    # ── 2. Noise reduction ─────────────────────────────────────────────────
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

    # ── 3. Canny edges ─────────────────────────────────────────────────────
    edges = cv2.Canny(blurred, canny_low, canny_high, apertureSize=3)

    # ── 4. Probabilistic Hough ─────────────────────────────────────────────
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=hough_threshold,
        minLineLength=min_len,
        maxLineGap=max_gap,
    )

    if lines is None or len(lines) == 0:
        return ScaleBarDetectionResult(
            detected=False,
            scale_bar_px=0.0,
            confidence=0.0,
            num_line_groups=0,
            method="hough",
            message="No line segments detected. Check image contrast and orientation.",
        )

    # ── 5. Filter to near-horizontal lines ─────────────────────────────────
    horizontal_segments: list[tuple[int, int, int, int]] = []
    for seg in lines:
        x1, y1, x2, y2 = seg[0]
        angle = float(np.abs(np.arctan2(y2 - y1, x2 - x1)))
        if angle <= max_angle_rad:
            horizontal_segments.append((x1, y1, x2, y2))

    if not horizontal_segments:
        return ScaleBarDetectionResult(
            detected=False,
            scale_bar_px=0.0,
            confidence=0.0,
            num_line_groups=0,
            method="hough",
            message=(
                f"Lines detected but none are near-horizontal (max angle {max_angle_deg}°). "
                "Rotate the image or increase max_angle_deg."
            ),
        )

    # ── 6. Cluster by Y position (within 10 px) and merge spans ───────────
    # Sort by Y centre, then cluster rows within 10 pixels of each other.
    segs_sorted = sorted(horizontal_segments, key=lambda s: (s[1] + s[3]) // 2)

    clusters: list[list[tuple[int, int, int, int]]] = []
    current_cluster: list[tuple[int, int, int, int]] = [segs_sorted[0]]
    current_y = (segs_sorted[0][1] + segs_sorted[0][3]) // 2

    for seg in segs_sorted[1:]:
        y_mid = (seg[1] + seg[3]) // 2
        if abs(y_mid - current_y) <= 10:
            current_cluster.append(seg)
        else:
            clusters.append(current_cluster)
            current_cluster = [seg]
            current_y = y_mid
    clusters.append(current_cluster)

    # For each cluster, compute the total span (min_x → max_x)
    cluster_spans: list[float] = []
    for cluster in clusters:
        all_x = [s[0] for s in cluster] + [s[2] for s in cluster]
        span = float(max(all_x) - min(all_x))
        cluster_spans.append(span)

    longest_span = max(cluster_spans)

    # ── 7. Compute confidence ──────────────────────────────────────────────
    # Heuristic: confidence scales with span relative to image width.
    # A scale bar spanning > 50% of the image = high confidence.
    span_ratio = longest_span / w
    confidence = float(np.clip(span_ratio * 2, 0.0, 1.0))

    if longest_span < min_len:
        return ScaleBarDetectionResult(
            detected=False,
            scale_bar_px=longest_span,
            confidence=confidence,
            num_line_groups=len(clusters),
            method="hough",
            message=(
                f"Detected span ({longest_span:.0f}px) too short. "
                "Increase min_line_length_frac or check image."
            ),
        )

    return ScaleBarDetectionResult(
        detected=True,
        scale_bar_px=longest_span,
        confidence=confidence,
        num_line_groups=len(clusters),
        method="hough",
        message=(
            f"Scale bar detected: {longest_span:.1f} px "
            f"({len(horizontal_segments)} segments in {len(clusters)} row-clusters, "
            f"confidence={confidence:.2f})"
        ),
    )
