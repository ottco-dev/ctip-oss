"""
video_pipeline.domain.motion — Motion estimation and stability analysis.

Detects and quantifies camera/stage motion between consecutive microscopy frames.
Motion estimation is used to:
1. Skip frames during fast motion (unusable for analysis)
2. Identify stable periods suitable for frame extraction
3. Characterize sample drift for temporal tracking
4. Estimate shake/vibration level (quality indicator)

ALGORITHM: Optical flow-based motion estimation.
  - Lucas-Kanade sparse optical flow on Shi-Tomasi corners
  - Fast, CPU-only, works on small ROI
  - Alternative: Phase correlation (frequency domain) for whole-frame motion

OUTPUT:
  - Translation vector (dx, dy) in pixels
  - Rotation angle (degrees)
  - Motion magnitude (pixels/frame)
  - Is_stable flag

Reference:
  Lucas, B.D. & Kanade, T. (1981). An iterative image registration technique
  with an application to stereo vision. IJCAI-81, pp. 674-679.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
from numpy.typing import NDArray


@dataclass
class MotionEstimate:
    """Motion estimate between two consecutive frames."""

    dx: float
    """Horizontal translation in pixels (positive = right)."""

    dy: float
    """Vertical translation in pixels (positive = down)."""

    rotation_deg: float
    """Estimated rotation in degrees."""

    magnitude: float
    """Euclidean magnitude of translation: √(dx² + dy²)."""

    n_tracked_points: int
    """Number of feature points used for estimation (reliability indicator)."""

    confidence: float
    """Confidence in the estimate [0, 1]. Low if few points tracked."""

    @property
    def is_static(self) -> bool:
        """True if motion is below 2-pixel threshold (essentially static)."""
        return self.magnitude < 2.0

    @property
    def is_highly_dynamic(self) -> bool:
        """True if motion is > 20 pixels (frame likely unusable)."""
        return self.magnitude > 20.0

    @property
    def direction(self) -> str:
        """Cardinal direction of dominant motion."""
        if self.magnitude < 1.0:
            return "static"
        angle = float(np.degrees(np.arctan2(self.dy, self.dx)))
        if -45 <= angle < 45:
            return "right"
        elif 45 <= angle < 135:
            return "down"
        elif angle >= 135 or angle < -135:
            return "left"
        else:
            return "up"


def estimate_motion(
    frame_prev: NDArray[np.uint8],
    frame_curr: NDArray[np.uint8],
    *,
    max_points: int = 200,
    quality_level: float = 0.01,
    min_distance: float = 10.0,
) -> MotionEstimate:
    """
    Estimate motion between two consecutive frames using sparse optical flow.

    Args:
        frame_prev:    Previous frame (RGB or grayscale, uint8).
        frame_curr:    Current frame (RGB or grayscale, uint8).
        max_points:    Maximum number of feature points to track.
        quality_level: Shi-Tomasi corner quality threshold.
        min_distance:  Minimum distance between detected corners.

    Returns:
        MotionEstimate with translation, rotation, magnitude, confidence.
    """
    # Convert to grayscale
    def _gray(f: NDArray[np.uint8]) -> NDArray[np.uint8]:
        if f.ndim == 3:
            return cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
        return f

    gray_prev = _gray(frame_prev)
    gray_curr = _gray(frame_curr)

    # Detect feature points in previous frame
    corners = cv2.goodFeaturesToTrack(
        gray_prev,
        maxCorners=max_points,
        qualityLevel=quality_level,
        minDistance=min_distance,
        blockSize=7,
    )

    if corners is None or len(corners) < 4:
        return MotionEstimate(
            dx=0.0, dy=0.0, rotation_deg=0.0,
            magnitude=0.0, n_tracked_points=0, confidence=0.0,
        )

    # Lucas-Kanade optical flow
    lk_params = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
        gray_prev, gray_curr, corners, None, **lk_params
    )

    if next_pts is None or status is None:
        return MotionEstimate(
            dx=0.0, dy=0.0, rotation_deg=0.0,
            magnitude=0.0, n_tracked_points=0, confidence=0.0,
        )

    # Filter to successfully tracked points
    mask = status.ravel() == 1
    prev_good = corners[mask].reshape(-1, 2)
    curr_good = next_pts[mask].reshape(-1, 2)

    n_tracked = int(mask.sum())
    confidence = min(1.0, n_tracked / max(max_points * 0.3, 10))

    if n_tracked < 4:
        return MotionEstimate(
            dx=0.0, dy=0.0, rotation_deg=0.0,
            magnitude=0.0, n_tracked_points=n_tracked, confidence=confidence,
        )

    # Estimate affine transform (translation + rotation)
    transform, inlier_mask = cv2.estimateAffinePartial2D(
        prev_good, curr_good,
        method=cv2.RANSAC,
        ransacReprojThreshold=2.0,
    )

    if transform is None:
        # Fall back to mean displacement
        disp = curr_good - prev_good
        dx = float(np.median(disp[:, 0]))
        dy = float(np.median(disp[:, 1]))
        magnitude = float(np.sqrt(dx ** 2 + dy ** 2))
        return MotionEstimate(
            dx=dx, dy=dy, rotation_deg=0.0,
            magnitude=magnitude, n_tracked_points=n_tracked, confidence=confidence * 0.5,
        )

    dx = float(transform[0, 2])
    dy = float(transform[1, 2])
    # Extract rotation from 2×2 upper-left rotation matrix
    rotation_deg = float(np.degrees(np.arctan2(transform[1, 0], transform[0, 0])))
    magnitude = float(np.sqrt(dx ** 2 + dy ** 2))

    # Boost confidence if RANSAC found many inliers
    if inlier_mask is not None:
        n_inliers = int(inlier_mask.sum())
        confidence = float(np.clip(n_inliers / max(n_tracked, 1), 0.0, 1.0))

    return MotionEstimate(
        dx=dx,
        dy=dy,
        rotation_deg=rotation_deg,
        magnitude=magnitude,
        n_tracked_points=n_tracked,
        confidence=confidence,
    )


def classify_motion_sequence(
    motions: List[MotionEstimate],
) -> dict:
    """
    Classify a sequence of motion estimates.

    Returns summary statistics useful for video quality assessment.
    """
    if not motions:
        return {}

    magnitudes = [m.magnitude for m in motions]
    return {
        "mean_magnitude_px": float(np.mean(magnitudes)),
        "max_magnitude_px": float(np.max(magnitudes)),
        "std_magnitude_px": float(np.std(magnitudes)),
        "n_static_frames": sum(1 for m in motions if m.is_static),
        "n_dynamic_frames": sum(1 for m in motions if m.is_highly_dynamic),
        "static_fraction": sum(1 for m in motions if m.is_static) / len(motions),
        "overall_drift_px": float(
            np.sqrt(
                sum(m.dx for m in motions) ** 2
                + sum(m.dy for m in motions) ** 2
            )
        ),
    }
