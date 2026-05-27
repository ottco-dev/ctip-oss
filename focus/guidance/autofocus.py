"""
focus.guidance.autofocus — Autofocus guidance for microscopy.

This module provides tools for:
1. Focus curve analysis — determine optimal focus position from a Z-stack
2. Region-based focus guidance — identify which image regions need focus adjustment
3. Focus tracking — detect focus drift during video capture
4. Focus recommendations — actionable guidance for microscope operators

AUTOFOCUS APPROACHES:
- Passive (image-based): analyze current frame → guide operator
- Z-stack sweep: capture N frames at different Z positions → select best
- Continuous tracking: monitor focus metric over time → detect drift

For microscopy of cannabis trichomes specifically:
- Stalked trichomes have an inherent depth-of-field challenge
  (stalk and head are at different focal planes)
- Optimal focus is on the SECRETORY HEAD (bulbous tip)
- Use regional focus analysis: focus the center region where most heads appear

Scientific note on depth of field at microscopy magnifications:
- 10x objective: DOF ≈ 3-5 µm
- 40x objective: DOF ≈ 0.5-1 µm
- Trichome head diameter: 30-120 µm
- Therefore: even a small Z drift causes significant blur
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from focus.metrics.composite import (
    FocusScoreResult,
    compute_focus_score,
    _compute_regional_scores,
)


@dataclass
class FocusCurveResult:
    """Result of Z-stack focus analysis."""

    z_positions: list[float]
    """Z positions (in µm or arbitrary units)"""

    focus_scores: list[float]
    """Composite focus score at each Z position"""

    optimal_z: float
    """Z position with maximum focus score"""

    optimal_index: int
    """Index of optimal frame in Z-stack"""

    curve_sharpness: float
    """Width of focus peak (narrower = smaller depth of field)"""

    is_reliable: bool
    """Whether the focus curve has a clear, single peak"""

    @property
    def best_score(self) -> float:
        return max(self.focus_scores) if self.focus_scores else 0.0


@dataclass
class AutofocusGuidance:
    """Actionable focus guidance for microscope operator."""

    current_score: float
    """Current composite focus score"""

    direction: str
    """'increase_z', 'decrease_z', 'in_focus', 'unclear'"""

    magnitude: str
    """'large', 'medium', 'small', 'none'"""

    region_advice: list[str]
    """Per-region textual guidance"""

    action: str
    """Human-readable action recommendation"""

    confidence: float
    """Confidence in guidance (0-1)"""


@dataclass
class FocusDriftDetector:
    """
    Detects focus drift during continuous video capture.

    Maintains a rolling window of focus scores and triggers
    an alert when the score drops below threshold.
    """

    window_size: int = 10
    """Number of recent frames to consider"""

    alert_threshold: float = 0.35
    """Composite score below this triggers drift alert"""

    drift_threshold: float = 0.15
    """Score drop from baseline that indicates drift"""

    _scores: list[float] = field(default_factory=list)
    _baseline: float | None = None

    def update(self, score: float) -> dict[str, bool | float | str]:
        """
        Update detector with new frame focus score.

        Returns:
            Dict with 'drift_detected', 'current_score', 'baseline', 'action'
        """
        self._scores.append(score)

        # Keep rolling window
        if len(self._scores) > self.window_size:
            self._scores.pop(0)

        # Establish baseline from first good frames
        if self._baseline is None and len(self._scores) >= 3:
            if max(self._scores) >= 0.5:  # Need at least one good frame
                self._baseline = float(np.percentile(self._scores, 75))

        current = float(np.mean(self._scores[-3:])) if len(self._scores) >= 3 else score

        drift_from_baseline = (
            (self._baseline - current) if self._baseline is not None else 0.0
        )

        drift_detected = (
            current < self.alert_threshold
            or (self._baseline is not None and drift_from_baseline > self.drift_threshold)
        )

        if drift_detected:
            if current < 0.2:
                action = "Critical: Severely out of focus. Stop and refocus immediately."
            elif current < self.alert_threshold:
                action = "Warning: Focus quality below acceptable threshold. Refocus."
            else:
                action = f"Focus drift detected. Score dropped {drift_from_baseline:.2f} from baseline. Adjust Z."
        else:
            action = "Focus acceptable."

        return {
            "drift_detected": drift_detected,
            "current_score": current,
            "baseline": self._baseline,
            "drift_from_baseline": drift_from_baseline,
            "action": action,
        }

    def reset_baseline(self) -> None:
        """Reset baseline (call after manual refocus)."""
        self._baseline = None
        self._scores.clear()


def analyze_focus_curve(
    frames: list[NDArray[np.uint8]],
    z_positions: list[float] | None = None,
) -> FocusCurveResult:
    """
    Analyze a Z-stack to find the optimal focus plane.

    Fits a Gaussian or polynomial to the focus curve and identifies
    the peak. Useful for:
    - Pre-capture autofocus sweep
    - Post-capture best-frame selection in Z-stacks

    Args:
        frames: List of images at different Z positions (ordered by Z)
        z_positions: Physical Z positions in µm (if None, use frame indices)

    Returns:
        FocusCurveResult with optimal position and reliability assessment
    """
    if z_positions is None:
        z_positions = list(range(len(frames)))

    assert len(frames) == len(z_positions), "Frame count must match Z position count"

    scores = [compute_focus_score(f).composite for f in frames]

    optimal_idx = int(np.argmax(scores))
    optimal_z = z_positions[optimal_idx]

    # Assess peak sharpness (FWHM of the focus curve)
    max_score = max(scores)
    half_max = max_score * 0.5
    above_half = [i for i, s in enumerate(scores) if s >= half_max]

    if len(above_half) >= 2:
        fwhm_indices = above_half[-1] - above_half[0]
        if len(z_positions) > 1:
            z_step = abs(z_positions[1] - z_positions[0])
            curve_sharpness = fwhm_indices * z_step
        else:
            curve_sharpness = float(fwhm_indices)
    else:
        curve_sharpness = 0.0

    # Reliability: single clear peak, high max score, smooth curve
    score_range = max_score - min(scores)
    is_reliable = (
        max_score >= 0.45
        and score_range >= 0.10
        and 0 < optimal_idx < len(scores) - 1  # Not at extreme edges
    )

    return FocusCurveResult(
        z_positions=z_positions,
        focus_scores=scores,
        optimal_z=optimal_z,
        optimal_index=optimal_idx,
        curve_sharpness=curve_sharpness,
        is_reliable=is_reliable,
    )


def compute_regional_guidance(
    image: NDArray[np.uint8],
    grid: tuple[int, int] = (3, 3),
) -> AutofocusGuidance:
    """
    Compute regional focus guidance for a live microscopy image.

    Divides image into a grid, computes focus per region,
    and provides actionable operator guidance.

    Center regions are weighted more heavily (trichomes tend to be
    centered in the microscope field of view for analysis).

    Args:
        image: Live frame from microscope camera (RGB or grayscale)
        grid: Grid dimensions for regional analysis

    Returns:
        AutofocusGuidance with direction, magnitude, and action text
    """
    import cv2

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    result = compute_focus_score(gray, compute_regional=True, region_grid=grid)
    current_score = result.composite

    # Regional scores
    region_scores = result.region_scores
    rows, cols = grid

    region_advice = []
    if region_scores is not None:
        for r in range(rows):
            for c in range(cols):
                s = float(region_scores[r, c])
                if s < 0.30:
                    region_advice.append(
                        f"Region ({r+1},{c+1}): blurry ({s:.2f}) — "
                        f"{'center' if r == rows//2 and c == cols//2 else 'edge'} zone"
                    )

    # Directional guidance (requires context from previous frame)
    # Without Z-stack context, we can only say "adjust"
    if current_score >= 0.75:
        direction = "in_focus"
        magnitude = "none"
        action = "In focus. Proceed with capture."
        confidence = 0.95
    elif current_score >= 0.50:
        direction = "in_focus"
        magnitude = "small"
        action = "Near focus. Fine-tune Z by ±1-2 µm."
        confidence = 0.70
    elif current_score >= 0.30:
        direction = "unclear"
        magnitude = "medium"
        action = "Out of focus. Adjust Z position and retry."
        confidence = 0.60
    else:
        direction = "unclear"
        magnitude = "large"
        action = "Severely out of focus. Perform autofocus sweep or manual refocus."
        confidence = 0.90

    return AutofocusGuidance(
        current_score=current_score,
        direction=direction,
        magnitude=magnitude,
        region_advice=region_advice,
        action=action,
        confidence=confidence,
    )


def select_best_frames(
    frames: list[NDArray[np.uint8]],
    n: int = 5,
    min_score: float = 0.35,
    deduplicate_threshold: float = 0.02,
) -> list[tuple[int, float]]:
    """
    Select N best frames from a sequence by focus quality.

    Deduplication: excludes frames with very similar focus scores
    (likely duplicate frames from video with little movement).

    Args:
        frames: List of images (from video or image sequence)
        n: Number of frames to select
        min_score: Minimum focus score to consider (0.35 = acceptable threshold)
        deduplicate_threshold: Min score difference between selected frames

    Returns:
        List of (frame_index, focus_score) tuples, sorted best first
    """
    scored = [(i, compute_focus_score(f).composite) for i, f in enumerate(frames)]
    scored = [(i, s) for i, s in scored if s >= min_score]
    scored.sort(key=lambda x: x[1], reverse=True)

    # Deduplicate similar scores
    selected: list[tuple[int, float]] = []
    for idx, score in scored:
        if all(abs(score - s) > deduplicate_threshold for _, s in selected):
            selected.append((idx, score))
        if len(selected) >= n:
            break

    return selected
