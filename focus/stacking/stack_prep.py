"""
focus.stacking.stack_prep — Focus stack preparation utilities.

Focus stacking is a computational photography technique that combines
multiple images taken at different focal depths to produce an
extended-depth-of-field (EDF) result with everything in focus.

For trichome microscopy, focus stacking is particularly valuable because:
- At 10x-40x magnification, depth of field is extremely shallow (1-5 µm)
- Stalked trichomes have 3D structure spanning multiple focal planes
- The secretory head and stalk are rarely simultaneously in focus
- EDF images enable accurate stalk length + head size measurement

PIPELINE:
1. Frame selection: filter frames suitable for stacking
2. Quality assessment: score each frame for focus + exposure + noise
3. Alignment: detect if frames need registration (drift/vibration)
4. Stack ordering: organize frames by Z position for stacking algorithms
5. Layer assignment: map each frame to its sharpest image regions

Note: This module handles PREPARATION only.
Actual focus stacking computation should be done with:
- Helicon Focus (commercial)
- Zerene Stacker (commercial)
- focus-stack (OSS, https://github.com/PetteriAimonen/focus-stack)
- OpenCV custom implementation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from focus.metrics.composite import compute_focus_score, FocusScoreResult


@dataclass
class StackFrame:
    """A single frame in a focus stack sequence."""

    index: int
    """Frame index in original sequence"""

    path: Path | None
    """Source file path (if from disk)"""

    focus_result: FocusScoreResult
    """Focus quality assessment"""

    exposure_ok: bool
    """Whether exposure is acceptable for stacking"""

    sharpest_regions: NDArray[np.float32] | None
    """Per-region sharpness map (for layer-based stacking)"""

    z_position: float | None = None
    """Z position in µm (if known from microscope metadata)"""

    @property
    def is_usable(self) -> bool:
        return self.focus_result.is_acceptable and self.exposure_ok

    @property
    def composite_score(self) -> float:
        return self.focus_result.composite


@dataclass
class StackPrepResult:
    """Complete focus stack preparation result."""

    frames: list[StackFrame]
    """All analyzed frames"""

    usable_frames: list[StackFrame]
    """Frames passing quality thresholds"""

    recommended_stack: list[StackFrame]
    """Recommended subset for focus stacking"""

    n_total: int
    estimated_z_range: float | None
    """Estimated Z range covered (µm), if positions available"""

    alignment_required: bool
    """Whether frames need alignment before stacking"""

    warnings: list[str] = field(default_factory=list)


def assess_exposure(
    image: NDArray[np.uint8],
    min_mean: float = 30.0,
    max_mean: float = 220.0,
    saturation_threshold: float = 0.01,
) -> tuple[bool, dict[str, float]]:
    """
    Assess if image exposure is suitable for focus stacking.

    Over-exposed or under-exposed frames degrade stacking quality.
    Saturated pixels contain no useful detail information.

    Args:
        image: RGB or grayscale image
        min_mean: Minimum acceptable mean intensity (too dark)
        max_mean: Maximum acceptable mean intensity (too bright)
        saturation_threshold: Maximum fraction of saturated pixels

    Returns:
        (exposure_ok, stats_dict)
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    mean_val = float(gray.mean())
    std_val = float(gray.std())
    saturated_fraction = float((gray >= 250).mean())
    dark_fraction = float((gray <= 5).mean())

    exposure_ok = (
        min_mean <= mean_val <= max_mean
        and saturated_fraction <= saturation_threshold
        and dark_fraction <= 0.10
    )

    return exposure_ok, {
        "mean": mean_val,
        "std": std_val,
        "saturated_fraction": saturated_fraction,
        "dark_fraction": dark_fraction,
    }


def detect_misalignment(
    frame1: NDArray[np.uint8],
    frame2: NDArray[np.uint8],
    max_shift_px: float = 10.0,
) -> tuple[bool, float]:
    """
    Detect if two frames are misaligned (shifted) using phase correlation.

    Uses FFT-based phase correlation for fast, subpixel-accurate shift detection.
    Returns True if the frames are misaligned beyond max_shift_px.

    Args:
        frame1: Reference frame (grayscale or RGB)
        frame2: Target frame (grayscale or RGB)
        max_shift_px: Maximum acceptable shift in pixels

    Returns:
        (misaligned, shift_magnitude_px)
    """
    def to_gray(img: NDArray) -> NDArray[np.uint8]:
        if img.ndim == 3:
            return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return img.copy()

    g1 = to_gray(frame1).astype(np.float32)
    g2 = to_gray(frame2).astype(np.float32)

    # Resize to same size if needed
    if g1.shape != g2.shape:
        g2 = cv2.resize(g2, (g1.shape[1], g1.shape[0]))

    # Phase correlation
    shift, response = cv2.phaseCorrelate(g1, g2)
    shift_mag = float(np.sqrt(shift[0] ** 2 + shift[1] ** 2))

    return shift_mag > max_shift_px, shift_mag


def prepare_focus_stack(
    images: list[NDArray[np.uint8]],
    z_positions: list[float] | None = None,
    min_focus_score: float = 0.30,
    max_frames: int = 15,
    check_alignment: bool = True,
) -> StackPrepResult:
    """
    Prepare a set of images for focus stacking.

    Performs:
    1. Focus quality assessment per frame
    2. Exposure assessment per frame
    3. Usability filtering
    4. Frame selection (best N frames)
    5. Alignment check (optional)

    Args:
        images: List of images at different Z positions
        z_positions: Z positions in µm (optional; use indices if None)
        min_focus_score: Minimum focus score for a frame to be usable
        max_frames: Maximum number of frames in final stack
        check_alignment: Whether to check for inter-frame misalignment

    Returns:
        StackPrepResult with assessment and recommendations
    """
    warnings: list[str] = []

    if not images:
        return StackPrepResult(
            frames=[], usable_frames=[], recommended_stack=[],
            n_total=0, estimated_z_range=None,
            alignment_required=False, warnings=["No images provided"]
        )

    # Assign Z positions
    if z_positions is None:
        z_pos_list = [float(i) for i in range(len(images))]
    else:
        if len(z_positions) != len(images):
            warnings.append(
                f"Z positions count ({len(z_positions)}) != frames ({len(images)}). Using indices."
            )
            z_pos_list = [float(i) for i in range(len(images))]
        else:
            z_pos_list = list(z_positions)

    # Assess each frame
    all_frames: list[StackFrame] = []
    for idx, (img, z) in enumerate(zip(images, z_pos_list)):
        focus_result = compute_focus_score(img, compute_regional=True, region_grid=(4, 4))
        exposure_ok, _ = assess_exposure(img)

        all_frames.append(StackFrame(
            index=idx,
            path=None,
            focus_result=focus_result,
            exposure_ok=exposure_ok,
            sharpest_regions=focus_result.region_scores,
            z_position=z,
        ))

    # Filter usable frames
    usable = [f for f in all_frames if f.composite_score >= min_focus_score and f.exposure_ok]

    if len(usable) == 0:
        warnings.append(
            f"No usable frames found (min_score={min_focus_score}). "
            "Lower min_focus_score or check image quality."
        )
        return StackPrepResult(
            frames=all_frames, usable_frames=[],
            recommended_stack=[], n_total=len(all_frames),
            estimated_z_range=None, alignment_required=False, warnings=warnings
        )

    if len(usable) < 3:
        warnings.append(
            f"Only {len(usable)} usable frames. Focus stack may have limited quality. "
            "Capture more frames with finer Z steps."
        )

    # Select best frames for stack (evenly distributed + best quality)
    if len(usable) <= max_frames:
        recommended = usable
    else:
        # Keep frames with highest focus scores, but ensure Z coverage
        # Sort by Z position to maintain stack order
        usable_sorted = sorted(usable, key=lambda f: f.z_position or 0)

        # Sample evenly across Z range, prefer higher-scoring frames in each zone
        zone_size = len(usable_sorted) // max_frames
        recommended = []
        for i in range(max_frames):
            start = i * zone_size
            end = start + zone_size if i < max_frames - 1 else len(usable_sorted)
            zone = usable_sorted[start:end]
            if zone:
                best_in_zone = max(zone, key=lambda f: f.composite_score)
                recommended.append(best_in_zone)

    # Estimated Z range
    z_vals = [f.z_position for f in recommended if f.z_position is not None]
    estimated_z_range = (max(z_vals) - min(z_vals)) if len(z_vals) >= 2 else None

    # Check alignment between consecutive usable frames
    alignment_required = False
    if check_alignment and len(recommended) >= 2:
        shifts: list[float] = []
        for i in range(min(5, len(recommended) - 1)):  # Check first 5 pairs
            img1 = images[recommended[i].index]
            img2 = images[recommended[i + 1].index]
            misaligned, shift_mag = detect_misalignment(img1, img2)
            shifts.append(shift_mag)
            if misaligned:
                alignment_required = True

        if alignment_required:
            avg_shift = float(np.mean(shifts))
            warnings.append(
                f"Frame misalignment detected (avg shift: {avg_shift:.1f}px). "
                "Apply image registration before stacking."
            )

    # Check for adequate Z coverage
    if estimated_z_range is not None and estimated_z_range < 2.0:
        warnings.append(
            f"Z range is only {estimated_z_range:.1f} units. "
            "May not capture full trichome depth. Consider larger Z sweep."
        )

    return StackPrepResult(
        frames=all_frames,
        usable_frames=usable,
        recommended_stack=recommended,
        n_total=len(all_frames),
        estimated_z_range=estimated_z_range,
        alignment_required=alignment_required,
        warnings=warnings,
    )


def select_sharpest_per_region(
    stack: list[StackFrame],
    grid: tuple[int, int] = (4, 4),
) -> NDArray[np.int32]:
    """
    Build a per-region frame assignment map for focus stacking.

    Returns a 2D array where each cell contains the index of the
    frame with the highest focus score for that region.

    This map can be used by a focus stacking algorithm to perform
    region-wise selection (simpler than full EDF algorithms).

    Args:
        stack: List of StackFrame objects (with regional scores)
        grid: Grid dimensions

    Returns:
        2D array (rows, cols) of frame indices
    """
    rows, cols = grid
    assignment = np.zeros((rows, cols), dtype=np.int32)
    best_scores = np.zeros((rows, cols), dtype=np.float32)

    for frame in stack:
        if frame.sharpest_regions is None:
            continue
        # Resize to grid if needed
        scores = frame.sharpest_regions
        if scores.shape != (rows, cols):
            scores = cv2.resize(scores, (cols, rows), interpolation=cv2.INTER_LINEAR)

        improved = scores > best_scores
        assignment[improved] = frame.index
        best_scores[improved] = scores[improved]

    return assignment
