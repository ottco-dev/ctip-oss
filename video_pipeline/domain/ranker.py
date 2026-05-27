"""
video_pipeline.domain.ranker — Frame ranking and best-frame selection.

Selects the N best frames from a scored set, with optional temporal
diversity constraints (prevent all best frames clustering in one segment).

SELECTION STRATEGIES:
1. Top-N: Take the N highest-scoring frames.
2. Diverse-N: Temporally spread — divide video into N segments,
   take the best frame from each segment.
3. Adaptive-N: Combine focus threshold + temporal diversity.
   Select frames above a minimum focus threshold, then apply
   temporal spreading to ensure coverage of the full video.

SCIENTIFIC RATIONALE:
Selecting the single best frame from a 60-second microscopy video may
miss regions where trichomes are in better focus due to:
- Z-axis drift during recording
- Different trichome depths across the sample
- Sample drift causing temporal focus variation

Temporal diversity ensures comprehensive sample coverage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from video_pipeline.domain.scorer import FrameQualityScore


@dataclass
class RankedFrame:
    """A frame with its quality score and index."""

    frame_index: int
    """Original frame index in the video."""

    timestamp_s: float
    """Timestamp in seconds."""

    quality: FrameQualityScore
    """Quality assessment for this frame."""

    phash: Optional[int] = None
    """Perceptual hash (for deduplication)."""

    @property
    def score(self) -> float:
        return self.quality.composite

    def __lt__(self, other: "RankedFrame") -> bool:
        return self.score < other.score

    def __gt__(self, other: "RankedFrame") -> bool:
        return self.score > other.score


def rank_top_n(
    frames: List[RankedFrame],
    n: int,
    *,
    min_score: float = 0.0,
) -> List[RankedFrame]:
    """
    Select the N highest-scoring frames.

    Args:
        frames:    All scored frames.
        n:         Number of frames to select.
        min_score: Minimum composite score to include.

    Returns:
        Up to N frames sorted by score descending.
    """
    filtered = [f for f in frames if f.score >= min_score]
    return sorted(filtered, key=lambda f: f.score, reverse=True)[:n]


def rank_diverse_n(
    frames: List[RankedFrame],
    n: int,
    *,
    min_score: float = 0.20,
) -> List[RankedFrame]:
    """
    Select N frames with temporal diversity.

    Divides the timeline into N equal segments and selects
    the highest-scoring frame from each segment.

    Args:
        frames:    All scored frames (must include timestamp_s).
        n:         Number of frames to select.
        min_score: Minimum score; segments where no frame meets
                   this threshold use the best available.

    Returns:
        Up to N frames, one per temporal segment, sorted by timestamp.
    """
    if not frames:
        return []

    n = max(1, n)
    ts_min = min(f.timestamp_s for f in frames)
    ts_max = max(f.timestamp_s for f in frames)

    if ts_max == ts_min:
        # All frames at same timestamp — fall back to top-N
        return rank_top_n(frames, n, min_score=min_score)

    segment_duration = (ts_max - ts_min) / n
    selected: List[RankedFrame] = []

    for seg in range(n):
        seg_start = ts_min + seg * segment_duration
        seg_end = ts_min + (seg + 1) * segment_duration + 0.001  # epsilon for last seg

        seg_frames = [
            f for f in frames
            if seg_start <= f.timestamp_s < seg_end
        ]

        if not seg_frames:
            continue

        # Prefer frames above min_score
        good = [f for f in seg_frames if f.score >= min_score]
        pool = good if good else seg_frames
        best = max(pool, key=lambda f: f.score)
        selected.append(best)

    # Sort by timestamp for chronological output
    return sorted(selected, key=lambda f: f.timestamp_s)


def rank_adaptive(
    frames: List[RankedFrame],
    n: int,
    *,
    min_focus: float = 0.30,
    diversity_weight: float = 0.5,
) -> List[RankedFrame]:
    """
    Adaptive ranking combining quality and temporal diversity.

    Score formula:
      adaptive_score = (1 - diversity_weight) × quality_score
                     + diversity_weight × temporal_spread_bonus

    Temporal spread bonus rewards frames that are far from already-selected
    frames (greedy selection similar to maximum coverage).

    Args:
        frames:           All scored frames.
        n:                Number of frames to select.
        min_focus:        Minimum focus score to be eligible.
        diversity_weight: Balance between quality (0) and diversity (1).

    Returns:
        Up to N frames, ordered by selection priority.
    """
    if not frames:
        return []

    eligible = [f for f in frames if f.quality.focus >= min_focus]
    if not eligible:
        eligible = frames  # Fall back to all frames if none meet focus threshold

    if len(eligible) <= n:
        return sorted(eligible, key=lambda f: f.score, reverse=True)

    # Duration for temporal spread normalization
    ts_min = min(f.timestamp_s for f in eligible)
    ts_max = max(f.timestamp_s for f in eligible)
    ts_range = ts_max - ts_min if ts_max > ts_min else 1.0

    selected: List[RankedFrame] = []
    remaining = list(eligible)

    # Seed: pick the best quality frame first
    best_first = max(remaining, key=lambda f: f.score)
    selected.append(best_first)
    remaining.remove(best_first)

    while len(selected) < n and remaining:
        # For each remaining frame, compute combined score
        best_adaptive = None
        best_adaptive_score = -1.0

        for f in remaining:
            # Temporal spread: min distance to any already selected frame
            min_dist = min(
                abs(f.timestamp_s - s.timestamp_s) / ts_range
                for s in selected
            )

            adaptive = (
                (1 - diversity_weight) * f.score
                + diversity_weight * min_dist
            )

            if adaptive > best_adaptive_score:
                best_adaptive_score = adaptive
                best_adaptive = f

        if best_adaptive is not None:
            selected.append(best_adaptive)
            remaining.remove(best_adaptive)

    return sorted(selected, key=lambda f: f.timestamp_s)
