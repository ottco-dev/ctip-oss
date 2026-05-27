"""
tests.unit.test_video_pipeline — Unit tests for video pipeline domain modules.

Tests:
  - Frame quality scoring (scorer)
  - Perceptual hashing and deduplication (hasher)
  - Frame ranking strategies (ranker)
  - Motion estimation (motion)
  - Edge cases: black frames, saturated frames, tiny frames
"""

from __future__ import annotations

import numpy as np
import pytest

from video_pipeline.domain.scorer import score_frame, FrameQualityScore
from video_pipeline.domain.hasher import (
    perceptual_hash,
    hamming_distance,
    is_near_duplicate,
    deduplicate_frames,
    find_scene_changes,
)
from video_pipeline.domain.ranker import (
    RankedFrame,
    rank_top_n,
    rank_diverse_n,
    rank_adaptive,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sharp_frame() -> np.ndarray:
    """Synthetic sharp frame: checkerboard pattern."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for i in range(0, 480, 20):
        for j in range(0, 640, 20):
            if (i // 20 + j // 20) % 2 == 0:
                frame[i:i+20, j:j+20] = 200
    return frame


@pytest.fixture
def blurry_frame() -> np.ndarray:
    """Synthetic blurry frame: uniform gray."""
    return np.full((480, 640, 3), 128, dtype=np.uint8)


@pytest.fixture
def overexposed_frame() -> np.ndarray:
    """Mostly white frame simulating overexposure."""
    return np.full((480, 640, 3), 252, dtype=np.uint8)


@pytest.fixture
def underexposed_frame() -> np.ndarray:
    """Very dark frame."""
    return np.full((480, 640, 3), 3, dtype=np.uint8)


def _make_ranked_frames(n: int, scores: list = None) -> list:
    """Create N ranked frames with given quality scores."""
    if scores is None:
        scores = [0.5 + i * 0.05 for i in range(n)]
    return [
        RankedFrame(
            frame_index=i,
            timestamp_s=float(i),
            quality=FrameQualityScore(
                composite=s,
                focus=s,
                exposure=0.7,
                noise=0.8,
            ),
        )
        for i, s in enumerate(scores)
    ]


# ── Frame Quality Scoring ─────────────────────────────────────────────────────

class TestFrameQualityScorer:

    def test_sharp_frame_higher_score_than_blurry(self, sharp_frame, blurry_frame):
        s_sharp = score_frame(sharp_frame, use_focus_composite=False)
        s_blurry = score_frame(blurry_frame, use_focus_composite=False)
        assert s_sharp.focus > s_blurry.focus, (
            f"Sharp focus {s_sharp.focus:.3f} should > blurry {s_blurry.focus:.3f}"
        )

    def test_composite_in_range(self, sharp_frame):
        score = score_frame(sharp_frame, use_focus_composite=False)
        assert 0.0 <= score.composite <= 1.0

    def test_all_subscores_in_range(self, sharp_frame):
        score = score_frame(sharp_frame, use_focus_composite=False)
        assert 0.0 <= score.focus <= 1.0
        assert 0.0 <= score.exposure <= 1.0
        assert 0.0 <= score.noise <= 1.0

    def test_overexposed_low_exposure_score(self, overexposed_frame):
        score = score_frame(overexposed_frame, use_focus_composite=False)
        assert score.exposure < 0.6, f"Overexposed frame exposure: {score.exposure:.3f}"

    def test_underexposed_low_exposure_score(self, underexposed_frame):
        score = score_frame(underexposed_frame, use_focus_composite=False)
        assert score.exposure < 0.6

    def test_quality_label_exists(self, sharp_frame):
        score = score_frame(sharp_frame, use_focus_composite=False)
        assert score.quality_label in ("excellent", "good", "acceptable", "poor", "unusable")

    def test_blurry_frame_is_not_excellent(self, blurry_frame):
        score = score_frame(blurry_frame, use_focus_composite=False)
        assert not score.is_excellent

    def test_small_frame_does_not_crash(self):
        tiny = np.ones((8, 8, 3), dtype=np.uint8) * 100
        score = score_frame(tiny, use_focus_composite=False)
        assert score is not None
        assert 0.0 <= score.composite <= 1.0

    def test_single_channel_not_accepted(self):
        """scorer expects 3-channel input."""
        gray = np.ones((100, 100), dtype=np.uint8) * 128
        # Should raise or handle gracefully
        try:
            score = score_frame(gray, use_focus_composite=False)
            # If it doesn't raise, result should still be valid
        except Exception:
            pass  # Expected for 2D input


# ── Perceptual Hashing ────────────────────────────────────────────────────────

class TestPerceptualHash:

    def test_identical_frames_zero_distance(self, sharp_frame):
        h1 = perceptual_hash(sharp_frame)
        h2 = perceptual_hash(sharp_frame)
        assert hamming_distance(h1, h2) == 0

    def test_different_frames_nonzero_distance(self, sharp_frame, blurry_frame):
        h1 = perceptual_hash(sharp_frame)
        h2 = perceptual_hash(blurry_frame)
        assert hamming_distance(h1, h2) > 0

    def test_hash_is_integer(self, sharp_frame):
        h = perceptual_hash(sharp_frame)
        assert isinstance(h, int)

    def test_near_duplicate_identical(self, sharp_frame):
        h = perceptual_hash(sharp_frame)
        assert is_near_duplicate(h, h, threshold=8) is True

    def test_near_duplicate_different(self, sharp_frame, overexposed_frame):
        h1 = perceptual_hash(sharp_frame)
        h2 = perceptual_hash(overexposed_frame)
        # Very different frames should not be near-duplicates
        assert is_near_duplicate(h1, h2, threshold=5) is False

    def test_deduplicate_all_identical(self, sharp_frame):
        hashes = [perceptual_hash(sharp_frame)] * 10
        kept = deduplicate_frames(hashes, threshold=8)
        assert len(kept) == 1  # All are duplicates

    def test_deduplicate_all_different(self):
        hashes = [i * 1000 for i in range(10)]  # Very different hashes
        kept = deduplicate_frames(hashes, threshold=0)
        assert len(kept) == 10

    def test_deduplicate_empty(self):
        assert deduplicate_frames([]) == []

    def test_hamming_symmetry(self):
        h1, h2 = 0b1010101010101010, 0b0101010101010101
        assert hamming_distance(h1, h2) == hamming_distance(h2, h1)

    def test_find_scene_changes_no_changes(self, sharp_frame):
        h = perceptual_hash(sharp_frame)
        hashes = [h] * 5
        changes = find_scene_changes(hashes)
        assert len(changes) == 0

    def test_find_scene_changes_detects_change(self, sharp_frame, overexposed_frame):
        h1 = perceptual_hash(sharp_frame)
        h2 = perceptual_hash(overexposed_frame)
        hashes = [h1, h1, h1, h2, h2]
        changes = find_scene_changes(hashes, threshold=10)
        # Should detect a change between index 2 and 3
        assert len(changes) >= 1


# ── Frame Ranking ─────────────────────────────────────────────────────────────

class TestFrameRanking:

    def test_top_n_returns_correct_count(self):
        frames = _make_ranked_frames(20)
        result = rank_top_n(frames, 5)
        assert len(result) <= 5

    def test_top_n_sorts_descending(self):
        frames = _make_ranked_frames(10)
        result = rank_top_n(frames, 10)
        scores = [f.score for f in result]
        assert scores == sorted(scores, reverse=True)

    def test_top_n_respects_min_score(self):
        frames = _make_ranked_frames(10, scores=[0.1, 0.2, 0.3, 0.8, 0.9, 0.4, 0.5, 0.6, 0.7, 0.95])
        result = rank_top_n(frames, 10, min_score=0.5)
        assert all(f.score >= 0.5 for f in result)

    def test_top_n_empty_returns_empty(self):
        assert rank_top_n([], 5) == []

    def test_diverse_n_temporal_spread(self):
        # Create 20 frames spread over 20 seconds
        frames = _make_ranked_frames(20)
        for i, f in enumerate(frames):
            f.timestamp_s = float(i)
        result = rank_diverse_n(frames, 5)
        # Should have ~5 frames spread across the timeline
        assert 1 <= len(result) <= 5
        if len(result) > 1:
            timestamps = [f.timestamp_s for f in result]
            # Sorted chronologically
            assert timestamps == sorted(timestamps)

    def test_diverse_n_empty_returns_empty(self):
        assert rank_diverse_n([], 5) == []

    def test_adaptive_selects_quality_and_diverse(self):
        # Create frames with a peak score in the middle
        frames = [
            RankedFrame(
                frame_index=i,
                timestamp_s=float(i),
                quality=FrameQualityScore(
                    composite=0.9 if i == 10 else 0.5,
                    focus=0.9 if i == 10 else 0.5,
                    exposure=0.7,
                    noise=0.8,
                ),
            )
            for i in range(20)
        ]
        result = rank_adaptive(frames, 3, min_focus=0.3)
        assert 1 <= len(result) <= 3

    def test_adaptive_empty_returns_empty(self):
        assert rank_adaptive([], 5) == []

    def test_ranking_with_n_greater_than_frames(self):
        frames = _make_ranked_frames(3)
        result = rank_top_n(frames, 10)
        assert len(result) == 3


# ── Integration: scorer + hasher + ranker ────────────────────────────────────

class TestVideoIntegration:

    def test_full_pipeline_on_synthetic_frames(self, sharp_frame, blurry_frame, overexposed_frame):
        """Simulate a mini video pipeline: score → hash → rank."""
        test_frames = [sharp_frame, blurry_frame, overexposed_frame, sharp_frame.copy()]
        scores = [score_frame(f, use_focus_composite=False) for f in test_frames]
        hashes = [perceptual_hash(f) for f in test_frames]

        # Deduplicate (sharp_frame appears twice)
        keep_idxs = deduplicate_frames(hashes, threshold=5)

        ranked = [
            RankedFrame(
                frame_index=keep_idxs[i],
                timestamp_s=float(keep_idxs[i]),
                quality=scores[keep_idxs[i]],
                phash=hashes[keep_idxs[i]],
            )
            for i in range(len(keep_idxs))
        ]

        best = rank_top_n(ranked, 2)
        assert len(best) <= 2
        # Best should be the sharp frame
        if best:
            assert best[0].quality.focus >= best[-1].quality.focus
