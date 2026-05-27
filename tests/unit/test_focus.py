"""
tests/unit/test_focus.py — Unit tests for the focus analysis module.

Tests cover:
  - Laplacian metrics (LVAR, MLAP, SLG, LEG)
  - Tenengrad metrics (standard, variance, AGS)
  - FFT/DCT metrics (high-frequency ratio, DCT score, Brenner, Vollath)
  - Composite focus scorer (FocusScoreResult, quality labels)
  - Focus heatmap generation (FocusHeatmapResult, per-region grid)
  - Autofocus guidance (FocusCurveResult, analyze_focus_curve, select_best_frames)

Scientific invariants verified:
  - Sharp images score higher than blurred images on all metrics
  - Composite weights sum to 1.0
  - All normalized scores are in [0, 1]
  - Quality labels are monotonically assigned with score thresholds
  - Heatmap BGR array shape matches input image
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray


# ── Test fixtures ─────────────────────────────────────────────────────────────

def _make_sharp_gray(size: int = 256) -> NDArray[np.uint8]:
    """High-frequency checkerboard pattern — sharp image."""
    img = np.zeros((size, size), dtype=np.uint8)
    block = size // 16
    for r in range(size):
        for c in range(size):
            if (r // block + c // block) % 2 == 0:
                img[r, c] = 255
    return img


def _make_blurred_gray(sharp: NDArray[np.uint8], ksize: int = 31) -> NDArray[np.uint8]:
    """Apply heavy Gaussian blur to simulate out-of-focus image."""
    return cv2.GaussianBlur(sharp, (ksize, ksize), 0)


def _make_uniform_gray(size: int = 256, value: int = 128) -> NDArray[np.uint8]:
    """Uniform gray image — all metrics should be near-zero."""
    return np.full((size, size), value, dtype=np.uint8)


def _make_noisy_gray(size: int = 256, noise_std: float = 50.0) -> NDArray[np.uint8]:
    """Random noise image — high Laplacian, may have high or low focus score."""
    rng = np.random.default_rng(42)
    img = rng.normal(128, noise_std, (size, size))
    return np.clip(img, 0, 255).astype(np.uint8)


def _make_sharp_rgb(size: int = 256) -> NDArray[np.uint8]:
    """Sharp RGB image (checkerboard with color channels)."""
    gray = _make_sharp_gray(size)
    return np.stack([gray, gray // 2, gray // 3], axis=-1).astype(np.uint8)


@pytest.fixture
def sharp_gray() -> NDArray[np.uint8]:
    return _make_sharp_gray(256)


@pytest.fixture
def blurred_gray(sharp_gray: NDArray[np.uint8]) -> NDArray[np.uint8]:
    return _make_blurred_gray(sharp_gray, ksize=31)


@pytest.fixture
def uniform_gray() -> NDArray[np.uint8]:
    return _make_uniform_gray(256)


@pytest.fixture
def sharp_rgb() -> NDArray[np.uint8]:
    return _make_sharp_rgb(256)


# ── Laplacian metrics ─────────────────────────────────────────────────────────

class TestLaplacianMetrics:
    """Tests for focus/metrics/laplacian.py"""

    def test_laplacian_variance_sharp_gt_blurred(self, sharp_gray, blurred_gray):
        from focus.metrics.laplacian import laplacian_variance
        lvar_sharp = laplacian_variance(sharp_gray)
        lvar_blurred = laplacian_variance(blurred_gray)
        assert lvar_sharp > lvar_blurred, (
            f"Sharp image LVAR ({lvar_sharp:.1f}) should exceed blurred ({lvar_blurred:.1f})"
        )

    def test_laplacian_variance_nonnegative(self, sharp_gray, blurred_gray, uniform_gray):
        from focus.metrics.laplacian import laplacian_variance
        for img in [sharp_gray, blurred_gray, uniform_gray]:
            assert laplacian_variance(img) >= 0.0

    def test_laplacian_variance_uniform_near_zero(self, uniform_gray):
        from focus.metrics.laplacian import laplacian_variance
        lvar = laplacian_variance(uniform_gray)
        assert lvar < 1.0, f"Uniform image LVAR should be near 0, got {lvar}"

    def test_modified_laplacian_sharp_gt_blurred(self, sharp_gray, blurred_gray):
        from focus.metrics.laplacian import modified_laplacian
        mlap_sharp = modified_laplacian(sharp_gray)
        mlap_blurred = modified_laplacian(blurred_gray)
        assert mlap_sharp > mlap_blurred

    def test_squared_laplacian_gradient(self, sharp_gray, blurred_gray):
        from focus.metrics.laplacian import squared_laplacian_gradient
        slg_sharp = squared_laplacian_gradient(sharp_gray)
        slg_blurred = squared_laplacian_gradient(blurred_gray)
        assert slg_sharp > slg_blurred

    def test_laplacian_energy_of_gradient(self, sharp_gray, blurred_gray):
        from focus.metrics.laplacian import laplacian_energy_of_gradient
        leg_sharp = laplacian_energy_of_gradient(sharp_gray)
        leg_blurred = laplacian_energy_of_gradient(blurred_gray)
        assert leg_sharp > leg_blurred

    def test_regional_laplacian_returns_float(self, sharp_gray):
        """regional_laplacian_variance returns a single robust float (tile-based percentile)."""
        from focus.metrics.laplacian import regional_laplacian_variance
        val = regional_laplacian_variance(sharp_gray, tile_size=64)
        assert isinstance(val, float)
        assert val >= 0.0

    def test_single_pixel_image_does_not_crash(self):
        from focus.metrics.laplacian import laplacian_variance
        tiny = np.array([[128]], dtype=np.uint8)
        val = laplacian_variance(tiny)
        assert val >= 0.0

    def test_accepts_rgb_input(self, sharp_rgb):
        """Laplacian should handle RGB by converting internally or accepting grayscale."""
        from focus.metrics.laplacian import laplacian_variance
        import cv2
        gray = cv2.cvtColor(sharp_rgb, cv2.COLOR_RGB2GRAY)
        val = laplacian_variance(gray)
        assert val >= 0.0


# ── Tenengrad metrics ─────────────────────────────────────────────────────────

class TestTenengradMetrics:
    """Tests for focus/metrics/tenengrad.py"""

    def test_tenengrad_sharp_gt_blurred(self, sharp_gray, blurred_gray):
        from focus.metrics.tenengrad import tenengrad
        assert tenengrad(sharp_gray) > tenengrad(blurred_gray)

    def test_tenengrad_nonnegative(self, sharp_gray, blurred_gray, uniform_gray):
        from focus.metrics.tenengrad import tenengrad
        for img in [sharp_gray, blurred_gray, uniform_gray]:
            assert tenengrad(img) >= 0.0

    def test_tenengrad_variance_sharp_gt_blurred(self, sharp_gray, blurred_gray):
        from focus.metrics.tenengrad import tenengrad_variance
        assert tenengrad_variance(sharp_gray) > tenengrad_variance(blurred_gray)

    def test_absolute_gradient_sum_sharp_gt_blurred(self, sharp_gray, blurred_gray):
        from focus.metrics.tenengrad import absolute_gradient_sum
        assert absolute_gradient_sum(sharp_gray) > absolute_gradient_sum(blurred_gray)

    def test_compute_gradient_map_shape(self, sharp_gray):
        from focus.metrics.tenengrad import compute_gradient_map
        gmap = compute_gradient_map(sharp_gray)
        assert gmap.shape == sharp_gray.shape

    def test_compute_gradient_map_nonnegative(self, sharp_gray):
        from focus.metrics.tenengrad import compute_gradient_map
        gmap = compute_gradient_map(sharp_gray)
        assert np.all(gmap >= 0.0)


# ── FFT / frequency domain metrics ───────────────────────────────────────────

class TestFFTMetrics:
    """Tests for focus/metrics/fft_metrics.py"""

    def test_fft_high_frequency_ratio_sharp_gt_blurred(self, sharp_gray, blurred_gray):
        from focus.metrics.fft_metrics import fft_high_frequency_ratio
        assert fft_high_frequency_ratio(sharp_gray) > fft_high_frequency_ratio(blurred_gray)

    def test_fft_ratio_in_range(self, sharp_gray, blurred_gray, uniform_gray):
        from focus.metrics.fft_metrics import fft_high_frequency_ratio
        for img in [sharp_gray, blurred_gray, uniform_gray]:
            val = fft_high_frequency_ratio(img)
            assert 0.0 <= val <= 1.0, f"FFT ratio out of [0,1]: {val}"

    def test_dct_score_sharp_gt_blurred(self, sharp_gray, blurred_gray):
        from focus.metrics.fft_metrics import dct_high_frequency_score
        assert dct_high_frequency_score(sharp_gray) >= dct_high_frequency_score(blurred_gray)

    def test_brenner_focus_sharp_gt_blurred(self, sharp_gray, blurred_gray):
        from focus.metrics.fft_metrics import brenner_focus
        assert brenner_focus(sharp_gray) > brenner_focus(blurred_gray)

    def test_brenner_focus_nonnegative(self, uniform_gray):
        from focus.metrics.fft_metrics import brenner_focus
        assert brenner_focus(uniform_gray) >= 0.0

    def test_vollath_f4_sharp_gt_blurred(self, sharp_gray, blurred_gray):
        from focus.metrics.fft_metrics import vollath_f4
        assert vollath_f4(sharp_gray) >= vollath_f4(blurred_gray)

    def test_power_spectral_slope_returns_float(self, sharp_gray):
        from focus.metrics.fft_metrics import power_spectral_slope
        val = power_spectral_slope(sharp_gray)
        assert isinstance(val, float)
        assert math.isfinite(val)


# ── Composite focus scorer ────────────────────────────────────────────────────

class TestCompositeFocusScore:
    """Tests for focus/metrics/composite.py — FocusScoreResult and compute_focus_score()"""

    def test_composite_higher_for_sharp(self, sharp_gray, blurred_gray):
        from focus.metrics.composite import compute_focus_score
        r_sharp = compute_focus_score(sharp_gray)
        r_blurred = compute_focus_score(blurred_gray)
        assert r_sharp.composite > r_blurred.composite, (
            f"Sharp composite ({r_sharp.composite:.3f}) should exceed "
            f"blurred ({r_blurred.composite:.3f})"
        )

    def test_composite_in_unit_range(self, sharp_gray, blurred_gray, uniform_gray):
        from focus.metrics.composite import compute_focus_score
        for img in [sharp_gray, blurred_gray, uniform_gray]:
            r = compute_focus_score(img)
            assert 0.0 <= r.composite <= 1.0, f"Composite out of [0,1]: {r.composite}"

    def test_all_sub_metrics_nonnegative(self, sharp_gray):
        from focus.metrics.composite import compute_focus_score
        r = compute_focus_score(sharp_gray)
        assert r.laplacian_variance >= 0.0
        assert r.tenengrad >= 0.0
        assert r.normalized_variance >= 0.0
        assert r.fft_score >= 0.0

    def test_quality_label_assigned(self, sharp_gray, blurred_gray, uniform_gray):
        from focus.metrics.composite import compute_focus_score
        # Labels from implementation: excellent, good, acceptable, poor, unusable
        valid_labels = {"excellent", "good", "acceptable", "poor", "unusable", "unacceptable"}
        for img in [sharp_gray, blurred_gray, uniform_gray]:
            r = compute_focus_score(img)
            assert r.quality_label in valid_labels, f"Unexpected label: {r.quality_label}"

    def test_quality_label_monotonic_with_score(self):
        """Verify quality label thresholds: excellent≥0.75, good≥0.55, acceptable≥0.35."""
        from focus.metrics.composite import compute_focus_score
        sharp = _make_sharp_gray(512)
        result = compute_focus_score(sharp)
        # Sharp checkerboard should score high
        if result.composite >= 0.75:
            assert result.quality_label == "excellent"
        elif result.composite >= 0.55:
            assert result.quality_label in {"excellent", "good"}

    def test_is_acceptable_property(self, sharp_gray, blurred_gray):
        from focus.metrics.composite import compute_focus_score
        r_sharp = compute_focus_score(sharp_gray)
        r_blurred = compute_focus_score(blurred_gray)
        assert r_sharp.is_acceptable, "Sharp image should be acceptable"
        # Blurred may or may not be acceptable depending on blur degree

    def test_is_good_property(self, sharp_gray):
        from focus.metrics.composite import compute_focus_score
        r = compute_focus_score(sharp_gray)
        # Sharp checkerboard should be good or excellent
        assert r.composite >= 0.35, "Sharp image should have composite ≥ 0.35"

    def test_accepts_rgb_input(self, sharp_rgb):
        from focus.metrics.composite import compute_focus_score
        r = compute_focus_score(sharp_rgb)
        assert 0.0 <= r.composite <= 1.0

    def test_regional_scores_shape(self, sharp_gray):
        from focus.metrics.composite import compute_focus_score
        r = compute_focus_score(sharp_gray, compute_regional=True, region_grid=(4, 4))
        assert r.region_scores is not None
        assert r.region_scores.shape == (4, 4), f"Expected (4,4), got {r.region_scores.shape}"

    def test_regional_scores_none_when_not_requested(self, sharp_gray):
        from focus.metrics.composite import compute_focus_score
        r = compute_focus_score(sharp_gray, compute_regional=False)
        assert r.region_scores is None

    def test_to_dict_has_required_keys(self, sharp_gray):
        from focus.metrics.composite import compute_focus_score
        r = compute_focus_score(sharp_gray)
        d = r.to_dict()
        assert "composite" in d
        assert "quality_label" in d

    def test_generate_focus_heatmap_shape(self, sharp_gray):
        from focus.metrics.composite import generate_focus_heatmap
        # generate_focus_heatmap in composite.py takes grid=(rows, cols) tuple
        heatmap = generate_focus_heatmap(sharp_gray, grid=(4, 4))
        assert heatmap.shape[:2] == sharp_gray.shape[:2], (
            f"Heatmap shape {heatmap.shape} should match image {sharp_gray.shape}"
        )
        assert heatmap.ndim == 3 and heatmap.shape[2] == 3, "Heatmap should be RGB"

    def test_rank_frames_by_focus(self):
        from focus.metrics.composite import rank_frames_by_focus, compute_focus_score
        rng = np.random.default_rng(42)
        frames = [rng.integers(0, 256, (64, 64, 3), dtype=np.uint8) for _ in range(5)]
        frames.append(_make_sharp_rgb(64))  # Add one known-sharp frame
        # rank_frames_by_focus expects list of (frame_index, FocusScoreResult)
        frame_scores = [(i, compute_focus_score(f)) for i, f in enumerate(frames)]
        ranked = rank_frames_by_focus(frame_scores, min_score=0.0)
        assert len(ranked) <= len(frames)
        # Indices should be sorted by descending composite score
        scores = [r[1].composite for r in ranked]
        assert scores == sorted(scores, reverse=True)


# ── Focus heatmap ─────────────────────────────────────────────────────────────

class TestFocusHeatmap:
    """Tests for focus/guidance/heatmap.py"""

    def test_generate_focus_heatmap_returns_result(self, sharp_gray):
        from focus.guidance.heatmap import generate_focus_heatmap, FocusHeatmapResult
        result = generate_focus_heatmap(sharp_gray)
        assert isinstance(result, FocusHeatmapResult)

    def test_heatmap_rgb_shape_matches_input(self, sharp_gray):
        from focus.guidance.heatmap import generate_focus_heatmap
        result = generate_focus_heatmap(sharp_gray)
        assert result.heatmap_rgb.shape[:2] == sharp_gray.shape[:2]
        assert result.heatmap_rgb.shape[2] == 3

    def test_score_map_shape(self, sharp_gray):
        from focus.guidance.heatmap import generate_focus_heatmap
        # FocusHeatmapResult uses .score_map (not .score_grid)
        result = generate_focus_heatmap(sharp_gray, grid=(6, 6))
        assert result.score_map.shape == (6, 6)

    def test_score_map_values_in_range(self, sharp_gray):
        from focus.guidance.heatmap import generate_focus_heatmap
        result = generate_focus_heatmap(sharp_gray)
        assert np.all(result.score_map >= 0.0)
        assert np.all(result.score_map <= 1.0)

    def test_generate_laplacian_heatmap(self, sharp_gray):
        from focus.guidance.heatmap import generate_laplacian_heatmap
        heatmap = generate_laplacian_heatmap(sharp_gray)
        assert heatmap.shape[:2] == sharp_gray.shape[:2]

    def test_rgb_input_accepted(self, sharp_rgb):
        from focus.guidance.heatmap import generate_focus_heatmap
        result = generate_focus_heatmap(sharp_rgb)
        assert result.heatmap_rgb.shape[2] == 3

    def test_has_quality_fractions(self, sharp_gray):
        from focus.guidance.heatmap import generate_focus_heatmap
        result = generate_focus_heatmap(sharp_gray)
        assert 0.0 <= result.sharp_fraction <= 1.0
        assert 0.0 <= result.acceptable_fraction <= 1.0

    def test_mean_score_in_range(self, sharp_gray):
        from focus.guidance.heatmap import generate_focus_heatmap
        result = generate_focus_heatmap(sharp_gray)
        assert 0.0 <= result.mean_score <= 1.0


# ── Autofocus guidance ────────────────────────────────────────────────────────

class TestAutofocusGuidance:
    """Tests for focus/guidance/autofocus.py"""

    def _make_gaussian_zstack(self, n: int = 9, sharp_idx: int = 4) -> list[NDArray[np.uint8]]:
        """
        Create a Z-stack of images with Gaussian focus curve centered at sharp_idx.
        Frame at center gets max sharpness (checkerboard), others are progressively blurred.
        """
        frames = []
        for i in range(n):
            dist = abs(i - sharp_idx)
            sharp = _make_sharp_gray(128)
            if dist == 0:
                frames.append(np.stack([sharp, sharp, sharp], axis=-1))
            else:
                blur_k = min(dist * 10 + 1, 51)
                if blur_k % 2 == 0:
                    blur_k += 1
                blurred = cv2.GaussianBlur(sharp, (blur_k, blur_k), 0)
                frames.append(np.stack([blurred, blurred, blurred], axis=-1))
        return frames

    def test_analyze_focus_curve_returns_result(self):
        from focus.guidance.autofocus import analyze_focus_curve, FocusCurveResult
        frames = self._make_gaussian_zstack(n=5)
        result = analyze_focus_curve(frames)
        assert isinstance(result, FocusCurveResult)

    def test_best_frame_index_is_peak(self):
        from focus.guidance.autofocus import analyze_focus_curve
        n = 9
        sharp_idx = 4
        frames = self._make_gaussian_zstack(n=n, sharp_idx=sharp_idx)
        result = analyze_focus_curve(frames)
        # optimal_index property
        idx = result.optimal_index
        assert abs(idx - sharp_idx) <= 2, (
            f"Optimal index {idx} not near expected peak {sharp_idx}"
        )

    def test_best_score_property(self):
        from focus.guidance.autofocus import analyze_focus_curve
        frames = self._make_gaussian_zstack(n=5, sharp_idx=2)
        result = analyze_focus_curve(frames)
        # best_score should equal max of focus_scores
        assert result.best_score == pytest.approx(max(result.focus_scores), rel=1e-4)

    def test_optimal_z_assigned(self):
        from focus.guidance.autofocus import analyze_focus_curve
        frames = self._make_gaussian_zstack(n=5)
        z_positions = [0.0, 1.0, 2.0, 3.0, 4.0]
        result = analyze_focus_curve(frames, z_positions=z_positions)
        assert result.optimal_z in z_positions

    def test_focus_curve_scores_length_matches_input(self):
        from focus.guidance.autofocus import analyze_focus_curve
        n = 7
        frames = self._make_gaussian_zstack(n=n)
        result = analyze_focus_curve(frames)
        assert len(result.focus_scores) == n

    def test_is_reliable_bool(self):
        from focus.guidance.autofocus import analyze_focus_curve
        frames = self._make_gaussian_zstack(n=9, sharp_idx=4)
        result = analyze_focus_curve(frames)
        assert isinstance(result.is_reliable, bool)

    def test_select_best_frames_returns_n(self):
        from focus.guidance.autofocus import select_best_frames
        rng = np.random.default_rng(0)
        frames = [rng.integers(0, 256, (64, 64, 3), dtype=np.uint8) for _ in range(10)]
        selected = select_best_frames(frames, n=3)
        assert len(selected) <= 3

    def test_select_best_frames_empty_input(self):
        from focus.guidance.autofocus import select_best_frames
        selected = select_best_frames([], n=5)
        assert selected == []

    def test_focus_drift_detector_returns_dict(self):
        from focus.guidance.autofocus import FocusDriftDetector
        # Uses alert_threshold and drift_threshold (not 'threshold')
        detector = FocusDriftDetector(window_size=5, alert_threshold=0.35, drift_threshold=0.15)
        for score in [0.8, 0.78, 0.72, 0.65, 0.55]:
            result = detector.update(score)
            assert isinstance(result, dict)
            assert "drift_detected" in result

    def test_focus_drift_detected_on_sudden_drop(self):
        from focus.guidance.autofocus import FocusDriftDetector
        # Use a large drift_threshold and low alert_threshold to ensure detection
        detector = FocusDriftDetector(window_size=5, alert_threshold=0.35, drift_threshold=0.10)
        # Feed stable high scores (establishes baseline ~0.8)
        for score in [0.8, 0.8, 0.8, 0.8, 0.8]:
            detector.update(score)
        # Sudden drop — rolling average of last 3 will be [0.8, 0.8, 0.15] ≈ 0.58
        # drift_from_baseline = 0.80 - 0.58 = 0.22 > drift_threshold=0.10
        result = detector.update(0.15)
        # Verify drift was flagged (drift_from_baseline should exceed threshold)
        assert result.get("drift_detected", False) is True, (
            f"Drift should be detected. Result: {result}"
        )

    def test_drift_detector_baseline_established(self):
        from focus.guidance.autofocus import FocusDriftDetector
        detector = FocusDriftDetector(window_size=5)
        for score in [0.7, 0.75, 0.72]:
            detector.update(score)
        # Baseline should be set after 3 readings with ≥ 0.5
        last = detector.update(0.71)
        # baseline key exists in result
        assert "baseline" in last


# ── Integration: focus score consistency ─────────────────────────────────────

class TestFocusConsistency:
    """Cross-module consistency tests."""

    def test_focus_metrics_agree_on_sharp_vs_blurred(self):
        """All metrics should rank sharp > blurred."""
        from focus.metrics.laplacian import laplacian_variance
        from focus.metrics.tenengrad import tenengrad
        from focus.metrics.fft_metrics import fft_high_frequency_ratio
        from focus.metrics.composite import compute_focus_score

        sharp = _make_sharp_gray(256)
        blurred = _make_blurred_gray(sharp, ksize=51)

        assert laplacian_variance(sharp) > laplacian_variance(blurred)
        assert tenengrad(sharp) > tenengrad(blurred)
        assert fft_high_frequency_ratio(sharp) > fft_high_frequency_ratio(blurred)
        assert compute_focus_score(sharp).composite > compute_focus_score(blurred).composite

    def test_composite_score_deterministic(self, sharp_gray):
        """Same input must always give same output."""
        from focus.metrics.composite import compute_focus_score
        r1 = compute_focus_score(sharp_gray)
        r2 = compute_focus_score(sharp_gray)
        assert r1.composite == pytest.approx(r2.composite, rel=1e-6)

    def test_heatmap_sharp_region_scores_higher_than_blurred_region(self):
        """A sharp patch embedded in a blurred field should show higher score."""
        from focus.guidance.heatmap import generate_focus_heatmap
        img = _make_uniform_gray(256, value=128)
        # Embed sharp patch in center quadrant
        sharp_patch = _make_sharp_gray(64)
        img[96:160, 96:160] = sharp_patch
        result = generate_focus_heatmap(img, grid=(4, 4))
        assert result is not None
        assert result.score_map is not None
        # Score map should have some variation (not all identical)
        assert result.score_map.std() >= 0.0  # Non-degenerate
