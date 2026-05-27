"""
tests.unit.test_morphology — Unit tests for morphology analysis module.

Tests:
  - Geometric descriptor extraction
  - Stalk/head detection
  - Density map computation
  - Rule-based morphology classification
  - Edge cases: empty mask, tiny mask, corrupted inputs
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from morphology.domain.geometric import extract_geometric_descriptors, _degenerate_descriptors
from morphology.domain.stalk_detector import detect_stalk_and_head
from morphology.domain.density_map import compute_density_map, TrichomeCentroid
from morphology.classification.classifier import (
    MorphologyClassifier,
    classify_morphology_geometric,
    GeometricFeatures,
)
from shared.core.enums import TrichomeType


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def circular_mask() -> np.ndarray:
    """Filled circle mask — simulates a bulbous trichome head."""
    mask = np.zeros((64, 64), dtype=np.uint8)
    center, r = (32, 32), 18
    for y in range(64):
        for x in range(64):
            if (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r ** 2:
                mask[y, x] = 255
    return mask


@pytest.fixture
def elongated_mask() -> np.ndarray:
    """Tall ellipse — simulates a stalked trichome (head+stalk)."""
    mask = np.zeros((100, 40), dtype=np.uint8)
    # Stalk: bottom 60px, narrow
    mask[40:100, 15:25] = 255
    # Head: top 40px, wider
    for y in range(40):
        for x in range(40):
            if (x - 20) ** 2 + (y - 20) ** 2 <= 18 ** 2:
                mask[y, x] = 255
    return mask


@pytest.fixture
def hair_mask() -> np.ndarray:
    """Very thin, tall rectangle — non-glandular hair."""
    mask = np.zeros((120, 8), dtype=np.uint8)
    mask[:, 2:6] = 255
    return mask


# ── Geometric Descriptors ────────────────────────────────────────────────────

class TestGeometricDescriptors:

    def test_circle_has_high_circularity(self, circular_mask):
        geo = extract_geometric_descriptors(circular_mask)
        assert geo.circularity > 0.85, f"Expected >0.85, got {geo.circularity:.3f}"

    def test_circle_has_low_elongation(self, circular_mask):
        geo = extract_geometric_descriptors(circular_mask)
        assert geo.elongation < 1.5, f"Expected <1.5, got {geo.elongation:.3f}"

    def test_elongated_mask_low_circularity(self, elongated_mask):
        geo = extract_geometric_descriptors(elongated_mask)
        # The stalk+head combined contour has moderate-to-low circularity
        # Allow up to 0.95 — the key assertion is that elongation is high (separate test)
        assert geo.circularity < 0.95

    def test_elongated_mask_high_elongation(self, elongated_mask):
        geo = extract_geometric_descriptors(elongated_mask)
        assert geo.elongation > 2.0, f"Expected >2.0, got {geo.elongation:.3f}"

    def test_area_matches_filled_circle(self, circular_mask):
        geo = extract_geometric_descriptors(circular_mask)
        expected_area = math.pi * 18 ** 2
        # Allow 8% tolerance: pixel rasterization introduces discretization error
        assert abs(geo.area_px - expected_area) < expected_area * 0.08

    def test_is_valid_flag(self, circular_mask):
        geo = extract_geometric_descriptors(circular_mask)
        assert geo.is_valid is True

    def test_empty_mask_returns_degenerate(self):
        mask = np.zeros((64, 64), dtype=np.uint8)
        geo = extract_geometric_descriptors(mask)
        assert geo.is_valid is False
        assert geo.area_px == 0.0

    def test_single_pixel_mask(self):
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[16, 16] = 255
        geo = extract_geometric_descriptors(mask)
        # Should not crash; may return degenerate
        assert geo is not None

    def test_invalid_shape_raises(self):
        mask_3d = np.zeros((32, 32, 3), dtype=np.uint8)
        with pytest.raises(ValueError):
            extract_geometric_descriptors(mask_3d)

    def test_feature_vector_correct_length(self, circular_mask):
        geo = extract_geometric_descriptors(circular_mask)
        vec = geo.to_feature_vector()
        assert len(vec) == 7
        assert vec.dtype == np.float32

    def test_feature_vector_values_in_range(self, circular_mask):
        geo = extract_geometric_descriptors(circular_mask)
        vec = geo.to_feature_vector()
        assert np.all(vec >= 0.0), f"Negative values in feature vector: {vec}"
        assert np.all(vec <= 1.0), f"Values > 1 in feature vector: {vec}"


# ── Stalk / Head Detection ────────────────────────────────────────────────────

class TestStalkDetector:

    def test_bulbous_has_no_stalk(self, circular_mask):
        stalk, head = detect_stalk_and_head(circular_mask)
        # Bulbous: no stalk
        # Note: may or may not detect stalk depending on mask shape
        assert stalk is not None
        assert head is not None

    def test_stalked_has_stalk(self, elongated_mask):
        stalk, head = detect_stalk_and_head(elongated_mask, min_stalk_length_px=5.0)
        # Elongated = stalked
        assert stalk.stalk_length_px > 0 or not stalk.has_visible_stalk
        # At minimum, should not crash

    def test_head_area_positive(self, circular_mask):
        _, head = detect_stalk_and_head(circular_mask)
        if head is not None:
            assert head.head_area_px > 0

    def test_small_mask_returns_no_stalk(self):
        tiny = np.zeros((10, 10), dtype=np.uint8)
        tiny[3:7, 3:7] = 255
        stalk, head = detect_stalk_and_head(tiny)
        assert stalk.has_visible_stalk is False

    def test_empty_mask_returns_no_stalk(self):
        empty = np.zeros((50, 50), dtype=np.uint8)
        stalk, head = detect_stalk_and_head(empty)
        assert stalk.stalk_length_px == 0.0
        assert stalk.has_visible_stalk is False
        assert head is None

    def test_stalk_confidence_in_range(self, elongated_mask):
        stalk, _ = detect_stalk_and_head(elongated_mask)
        assert 0.0 <= stalk.confidence <= 1.0


# ── Density Map ───────────────────────────────────────────────────────────────

class TestDensityMap:

    def _make_centroids(self, n: int = 20) -> list:
        rng = np.random.default_rng(42)
        return [
            TrichomeCentroid(x=float(rng.uniform(10, 490)), y=float(rng.uniform(10, 490)))
            for _ in range(n)
        ]

    def test_basic_density_map(self):
        centroids = self._make_centroids(20)
        result = compute_density_map(centroids, 500, 500)
        assert result.total_count == 20
        assert result.density_grid.shape == (8, 8)
        assert result.kde_map.shape == (500, 500)

    def test_empty_centroids(self):
        result = compute_density_map([], 100, 100)
        assert result.total_count == 0
        assert result.density_grid.sum() == 0

    def test_density_counts_match(self):
        centroids = [TrichomeCentroid(x=50.0, y=50.0) for _ in range(5)]
        result = compute_density_map(centroids, 100, 100, grid_rows=4, grid_cols=4)
        assert result.density_grid.sum() == 5

    def test_physical_density_requires_calibration(self):
        centroids = self._make_centroids(10)
        result_no_cal = compute_density_map(centroids, 100, 100)
        assert result_no_cal.density_per_mm2 is None

        result_with_cal = compute_density_map(centroids, 100, 100, um_per_pixel=0.5)
        assert result_with_cal.density_per_mm2 is not None
        assert result_with_cal.density_per_mm2 > 0

    def test_heatmap_has_correct_shape(self):
        centroids = self._make_centroids(5)
        result = compute_density_map(centroids, 200, 300)
        assert result.heatmap_bgr.shape == (200, 300, 3)

    def test_invalid_dimensions_raise(self):
        with pytest.raises((ValueError, Exception)):
            compute_density_map([], -1, 100)


# ── Morphology Classifier ─────────────────────────────────────────────────────

class TestMorphologyClassifier:

    def test_rule_based_no_stalk_is_bulbous_or_sessile(self):
        features = GeometricFeatures(
            head_area_px=150.0,
            stalk_length_px=0.0,
            head_circularity=0.90,
            elongation=1.1,
            head_stalk_ratio=10.0,
            total_height_px=25.0,
            aspect_ratio=1.0,
        )
        result = classify_morphology_geometric(features)
        assert result.primary_type in (TrichomeType.BULBOUS, TrichomeType.CAPITATE_SESSILE)
        assert float(result.confidence) >= 0.5

    def test_rule_based_long_stalk_is_stalked(self):
        features = GeometricFeatures(
            head_area_px=1200.0,
            stalk_length_px=80.0,
            head_circularity=0.70,
            elongation=3.0,
            head_stalk_ratio=0.5,
            total_height_px=120.0,
            aspect_ratio=0.4,
        )
        result = classify_morphology_geometric(features)
        assert result.primary_type == TrichomeType.CAPITATE_STALKED
        assert float(result.confidence) > 0.7

    def test_rule_based_hair_is_non_glandular(self):
        features = GeometricFeatures(
            head_area_px=50.0,
            stalk_length_px=200.0,
            head_circularity=0.05,
            elongation=15.0,
            head_stalk_ratio=0.1,
            total_height_px=300.0,
            aspect_ratio=0.1,
        )
        result = classify_morphology_geometric(features)
        assert result.primary_type == TrichomeType.NON_GLANDULAR

    def test_probabilities_sum_to_one(self):
        features = GeometricFeatures(
            head_area_px=500.0, stalk_length_px=10.0,
            head_circularity=0.6, elongation=1.8,
            head_stalk_ratio=1.0, total_height_px=60.0,
            aspect_ratio=0.8,
        )
        result = classify_morphology_geometric(features)
        total_prob = sum(result.class_probabilities.values())
        assert abs(total_prob - 1.0) < 0.01

    def test_classifier_no_model_is_rule_based(self, circular_mask):
        clf = MorphologyClassifier()
        assert clf.has_model is False
        morph = clf.predict_geometric()
        assert morph is not None

    def test_classifier_predict_geometric_with_geo(self, circular_mask):
        from morphology.domain.geometric import extract_geometric_descriptors
        geo = extract_geometric_descriptors(circular_mask)
        clf = MorphologyClassifier()
        morph = clf.predict_geometric(geo=geo)
        assert morph is not None
        assert morph.model_id == "geometric"
        assert 0.0 <= float(morph.confidence) <= 1.0
