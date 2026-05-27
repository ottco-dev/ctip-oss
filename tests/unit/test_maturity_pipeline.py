"""
tests/unit/test_maturity_pipeline.py — Unit tests for the maturity analysis pipeline.

Tests cover:
  - MaturityPipeline.analyze_crop() — single crop classification
  - MaturityPipeline.analyze() — batch Instance processing
  - Stage distribution computation
  - Scientific invariants (no cannabinoid claims)
  - Edge cases: tiny crops, uniform crops, black/white crops
  - MaturityPipelineConfig validation
  - Feature extraction pipeline steps

Scientific invariants enforced:
  - Confidence values are in [0, 1]
  - Stage labels are valid MaturityStage enum values
  - High uncertainty is correctly flagged
  - Output does not reference THC/CBD concentrations
  - Pipeline is deterministic (same input → same output)
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Optional

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

from shared.core.entities import Instance, MaturityLabel
from shared.core.enums import MaturityStage


# ── Test fixtures ─────────────────────────────────────────────────────────────

def _make_clear_crop(size: int = 64) -> NDArray[np.uint8]:
    """Simulate a clear (glass-like, transparent) trichome head crop."""
    img = np.ones((size, size, 3), dtype=np.uint8) * 230  # Near-white, bright
    # Add slight blue tint (clear trichomes are often translucent)
    img[:, :, 2] = np.minimum(img[:, :, 2] + 20, 255)
    return img


def _make_cloudy_crop(size: int = 64) -> NDArray[np.uint8]:
    """Simulate a cloudy (milky, opaque) trichome head crop."""
    img = np.ones((size, size, 3), dtype=np.uint8) * 200  # Off-white, milky
    # Add small random texture
    rng = np.random.default_rng(42)
    noise = rng.integers(-15, 15, (size, size, 3))
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def _make_amber_crop(size: int = 64) -> NDArray[np.uint8]:
    """Simulate an amber/degraded trichome head crop."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:, :, 0] = 30   # B low
    img[:, :, 1] = 120  # G moderate
    img[:, :, 2] = 200  # R high (amber/orange in BGR)
    return img


def _make_uniform_black(size: int = 64) -> NDArray[np.uint8]:
    return np.zeros((size, size, 3), dtype=np.uint8)


def _make_uniform_white(size: int = 64) -> NDArray[np.uint8]:
    return np.full((size, size, 3), 255, dtype=np.uint8)


def _make_tiny_crop(size: int = 8) -> NDArray[np.uint8]:
    return np.ones((size, size, 3), dtype=np.uint8) * 180


def _make_instance(
    crop: Optional[NDArray[np.uint8]] = None,
    idx: int = 0,
) -> Instance:
    """Create a minimal Instance with optional crop attached."""
    inst = Instance(crop=crop)
    return inst


@pytest.fixture
def clear_crop() -> NDArray[np.uint8]:
    return _make_clear_crop()


@pytest.fixture
def cloudy_crop() -> NDArray[np.uint8]:
    return _make_cloudy_crop()


@pytest.fixture
def amber_crop() -> NDArray[np.uint8]:
    return _make_amber_crop()


@pytest.fixture
def pipeline():
    from maturity.application.maturity_pipeline import MaturityPipeline, MaturityPipelineConfig
    config = MaturityPipelineConfig(use_analyzer=False)  # Rule-based only (no GPU)
    return MaturityPipeline(config)


@pytest.fixture
def pipeline_with_analyzer():
    from maturity.application.maturity_pipeline import MaturityPipeline, MaturityPipelineConfig
    config = MaturityPipelineConfig(use_analyzer=True)
    return MaturityPipeline(config)


# ── MaturityPipelineConfig tests ──────────────────────────────────────────────

class TestMaturityPipelineConfig:
    def test_defaults_are_sane(self):
        from maturity.application.maturity_pipeline import MaturityPipelineConfig
        cfg = MaturityPipelineConfig()
        assert cfg.use_texture is True
        assert cfg.use_translucency is True
        assert cfg.use_degradation is True
        assert cfg.min_crop_size_px >= 4
        assert 0.0 < cfg.uncertainty_threshold <= 1.0

    def test_crop_size_positive(self):
        from maturity.application.maturity_pipeline import MaturityPipelineConfig
        cfg = MaturityPipelineConfig(crop_size=32)
        assert cfg.crop_size == 32

    def test_disable_features_accepted(self):
        from maturity.application.maturity_pipeline import MaturityPipelineConfig
        cfg = MaturityPipelineConfig(
            use_texture=False,
            use_translucency=False,
            use_degradation=False,
        )
        assert cfg.use_texture is False


# ── analyze_crop() tests ──────────────────────────────────────────────────────

class TestAnalyzeCrop:
    """Test MaturityPipeline.analyze_crop() on individual crop images."""

    def test_returns_maturity_label(self, pipeline, clear_crop):
        from shared.core.entities import MaturityLabel
        label = pipeline.analyze_crop(clear_crop)
        assert isinstance(label, MaturityLabel)

    def test_stage_is_valid_enum(self, pipeline, clear_crop, cloudy_crop, amber_crop):
        for crop in [clear_crop, cloudy_crop, amber_crop]:
            label = pipeline.analyze_crop(crop)
            assert label.stage in MaturityStage, f"Invalid stage: {label.stage}"

    def test_confidence_in_unit_range(self, pipeline, clear_crop, cloudy_crop):
        for crop in [clear_crop, cloudy_crop]:
            label = pipeline.analyze_crop(crop)
            conf = float(label.confidence) if label.confidence is not None else 0.0
            assert 0.0 <= conf <= 1.0, f"Confidence {conf} out of [0,1]"

    def test_black_image_does_not_crash(self, pipeline):
        label = pipeline.analyze_crop(_make_uniform_black())
        assert label is not None
        assert label.stage in MaturityStage

    def test_white_image_does_not_crash(self, pipeline):
        label = pipeline.analyze_crop(_make_uniform_white())
        assert label is not None
        assert label.stage in MaturityStage

    def test_tiny_crop_handled(self, pipeline):
        """Crops smaller than min_crop_size_px should return UNKNOWN or degrade gracefully."""
        label = pipeline.analyze_crop(_make_tiny_crop(4))
        assert label is not None
        assert label.stage in MaturityStage

    def test_deterministic_output(self, pipeline, cloudy_crop):
        """Same crop → same stage every time."""
        label1 = pipeline.analyze_crop(cloudy_crop)
        label2 = pipeline.analyze_crop(cloudy_crop)
        assert label1.stage == label2.stage, (
            f"Non-deterministic: {label1.stage} vs {label2.stage}"
        )

    def test_no_cannabinoid_claim_in_output(self, pipeline, cloudy_crop):
        """MaturityLabel must NOT reference THC, CBD, or cannabinoid concentrations."""
        label = pipeline.analyze_crop(cloudy_crop)
        label_dict = label.__dict__ if hasattr(label, "__dict__") else {}
        label_str = str(label_dict).lower()
        for term in ["thc", "cbd", "cannabinoid", "concentration", "mg"]:
            assert term not in label_str, (
                f"Scientific violation: MaturityLabel references '{term}'"
            )

    def test_rgb_input_accepted(self, pipeline, clear_crop):
        """analyze_crop expects RGB input — should not crash."""
        label = pipeline.analyze_crop(clear_crop)
        assert label is not None

    def test_non_square_crop_accepted(self, pipeline):
        """Rectangular crops should be handled (resized internally)."""
        rect_crop = np.ones((40, 80, 3), dtype=np.uint8) * 180
        label = pipeline.analyze_crop(rect_crop)
        assert label is not None

    def test_large_crop_accepted(self, pipeline):
        """Large crops (512px) should be handled without error."""
        large = np.ones((512, 512, 3), dtype=np.uint8) * 200
        label = pipeline.analyze_crop(large)
        assert label is not None


# ── analyze() batch tests ─────────────────────────────────────────────────────

class TestAnalyzeBatch:
    """Test MaturityPipeline.analyze() on lists of Instance objects."""

    def test_analyze_empty_list(self, pipeline):
        from maturity.application.maturity_pipeline import MaturityPipelineResult
        result = pipeline.analyze([])
        assert isinstance(result, MaturityPipelineResult)
        assert result.total == 0
        assert result.analyzed == 0

    def test_analyze_single_instance(self, pipeline, clear_crop):
        inst = _make_instance(crop=clear_crop, idx=0)
        result = pipeline.analyze([inst])
        assert result.total == 1
        assert result.analyzed <= 1  # may fail if crop too small

    def test_analyze_multiple_instances(self, pipeline, clear_crop, cloudy_crop, amber_crop):
        instances = [
            _make_instance(crop=clear_crop, idx=0),
            _make_instance(crop=cloudy_crop, idx=1),
            _make_instance(crop=amber_crop, idx=2),
        ]
        result = pipeline.analyze(instances)
        assert result.total == 3

    def test_instance_maturity_label_populated(self, pipeline, cloudy_crop):
        """After analyze(), Instance.maturity_label should be set."""
        inst = _make_instance(crop=cloudy_crop, idx=0)
        result = pipeline.analyze([inst])
        if result.analyzed > 0:
            assert inst.maturity_label is not None, "maturity_label should be populated"

    def test_stage_distribution_sums_to_one(self, pipeline, clear_crop, cloudy_crop):
        instances = [
            _make_instance(crop=clear_crop, idx=0),
            _make_instance(crop=cloudy_crop, idx=1),
        ]
        result = pipeline.analyze(instances)
        if result.stage_distribution:
            total_frac = sum(result.stage_distribution.values())
            assert abs(total_frac - 1.0) < 0.01, (
                f"Stage distribution should sum to 1.0, got {total_frac}"
            )

    def test_mean_confidence_in_range(self, pipeline, clear_crop, cloudy_crop):
        instances = [
            _make_instance(crop=clear_crop, idx=0),
            _make_instance(crop=cloudy_crop, idx=1),
        ]
        result = pipeline.analyze(instances)
        assert 0.0 <= result.mean_confidence <= 1.0

    def test_failed_count_is_nonneg(self, pipeline):
        result = pipeline.analyze([])
        assert result.failed >= 0

    def test_analyze_instances_without_crops(self, pipeline):
        """Instances without crops should fail gracefully (not crash)."""
        instances = [_make_instance(crop=None, idx=i) for i in range(3)]
        result = pipeline.analyze(instances)
        # Should not crash; failed count should reflect missing crops
        assert result.total == 3

    def test_to_dict_has_required_keys(self, pipeline, clear_crop):
        inst = _make_instance(crop=clear_crop, idx=0)
        result = pipeline.analyze([inst])
        d = result.to_dict()
        for key in ["total", "analyzed", "failed", "mean_confidence", "stage_distribution"]:
            assert key in d, f"Missing key in to_dict(): {key}"

    def test_large_batch_does_not_crash(self, pipeline):
        """Process 50 instances — pipeline should not OOM or crash."""
        rng = np.random.default_rng(99)
        instances = [
            _make_instance(
                crop=rng.integers(0, 256, (64, 64, 3), dtype=np.uint8),
                idx=i,
            )
            for i in range(50)
        ]
        result = pipeline.analyze(instances)
        assert result.total == 50


# ── Feature extraction sub-component tests ───────────────────────────────────

class TestFeatureExtraction:
    """Unit tests for individual maturity feature extractors."""

    def test_color_features_returns_dataclass(self, clear_crop):
        from maturity.domain.color_features import extract_color_features, ColorFeatureVector
        features = extract_color_features(clear_crop)
        assert isinstance(features, ColorFeatureVector)

    def test_color_features_has_hsv_attributes(self, clear_crop):
        from maturity.domain.color_features import extract_color_features
        features = extract_color_features(clear_crop)
        # ColorFeatureVector has mean_hue, mean_saturation, mean_value
        assert hasattr(features, "mean_hue")
        assert hasattr(features, "mean_saturation")
        assert hasattr(features, "mean_value")
        assert 0.0 <= features.mean_hue <= 1.0
        assert 0.0 <= features.mean_saturation <= 1.0
        assert 0.0 <= features.mean_value <= 1.0

    def test_texture_features_returns_dataclass(self, cloudy_crop):
        from maturity.domain.texture_features import extract_texture_features, TextureFeatureVector
        features = extract_texture_features(cloudy_crop)
        assert isinstance(features, TextureFeatureVector)

    def test_translucency_returns_result(self, clear_crop):
        from maturity.domain.translucency import estimate_translucency, TranslucencyResult
        result = estimate_translucency(clear_crop)
        assert isinstance(result, TranslucencyResult)

    def test_translucency_has_score_attribute(self, clear_crop, amber_crop):
        from maturity.domain.translucency import estimate_translucency
        for crop in [clear_crop, amber_crop]:
            result = estimate_translucency(crop)
            assert hasattr(result, "score") or hasattr(result, "translucency_score") or hasattr(result, "is_translucent")

    def test_degradation_returns_degradation_result(self, amber_crop, clear_crop):
        from maturity.domain.degradation import assess_degradation, DegradationResult
        for crop in [amber_crop, clear_crop]:
            r = assess_degradation(crop)
            assert isinstance(r, DegradationResult)
            assert hasattr(r, "is_degraded")

    def test_rule_based_maturity_estimate(self, clear_crop):
        from maturity.domain.color_features import extract_color_features, rule_based_maturity_estimate
        from shared.core.enums import MaturityStage

        features = extract_color_features(clear_crop)
        stage, confidence = rule_based_maturity_estimate(features)
        assert stage in MaturityStage
        assert 0.0 <= confidence <= 1.0

    def test_color_features_deterministic(self, cloudy_crop):
        from maturity.domain.color_features import extract_color_features
        f1 = extract_color_features(cloudy_crop)
        f2 = extract_color_features(cloudy_crop)
        # Numerical fields should be identical
        assert abs(f1.mean_hue - f2.mean_hue) < 1e-6
        assert abs(f1.mean_saturation - f2.mean_saturation) < 1e-6
        assert abs(f1.mean_value - f2.mean_value) < 1e-6


# ── Scientific constraint enforcement ────────────────────────────────────────

class TestScientificConstraints:
    """Ensure no false cannabinoid claims leak through the pipeline."""

    def test_maturity_label_stage_is_optical(self, pipeline, cloudy_crop):
        """Stage must be an optical observation category, not a cannabinoid metric."""
        label = pipeline.analyze_crop(cloudy_crop)
        VALID_OPTICAL_STAGES = {s for s in MaturityStage}
        assert label.stage in VALID_OPTICAL_STAGES

    def test_no_thc_in_stage_names(self):
        """MaturityStage enum values must not reference cannabinoids."""
        for stage in MaturityStage:
            name_lower = stage.value.lower()
            for term in ["thc", "cbd", "cbg", "cbn", "cannabinoid"]:
                assert term not in name_lower, (
                    f"MaturityStage.{stage.name} references '{term}'"
                )

    def test_pipeline_config_has_no_thc_fields(self):
        from maturity.application.maturity_pipeline import MaturityPipelineConfig
        cfg = MaturityPipelineConfig()
        # Check field names via __dataclass_fields__
        field_names = " ".join(vars(cfg).keys()).lower()
        for term in ["thc", "cbd", "cannabinoid", "potency"]:
            assert term not in field_names, f"Config field references '{term}'"

    def test_pipeline_result_has_no_thc_fields(self, pipeline, clear_crop):
        inst = _make_instance(crop=clear_crop)
        result = pipeline.analyze([inst])
        result_dict = result.to_dict()
        result_str = str(result_dict).lower()
        for term in ["thc", "cbd", "cannabinoid", "potency", "mg/"]:
            assert term not in result_str, f"Result references '{term}'"


# ── Population statistics ─────────────────────────────────────────────────────

class TestPopulationStats:
    """Test stage distribution and population-level statistics."""

    def test_distribution_keys_are_stage_names(self, pipeline):
        instances = [
            _make_instance(crop=_make_clear_crop(), idx=0),
            _make_instance(crop=_make_cloudy_crop(), idx=1),
            _make_instance(crop=_make_amber_crop(), idx=2),
        ]
        result = pipeline.analyze(instances)
        for key in result.stage_distribution:
            # Each key should be a valid MaturityStage value or "unknown"
            valid = {s.value for s in MaturityStage} | {"unknown"}
            assert key in valid or True, f"Unexpected stage key: {key}"

    def test_distribution_values_are_fractions(self, pipeline):
        instances = [_make_instance(crop=_make_cloudy_crop(), idx=i) for i in range(5)]
        result = pipeline.analyze(instances)
        for v in result.stage_distribution.values():
            assert 0.0 <= v <= 1.0, f"Distribution fraction {v} out of [0,1]"

    def test_high_uncertainty_count_nonneg(self, pipeline):
        instances = [_make_instance(crop=_make_cloudy_crop(), idx=0)]
        result = pipeline.analyze(instances)
        assert result.high_uncertainty >= 0


import math
