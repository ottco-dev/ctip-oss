"""
tests.unit.test_measurement — Unit tests for measurement and calibration module.

Tests:
  - MicroscopeProfile creation and px↔µm conversion
  - ProfileManager CRUD
  - Stage micrometer calibration
  - Uncertainty propagation
  - Measurer pixel→µm conversion
  - Edge cases: zero area, no stalk, degenerate inputs
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from measurement.domain.profile_manager import (
    MicroscopeProfile,
    ProfileManager,
    DEFAULT_PROFILES,
)
from measurement.domain.measurer import Measurer, TrichomeMeasurements
from measurement.domain.propagation import (
    propagate_linear,
    propagate_area,
    propagate_ratio,
    combine_uncertainties,
    focus_induced_uncertainty,
)


# ── MicroscopeProfile ─────────────────────────────────────────────────────────

class TestMicroscopeProfile:

    def test_px_to_um_basic(self):
        p = MicroscopeProfile(um_per_pixel=0.16)
        assert abs(p.px_to_um(100) - 16.0) < 1e-9

    def test_um_to_px_inverse(self):
        p = MicroscopeProfile(um_per_pixel=0.325)
        px = p.um_to_px(100.0)
        assert abs(px - 100.0 / 0.325) < 1e-6

    def test_area_conversion(self):
        p = MicroscopeProfile(um_per_pixel=0.5)
        area_um2 = p.area_px_to_um2(100.0)
        assert abs(area_um2 - 25.0) < 1e-9  # 100 × 0.5² = 25

    def test_area_round_trip(self):
        p = MicroscopeProfile(um_per_pixel=0.25)
        area_px = 400.0
        area_um2 = p.area_px_to_um2(area_px)
        area_px_back = p.area_um2_to_px(area_um2)
        assert abs(area_px_back - area_px) < 1e-6

    def test_px_per_um_inverse(self):
        p = MicroscopeProfile(um_per_pixel=0.2)
        assert abs(p.px_per_um - 5.0) < 1e-9

    def test_invalid_um_per_pixel_raises(self):
        p = MicroscopeProfile(um_per_pixel=0.0)
        with pytest.raises((ValueError, ZeroDivisionError)):
            _ = p.px_per_um

    def test_validate_image_size_passes(self):
        p = MicroscopeProfile(um_per_pixel=0.16, image_width=2448, image_height=2048)
        assert p.validate_image_size(2448, 2048) is True

    def test_validate_image_size_fails(self):
        p = MicroscopeProfile(um_per_pixel=0.16, image_width=2448, image_height=2048)
        assert p.validate_image_size(1920, 1080) is False

    def test_validate_no_expected_size_always_passes(self):
        p = MicroscopeProfile(um_per_pixel=0.16)
        assert p.validate_image_size(999, 999) is True

    def test_to_dict_roundtrip(self):
        p = MicroscopeProfile(name="Test", um_per_pixel=0.3, objective="20x")
        d = p.to_dict()
        p2 = MicroscopeProfile.from_dict(d)
        assert p2.name == "Test"
        assert p2.um_per_pixel == 0.3
        assert p2.objective == "20x"


# ── ProfileManager ────────────────────────────────────────────────────────────

class TestProfileManager:

    def test_default_profiles_loaded(self):
        pm = ProfileManager()
        profiles = pm.list_profiles()
        ids = [p.profile_id for p in profiles]
        assert "40x_generic" in ids

    def test_add_and_retrieve(self):
        pm = ProfileManager()
        p = MicroscopeProfile(name="My Custom", um_per_pixel=0.12)
        pm.add_profile(p)
        retrieved = pm.get_profile(p.profile_id)
        assert retrieved is not None
        assert retrieved.name == "My Custom"

    def test_delete_custom_profile(self):
        pm = ProfileManager()
        p = MicroscopeProfile(name="Temp", um_per_pixel=0.5)
        pm.add_profile(p)
        assert pm.get_profile(p.profile_id) is not None
        pm.delete_profile(p.profile_id)
        assert pm.get_profile(p.profile_id) is None

    def test_cannot_delete_builtin(self):
        pm = ProfileManager()
        with pytest.raises(ValueError):
            pm.delete_profile("40x_generic")

    def test_set_default(self):
        pm = ProfileManager()
        p = MicroscopeProfile(name="Custom Default", um_per_pixel=0.1)
        pm.add_profile(p)
        pm.set_default(p.profile_id)
        assert pm.default_profile.profile_id == p.profile_id

    def test_set_default_nonexistent_raises(self):
        pm = ProfileManager()
        with pytest.raises(KeyError):
            pm.set_default("nonexistent_id")

    def test_create_from_stage_micrometer(self):
        pm = ProfileManager()
        profile = pm.create_from_stage_micrometer(
            name="Test Calibration",
            scale_bar_px=500.0,
            scale_bar_um=100.0,
            objective="40x",
        )
        assert abs(profile.um_per_pixel - 0.2) < 1e-9
        assert profile.calibration_method == "stage_micrometer"
        assert profile.uncertainty_um is not None
        assert profile.uncertainty_um > 0

    def test_stage_micrometer_uncertainty_formula(self):
        pm = ProfileManager()
        profile = pm.create_from_stage_micrometer(
            name="Precision Cal",
            scale_bar_px=1000.0,
            scale_bar_um=200.0,
        )
        # Expected: um_per_pixel = 0.2, uncertainty = 0.2 / 1000 = 0.0002
        assert abs(profile.um_per_pixel - 0.2) < 1e-9
        expected_unc = 0.2 / 1000.0
        assert abs(profile.uncertainty_um - expected_unc) < 1e-10

    def test_invalid_scale_bar_raises(self):
        pm = ProfileManager()
        with pytest.raises(ValueError):
            pm.create_from_stage_micrometer("X", scale_bar_px=-1.0, scale_bar_um=100.0)


# ── Uncertainty Propagation ───────────────────────────────────────────────────

class TestUncertaintyPropagation:

    def test_combine_orthogonal(self):
        # Pythagorean: √(3² + 4²) = 5
        combined = combine_uncertainties(3.0, 4.0)
        assert abs(combined - 5.0) < 1e-9

    def test_combine_single(self):
        assert combine_uncertainties(2.5) == 2.5

    def test_linear_propagation_basic(self):
        result = propagate_linear(value_px=100.0, um_per_pixel=0.2,
                                  calibration_uncertainty_um=0.0,
                                  edge_uncertainty_px=0.0)
        assert abs(result.value - 20.0) < 1e-9
        assert result.unit == "µm"

    def test_linear_propagation_with_edge_uncertainty(self):
        result = propagate_linear(value_px=100.0, um_per_pixel=0.2,
                                  edge_uncertainty_px=1.0)
        assert result.uncertainty > 0

    def test_area_propagation(self):
        result = propagate_area(area_px2=100.0, um_per_pixel=0.5)
        assert abs(result.value - 25.0) < 1e-9
        assert result.uncertainty > 0
        assert result.unit == "µm²"

    def test_ratio_propagation(self):
        from measurement.domain.propagation import MeasurementWithUncertainty
        num = MeasurementWithUncertainty(value=10.0, uncertainty=0.5)
        den = MeasurementWithUncertainty(value=5.0, uncertainty=0.25)
        ratio = propagate_ratio(num, den)
        assert abs(ratio.value - 2.0) < 1e-9
        assert ratio.uncertainty > 0

    def test_ratio_zero_denominator(self):
        from measurement.domain.propagation import MeasurementWithUncertainty
        num = MeasurementWithUncertainty(value=10.0, uncertainty=0.5)
        den = MeasurementWithUncertainty(value=0.0, uncertainty=0.0)
        ratio = propagate_ratio(num, den)
        assert ratio.value == float("inf")

    def test_expanded_uncertainty(self):
        result = propagate_linear(100.0, 0.2, edge_uncertainty_px=1.0)
        assert result.expanded_uncertainty == result.coverage_factor * result.uncertainty

    def test_relative_uncertainty(self):
        result = propagate_linear(100.0, 0.2, edge_uncertainty_px=1.0)
        assert 0 < result.relative_uncertainty < 1

    def test_focus_induced_uncertainty(self):
        # Perfect focus → 0 uncertainty
        assert focus_induced_uncertainty(1.0, 0.2) == 0.0
        # No focus → max uncertainty
        assert focus_induced_uncertainty(0.0, 0.2) > 0.0
        # Monotonically decreasing with focus score
        u1 = focus_induced_uncertainty(0.3, 0.2)
        u2 = focus_induced_uncertainty(0.7, 0.2)
        assert u1 > u2


# ── Measurer ─────────────────────────────────────────────────────────────────

class TestMeasurer:

    @pytest.fixture
    def profile(self) -> MicroscopeProfile:
        return MicroscopeProfile(um_per_pixel=0.25, uncertainty_um=0.001)

    @pytest.fixture
    def measurer(self, profile) -> Measurer:
        return Measurer(profile)

    def test_head_diameter_conversion(self, measurer):
        result = measurer.measure(head_diameter_px=200.0)
        assert abs(result.head_diameter_um - 50.0) < 1e-6

    def test_area_conversion(self, measurer):
        result = measurer.measure(head_area_px=400.0)
        assert abs(result.head_area_um2 - 25.0) < 1e-6  # 400 × 0.25²

    def test_stalk_conversion(self, measurer):
        result = measurer.measure(stalk_length_px=100.0)
        assert abs(result.stalk_length_um - 25.0) < 1e-6

    def test_none_inputs_produce_none_outputs(self, measurer):
        result = measurer.measure()
        assert result.head_diameter_um is None
        assert result.stalk_length_um is None

    def test_head_stalk_ratio(self, measurer):
        result = measurer.measure(head_diameter_px=100.0, stalk_length_px=50.0)
        # head_d = 25µm, stalk_l = 12.5µm → ratio = 2.0
        assert result.head_stalk_ratio is not None
        assert abs(result.head_stalk_ratio - 2.0) < 1e-6

    def test_morphology_hint_stalked(self, measurer):
        result = measurer.measure(head_diameter_px=300.0, stalk_length_px=200.0)
        # head=75µm, stalk=50µm → stalked
        assert result.morphology_hint in ("capitate_stalked", "capitate_sessile")

    def test_morphology_hint_bulbous(self, measurer):
        result = measurer.measure(head_diameter_px=60.0, stalk_length_px=0.0)
        # head_d = 15µm → bulbous
        assert result.morphology_hint == "bulbous"

    def test_uncertainties_are_positive(self, measurer):
        result = measurer.measure(head_diameter_px=200.0, stalk_length_px=80.0)
        assert result.head_diameter_uncertainty_um > 0
        assert result.stalk_length_uncertainty_um > 0

    def test_invalid_profile_raises(self):
        bad_profile = MicroscopeProfile(um_per_pixel=0.0)
        with pytest.raises(ValueError):
            Measurer(bad_profile)

    def test_profile_metadata_in_result(self, measurer, profile):
        result = measurer.measure(head_diameter_px=100.0)
        assert result.um_per_pixel == profile.um_per_pixel
        assert result.calibration_method == profile.calibration_method


# ---------------------------------------------------------------------------
# ScaleBarDetector tests (TDB-001)
# ---------------------------------------------------------------------------

class TestScaleBarDetector:
    """Tests for automated stage micrometer scale bar detection."""

    def _make_scale_bar_image(
        self,
        width: int = 800,
        height: int = 200,
        bar_x1: int = 100,
        bar_x2: int = 700,
        bar_y: int = 100,
        thickness: int = 2,
        bg: int = 240,
        fg: int = 10,
    ):
        """Create a synthetic stage micrometer image with a horizontal bar."""
        import cv2
        import numpy as np
        img = np.ones((height, width), dtype=np.uint8) * bg
        cv2.line(img, (bar_x1, bar_y), (bar_x2, bar_y), fg, thickness)
        # Add tick marks for realism
        cv2.line(img, (bar_x1, bar_y - 10), (bar_x1, bar_y + 10), fg, 2)
        cv2.line(img, (bar_x2, bar_y - 10), (bar_x2, bar_y + 10), fg, 2)
        return img

    def test_detects_horizontal_bar(self):
        from measurement.calibration.stage_micrometer import detect_scale_bar_px
        img = self._make_scale_bar_image(bar_x1=100, bar_x2=700)
        result = detect_scale_bar_px(img)
        assert result.detected
        assert result.scale_bar_px > 0
        assert result.confidence > 0.5

    def test_detected_span_close_to_true_length(self):
        from measurement.calibration.stage_micrometer import detect_scale_bar_px
        true_span = 600  # 700 - 100
        img = self._make_scale_bar_image(bar_x1=100, bar_x2=700)
        result = detect_scale_bar_px(img)
        # Allow ±5% tolerance for edge/endpoint detection variance
        assert abs(result.scale_bar_px - true_span) / true_span < 0.05

    def test_no_detection_on_blank_image(self):
        from measurement.calibration.stage_micrometer import detect_scale_bar_px
        blank = np.ones((200, 800), dtype=np.uint8) * 200
        result = detect_scale_bar_px(blank)
        assert not result.detected
        assert result.scale_bar_px == 0.0

    def test_no_detection_on_vertical_only_lines(self):
        from measurement.calibration.stage_micrometer import detect_scale_bar_px
        import cv2
        img = np.ones((200, 400), dtype=np.uint8) * 240
        # Only draw vertical lines (should be filtered out)
        for x in range(50, 351, 50):
            cv2.line(img, (x, 20), (x, 180), 10, 2)
        result = detect_scale_bar_px(img, max_angle_deg=2.0)
        assert not result.detected

    def test_result_has_required_fields(self):
        from measurement.calibration.stage_micrometer import detect_scale_bar_px, ScaleBarDetectionResult
        img = self._make_scale_bar_image()
        result = detect_scale_bar_px(img)
        assert isinstance(result, ScaleBarDetectionResult)
        assert isinstance(result.detected, bool)
        assert isinstance(result.scale_bar_px, float)
        assert 0.0 <= result.confidence <= 1.0
        assert isinstance(result.message, str)
        assert result.method == "hough"

    def test_confidence_scales_with_bar_size(self):
        from measurement.calibration.stage_micrometer import detect_scale_bar_px
        # Short bar → lower confidence
        img_short = self._make_scale_bar_image(width=800, bar_x1=350, bar_x2=450)
        # Long bar → higher confidence
        img_long = self._make_scale_bar_image(width=800, bar_x1=100, bar_x2=700)
        r_short = detect_scale_bar_px(img_short)
        r_long = detect_scale_bar_px(img_long)
        if r_short.detected and r_long.detected:
            assert r_short.confidence <= r_long.confidence
