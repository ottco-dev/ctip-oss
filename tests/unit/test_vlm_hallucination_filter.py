"""
tests/unit/test_vlm_hallucination_filter.py — VLM hallucination filter tests.

Tests the HITL safety gate:
  HallucinationFilter — maturity, quality, morphology, cross-model filtering
  FilterResult — flags, priority, confidence penalty
  HallucinationFilterConfig — threshold effects

Scientific rationale: VLM hallucinations in training data corrupt the model.
Detected hallucinations are NOT rejected — they are escalated to human review.
"""

from __future__ import annotations

import pytest

from vlm_labeling.filtering.hallucination import (
    HallucinationFilter,
    HallucinationFilterConfig,
    HallucinationFlag,
    FilterResult,
    VALID_MATURITY_STAGES,
    VALID_QUALITY_LEVELS,
    VALID_MORPHOLOGY_TYPES,
)


# ─────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────

@pytest.fixture
def filt():
    return HallucinationFilter()


def _maturity_response(
    stage: str = "cloudy",
    confidence: float = 0.75,
    clear: float = 0.1,
    cloudy: float = 0.8,
    amber: float = 0.1,
) -> dict:
    return {
        "maturity_stage": stage,
        "confidence": confidence,
        "clear_fraction_estimate": clear,
        "cloudy_fraction_estimate": cloudy,
        "amber_fraction_estimate": amber,
        "reasoning": "Test response",
    }


# ─────────────────────────────────────────────────────────────────
# 1. None / invalid JSON → INVALID_JSON flag
# ─────────────────────────────────────────────────────────────────

class TestInvalidJsonHandling:

    def test_none_response_is_not_passed(self, filt):
        r = filt.filter_maturity(None)
        assert r.passed is False

    def test_none_response_sets_invalid_json_flag(self, filt):
        r = filt.filter_maturity(None)
        assert HallucinationFlag.INVALID_JSON in r.flags

    def test_none_confidence_is_zero(self, filt):
        r = filt.filter_maturity(None)
        assert r.adjusted_confidence == 0.0

    def test_none_priority_is_critical(self, filt):
        r = filt.filter_maturity(None)
        assert r.review_priority == 3

    def test_none_quality_response_flagged(self, filt):
        r = filt.filter_quality(None)
        assert HallucinationFlag.INVALID_JSON in r.flags
        assert r.passed is False

    def test_none_morphology_response_flagged(self, filt):
        r = filt.filter_morphology(None)
        assert HallucinationFlag.INVALID_JSON in r.flags
        assert r.passed is False


# ─────────────────────────────────────────────────────────────────
# 2. Clean passing response
# ─────────────────────────────────────────────────────────────────

class TestCleanResponse:

    def test_clean_cloudy_passes(self, filt):
        r = filt.filter_maturity(_maturity_response("cloudy", confidence=0.85))
        assert r.passed is True
        assert r.flags == []

    def test_clean_clear_passes(self, filt):
        r = filt.filter_maturity(_maturity_response(
            "clear", confidence=0.82, clear=0.85, cloudy=0.10, amber=0.05
        ))
        assert r.passed is True

    def test_clean_amber_passes(self, filt):
        r = filt.filter_maturity(_maturity_response(
            "amber", confidence=0.78, clear=0.05, cloudy=0.25, amber=0.70
        ))
        assert r.passed is True

    def test_confidence_preserved_on_clean(self, filt):
        r = filt.filter_maturity(_maturity_response("cloudy", confidence=0.80))
        assert r.adjusted_confidence == pytest.approx(0.80, abs=0.01)

    def test_passed_result_is_reliable(self, filt):
        r = filt.filter_maturity(_maturity_response("cloudy", confidence=0.85))
        assert r.is_reliable is True

    def test_passed_result_not_flagged(self, filt):
        r = filt.filter_maturity(_maturity_response("cloudy", confidence=0.85))
        assert r.is_flagged is False


# ─────────────────────────────────────────────────────────────────
# 3. Unknown class → UNKNOWN_CLASS flag
# ─────────────────────────────────────────────────────────────────

class TestUnknownClass:

    def test_unknown_maturity_stage_flagged(self, filt):
        r = filt.filter_maturity(_maturity_response("super_cloudy", confidence=0.80))
        assert HallucinationFlag.UNKNOWN_CLASS in r.flags

    def test_unknown_class_penalises_confidence_heavily(self, filt):
        r = filt.filter_maturity(_maturity_response("??", confidence=0.90))
        assert r.adjusted_confidence < 0.40  # 0.3× penalty

    def test_all_valid_maturity_stages_pass_class_check(self, filt):
        for stage in VALID_MATURITY_STAGES:
            base = _maturity_response(confidence=0.85)
            base["maturity_stage"] = stage
            # Adjust fractions to match stage (avoid semantic flag)
            if stage == "amber":
                base.update({"amber_fraction_estimate": 0.6, "cloudy_fraction_estimate": 0.3, "clear_fraction_estimate": 0.1})
            elif stage == "clear":
                base.update({"clear_fraction_estimate": 0.8, "cloudy_fraction_estimate": 0.15, "amber_fraction_estimate": 0.05})
            r = filt.filter_maturity(base)
            assert HallucinationFlag.UNKNOWN_CLASS not in r.flags, f"Stage {stage} wrongly flagged"

    def test_unknown_quality_level_flagged(self, filt):
        r = filt.filter_quality({"overall_quality": "perfect", "confidence": 0.80})
        assert HallucinationFlag.UNKNOWN_CLASS in r.flags

    def test_all_valid_quality_levels_pass(self, filt):
        for level in VALID_QUALITY_LEVELS:
            r = filt.filter_quality({"overall_quality": level, "confidence": 0.75})
            assert HallucinationFlag.UNKNOWN_CLASS not in r.flags

    def test_unknown_morphology_type_flagged(self, filt):
        r = filt.filter_morphology({"dominant_type": "glandular_giant", "confidence": 0.80})
        assert HallucinationFlag.UNKNOWN_CLASS in r.flags

    def test_all_valid_morphology_types_pass(self, filt):
        for t in VALID_MORPHOLOGY_TYPES:
            r = filt.filter_morphology({"dominant_type": t, "confidence": 0.75})
            assert HallucinationFlag.UNKNOWN_CLASS not in r.flags


# ─────────────────────────────────────────────────────────────────
# 4. Low confidence → LOW_CONFIDENCE flag
# ─────────────────────────────────────────────────────────────────

class TestLowConfidence:

    def test_below_threshold_flagged(self, filt):
        r = filt.filter_maturity(_maturity_response("cloudy", confidence=0.20))
        assert HallucinationFlag.LOW_CONFIDENCE in r.flags

    def test_above_threshold_not_flagged(self, filt):
        r = filt.filter_maturity(_maturity_response("cloudy", confidence=0.75))
        assert HallucinationFlag.LOW_CONFIDENCE not in r.flags

    def test_zero_confidence_flagged(self, filt):
        r = filt.filter_maturity(_maturity_response("cloudy", confidence=0.0))
        assert HallucinationFlag.LOW_CONFIDENCE in r.flags

    def test_custom_threshold_respected(self):
        cfg = HallucinationFilterConfig(min_confidence=0.60)
        filt = HallucinationFilter(config=cfg)
        # 0.55 should fail with threshold=0.60
        r = filt.filter_maturity(_maturity_response("cloudy", confidence=0.55))
        assert HallucinationFlag.LOW_CONFIDENCE in r.flags
        # 0.65 should pass
        r2 = filt.filter_maturity(_maturity_response("cloudy", confidence=0.65))
        assert HallucinationFlag.LOW_CONFIDENCE not in r2.flags


# ─────────────────────────────────────────────────────────────────
# 5. Constraint violations
# ─────────────────────────────────────────────────────────────────

class TestConstraintViolations:

    def test_fraction_out_of_range_flagged(self, filt):
        r = filt.filter_maturity(_maturity_response(
            "cloudy", confidence=0.80, clear=-0.1, cloudy=0.9, amber=0.1
        ))
        assert HallucinationFlag.CONSTRAINT_VIOLATION in r.flags

    def test_fraction_above_one_flagged(self, filt):
        r = filt.filter_maturity(_maturity_response(
            "cloudy", confidence=0.80, clear=0.1, cloudy=1.5, amber=0.1
        ))
        assert HallucinationFlag.CONSTRAINT_VIOLATION in r.flags

    def test_fraction_sum_far_from_one_flagged(self, filt):
        # Sum = 0.1 + 0.1 + 0.1 = 0.3 — far from 1.0
        r = filt.filter_maturity(_maturity_response(
            "cloudy", confidence=0.80, clear=0.1, cloudy=0.1, amber=0.1
        ))
        assert HallucinationFlag.CONSTRAINT_VIOLATION in r.flags

    def test_fraction_sum_near_one_passes(self, filt):
        # Sum = 0.10 + 0.80 + 0.10 = 1.00
        r = filt.filter_maturity(_maturity_response(
            "cloudy", confidence=0.85, clear=0.10, cloudy=0.80, amber=0.10
        ))
        assert HallucinationFlag.CONSTRAINT_VIOLATION not in r.flags

    def test_constraint_violation_penalises_confidence(self, filt):
        r_clean = filt.filter_maturity(_maturity_response("cloudy", confidence=0.80))
        r_bad = filt.filter_maturity(_maturity_response(
            "cloudy", confidence=0.80, clear=0.0, cloudy=0.0, amber=0.0
        ))
        assert r_bad.adjusted_confidence < r_clean.adjusted_confidence


# ─────────────────────────────────────────────────────────────────
# 6. Semantic inconsistency
# ─────────────────────────────────────────────────────────────────

class TestSemanticConsistency:

    def test_clear_stage_with_high_amber_flagged(self, filt):
        # "clear" stage but amber_fraction = 0.80 — inconsistent
        r = filt.filter_maturity(_maturity_response(
            "clear", confidence=0.80, clear=0.10, cloudy=0.10, amber=0.80
        ))
        assert HallucinationFlag.SEMANTIC_INCONSISTENCY in r.flags

    def test_amber_stage_with_zero_amber_flagged(self, filt):
        # "amber" stage but amber_fraction = 0.0 — inconsistent
        r = filt.filter_maturity(_maturity_response(
            "amber", confidence=0.80, clear=0.50, cloudy=0.50, amber=0.00
        ))
        assert HallucinationFlag.SEMANTIC_INCONSISTENCY in r.flags

    def test_degraded_with_high_clear_flagged(self, filt):
        # "degraded" should not have clear_fraction > 0.20
        r = filt.filter_maturity({
            "maturity_stage": "degraded",
            "confidence": 0.75,
            "clear_fraction_estimate": 0.80,
        })
        assert HallucinationFlag.SEMANTIC_INCONSISTENCY in r.flags

    def test_consistent_cloudy_not_flagged_semantically(self, filt):
        r = filt.filter_maturity(_maturity_response(
            "cloudy", confidence=0.85, clear=0.05, cloudy=0.85, amber=0.10
        ))
        assert HallucinationFlag.SEMANTIC_INCONSISTENCY not in r.flags

    def test_semantic_check_disabled_skips_check(self):
        cfg = HallucinationFilterConfig(enable_semantic_check=False)
        filt = HallucinationFilter(config=cfg)
        # Would normally trigger semantic inconsistency
        r = filt.filter_maturity(_maturity_response(
            "clear", confidence=0.80, clear=0.0, cloudy=0.0, amber=1.0
        ))
        assert HallucinationFlag.SEMANTIC_INCONSISTENCY not in r.flags


# ─────────────────────────────────────────────────────────────────
# 7. Cross-model agreement
# ─────────────────────────────────────────────────────────────────

class TestCrossModelAgreement:

    def test_two_agreeing_models_pass(self, filt):
        results = [
            {"maturity_stage": "cloudy", "confidence": 0.82},
            {"maturity_stage": "cloudy", "confidence": 0.78},
        ]
        r = filt.filter_cross_model(results, prediction_key="maturity_stage")
        assert HallucinationFlag.CROSS_MODEL_DISAGREEMENT not in r.flags
        assert r.passed is True

    def test_complete_disagreement_flagged(self, filt):
        results = [
            {"maturity_stage": "clear", "confidence": 0.80},
            {"maturity_stage": "amber", "confidence": 0.80},
            {"maturity_stage": "degraded", "confidence": 0.80},
        ]
        r = filt.filter_cross_model(results, prediction_key="maturity_stage")
        assert HallucinationFlag.CROSS_MODEL_DISAGREEMENT in r.flags
        assert r.passed is False

    def test_single_model_result_passes_trivially(self, filt):
        results = [{"maturity_stage": "cloudy", "confidence": 0.85}]
        r = filt.filter_cross_model(results, prediction_key="maturity_stage")
        assert r.passed is True

    def test_empty_results_handled(self, filt):
        r = filt.filter_cross_model([], prediction_key="maturity_stage")
        assert r.passed is True  # Trivially passes with < 2 results

    def test_agreement_rate_reduces_confidence(self, filt):
        # 2/3 agree → agreement_rate ≈ 0.67 → conf penalised
        results = [
            {"maturity_stage": "cloudy", "confidence": 0.80},
            {"maturity_stage": "cloudy", "confidence": 0.80},
            {"maturity_stage": "amber",  "confidence": 0.80},
        ]
        r = filt.filter_cross_model(results)
        # avg_conf = 0.80, agreement_rate = 0.67 → adjusted ≈ 0.53
        assert r.adjusted_confidence < 0.80

    def test_cross_model_avg_confidence_computed(self, filt):
        results = [
            {"maturity_stage": "cloudy", "confidence": 0.60},
            {"maturity_stage": "cloudy", "confidence": 0.80},
        ]
        r = filt.filter_cross_model(results)
        # avg_conf = 0.70; perfect agreement → adjusted = 0.70
        assert r.adjusted_confidence == pytest.approx(0.70, abs=0.05)


# ─────────────────────────────────────────────────────────────────
# 8. Review priority
# ─────────────────────────────────────────────────────────────────

class TestReviewPriority:

    def test_clean_result_low_priority(self, filt):
        r = filt.filter_maturity(_maturity_response("cloudy", confidence=0.85))
        assert r.review_priority == 0

    def test_invalid_json_is_critical(self, filt):
        r = filt.filter_maturity(None)
        assert r.review_priority == 3

    def test_multiple_flags_increase_priority(self, filt):
        # Trigger multiple flags: unknown class + constraint violation
        r = filt.filter_maturity({
            "maturity_stage": "ufo_stage",   # UNKNOWN_CLASS
            "confidence": 0.85,
            "amber_fraction_estimate": 2.0,  # CONSTRAINT_VIOLATION (>1.0)
        })
        single_flag = filt.filter_maturity(_maturity_response("cloudy", confidence=0.20))  # LOW_CONFIDENCE only
        # Multiple flags should produce >= priority of single flag
        assert r.review_priority >= single_flag.review_priority

    def test_priority_in_valid_range(self, filt):
        for conf in [0.0, 0.3, 0.5, 0.8, 1.0]:
            r = filt.filter_maturity(_maturity_response("cloudy", confidence=conf))
            assert 0 <= r.review_priority <= 3


# ─────────────────────────────────────────────────────────────────
# 9. FilterResult properties
# ─────────────────────────────────────────────────────────────────

class TestFilterResultProperties:

    def test_flag_names_returns_string_list(self, filt):
        r = filt.filter_maturity(None)
        assert isinstance(r.flag_names, list)
        assert all(isinstance(n, str) for n in r.flag_names)

    def test_is_flagged_true_when_flags_present(self, filt):
        r = filt.filter_maturity(_maturity_response("cloudy", confidence=0.10))
        assert r.is_flagged is True

    def test_is_reliable_false_when_not_passed(self, filt):
        r = filt.filter_maturity(None)
        assert r.is_reliable is False

    def test_adjusted_confidence_clamped_to_unit_interval(self, filt):
        # Even with many penalties, confidence stays >= 0
        r = filt.filter_maturity({
            "maturity_stage": "bad_stage",
            "confidence": 0.01,
            "amber_fraction_estimate": 5.0,
        })
        assert 0.0 <= r.adjusted_confidence <= 1.0

    def test_flag_details_nonempty_when_flagged(self, filt):
        r = filt.filter_maturity(_maturity_response("bad_stage", confidence=0.20))
        # At least UNKNOWN_CLASS and LOW_CONFIDENCE should have details
        assert len(r.flag_details) > 0
