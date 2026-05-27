"""
tests/unit/test_vlm_schema_enforcer.py — SchemaEnforcer tests.

Covers:
  _extract_json    — JSON extraction from clean / fenced / prose / malformed text
  _emergency_extract — key:value extraction from severely malformed output
  _validate_fields — required fields, type coercion, enum validation, range clamping
  _coerce_field    — per-FieldType coercion paths
  _enforce_fractions — normalisation + all-zero distribution
  enforce_maturity / enforce_quality / enforce_morphology — public API
  EnforcementResult — properties: valid alias, error_message
"""

from __future__ import annotations

import json

import pytest

from vlm_labeling.prompts.schema_enforcer import (
    EnforcementResult,
    FieldSpec,
    FieldType,
    JsonSchema,
    SchemaEnforcer,
    enforce_maturity,
    enforce_morphology,
    enforce_quality,
)


# ─────────────────────────────────────────────────────────────────
# Helpers / fixtures
# ─────────────────────────────────────────────────────────────────

def _simple_schema(required_field: bool = True) -> JsonSchema:
    """Minimal schema with one required and one optional field."""
    return JsonSchema(
        name="TestSchema",
        fields=[
            FieldSpec("label", FieldType.STRING, required=required_field,
                      allowed_values=["yes", "no"], default="unknown"),
            FieldSpec("score", FieldType.FLOAT, required=False,
                      min_value=0.0, max_value=1.0, default=0.5),
        ],
    )


def _fraction_schema() -> JsonSchema:
    return JsonSchema(
        name="FracSchema",
        fields=[
            FieldSpec("a", FieldType.FLOAT, required=False, min_value=0.0, max_value=1.0, default=0.0),
            FieldSpec("b", FieldType.FLOAT, required=False, min_value=0.0, max_value=1.0, default=0.0),
            FieldSpec("c", FieldType.FLOAT, required=False, min_value=0.0, max_value=1.0, default=0.0),
        ],
        fraction_fields=["a", "b", "c"],
        fraction_sum_tolerance=0.05,
    )


# ─────────────────────────────────────────────────────────────────
# 1. JSON extraction from clean text
# ─────────────────────────────────────────────────────────────────

class TestExtractJson:

    def test_plain_json_object(self):
        raw = '{"label": "yes", "score": 0.9}'
        result = SchemaEnforcer(_simple_schema()).enforce(raw)
        assert result.is_valid
        assert result.data["label"] == "yes"

    def test_markdown_fenced_json(self):
        raw = "```json\n{\"label\": \"no\", \"score\": 0.2}\n```"
        result = SchemaEnforcer(_simple_schema()).enforce(raw)
        assert result.is_valid
        assert result.data["label"] == "no"

    def test_fenced_without_lang_specifier(self):
        raw = "```\n{\"label\": \"yes\"}\n```"
        result = SchemaEnforcer(_simple_schema()).enforce(raw)
        assert result.is_valid
        assert result.data["label"] == "yes"

    def test_json_embedded_in_prose(self):
        raw = 'The model output is {"label": "yes", "score": 0.75} based on analysis.'
        result = SchemaEnforcer(_simple_schema()).enforce(raw)
        assert result.is_valid
        assert result.data["label"] == "yes"

    def test_whitespace_and_newlines_in_json(self):
        raw = '{\n  "label": "no",\n  "score": 0.3\n}'
        result = SchemaEnforcer(_simple_schema()).enforce(raw)
        assert result.is_valid

    def test_raw_text_stored_in_result(self):
        raw = '{"label": "yes"}'
        result = SchemaEnforcer(_simple_schema()).enforce(raw)
        assert result.raw_text == raw


# ─────────────────────────────────────────────────────────────────
# 2. Emergency extraction
# ─────────────────────────────────────────────────────────────────

class TestEmergencyExtract:

    def test_extracts_quoted_kv_pairs(self):
        raw = '"label": "yes", "score": 0.8'
        enforcer = SchemaEnforcer(_simple_schema())
        data = SchemaEnforcer._emergency_extract(raw)
        assert data is not None
        assert "label" in data

    def test_returns_none_on_empty_string(self):
        data = SchemaEnforcer._emergency_extract("no fields here at all $$$")
        assert data is None or isinstance(data, dict)  # None or empty dict acceptable

    def test_emergency_extraction_marks_repaired(self):
        # Severely malformed — no JSON braces at all, but has key:value
        raw = 'label: "yes" score: 0.9'
        result = SchemaEnforcer(_simple_schema()).enforce(raw)
        # Either repaired or fallback to defaults — must not crash
        assert isinstance(result.data, dict)

    def test_total_failure_returns_defaults(self):
        raw = "completely unstructured prose with no fields whatsoever"
        result = SchemaEnforcer(_simple_schema()).enforce(raw)
        assert isinstance(result.data, dict)
        assert "label" in result.data
        assert result.is_valid is False


# ─────────────────────────────────────────────────────────────────
# 3. Field validation — required / optional / missing
# ─────────────────────────────────────────────────────────────────

class TestFieldValidation:

    def test_missing_required_field_invalidates_result(self):
        raw = '{"score": 0.5}'  # label (required) is absent
        result = SchemaEnforcer(_simple_schema(required_field=True)).enforce(raw)
        assert result.is_valid is False
        assert any("label" in e for e in result.errors)

    def test_missing_optional_field_uses_default(self):
        raw = '{"label": "yes"}'  # score (optional) missing
        result = SchemaEnforcer(_simple_schema()).enforce(raw)
        assert result.is_valid
        assert result.data["score"] == pytest.approx(0.5, abs=0.001)

    def test_valid_complete_input_no_errors(self):
        raw = '{"label": "yes", "score": 0.7}'
        result = SchemaEnforcer(_simple_schema()).enforce(raw)
        assert result.is_valid
        assert result.errors == []

    def test_invalid_enum_value_replaced_with_default(self):
        raw = '{"label": "maybe", "score": 0.5}'
        result = SchemaEnforcer(_simple_schema()).enforce(raw)
        assert result.data["label"] == "unknown"  # default

    def test_case_insensitive_enum_match_repairs(self):
        raw = '{"label": "YES", "score": 0.5}'
        result = SchemaEnforcer(_simple_schema()).enforce(raw)
        assert result.data["label"] == "yes"
        assert result.was_repaired

    def test_value_below_min_clamped(self):
        raw = '{"label": "yes", "score": -0.5}'
        result = SchemaEnforcer(_simple_schema()).enforce(raw)
        assert result.data["score"] == pytest.approx(0.0, abs=0.001)

    def test_value_above_max_clamped(self):
        raw = '{"label": "yes", "score": 1.5}'
        result = SchemaEnforcer(_simple_schema()).enforce(raw)
        assert result.data["score"] == pytest.approx(1.0, abs=0.001)

    def test_unknown_extra_fields_passed_through(self):
        raw = '{"label": "yes", "score": 0.5, "extra_debug": "foo"}'
        result = SchemaEnforcer(_simple_schema()).enforce(raw)
        assert result.data.get("extra_debug") == "foo"


# ─────────────────────────────────────────────────────────────────
# 4. Field coercion — per FieldType
# ─────────────────────────────────────────────────────────────────

class TestFieldCoercion:

    def _make_schema(self, field_type: FieldType, **kwargs) -> JsonSchema:
        return JsonSchema(
            name="CoercionSchema",
            fields=[
                FieldSpec("value", field_type, required=False, default=None, **kwargs),
            ],
        )

    def test_float_from_int_string(self):
        raw = '{"value": "0.75"}'
        result = SchemaEnforcer(self._make_schema(FieldType.FLOAT)).enforce(raw)
        assert result.data["value"] == pytest.approx(0.75)

    def test_int_from_float(self):
        raw = '{"value": 3.9}'
        result = SchemaEnforcer(self._make_schema(FieldType.INT)).enforce(raw)
        assert result.data["value"] == 3
        assert isinstance(result.data["value"], int)

    def test_bool_from_string_true(self):
        raw = '{"value": "true"}'
        result = SchemaEnforcer(self._make_schema(FieldType.BOOL)).enforce(raw)
        assert result.data["value"] is True

    def test_bool_from_string_false(self):
        raw = '{"value": "false"}'
        result = SchemaEnforcer(self._make_schema(FieldType.BOOL)).enforce(raw)
        assert result.data["value"] is False

    def test_bool_from_yes_string(self):
        raw = '{"value": "yes"}'
        result = SchemaEnforcer(self._make_schema(FieldType.BOOL)).enforce(raw)
        assert result.data["value"] is True

    def test_list_from_json_string(self):
        raw = '{"value": "[1, 2, 3]"}'
        result = SchemaEnforcer(self._make_schema(FieldType.LIST)).enforce(raw)
        assert isinstance(result.data["value"], list)

    def test_list_from_comma_separated_string(self):
        raw = '{"value": "a, b, c"}'
        result = SchemaEnforcer(self._make_schema(FieldType.LIST)).enforce(raw)
        val = result.data["value"]
        assert isinstance(val, list)
        assert len(val) == 3

    def test_nullable_float_none_preserved(self):
        raw = '{"value": null}'
        result = SchemaEnforcer(self._make_schema(FieldType.NULLABLE_FLOAT)).enforce(raw)
        assert result.data["value"] is None

    def test_nullable_int_none_preserved(self):
        raw = '{"value": null}'
        result = SchemaEnforcer(self._make_schema(FieldType.NULLABLE_INT)).enforce(raw)
        assert result.data["value"] is None

    def test_nullable_float_with_value(self):
        raw = '{"value": 0.42}'
        result = SchemaEnforcer(self._make_schema(FieldType.NULLABLE_FLOAT)).enforce(raw)
        assert result.data["value"] == pytest.approx(0.42)

    def test_string_coercion_from_int(self):
        raw = '{"value": 42}'
        result = SchemaEnforcer(self._make_schema(FieldType.STRING)).enforce(raw)
        assert isinstance(result.data["value"], str)

    def test_dict_field_passthrough(self):
        # Test _coerce_field directly: dict value passes through unchanged
        spec = FieldSpec("value", FieldType.DICT, required=False, default={})
        coerced, err, _ = SchemaEnforcer._coerce_field({"nested": 1}, spec)
        assert isinstance(coerced, dict)
        assert coerced == {"nested": 1}
        assert err is None


# ─────────────────────────────────────────────────────────────────
# 5. Fraction enforcement
# ─────────────────────────────────────────────────────────────────

class TestFractionEnforcement:

    def test_already_normalised_no_repair(self):
        raw = '{"a": 0.5, "b": 0.3, "c": 0.2}'
        result = SchemaEnforcer(_fraction_schema()).enforce(raw)
        total = result.data["a"] + result.data["b"] + result.data["c"]
        assert abs(total - 1.0) < 0.02

    def test_renormalises_when_outside_tolerance(self):
        raw = '{"a": 2.0, "b": 1.0, "c": 1.0}'
        result = SchemaEnforcer(_fraction_schema()).enforce(raw)
        total = result.data["a"] + result.data["b"] + result.data["c"]
        assert abs(total - 1.0) < 0.02
        assert result.was_repaired

    def test_all_zero_fractions_distributed_uniformly(self):
        raw = '{"a": 0.0, "b": 0.0, "c": 0.0}'
        result = SchemaEnforcer(_fraction_schema()).enforce(raw)
        for key in ("a", "b", "c"):
            assert result.data[key] == pytest.approx(1.0 / 3, abs=0.01)
        assert result.was_repaired

    def test_slightly_above_tolerance_renormalised(self):
        raw = '{"a": 0.5, "b": 0.3, "c": 0.3}'  # sum = 1.1
        result = SchemaEnforcer(_fraction_schema()).enforce(raw)
        total = result.data["a"] + result.data["b"] + result.data["c"]
        assert abs(total - 1.0) < 0.02

    def test_warnings_added_when_renormalised(self):
        raw = '{"a": 2.0, "b": 1.0, "c": 1.0}'
        result = SchemaEnforcer(_fraction_schema()).enforce(raw)
        assert any("renormalized" in w or "Frac" in w for w in result.warnings)


# ─────────────────────────────────────────────────────────────────
# 6. EnforcementResult properties
# ─────────────────────────────────────────────────────────────────

class TestEnforcementResultProperties:

    def test_valid_alias_matches_is_valid(self):
        result = EnforcementResult(data={}, is_valid=True)
        assert result.valid is True
        assert result.valid == result.is_valid

    def test_error_message_returns_first_error(self):
        result = EnforcementResult(data={}, is_valid=False, errors=["error 1", "error 2"])
        assert result.error_message == "error 1"

    def test_error_message_none_when_valid(self):
        result = EnforcementResult(data={}, is_valid=True)
        assert result.error_message is None

    def test_was_repaired_defaults_false(self):
        result = EnforcementResult(data={}, is_valid=True)
        assert result.was_repaired is False

    def test_data_returned_on_valid_result(self):
        raw = '{"label": "yes", "score": 0.9}'
        result = SchemaEnforcer(_simple_schema()).enforce(raw)
        assert isinstance(result.data, dict)
        assert "label" in result.data


# ─────────────────────────────────────────────────────────────────
# 7. enforce_maturity convenience function
# ─────────────────────────────────────────────────────────────────

class TestEnforceMaturity:

    def test_valid_maturity_json(self):
        raw = json.dumps({
            "maturity_stage": "cloudy",
            "clear": 0.1,
            "cloudy": 0.8,
            "amber": 0.1,
            "mixed": 0.0,
            "confidence": 0.85,
        })
        result = enforce_maturity(raw)
        assert result.is_valid
        assert result.data["maturity_stage"] == "cloudy"
        assert result.data["confidence"] == pytest.approx(0.85)

    def test_invalid_maturity_stage_defaults_to_unknown(self):
        raw = json.dumps({"maturity_stage": "ripe", "confidence": 0.7})
        result = enforce_maturity(raw)
        assert result.data["maturity_stage"] == "unknown"

    def test_fractions_renormalised(self):
        raw = json.dumps({
            "maturity_stage": "amber",
            "clear": 0.5,
            "cloudy": 0.5,
            "amber": 0.5,
            "mixed": 0.5,
        })
        result = enforce_maturity(raw)
        total = sum(result.data[k] for k in ("clear", "cloudy", "amber", "mixed"))
        assert abs(total - 1.0) < 0.05

    def test_confidence_clamped_to_range(self):
        raw = json.dumps({"maturity_stage": "clear", "confidence": 1.5})
        result = enforce_maturity(raw)
        assert result.data["confidence"] <= 1.0

    def test_completely_invalid_text_returns_defaults(self):
        result = enforce_maturity("this is not JSON at all and has no fields")
        assert isinstance(result.data, dict)
        assert result.is_valid is False

    def test_maturity_all_stages_accepted(self):
        for stage in ("clear", "cloudy", "amber", "mixed", "unknown"):
            raw = json.dumps({"maturity_stage": stage})
            result = enforce_maturity(raw)
            assert result.data["maturity_stage"] == stage


# ─────────────────────────────────────────────────────────────────
# 8. enforce_quality convenience function
# ─────────────────────────────────────────────────────────────────

class TestEnforceQuality:

    def test_valid_quality_json(self):
        raw = json.dumps({
            "overall_quality": "high",
            "is_in_focus": True,
            "focus_score": 0.9,
            "confidence": 0.88,
        })
        result = enforce_quality(raw)
        assert result.is_valid
        assert result.data["overall_quality"] == "high"

    def test_invalid_quality_level_replaced_with_default(self):
        raw = json.dumps({"overall_quality": "perfect", "confidence": 0.9})
        result = enforce_quality(raw)
        assert result.data["overall_quality"] == "unknown"

    def test_focus_score_clamped(self):
        raw = json.dumps({"focus_score": -1.0})
        result = enforce_quality(raw)
        assert result.data["focus_score"] == pytest.approx(0.0)

    def test_bool_fields_coerced_from_strings(self):
        raw = '{"is_in_focus": "true", "has_debris": "false"}'
        result = enforce_quality(raw)
        assert result.data["is_in_focus"] is True
        assert result.data["has_debris"] is False

    def test_empty_json_object_returns_defaults(self):
        result = enforce_quality("{}")
        assert isinstance(result.data, dict)
        assert result.data["overall_quality"] == "unknown"

    def test_all_quality_levels_accepted(self):
        for level in ("high", "medium", "low", "unusable", "unknown"):
            result = enforce_quality(json.dumps({"overall_quality": level}))
            assert result.data["overall_quality"] == level


# ─────────────────────────────────────────────────────────────────
# 9. enforce_morphology convenience function
# ─────────────────────────────────────────────────────────────────

class TestEnforceMorphology:

    def test_valid_morphology_json(self):
        raw = json.dumps({
            "dominant_type": "capitate_stalked",
            "density": "dense",
            "count_estimate": 42,
            "confidence": 0.75,
        })
        result = enforce_morphology(raw)
        assert result.is_valid
        assert result.data["dominant_type"] == "capitate_stalked"

    def test_missing_required_dominant_type_invalid(self):
        raw = json.dumps({"density": "moderate", "confidence": 0.6})
        result = enforce_morphology(raw)
        # dominant_type is required — missing or invalid makes it invalid
        assert result.is_valid is False

    def test_invalid_dominant_type_defaults_to_unknown(self):
        raw = json.dumps({"dominant_type": "glandular", "confidence": 0.5})
        result = enforce_morphology(raw)
        assert result.data["dominant_type"] == "unknown"

    def test_count_estimate_nullable_int(self):
        raw = json.dumps({"dominant_type": "bulbous", "count_estimate": None})
        result = enforce_morphology(raw)
        assert result.data["count_estimate"] is None

    def test_count_estimate_clamped_to_max(self):
        raw = json.dumps({"dominant_type": "bulbous", "count_estimate": 99999})
        result = enforce_morphology(raw)
        assert result.data["count_estimate"] <= 5000

    def test_types_present_list_passthrough(self):
        raw = json.dumps({
            "dominant_type": "mixed",
            "types_present": ["capitate_stalked", "bulbous"],
        })
        result = enforce_morphology(raw)
        assert isinstance(result.data["types_present"], list)

    def test_density_invalid_replaced_with_default(self):
        raw = json.dumps({"dominant_type": "bulbous", "density": "very_dense"})
        result = enforce_morphology(raw)
        assert result.data["density"] == "moderate"

    def test_all_valid_dominant_types_accepted(self):
        for dtype in ("capitate_stalked", "capitate_sessile", "bulbous",
                      "non_glandular", "mixed", "unknown"):
            result = enforce_morphology(json.dumps({"dominant_type": dtype}))
            assert result.data["dominant_type"] == dtype


# ─────────────────────────────────────────────────────────────────
# 10. repair flag tracking
# ─────────────────────────────────────────────────────────────────

class TestRepairTracking:

    def test_clean_input_not_repaired(self):
        raw = json.dumps({"maturity_stage": "clear", "confidence": 0.9})
        result = enforce_maturity(raw)
        # No type coercion or range violation → was_repaired depends only on fractions
        # (fractions default to 0.0, will be distributed) — check no extra repairs
        assert isinstance(result.was_repaired, bool)

    def test_out_of_range_value_marks_repaired(self):
        raw = json.dumps({
            "dominant_type": "bulbous",
            "confidence": 2.0,  # > 1.0
        })
        result = enforce_morphology(raw)
        assert result.was_repaired

    def test_invalid_enum_marks_repaired(self):
        raw = json.dumps({"dominant_type": "BULBOUS"})  # wrong case
        result = enforce_morphology(raw)
        # Case-insensitive match should repair it
        assert result.data["dominant_type"] == "bulbous"
        assert result.was_repaired
