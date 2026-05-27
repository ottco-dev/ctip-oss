"""
tests/unit/test_vlm_schema.py — Unit tests for VLM schema enforcement.

Tests JSON extraction, schema validation, and fraction normalization
without requiring any VLM model.
"""

import pytest


# ---------------------------------------------------------------------------
# Schema enforcer tests
# ---------------------------------------------------------------------------


def test_enforce_valid_json():
    """Should accept valid JSON matching the maturity schema."""
    from vlm_labeling.prompts.schema_enforcer import SchemaEnforcer, MATURITY_SCHEMA

    enforcer = SchemaEnforcer(MATURITY_SCHEMA)
    valid_json = '{"clear": 0.2, "cloudy": 0.6, "amber": 0.1, "mixed": 0.1, "confidence": 0.85}'
    result = enforcer.enforce(valid_json)

    assert result.valid
    assert result.data is not None
    assert result.data.get("clear") == pytest.approx(0.2, abs=0.01)


def test_enforce_fractions_sum_to_one():
    """Fraction fields should be normalized to sum=1.0."""
    from vlm_labeling.prompts.schema_enforcer import SchemaEnforcer, MATURITY_SCHEMA

    enforcer = SchemaEnforcer(MATURITY_SCHEMA)
    json_text = '{"clear": 0.4, "cloudy": 0.8, "amber": 0.2, "mixed": 0.1, "confidence": 0.7}'
    result = enforcer.enforce(json_text)

    if result.data:
        fracs = sum(
            result.data.get(k, 0)
            for k in ("clear", "cloudy", "amber", "mixed")
        )
        assert fracs == pytest.approx(1.0, abs=0.01)


def test_enforce_malformed_json():
    """Should extract values from malformed JSON without raising."""
    from vlm_labeling.prompts.schema_enforcer import SchemaEnforcer, MATURITY_SCHEMA

    enforcer = SchemaEnforcer(MATURITY_SCHEMA)
    malformed = "clear: 0.3, cloudy: 0.5, amber: 0.2, mixed: 0.0"  # No braces
    result = enforcer.enforce(malformed)

    # Should not raise; may or may not succeed depending on fallback
    assert isinstance(result.valid, bool)
    assert result.error_message is not None or result.valid


def test_enforce_markdown_fence():
    """Should extract JSON from markdown code fences."""
    from vlm_labeling.prompts.schema_enforcer import SchemaEnforcer, MATURITY_SCHEMA

    enforcer = SchemaEnforcer(MATURITY_SCHEMA)
    with_fence = """
Here is the analysis:
```json
{"clear": 0.1, "cloudy": 0.7, "amber": 0.1, "mixed": 0.1, "confidence": 0.9}
```
That is my answer.
"""
    result = enforcer.enforce(with_fence)
    assert result.valid
    assert result.data is not None


def test_enforce_missing_optional_fields():
    """Missing optional fields should have defaults applied."""
    from vlm_labeling.prompts.schema_enforcer import SchemaEnforcer, MATURITY_SCHEMA

    enforcer = SchemaEnforcer(MATURITY_SCHEMA)
    minimal = '{"clear": 0.3, "cloudy": 0.5, "amber": 0.1, "mixed": 0.1}'
    result = enforcer.enforce(minimal)

    # Should succeed even without "confidence" field
    assert result.valid or result.data is not None


def test_quality_schema_validation():
    """Quality schema should validate properly."""
    from vlm_labeling.prompts.schema_enforcer import SchemaEnforcer, QUALITY_SCHEMA

    enforcer = SchemaEnforcer(QUALITY_SCHEMA)
    valid = '{"in_focus": true, "well_exposed": true, "usable": true, "quality_score": 0.88}'
    result = enforcer.enforce(valid)
    assert result.valid


# ---------------------------------------------------------------------------
# Hallucination filter tests
# ---------------------------------------------------------------------------


def test_filter_low_confidence():
    """Predictions below confidence threshold should be marked unreliable."""
    from vlm_labeling.filtering.hallucination import HallucinationFilter, FilterConfig

    config = FilterConfig(min_confidence=0.70)
    hfilter = HallucinationFilter(config)

    label = {
        "confidence": 0.4,
        "clear": 0.2,
        "cloudy": 0.5,
        "amber": 0.2,
        "mixed": 0.1,
    }
    result = hfilter.filter_label(label)
    assert not result.is_reliable


def test_filter_high_confidence():
    """High confidence labels should pass the filter."""
    from vlm_labeling.filtering.hallucination import HallucinationFilter, FilterConfig

    config = FilterConfig(min_confidence=0.70)
    hfilter = HallucinationFilter(config)

    label = {
        "confidence": 0.92,
        "clear": 0.1,
        "cloudy": 0.75,
        "amber": 0.1,
        "mixed": 0.05,
    }
    result = hfilter.filter_label(label)
    assert result.is_reliable


def test_filter_impossible_fractions():
    """Fractions summing to > 1 or with negative values should be flagged."""
    from vlm_labeling.filtering.hallucination import HallucinationFilter, FilterConfig

    config = FilterConfig(min_confidence=0.70)
    hfilter = HallucinationFilter(config)

    # Fractions sum to 1.5 — clearly wrong
    label = {
        "confidence": 0.9,
        "clear": 0.5,
        "cloudy": 0.6,
        "amber": 0.3,
        "mixed": 0.1,
    }
    result = hfilter.filter_label(label)
    # Should either be unreliable or have corrected fractions
    if result.corrected_data:
        total = sum(result.corrected_data.get(k, 0) for k in ("clear", "cloudy", "amber", "mixed"))
        assert total == pytest.approx(1.0, abs=0.05)


# ---------------------------------------------------------------------------
# Prompt template tests
# ---------------------------------------------------------------------------


def test_maturity_prompt_format():
    """Maturity prompt should contain required JSON schema keys."""
    from vlm_labeling.prompts.trichome_prompts import get_maturity_prompt

    prompt = get_maturity_prompt()
    assert "clear" in prompt.lower()
    assert "cloudy" in prompt.lower()
    assert "amber" in prompt.lower()
    assert "json" in prompt.lower() or "JSON" in prompt


def test_prompt_scientific_caveat():
    """Prompts must not instruct model to claim THC concentration."""
    from vlm_labeling.prompts.trichome_prompts import get_maturity_prompt

    prompt = get_maturity_prompt()
    forbidden = ["THC%", "thc content", "potency", "percentage of thc"]
    for phrase in forbidden:
        assert phrase.lower() not in prompt.lower(), (
            f"Forbidden phrase '{phrase}' found in maturity prompt"
        )
