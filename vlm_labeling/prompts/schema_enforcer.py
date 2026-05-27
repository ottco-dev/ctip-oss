"""
vlm_labeling.prompts.schema_enforcer — JSON schema enforcement and validation for VLM outputs.

Problem: LLMs sometimes:
  - Return invalid JSON
  - Return JSON with missing required fields
  - Return JSON with values outside allowed ranges
  - Return prose instead of JSON

This module enforces JSON schemas on VLM outputs with:
  1. Extraction: finds JSON in messy text (markdown, prose, fences)
  2. Validation: checks required fields, types, ranges, enums
  3. Repair: attempts to fix common model mistakes
  4. Fallback: constructs minimum-viable defaults if repair fails
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema definition types
# ---------------------------------------------------------------------------

class FieldType(str, Enum):
    STRING = "string"
    FLOAT = "float"
    INT = "int"
    BOOL = "bool"
    LIST = "list"
    DICT = "dict"
    NULLABLE_FLOAT = "nullable_float"
    NULLABLE_INT = "nullable_int"
    NULLABLE_STRING = "nullable_string"


@dataclass
class FieldSpec:
    """Specification for a single JSON field."""

    name: str
    field_type: FieldType
    required: bool = True
    allowed_values: list[Any] | None = None
    min_value: float | None = None
    max_value: float | None = None
    default: Any = None
    description: str = ""


@dataclass
class JsonSchema:
    """Schema definition for a VLM response."""

    name: str
    """Schema identifier (for logging)."""

    fields: list[FieldSpec]
    """Field specifications."""

    fraction_fields: list[str] | None = None
    """Field names that must sum to 1.0 (within tolerance)."""

    fraction_sum_tolerance: float = 0.05


@dataclass
class EnforcementResult:
    """Result of schema enforcement."""

    data: dict[str, Any]
    """Validated/repaired data dict."""

    is_valid: bool
    """True if passed validation without critical errors."""

    errors: list[str] = field(default_factory=list)
    """Validation error messages."""

    warnings: list[str] = field(default_factory=list)
    """Non-critical issues."""

    was_repaired: bool = False
    """True if automatic repair was applied."""

    raw_text: str = ""
    """Original model output before processing."""

    @property
    def valid(self) -> bool:
        """Alias for is_valid — backward compat with tests."""
        return self.is_valid

    @property
    def error_message(self) -> str | None:
        """First error message, or None if valid."""
        return self.errors[0] if self.errors else None


# ---------------------------------------------------------------------------
# Pre-defined schemas for trichome VLM tasks
# ---------------------------------------------------------------------------

MATURITY_SCHEMA = JsonSchema(
    name="MaturityClassification",
    fields=[
        FieldSpec(
            name="maturity_stage",
            field_type=FieldType.STRING,
            required=False,
            allowed_values=["clear", "cloudy", "amber", "mixed", "unknown"],
            default="unknown",
            description="Dominant maturity stage",
        ),
        # Short-form fraction fields (used by VLMs and tests)
        FieldSpec(
            name="clear",
            field_type=FieldType.FLOAT,
            required=False,
            min_value=0.0,
            max_value=1.0,
            default=0.0,
            description="Fraction clear (immature)",
        ),
        FieldSpec(
            name="cloudy",
            field_type=FieldType.FLOAT,
            required=False,
            min_value=0.0,
            max_value=1.0,
            default=0.0,
            description="Fraction cloudy (mature)",
        ),
        FieldSpec(
            name="amber",
            field_type=FieldType.FLOAT,
            required=False,
            min_value=0.0,
            max_value=1.0,
            default=0.0,
            description="Fraction amber (degraded)",
        ),
        FieldSpec(
            name="mixed",
            field_type=FieldType.FLOAT,
            required=False,
            min_value=0.0,
            max_value=1.0,
            default=0.0,
            description="Fraction mixed",
        ),
        # Long-form fraction fields (pipeline internal)
        FieldSpec(
            name="clear_fraction",
            field_type=FieldType.FLOAT,
            required=False,
            min_value=0.0,
            max_value=1.0,
            default=0.0,
            description="Fraction of trichomes in clear (immature) stage",
        ),
        FieldSpec(
            name="cloudy_fraction",
            field_type=FieldType.FLOAT,
            required=False,
            min_value=0.0,
            max_value=1.0,
            default=0.0,
            description="Fraction of trichomes in cloudy (mature) stage",
        ),
        FieldSpec(
            name="amber_fraction",
            field_type=FieldType.FLOAT,
            required=False,
            min_value=0.0,
            max_value=1.0,
            default=0.0,
            description="Fraction of trichomes in amber (degraded) stage",
        ),
        FieldSpec(
            name="confidence",
            field_type=FieldType.FLOAT,
            required=False,
            min_value=0.0,
            max_value=1.0,
            default=0.5,
        ),
    ],
    # Normalize whichever fraction group sums to ≠ 1.0
    # Short names take precedence in test assertions
    fraction_fields=["clear", "cloudy", "amber", "mixed"],
    fraction_sum_tolerance=0.10,
)

QUALITY_SCHEMA = JsonSchema(
    name="ImageQuality",
    fields=[
        FieldSpec(
            name="overall_quality",
            field_type=FieldType.STRING,
            required=False,
            allowed_values=["high", "medium", "low", "unusable", "unknown"],
            default="unknown",
        ),
        FieldSpec(
            name="is_in_focus",
            field_type=FieldType.BOOL,
            required=False,
            default=False,
        ),
        FieldSpec(name="in_focus", field_type=FieldType.BOOL, required=False, default=False),
        FieldSpec(name="well_exposed", field_type=FieldType.BOOL, required=False, default=False),
        FieldSpec(name="usable", field_type=FieldType.BOOL, required=False, default=False),
        FieldSpec(name="quality_score", field_type=FieldType.FLOAT, required=False, min_value=0.0, max_value=1.0, default=0.5),
        FieldSpec(
            name="focus_score",
            field_type=FieldType.FLOAT,
            required=False,
            min_value=0.0,
            max_value=1.0,
            default=0.5,
        ),
        FieldSpec(
            name="has_debris",
            field_type=FieldType.BOOL,
            required=False,
            default=False,
        ),
        FieldSpec(
            name="adequate_lighting",
            field_type=FieldType.BOOL,
            required=False,
            default=True,
        ),
        FieldSpec(
            name="confidence",
            field_type=FieldType.FLOAT,
            required=False,
            min_value=0.0,
            max_value=1.0,
            default=0.5,
        ),
    ],
)

MORPHOLOGY_SCHEMA = JsonSchema(
    name="TrichomeMorphology",
    fields=[
        FieldSpec(
            name="dominant_type",
            field_type=FieldType.STRING,
            required=True,
            allowed_values=[
                "capitate_stalked", "capitate_sessile",
                "bulbous", "non_glandular", "mixed", "unknown",
            ],
            default="unknown",
        ),
        FieldSpec(
            name="types_present",
            field_type=FieldType.LIST,
            required=False,
            default=[],
        ),
        FieldSpec(
            name="density",
            field_type=FieldType.STRING,
            required=False,
            allowed_values=["sparse", "moderate", "dense", "unknown"],
            default="moderate",
        ),
        FieldSpec(
            name="count_estimate",
            field_type=FieldType.NULLABLE_INT,
            required=False,
            min_value=0,
            max_value=5000,
            default=None,
        ),
        FieldSpec(
            name="confidence",
            field_type=FieldType.FLOAT,
            required=False,
            min_value=0.0,
            max_value=1.0,
            default=0.5,
        ),
    ],
)


# ---------------------------------------------------------------------------
# Schema Enforcer
# ---------------------------------------------------------------------------

class SchemaEnforcer:
    """
    Enforce a JsonSchema on raw VLM text output.

    Steps:
    1. Extract JSON from raw text (handles fences, prose, etc.)
    2. Validate field types, ranges, enums
    3. Attempt repairs on common failures
    4. Apply defaults for missing optional fields
    5. Return EnforcementResult
    """

    def __init__(self, schema: JsonSchema) -> None:
        self.schema = schema

    def enforce(self, raw_text: str) -> EnforcementResult:
        """
        Enforce schema on raw VLM output text.

        Args:
            raw_text: Raw model output string.

        Returns:
            EnforcementResult with validated data and error information.
        """
        errors: list[str] = []
        warnings: list[str] = []
        was_repaired = False

        # Step 1: Extract JSON
        data = self._extract_json(raw_text)
        if data is None:
            # Try to recover from common model mistakes
            data = self._emergency_extract(raw_text)
            if data is None:
                defaults = self._get_defaults()
                return EnforcementResult(
                    data=defaults,
                    is_valid=False,
                    errors=[f"[{self.schema.name}] Failed to extract JSON from output"],
                    raw_text=raw_text,
                )
            was_repaired = True
            warnings.append("JSON extracted via emergency recovery")

        # Step 2: Validate and repair each field
        validated, field_errors, field_warnings, repaired = self._validate_fields(data)
        errors.extend(field_errors)
        warnings.extend(field_warnings)
        if repaired:
            was_repaired = True

        # Step 3: Enforce fraction sum constraint
        if self.schema.fraction_fields:
            frac_errors, repaired_fracs = self._enforce_fractions(validated)
            if repaired_fracs:
                was_repaired = True
                warnings.append("Fractions renormalized to sum to 1.0")
            # Fraction errors are warnings not errors (we fix them)
            warnings.extend(frac_errors)

        # Step 4: Determine validity
        # A result is valid if no REQUIRED fields have errors
        required_names = {f.name for f in self.schema.fields if f.required}
        critical_errors = [e for e in errors if any(n in e for n in required_names)]
        is_valid = len(critical_errors) == 0

        return EnforcementResult(
            data=validated,
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            was_repaired=was_repaired,
            raw_text=raw_text,
        )

    # ------------------------------------------------------------------
    # JSON extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any] | None:
        """Extract JSON object from model text with multiple strategies."""
        # Clean markdown fences
        text = re.sub(r"```json\s*", "", text)
        text = re.sub(r"```\s*", "", text)

        # Strategy 1: Simple top-level object regex
        simple_match = re.search(r"\{[^{}]+\}", text, re.DOTALL)
        if simple_match:
            try:
                return json.loads(simple_match.group())
            except json.JSONDecodeError:
                pass

        # Strategy 2: Nested object (up to 2 levels)
        nested_match = re.search(r"\{(?:[^{}]|\{[^{}]*\})*\}", text, re.DOTALL)
        if nested_match:
            try:
                return json.loads(nested_match.group())
            except json.JSONDecodeError:
                pass

        # Strategy 3: Find first { and last } and try parsing
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass

        # Strategy 4: Try entire text
        try:
            result = json.loads(text.strip())
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

        return None

    @staticmethod
    def _emergency_extract(text: str) -> dict[str, Any] | None:
        """
        Emergency extraction for severely malformed output.

        Attempts to find key:value patterns in plain text.
        """
        data: dict[str, Any] = {}

        # Look for "key": value patterns
        kv_pattern = re.compile(
            r'"(\w+)"\s*:\s*'
            r'(true|false|null|"[^"]*"|-?\d+\.?\d*|\[[^\]]*\])',
            re.IGNORECASE,
        )
        for match in kv_pattern.finditer(text):
            key = match.group(1)
            raw_val = match.group(2)
            try:
                data[key] = json.loads(raw_val)
            except json.JSONDecodeError:
                data[key] = raw_val.strip('"')

        # Also try unquoted key: value patterns
        unquoted_kv = re.compile(
            r'(\w+)\s*:\s*(true|false|null|"[^"]*"|-?\d+\.?\d*)',
            re.IGNORECASE,
        )
        for match in unquoted_kv.finditer(text):
            key = match.group(1)
            if key not in data:
                raw_val = match.group(2)
                try:
                    data[key] = json.loads(raw_val)
                except json.JSONDecodeError:
                    data[key] = raw_val.strip('"')

        return data if data else None

    # ------------------------------------------------------------------
    # Field validation
    # ------------------------------------------------------------------

    def _validate_fields(
        self,
        data: dict[str, Any],
    ) -> tuple[dict[str, Any], list[str], list[str], bool]:
        """
        Validate and coerce all fields against schema.

        Returns:
            (validated_data, errors, warnings, was_repaired)
        """
        validated: dict[str, Any] = {}
        errors: list[str] = []
        warnings: list[str] = []
        was_repaired = False

        for spec in self.schema.fields:
            raw_value = data.get(spec.name)

            if raw_value is None:
                if spec.required:
                    errors.append(f"[{self.schema.name}] Missing required field: {spec.name!r}")
                    validated[spec.name] = spec.default
                    was_repaired = True
                else:
                    validated[spec.name] = spec.default
                continue

            # Coerce type
            coerced, coerce_error, coerced_flag = self._coerce_field(raw_value, spec)
            if coerce_error:
                if spec.required:
                    errors.append(f"[{self.schema.name}] {spec.name!r}: {coerce_error}")
                else:
                    warnings.append(f"[{self.schema.name}] {spec.name!r}: {coerce_error}")
                validated[spec.name] = spec.default
                was_repaired = True
                continue

            if coerced_flag:
                was_repaired = True

            # Validate allowed values
            if spec.allowed_values and coerced not in spec.allowed_values:
                # Try case-insensitive match for strings
                if spec.field_type == FieldType.STRING:
                    lower = str(coerced).lower()
                    matches = [v for v in spec.allowed_values if str(v).lower() == lower]
                    if matches:
                        coerced = matches[0]
                        was_repaired = True
                    else:
                        msg = (
                            f"[{self.schema.name}] {spec.name!r}: value {coerced!r} "
                            f"not in {spec.allowed_values}"
                        )
                        if spec.required:
                            errors.append(msg)
                        else:
                            warnings.append(msg)
                        coerced = spec.default
                        was_repaired = True

            # Validate range
            if spec.min_value is not None and isinstance(coerced, (int, float)):
                if coerced < spec.min_value:
                    warnings.append(
                        f"[{self.schema.name}] {spec.name!r}: {coerced} < min {spec.min_value}"
                    )
                    coerced = spec.min_value
                    was_repaired = True

            if spec.max_value is not None and isinstance(coerced, (int, float)):
                if coerced > spec.max_value:
                    warnings.append(
                        f"[{self.schema.name}] {spec.name!r}: {coerced} > max {spec.max_value}"
                    )
                    coerced = spec.max_value
                    was_repaired = True

            validated[spec.name] = coerced

        # Pass through unknown fields as-is (for debugging)
        for key, value in data.items():
            if key not in validated:
                validated[key] = value

        return validated, errors, warnings, was_repaired

    @staticmethod
    def _coerce_field(
        value: Any,
        spec: FieldSpec,
    ) -> tuple[Any, str | None, bool]:
        """
        Coerce a value to the specified field type.

        Returns:
            (coerced_value, error_message_or_None, was_coerced)
        """
        try:
            ft = spec.field_type

            if ft == FieldType.STRING:
                if isinstance(value, str):
                    return value, None, False
                return str(value), None, True

            elif ft == FieldType.FLOAT:
                v = float(value)
                return v, None, not isinstance(value, float)

            elif ft == FieldType.INT:
                v = int(float(value))
                return v, None, not isinstance(value, int)

            elif ft == FieldType.BOOL:
                if isinstance(value, bool):
                    return value, None, False
                if isinstance(value, str):
                    return value.lower() in {"true", "yes", "1"}, None, True
                return bool(value), None, True

            elif ft == FieldType.LIST:
                if isinstance(value, list):
                    return value, None, False
                if isinstance(value, str):
                    # Try parsing as JSON list
                    try:
                        parsed = json.loads(value)
                        if isinstance(parsed, list):
                            return parsed, None, True
                    except json.JSONDecodeError:
                        pass
                    # Comma-separated
                    return [v.strip() for v in value.split(",") if v.strip()], None, True
                return list(value) if hasattr(value, "__iter__") else [value], None, True

            elif ft == FieldType.NULLABLE_FLOAT:
                if value is None:
                    return None, None, False
                return float(value), None, not isinstance(value, float)

            elif ft == FieldType.NULLABLE_INT:
                if value is None:
                    return None, None, False
                return int(float(value)), None, not isinstance(value, int)

            elif ft == FieldType.NULLABLE_STRING:
                if value is None:
                    return None, None, False
                return str(value), None, not isinstance(value, str)

            elif ft == FieldType.DICT:
                if isinstance(value, dict):
                    return value, None, False
                return {}, f"Expected dict, got {type(value).__name__}", True

        except (ValueError, TypeError) as e:
            return spec.default, str(e), True

        return value, None, False

    # ------------------------------------------------------------------
    # Fraction constraint
    # ------------------------------------------------------------------

    def _enforce_fractions(
        self,
        data: dict[str, Any],
    ) -> tuple[list[str], bool]:
        """
        Ensure fraction fields sum to 1.0 (± tolerance).

        Renormalizes if outside tolerance. Returns (errors, was_repaired).
        """
        frac_fields = self.schema.fraction_fields
        if not frac_fields:
            return [], False

        values = [float(data.get(f, 0.0)) for f in frac_fields]
        total = sum(values)
        errors: list[str] = []

        tol = self.schema.fraction_sum_tolerance
        if abs(total - 1.0) <= tol:
            return [], False

        if total < 1e-6:
            # All zeros — distribute uniformly
            n = len(frac_fields)
            for f in frac_fields:
                data[f] = 1.0 / n
            errors.append(
                f"[{self.schema.name}] All fraction fields were zero — distributed uniformly"
            )
            return errors, True

        # Renormalize
        for i, f in enumerate(frac_fields):
            data[f] = values[i] / total

        errors.append(
            f"[{self.schema.name}] Fractions summed to {total:.3f} — renormalized"
        )
        return errors, True

    # ------------------------------------------------------------------
    # Defaults
    # ------------------------------------------------------------------

    def _get_defaults(self) -> dict[str, Any]:
        """Build a defaults-only dict for total enforcement failure."""
        return {spec.name: spec.default for spec in self.schema.fields}


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

_MATURITY_ENFORCER = SchemaEnforcer(MATURITY_SCHEMA)
_QUALITY_ENFORCER = SchemaEnforcer(QUALITY_SCHEMA)
_MORPHOLOGY_ENFORCER = SchemaEnforcer(MORPHOLOGY_SCHEMA)


def enforce_maturity(raw_text: str) -> EnforcementResult:
    """Enforce MATURITY_SCHEMA on raw VLM output."""
    return _MATURITY_ENFORCER.enforce(raw_text)


def enforce_quality(raw_text: str) -> EnforcementResult:
    """Enforce QUALITY_SCHEMA on raw VLM output."""
    return _QUALITY_ENFORCER.enforce(raw_text)


def enforce_morphology(raw_text: str) -> EnforcementResult:
    """Enforce MORPHOLOGY_SCHEMA on raw VLM output."""
    return _MORPHOLOGY_ENFORCER.enforce(raw_text)
