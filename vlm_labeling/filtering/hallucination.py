"""
vlm_labeling.filtering.hallucination — VLM hallucination detection and filtering.

PROBLEM:
VLMs hallucinate. For scientific microscopy analysis, hallucinations are especially
dangerous because they produce confident-sounding but wrong labels that corrupt
training data.

DETECTION STRATEGIES:

1. Confidence Gate:
   - VLM reports confidence < threshold → flag as uncertain
   - Simple but effective: hallucinations often correlate with lower stated confidence

2. Cross-Model Agreement:
   - Run same image through 2+ VLMs
   - If predictions disagree by more than threshold → flag for human review
   - Requires multiple VLM backends loaded (expensive)

3. Rule-Based Consistency Check:
   - Cross-check VLM output against rule-based system (color features)
   - If VLM says "cloudy" but color analysis shows clear amber hue → flag
   - No extra VRAM required — just CPU color analysis

4. Constraint Violation Detection:
   - VLM fraction estimates should sum to ≤ 1.0
   - Fraction values must be in [0, 1]
   - Enum values must be from known set

5. Semantic Consistency:
   - "degraded" maturity should not have clear_fraction_estimate > 0.5
   - "clear" maturity should have amber_fraction_estimate < 0.1
   - Violated constraints indicate hallucination

DESIGN DECISION:
All flagged items go to human_review queue, NOT rejected outright.
VLM labels are PSEUDO-LABELS by design. Even correct VLM labels require review.
Hallucination detection increases the priority of review, not the discard rate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from shared.logging.logger import get_logger

logger = get_logger(__name__)


class HallucinationFlag(str, Enum):
    """Why a VLM result was flagged."""

    LOW_CONFIDENCE = "low_confidence"
    """VLM stated confidence below threshold."""

    CONSTRAINT_VIOLATION = "constraint_violation"
    """Output values violate domain constraints."""

    SEMANTIC_INCONSISTENCY = "semantic_inconsistency"
    """Values are internally inconsistent."""

    RULE_DISAGREEMENT = "rule_disagreement"
    """VLM disagrees significantly with rule-based classifier."""

    CROSS_MODEL_DISAGREEMENT = "cross_model_disagreement"
    """Multiple VLMs give different predictions."""

    INVALID_JSON = "invalid_json"
    """VLM failed to produce valid JSON."""

    UNKNOWN_CLASS = "unknown_class"
    """VLM predicted a class not in the closed set."""


@dataclass
class FilterResult:
    """Result from hallucination filtering."""

    passed: bool
    """True if result passed all filters (low hallucination risk)."""

    flags: list[HallucinationFlag] = field(default_factory=list)
    """List of hallucination flags triggered."""

    flag_details: dict[str, str] = field(default_factory=dict)
    """Human-readable explanation for each flag."""

    adjusted_confidence: float = 0.0
    """
    Confidence after applying hallucination penalty.
    Lower than original if hallucinations were detected.
    """

    review_priority: int = 0
    """
    Review queue priority:
    0 = low (routine), 1 = medium, 2 = high, 3 = critical.
    Higher → review sooner.
    """

    corrected_data: Optional[dict] = None
    """Corrected/normalized label data (e.g., renormalized fractions)."""

    @property
    def is_flagged(self) -> bool:
        return len(self.flags) > 0

    @property
    def is_reliable(self) -> bool:
        """True if label passed all hallucination checks (alias for `passed`)."""
        return self.passed

    @property
    def flag_names(self) -> list[str]:
        return [f.value for f in self.flags]


@dataclass
class HallucinationFilterConfig:
    """Configuration for hallucination filtering."""

    # Confidence thresholds
    min_confidence: float = 0.40
    """
    VLM results with confidence < this are flagged.
    0.40 is deliberately permissive — VLMs are often underconfident.
    """

    high_confidence_threshold: float = 0.80
    """Results above this skip some consistency checks (reduce false positives)."""

    # Rule-based cross-check
    enable_rule_check: bool = True
    """Cross-check VLM output against rule-based color analysis."""

    rule_disagreement_threshold: float = 0.40
    """
    How much the rule-based system must disagree to flag.
    0.40 = 40% probability assigned to different class by rule system.
    """

    # Fraction sum tolerance
    fraction_sum_tolerance: float = 0.15
    """
    Max allowed deviation from 1.0 for fraction estimates that should sum to 1.0.
    Allows for ~±15% tolerance due to VLM estimation errors.
    """

    # Semantic consistency
    enable_semantic_check: bool = True
    """Check that maturity stage is consistent with fraction estimates."""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SEMANTIC CONSISTENCY RULES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Expected fraction ranges for each maturity stage
MATURITY_SEMANTIC_RULES: dict[str, dict[str, tuple[float, float]]] = {
    "clear": {
        "clear_fraction_estimate": (0.40, 1.0),   # Should be mostly clear
        "amber_fraction_estimate": (0.0, 0.15),   # Should have minimal amber
    },
    "cloudy": {
        "cloudy_fraction_estimate": (0.40, 1.0),  # Should be mostly cloudy
        "amber_fraction_estimate": (0.0, 0.25),   # Some amber tolerated
    },
    "amber": {
        "amber_fraction_estimate": (0.30, 1.0),   # Must have significant amber
    },
    "cloudy_amber_mix": {
        # Both cloudy and amber should be present
        "cloudy_fraction_estimate": (0.15, 0.85),
        "amber_fraction_estimate": (0.15, 0.85),
    },
    "degraded": {
        "clear_fraction_estimate": (0.0, 0.20),   # Should not be mostly clear
    },
    "unknown": {},  # No constraints for unknown
}

VALID_MATURITY_STAGES = {
    "clear", "cloudy", "amber", "cloudy_amber_mix", "degraded", "unknown"
}

VALID_QUALITY_LEVELS = {
    "excellent", "good", "acceptable", "poor", "unusable"
}

VALID_MORPHOLOGY_TYPES = {
    "capitate_stalked", "capitate_sessile", "bulbous",
    "non_glandular", "mixed", "unknown"
}


class HallucinationFilter:
    """
    Multi-strategy hallucination detection for VLM outputs.

    Usage:
        filt = HallucinationFilter()

        # Filter a maturity classification result
        vlm_result = labeler.label_maturity(image)
        filter_result = filt.filter_maturity(vlm_result.parsed_response)

        if not filter_result.passed:
            print(f"Flagged: {filter_result.flag_names}")
            # → Send to human review with high priority

    Note: This filter does NOT reject VLM outputs — it adjusts review priority.
    All VLM pseudo-labels require human review before entering training data.
    """

    def __init__(self, config: HallucinationFilterConfig | None = None) -> None:
        self._config = config or HallucinationFilterConfig()

    def filter_maturity(
        self,
        parsed_response: dict[str, Any] | None,
        color_features: Any | None = None,  # Optional: ColorFeatureVector from maturity service
    ) -> FilterResult:
        """
        Filter a maturity classification VLM response.

        Args:
            parsed_response: Parsed JSON from VLM (may be None if VLM failed).
            color_features: Optional color features from rule-based analysis.
                           If provided, cross-checks VLM output vs. rules.

        Returns:
            FilterResult with flags and adjusted confidence.
        """
        flags: list[HallucinationFlag] = []
        details: dict[str, str] = {}

        # 1. Check for None (invalid JSON)
        if parsed_response is None:
            return FilterResult(
                passed=False,
                flags=[HallucinationFlag.INVALID_JSON],
                flag_details={"invalid_json": "VLM failed to produce parseable JSON"},
                adjusted_confidence=0.0,
                review_priority=3,
            )

        base_conf = float(parsed_response.get("confidence", 0.0))
        adjusted_conf = base_conf

        # 2. Validate enum value
        maturity_stage = parsed_response.get("maturity_stage", "")
        if maturity_stage not in VALID_MATURITY_STAGES:
            flags.append(HallucinationFlag.UNKNOWN_CLASS)
            details["unknown_class"] = f"Stage '{maturity_stage}' not in valid set: {VALID_MATURITY_STAGES}"
            adjusted_conf *= 0.3  # Heavy penalty

        # 3. Confidence gate
        if base_conf < self._config.min_confidence:
            flags.append(HallucinationFlag.LOW_CONFIDENCE)
            details["low_confidence"] = (
                f"VLM confidence {base_conf:.2f} < threshold {self._config.min_confidence:.2f}"
            )

        # 4. Constraint validation
        constraint_violations = self._check_fraction_constraints(parsed_response)
        if constraint_violations:
            flags.append(HallucinationFlag.CONSTRAINT_VIOLATION)
            details["constraint_violation"] = "; ".join(constraint_violations)
            adjusted_conf *= 0.70

        # 5. Semantic consistency check
        if self._config.enable_semantic_check and maturity_stage in VALID_MATURITY_STAGES:
            semantic_violations = self._check_semantic_consistency_maturity(
                maturity_stage, parsed_response
            )
            if semantic_violations:
                flags.append(HallucinationFlag.SEMANTIC_INCONSISTENCY)
                details["semantic_inconsistency"] = "; ".join(semantic_violations)
                adjusted_conf *= 0.60

        # 6. Rule-based cross-check (if color features provided)
        if self._config.enable_rule_check and color_features is not None:
            rule_flag, rule_detail = self._check_rule_agreement_maturity(
                maturity_stage, base_conf, color_features
            )
            if rule_flag is not None:
                flags.append(rule_flag)
                details[rule_flag.value] = rule_detail
                adjusted_conf *= 0.55

        # Clamp confidence
        adjusted_conf = float(np.clip(adjusted_conf, 0.0, 1.0))

        # Compute review priority
        priority = self._compute_priority(flags, adjusted_conf)

        passed = (
            len(flags) == 0
            or (
                len(flags) == 1
                and flags[0] == HallucinationFlag.LOW_CONFIDENCE
                and base_conf >= self._config.min_confidence * 0.85
            )
        )

        return FilterResult(
            passed=passed,
            flags=flags,
            flag_details=details,
            adjusted_confidence=adjusted_conf,
            review_priority=priority,
        )

    def filter_quality(
        self,
        parsed_response: dict[str, Any] | None,
    ) -> FilterResult:
        """Filter an image quality assessment response."""
        if parsed_response is None:
            return FilterResult(
                passed=False,
                flags=[HallucinationFlag.INVALID_JSON],
                flag_details={"invalid_json": "VLM failed to produce parseable JSON"},
                adjusted_confidence=0.0,
                review_priority=2,
            )

        flags: list[HallucinationFlag] = []
        details: dict[str, str] = {}

        base_conf = float(parsed_response.get("confidence", 0.0))
        adjusted_conf = base_conf

        quality = parsed_response.get("overall_quality", "")
        if quality not in VALID_QUALITY_LEVELS:
            flags.append(HallucinationFlag.UNKNOWN_CLASS)
            details["unknown_class"] = f"Quality '{quality}' not in valid set"
            adjusted_conf *= 0.3

        if base_conf < self._config.min_confidence:
            flags.append(HallucinationFlag.LOW_CONFIDENCE)
            details["low_confidence"] = f"Confidence {base_conf:.2f} below threshold"

        priority = self._compute_priority(flags, adjusted_conf)
        return FilterResult(
            passed=len(flags) == 0,
            flags=flags,
            flag_details=details,
            adjusted_confidence=float(np.clip(adjusted_conf, 0, 1)),
            review_priority=priority,
        )

    def filter_morphology(
        self,
        parsed_response: dict[str, Any] | None,
    ) -> FilterResult:
        """Filter a morphology classification response."""
        if parsed_response is None:
            return FilterResult(
                passed=False,
                flags=[HallucinationFlag.INVALID_JSON],
                flag_details={"invalid_json": "No parseable JSON"},
                adjusted_confidence=0.0,
                review_priority=2,
            )

        flags: list[HallucinationFlag] = []
        details: dict[str, str] = {}

        base_conf = float(parsed_response.get("confidence", 0.0))
        adjusted_conf = base_conf

        dominant_type = parsed_response.get("dominant_type", "")
        if dominant_type not in VALID_MORPHOLOGY_TYPES:
            flags.append(HallucinationFlag.UNKNOWN_CLASS)
            details["unknown_class"] = f"Type '{dominant_type}' not in valid set"
            adjusted_conf *= 0.3

        if base_conf < self._config.min_confidence:
            flags.append(HallucinationFlag.LOW_CONFIDENCE)
            details["low_confidence"] = f"Confidence {base_conf:.2f} below threshold"

        priority = self._compute_priority(flags, adjusted_conf)
        return FilterResult(
            passed=len(flags) == 0,
            flags=flags,
            flag_details=details,
            adjusted_confidence=float(np.clip(adjusted_conf, 0, 1)),
            review_priority=priority,
        )

    def filter_cross_model(
        self,
        results: list[dict[str, Any]],
        prediction_key: str = "maturity_stage",
    ) -> FilterResult:
        """
        Check agreement across multiple VLM model outputs.

        Args:
            results: List of parsed responses from different VLMs.
            prediction_key: Key to compare across models.

        Returns:
            FilterResult with cross-model agreement assessment.
        """
        if len(results) < 2:
            return FilterResult(passed=True, adjusted_confidence=0.5)

        predictions = [
            r.get(prediction_key, "unknown")
            for r in results
            if r is not None
        ]

        if not predictions:
            return FilterResult(
                passed=False,
                flags=[HallucinationFlag.INVALID_JSON],
                adjusted_confidence=0.0,
            )

        # Check majority agreement
        from collections import Counter
        counts = Counter(predictions)
        most_common, most_count = counts.most_common(1)[0]
        agreement_rate = most_count / len(predictions)

        flags: list[HallucinationFlag] = []
        details: dict[str, str] = {}

        if agreement_rate < 0.60:
            flags.append(HallucinationFlag.CROSS_MODEL_DISAGREEMENT)
            details["cross_model_disagreement"] = (
                f"Models disagree: {dict(counts)}. "
                f"Majority agreement only {agreement_rate:.0%}."
            )

        # Average confidence from all models
        confidences = [float(r.get("confidence", 0.5)) for r in results if r is not None]
        avg_conf = float(np.mean(confidences)) if confidences else 0.5

        # Reduce confidence if disagreement
        adjusted_conf = avg_conf * agreement_rate

        priority = self._compute_priority(flags, adjusted_conf)
        return FilterResult(
            passed=len(flags) == 0,
            flags=flags,
            flag_details=details,
            adjusted_confidence=float(np.clip(adjusted_conf, 0, 1)),
            review_priority=priority,
        )

    def _check_fraction_constraints(
        self,
        response: dict[str, Any],
    ) -> list[str]:
        """Check that fraction values are in [0,1] and sum reasonably."""
        violations = []

        fraction_keys = [
            "amber_fraction_estimate",
            "cloudy_fraction_estimate",
            "clear_fraction_estimate",
        ]

        present_fractions = {}
        for key in fraction_keys:
            val = response.get(key)
            if val is not None:
                try:
                    fval = float(val)
                    if not (0.0 <= fval <= 1.0):
                        violations.append(
                            f"{key}={fval:.2f} out of [0,1] range"
                        )
                    present_fractions[key] = fval
                except (TypeError, ValueError):
                    violations.append(f"{key} is not numeric: {val!r}")

        # Check sum if all three are present
        if len(present_fractions) == 3:
            total = sum(present_fractions.values())
            tol = self._config.fraction_sum_tolerance
            if abs(total - 1.0) > tol:
                violations.append(
                    f"Fraction estimates sum to {total:.2f}, "
                    f"expected ~1.0 (±{tol:.2f})"
                )

        return violations

    def _check_semantic_consistency_maturity(
        self,
        stage: str,
        response: dict[str, Any],
    ) -> list[str]:
        """Check that fraction estimates match the stated maturity stage."""
        violations = []

        rules = MATURITY_SEMANTIC_RULES.get(stage, {})
        for fraction_key, (min_val, max_val) in rules.items():
            val = response.get(fraction_key)
            if val is not None:
                try:
                    fval = float(val)
                    if not (min_val <= fval <= max_val):
                        violations.append(
                            f"Stage '{stage}' expects {fraction_key} in "
                            f"[{min_val:.2f}, {max_val:.2f}], got {fval:.2f}"
                        )
                except (TypeError, ValueError):
                    pass

        return violations

    def _check_rule_agreement_maturity(
        self,
        vlm_stage: str,
        vlm_confidence: float,
        color_features: Any,
    ) -> tuple[HallucinationFlag | None, str]:
        """
        Cross-check VLM maturity prediction against rule-based system.

        Returns (flag, detail) or (None, "") if they agree.
        """
        # Import here to avoid circular dependency
        try:
            from maturity.domain.color_features import rule_based_maturity_estimate
            rule_stage, rule_confidence = rule_based_maturity_estimate(color_features)
            rule_stage_str = rule_stage.value.lower()

            # Map "cloudy_amber_mix" → matches both "cloudy" and "amber" categories
            def stages_compatible(s1: str, s2: str) -> bool:
                if s1 == s2:
                    return True
                mix_stages = {"cloudy", "amber"}
                if s1 == "cloudy_amber_mix" and s2 in mix_stages:
                    return True
                if s2 == "cloudy_amber_mix" and s1 in mix_stages:
                    return True
                # "unknown" is always compatible
                if "unknown" in (s1, s2):
                    return True
                return False

            if not stages_compatible(vlm_stage, rule_stage_str):
                # Only flag if rule system is also reasonably confident
                if rule_confidence > self._config.rule_disagreement_threshold:
                    return (
                        HallucinationFlag.RULE_DISAGREEMENT,
                        f"VLM predicts '{vlm_stage}' (conf={vlm_confidence:.2f}) "
                        f"but rule system predicts '{rule_stage_str}' "
                        f"(conf={rule_confidence:.2f})"
                    )

        except ImportError:
            pass  # Rule-based system not available
        except Exception as e:
            logger.debug("Rule check failed", error=str(e))

        return None, ""

    def _compute_priority(
        self,
        flags: list[HallucinationFlag],
        adjusted_confidence: float,
    ) -> int:
        """Compute review priority based on flags and confidence."""
        if HallucinationFlag.INVALID_JSON in flags:
            return 3  # Critical — model failed completely
        if HallucinationFlag.UNKNOWN_CLASS in flags:
            return 3  # Critical — model produced impossible output
        if HallucinationFlag.CROSS_MODEL_DISAGREEMENT in flags:
            return 2  # High — models disagree
        if HallucinationFlag.SEMANTIC_INCONSISTENCY in flags:
            return 2  # High — internally inconsistent
        if HallucinationFlag.RULE_DISAGREEMENT in flags:
            return 2  # High — disagrees with physics-based system
        if HallucinationFlag.CONSTRAINT_VIOLATION in flags:
            return 2  # High — violated domain constraints
        if HallucinationFlag.LOW_CONFIDENCE in flags:
            if adjusted_confidence < 0.25:
                return 2
            return 1  # Medium
        if adjusted_confidence < 0.50:
            return 1  # Medium — no flags but low confidence
        return 0  # Low — routine review


# Backward-compatibility alias (tests use FilterConfig)
FilterConfig = HallucinationFilterConfig



# ---------------------------------------------------------------------------
# filter_label — generic convenience method for test compatibility
# ---------------------------------------------------------------------------

def _filter_label_generic(self, label: dict) -> "FilterResult":
    """
    Generic label filter. Accepts a dict with confidence + fraction fields.
    
    The test API: hfilter.filter_label({"confidence": 0.9, "clear": 0.1, ...})
    Returns FilterResult with is_reliable and corrected_data.
    """
    confidence = float(label.get("confidence", 0.5))
    frac_keys = [k for k in ("clear", "cloudy", "amber", "mixed",
                              "clear_fraction", "cloudy_fraction", "amber_fraction") 
                 if k in label]
    frac_values = {k: float(label[k]) for k in frac_keys}
    total = sum(frac_values.values())
    
    flags: list[HallucinationFlag] = []
    flag_details: dict = {}
    corrected: dict = dict(label)
    
    # Check confidence threshold
    if confidence < self._config.min_confidence:
        flags.append(HallucinationFlag.LOW_CONFIDENCE)
        flag_details[HallucinationFlag.LOW_CONFIDENCE.value] = (
            f"Confidence {confidence:.2f} < threshold {self._config.min_confidence:.2f}"
        )
    
    # Check and normalize fractions
    if frac_keys and abs(total - 1.0) > 0.05:
        if total > 0:
            for k in frac_keys:
                corrected[k] = frac_values[k] / total
        flags.append(HallucinationFlag.CONSTRAINT_VIOLATION)
        flag_details[HallucinationFlag.CONSTRAINT_VIOLATION.value] = (
            f"Fractions summed to {total:.3f} instead of 1.0"
        )
    
    # Check for impossible negatives
    if any(v < 0 for v in frac_values.values()):
        flags.append(HallucinationFlag.IMPOSSIBLE_VALUE)
        flag_details[HallucinationFlag.IMPOSSIBLE_VALUE.value] = "Negative fraction detected"
    
    penalty = min(0.3, len(flags) * 0.1)
    adjusted = max(0.0, confidence - penalty)
    passed = len(flags) == 0
    
    return FilterResult(
        passed=passed,
        flags=flags,
        flag_details=flag_details,
        adjusted_confidence=adjusted,
        review_priority=self._compute_priority(flags, adjusted),
        corrected_data=corrected if flags else None,
    )


HallucinationFilter.filter_label = _filter_label_generic  # type: ignore[attr-defined]
