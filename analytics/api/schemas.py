"""
analytics.api.schemas — Pydantic request/response models for the analytics API.

Covers:
  - Confidence calibration (ECE / reliability diagrams)
  - Per-run calibration queries
  - Population-level confidence histograms
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Calibration — request
# ---------------------------------------------------------------------------

class CalibrationRequest(BaseModel):
    """
    Input payload for ECE / reliability diagram computation.

    Provide either:
      - (confidences, is_correct) for raw predictions, OR
      - mlflow_run_id to pull logged prediction artifacts from MLflow.
    """

    confidences: Optional[List[float]] = Field(
        default=None,
        description="Model confidence scores in [0, 1]. Must match length of is_correct.",
        min_length=1,
        max_length=100_000,
    )
    is_correct: Optional[List[bool]] = Field(
        default=None,
        description="Boolean correctness flags for each prediction.",
        min_length=1,
        max_length=100_000,
    )
    mlflow_run_id: Optional[str] = Field(
        default=None,
        description="MLflow run ID to load confidence artifacts from.",
        pattern=r"^[a-f0-9]{32}$",
    )
    num_bins: int = Field(
        default=10,
        ge=5,
        le=50,
        description="Number of equal-width bins for calibration. Default 10 (Guo et al. 2017).",
    )

    model_config = {"json_schema_extra": {
        "examples": [{
            "confidences": [0.9, 0.8, 0.7, 0.6, 0.5],
            "is_correct":  [True, True, False, True, False],
            "num_bins": 10,
        }]
    }}


# ---------------------------------------------------------------------------
# Calibration — response
# ---------------------------------------------------------------------------

class BinStats(BaseModel):
    """Per-bin statistics for the reliability diagram."""

    bin_index: int
    confidence_lower: float = Field(description="Lower edge of this bin (inclusive).")
    confidence_upper: float = Field(description="Upper edge of this bin (exclusive, except last).")
    mean_confidence: float = Field(description="Mean confidence of samples in this bin.")
    accuracy: float = Field(description="Observed accuracy of samples in this bin.")
    count: int = Field(description="Number of samples in this bin.")
    gap: float = Field(
        description="Signed calibration gap: mean_confidence − accuracy. "
                    "Positive = overconfident, negative = underconfident."
    )
    abs_gap: float = Field(description="Absolute calibration error for this bin.")
    weight: float = Field(description="Fractional weight of this bin (count / total_n).")

    # Derived UI flags
    is_overconfident: bool = Field(
        description="True if mean_confidence > accuracy for this bin."
    )
    is_empty: bool = Field(
        description="True if count == 0 (no samples in this bin)."
    )


class CalibrationResponse(BaseModel):
    """
    Full calibration analysis result.

    Includes scalar metrics (ECE, MCE) and per-bin data for
    rendering reliability diagrams on the frontend.
    """

    # ── Scalar metrics ──────────────────────────────────────────────────────
    ece: float = Field(
        description="Expected Calibration Error ∈ [0, 1]. Lower is better. "
                    "ECE = Σ |acc(b) − conf(b)| × |B_b| / n across all bins b."
    )
    mce: float = Field(
        description="Maximum Calibration Error — worst single-bin |acc − conf|."
    )
    num_bins: int
    total_samples: int

    # ── Summary flags ───────────────────────────────────────────────────────
    is_overconfident: bool = Field(
        description="True if weighted-average confidence > weighted-average accuracy."
    )
    overconfident_bin_fraction: float = Field(
        description="Fraction of non-empty bins that are overconfident."
    )

    # ── Per-bin data (reliability diagram) ──────────────────────────────────
    bins: List[BinStats] = Field(
        description="Per-bin statistics. Length = num_bins. "
                    "Empty bins have accuracy=0, mean_confidence=0."
    )

    # ── Histogram (for confidence distribution overlay) ─────────────────────
    confidence_histogram: List[int] = Field(
        description="Confidence histogram with same bin edges as calibration bins. "
                    "Identical to [b.count for b in bins]."
    )

    # ── Interpretation ───────────────────────────────────────────────────────
    interpretation: str = Field(
        description="Human-readable summary: calibration quality assessment."
    )

    # ── Source metadata ──────────────────────────────────────────────────────
    mlflow_run_id: Optional[str] = None
    source: str = Field(
        default="direct",
        description="'direct' (payload) or 'mlflow' (loaded from run artifacts).",
    )

    model_config = {"json_schema_extra": {"examples": []}}


# ---------------------------------------------------------------------------
# Confidence histogram endpoint
# ---------------------------------------------------------------------------

class ConfidenceHistogramRequest(BaseModel):
    """Request for a standalone confidence histogram (no correctness labels needed)."""

    confidences: List[float] = Field(
        min_length=1,
        max_length=100_000,
        description="Raw confidence scores in [0, 1].",
    )
    num_bins: int = Field(default=20, ge=5, le=100)


class ConfidenceBin(BaseModel):
    bin_index: int
    lower: float
    upper: float
    count: int
    fraction: float


class ConfidenceHistogramResponse(BaseModel):
    bins: List[ConfidenceBin]
    total: int
    mean_confidence: float
    median_confidence: float
    std_confidence: float
    fraction_high_confidence: float = Field(
        description="Fraction of predictions with confidence ≥ 0.70."
    )
