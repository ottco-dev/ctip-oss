"""
analytics/reporting/session_report.py — Full analysis session summary report.

Aggregates results from one analysis session (one or more images processed
through the full pipeline) into a structured session summary.

Output formats:
  - SessionReport dataclass (in-memory)
  - JSON (via analytics/export/json_exporter.py)
  - PDF (via analytics/export/pdf_exporter.py)
  - CSV (via analytics/export/csv_exporter.py)
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Scientific disclaimer (required on all reports)
# ---------------------------------------------------------------------------

SCIENTIFIC_CAVEAT = (
    "IMPORTANT: Visual trichome maturity analysis provides a qualitative proxy for "
    "harvest timing guidance only. It does NOT allow quantitative determination of "
    "THC, CBD, or other cannabinoid concentrations. Identical visual appearance may "
    "correspond to 10-20% difference in THC content between strains. "
    "Reference: Elzinga et al. (2015). Cannabinoids and terpenes as chemotaxonomic "
    "markers in cannabis. Nat. Prod. Chem. Res. 3:181."
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class TrichomeTypeSummary:
    """Per-type detection summary."""

    type_name: str
    count: int
    fraction: float
    mean_confidence: float
    mean_area_px: float = 0.0
    mean_diameter_um: Optional[float] = None


@dataclass
class MaturitySummary:
    """Maturity distribution across session."""

    clear_fraction: float = 0.0
    cloudy_fraction: float = 0.0
    amber_fraction: float = 0.0
    mixed_fraction: float = 0.0
    dominant_stage: str = "unknown"
    harvest_recommendation: str = ""
    scientific_caveat: str = SCIENTIFIC_CAVEAT

    def __post_init__(self):
        # Ensure fractions sum to 1
        total = self.clear_fraction + self.cloudy_fraction + self.amber_fraction + self.mixed_fraction
        if total > 0 and abs(total - 1.0) > 0.01:
            self.clear_fraction /= total
            self.cloudy_fraction /= total
            self.amber_fraction /= total
            self.mixed_fraction /= total

        # Dominant stage
        stages = {
            "clear": self.clear_fraction,
            "cloudy": self.cloudy_fraction,
            "amber": self.amber_fraction,
            "mixed": self.mixed_fraction,
        }
        self.dominant_stage = max(stages, key=stages.get)

        # Harvest recommendation (conservative language)
        self.harvest_recommendation = _harvest_recommendation(
            self.cloudy_fraction,
            self.amber_fraction,
        )


def _harvest_recommendation(cloudy: float, amber: float) -> str:
    """
    Generate harvest timing guidance from maturity fractions.

    Uses conservative language — never claims THC quantification.
    """
    if amber > 0.3:
        return (
            "Amber fraction > 30%: trichomes showing significant oxidative degradation. "
            "Consider harvesting soon to preserve terpene profile. "
            "Note: amber coloration reflects oxidative degradation — optical observation only, not a measure of any specific compound."
        )
    elif cloudy > 0.7:
        return (
            "Cloudy fraction > 70%: trichomes appear opaque — often associated with "
            "peak resin density. Traditional harvest timing indicator. "
            "Verify with strain-specific guidance."
        )
    elif cloudy > 0.4:
        return (
            "Mixed cloudy/clear: harvest window approaching. "
            "Monitor daily for continued development."
        )
    else:
        return (
            "Predominantly clear trichomes: typically indicates continued development needed. "
            "Re-evaluate in 5-10 days."
        )


@dataclass
class ImageResult:
    """Per-image analysis result in a session."""

    image_path: str
    detection_count: int
    mean_confidence: float = 0.0
    maturity: Optional[MaturitySummary] = None
    focus_score: float = 0.0
    exposure_ok: bool = True
    processing_ms: float = 0.0
    error: Optional[str] = None


@dataclass
class SessionReport:
    """
    Aggregated session report across all analyzed images.

    Suitable for JSON/PDF export and display in the web UI.
    """

    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    session_name: str = "Trichome Analysis Session"
    created_at: datetime = field(default_factory=datetime.utcnow)

    # Image metadata
    total_images: int = 0
    successful_images: int = 0
    failed_images: int = 0

    # Detection aggregate
    total_trichomes: int = 0
    mean_trichomes_per_image: float = 0.0
    mean_confidence: float = 0.0
    type_summaries: list[TrichomeTypeSummary] = field(default_factory=list)

    # Maturity aggregate
    maturity_summary: Optional[MaturitySummary] = None

    # Quality metrics
    mean_focus_score: float = 0.0
    low_focus_images: int = 0
    overexposed_images: int = 0

    # Per-image results
    image_results: list[ImageResult] = field(default_factory=list)

    # Scientific compliance
    scientific_caveat: str = SCIENTIFIC_CAVEAT

    # Pipeline info
    detection_model: str = ""
    segmentation_model: str = ""
    processing_time_s: float = 0.0
    hardware: str = "RTX 4060"

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        import dataclasses

        def _convert(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            if isinstance(obj, list):
                return [_convert(i) for i in obj]
            if dataclasses.is_dataclass(obj):
                return {k: _convert(v) for k, v in dataclasses.asdict(obj).items()}
            return obj

        return _convert(self)


# ---------------------------------------------------------------------------
# Session builder
# ---------------------------------------------------------------------------


class SessionReportBuilder:
    """
    Collects per-image results and builds a SessionReport.

    Usage:
        builder = SessionReportBuilder("My Session")
        for image in images:
            result = pipeline.run(image)
            builder.add_result(image_path, result)
        report = builder.build()
    """

    def __init__(
        self,
        session_name: str = "Analysis Session",
        detection_model: str = "",
        segmentation_model: str = "",
    ) -> None:
        self.session_name = session_name
        self.detection_model = detection_model
        self.segmentation_model = segmentation_model
        self._results: list[ImageResult] = []
        self._start_time = datetime.utcnow()

    def add_image_result(self, result: ImageResult) -> None:
        """Add a per-image analysis result."""
        self._results.append(result)

    def add_raw(
        self,
        image_path: str,
        detection_count: int = 0,
        mean_confidence: float = 0.0,
        maturity_fractions: Optional[dict] = None,
        focus_score: float = 0.0,
        exposure_ok: bool = True,
        processing_ms: float = 0.0,
        error: Optional[str] = None,
    ) -> None:
        """Add a result from raw pipeline outputs."""
        maturity = None
        if maturity_fractions:
            maturity = MaturitySummary(
                clear_fraction=maturity_fractions.get("clear", 0.0),
                cloudy_fraction=maturity_fractions.get("cloudy", 0.0),
                amber_fraction=maturity_fractions.get("amber", 0.0),
                mixed_fraction=maturity_fractions.get("mixed", 0.0),
            )
        self.add_image_result(
            ImageResult(
                image_path=image_path,
                detection_count=detection_count,
                mean_confidence=mean_confidence,
                maturity=maturity,
                focus_score=focus_score,
                exposure_ok=exposure_ok,
                processing_ms=processing_ms,
                error=error,
            )
        )

    def build(self) -> SessionReport:
        """Build and return the final SessionReport."""
        successful = [r for r in self._results if r.error is None]
        failed = [r for r in self._results if r.error is not None]

        total_trichomes = sum(r.detection_count for r in successful)
        total_images = len(self._results)
        n_ok = len(successful)

        mean_conf = (
            sum(r.mean_confidence for r in successful) / n_ok if n_ok > 0 else 0.0
        )
        mean_focus = (
            sum(r.focus_score for r in successful) / n_ok if n_ok > 0 else 0.0
        )
        low_focus = sum(1 for r in successful if r.focus_score < 80.0)
        overexposed = sum(1 for r in successful if not r.exposure_ok)

        # Aggregate maturity
        maturity_agg: dict[str, list[float]] = {
            "clear": [], "cloudy": [], "amber": [], "mixed": []
        }
        for r in successful:
            if r.maturity:
                maturity_agg["clear"].append(r.maturity.clear_fraction)
                maturity_agg["cloudy"].append(r.maturity.cloudy_fraction)
                maturity_agg["amber"].append(r.maturity.amber_fraction)
                maturity_agg["mixed"].append(r.maturity.mixed_fraction)

        maturity_summary = None
        if any(len(v) > 0 for v in maturity_agg.values()):
            def _mean(lst):
                return sum(lst) / len(lst) if lst else 0.0

            maturity_summary = MaturitySummary(
                clear_fraction=round(_mean(maturity_agg["clear"]), 3),
                cloudy_fraction=round(_mean(maturity_agg["cloudy"]), 3),
                amber_fraction=round(_mean(maturity_agg["amber"]), 3),
                mixed_fraction=round(_mean(maturity_agg["mixed"]), 3),
            )

        elapsed = (datetime.utcnow() - self._start_time).total_seconds()

        return SessionReport(
            session_name=self.session_name,
            total_images=total_images,
            successful_images=n_ok,
            failed_images=len(failed),
            total_trichomes=total_trichomes,
            mean_trichomes_per_image=round(total_trichomes / n_ok if n_ok > 0 else 0.0, 1),
            mean_confidence=round(mean_conf, 4),
            maturity_summary=maturity_summary,
            mean_focus_score=round(mean_focus, 2),
            low_focus_images=low_focus,
            overexposed_images=overexposed,
            image_results=self._results,
            detection_model=self.detection_model,
            segmentation_model=self.segmentation_model,
            processing_time_s=round(elapsed, 2),
        )
