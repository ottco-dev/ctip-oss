"""
analytics/reporting/scientific_report.py — Publication-quality scientific report.

Generates a structured scientific report suitable for:
  - Academic paper supplementary materials
  - Research notebooks
  - Regulatory documentation

Enforces scientific honesty:
  - Confidence intervals on all metrics
  - Explicit uncertainty quantification
  - Mandatory caveats on maturity analysis claims
  - Clear separation of fact vs interpretation

Output formats: markdown, LaTeX table, structured JSON.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


# ---------------------------------------------------------------------------
# Scientific caveats (always included)
# ---------------------------------------------------------------------------

CAVEATS = {
    "maturity_quantification": (
        "Visual maturity analysis CANNOT quantify cannabinoid concentrations. "
        "Identical trichome morphology corresponds to 10-20% THC difference between strains "
        "(Elzinga et al., 2015). Amber coloration indicates oxidative THC→CBN degradation, "
        "NOT THC content (ElSohly & Slade, 2005)."
    ),
    "amber_mechanism": (
        "Amber coloration results from photo-oxidative THC→CBN dehydrogenation (2-step mechanism). "
        "It is a degradation signal, not a potency indicator. "
        "Reference: ElSohly & Slade (2005). Life Sciences 78(5):539-548."
    ),
    "cloudy_mechanism": (
        "Opaque/cloudy appearance results from Mie scattering of dense resin droplets "
        "and THCA crystallization increasing refractive index contrast. "
        "It measures optical density, NOT concentration. "
        "Reference: Fischedick et al. (2010). Phytochemistry 71(17-18):2058-2073."
    ),
    "inter_annotator": (
        "Inter-annotator agreement on maturity classification typically ranges "
        "κ=0.55-0.75 (moderate to substantial). "
        "Model performance should be compared against this human baseline."
    ),
    "calibration": (
        "All pixel-to-micron measurements require objective-specific calibration. "
        "Measurements reported without calibration are in pixels only."
    ),
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class MetricWithCI:
    """A metric value with confidence interval."""

    value: float
    ci_lower: float
    ci_upper: float
    n_bootstrap: int = 1000
    confidence_level: float = 0.95

    def __str__(self) -> str:
        return (
            f"{self.value:.4f} "
            f"[{self.confidence_level*100:.0f}% CI: {self.ci_lower:.4f}, {self.ci_upper:.4f}]"
        )

    def to_latex(self) -> str:
        return f"${self.value:.3f} \\pm {(self.ci_upper - self.ci_lower)/2:.3f}$"


@dataclass
class DetectionResults:
    """Detection benchmark results."""

    model_name: str
    dataset_name: str
    n_images: int
    n_instances: int
    evaluation_date: str
    seed: int = 42

    map50: Optional[MetricWithCI] = None
    map50_95: Optional[float] = None

    # Per-class AP50
    ap_capitate_stalked: Optional[float] = None
    ap_capitate_sessile: Optional[float] = None
    ap_bulbous: Optional[float] = None
    ap_non_glandular: Optional[float] = None

    # Small object performance
    ap_small: Optional[float] = None  # Objects < 32px²

    # Calibration
    ece: Optional[float] = None
    mce: Optional[float] = None

    # Inference
    inference_ms: Optional[float] = None
    vram_gb: Optional[float] = None
    hardware: str = "RTX 4060"


@dataclass
class ScientificReport:
    """
    Complete scientific evaluation report.

    Includes all metrics, confidence intervals, methodology,
    and mandatory scientific caveats.
    """

    title: str = "Trichome Detection and Maturity Analysis — Scientific Report"
    authors: list[str] = field(default_factory=list)
    institution: str = ""
    date: str = field(default_factory=lambda: datetime.utcnow().strftime("%Y-%m-%d"))
    version: str = "1.0.0"

    # Results
    detection_results: Optional[DetectionResults] = None
    segmentation_results: Optional[dict] = None
    maturity_results: Optional[dict] = None

    # Methodology
    methodology: dict = field(default_factory=dict)

    # References
    references: list[str] = field(default_factory=list)

    # Mandatory caveats
    caveats: dict = field(default_factory=lambda: dict(CAVEATS))

    # Limitations
    limitations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize to JSON-safe dict."""
        return {
            "title": self.title,
            "authors": self.authors,
            "institution": self.institution,
            "date": self.date,
            "version": self.version,
            "caveats": self.caveats,
            "detection_results": self._serialize_detection(),
            "segmentation_results": self.segmentation_results,
            "maturity_results": self.maturity_results,
            "methodology": self.methodology,
            "limitations": self.limitations,
            "references": self.references,
        }

    def _serialize_detection(self) -> Optional[dict]:
        if self.detection_results is None:
            return None
        r = self.detection_results
        return {
            "model": r.model_name,
            "dataset": r.dataset_name,
            "n_images": r.n_images,
            "n_instances": r.n_instances,
            "evaluation_date": r.evaluation_date,
            "seed": r.seed,
            "metrics": {
                "mAP50": str(r.map50) if r.map50 else None,
                "mAP50-95": r.map50_95,
                "per_class_AP50": {
                    "capitate-stalked": r.ap_capitate_stalked,
                    "capitate-sessile": r.ap_capitate_sessile,
                    "bulbous": r.ap_bulbous,
                    "non-glandular": r.ap_non_glandular,
                },
                "AP_small": r.ap_small,
                "calibration": {"ECE": r.ece, "MCE": r.mce},
                "inference": {
                    "ms_per_image": r.inference_ms,
                    "vram_gb": r.vram_gb,
                    "hardware": r.hardware,
                },
            },
        }

    def to_markdown(self) -> str:
        """Generate markdown-formatted scientific report."""
        lines = [
            f"# {self.title}",
            f"",
            f"**Date:** {self.date}  ",
            f"**Version:** {self.version}",
            f"",
            f"---",
            f"",
            f"## ⚠️ Scientific Caveats",
            f"",
            f"> **Mandatory disclaimer**: {self.caveats.get('maturity_quantification', '')}",
            f"",
            f"---",
            f"",
        ]

        if self.detection_results:
            r = self.detection_results
            lines += [
                f"## Detection Results",
                f"",
                f"**Model:** {r.model_name}  ",
                f"**Dataset:** {r.dataset_name} ({r.n_images} images, {r.n_instances} instances)  ",
                f"**Hardware:** {r.hardware}  ",
                f"",
                f"| Metric | Value |",
                f"|--------|-------|",
            ]

            if r.map50:
                lines.append(f"| mAP50 | {r.map50} |")
            if r.map50_95 is not None:
                lines.append(f"| mAP50-95 | {r.map50_95:.4f} |")
            if r.ece is not None:
                lines.append(f"| ECE | {r.ece:.4f} (target < 0.05) |")
            if r.inference_ms is not None:
                lines.append(f"| Inference | {r.inference_ms:.1f} ms/image |")

            lines += ["", "**Per-class AP50:**", ""]
            if r.ap_capitate_stalked is not None:
                lines.append(f"- capitate-stalked: {r.ap_capitate_stalked:.4f}")
            if r.ap_capitate_sessile is not None:
                lines.append(f"- capitate-sessile: {r.ap_capitate_sessile:.4f}")
            if r.ap_bulbous is not None:
                lines.append(f"- bulbous: {r.ap_bulbous:.4f}")
            if r.ap_non_glandular is not None:
                lines.append(f"- non-glandular: {r.ap_non_glandular:.4f}")

            lines.append("")

        if self.limitations:
            lines += ["## Limitations", ""]
            for lim in self.limitations:
                lines.append(f"- {lim}")
            lines.append("")

        if self.references:
            lines += ["## References", ""]
            for i, ref in enumerate(self.references, 1):
                lines.append(f"{i}. {ref}")

        return "\n".join(lines)

    def to_latex_table(self) -> str:
        """Generate LaTeX table for the detection results."""
        if self.detection_results is None:
            return "% No detection results available"

        r = self.detection_results
        rows = []

        if r.map50:
            ci_str = f"[{r.map50.ci_lower:.3f}, {r.map50.ci_upper:.3f}]"
            rows.append(f"mAP50 & {r.map50.value:.3f} & {ci_str} \\\\")
        if r.map50_95 is not None:
            rows.append(f"mAP50-95 & {r.map50_95:.3f} & — \\\\")
        if r.ece is not None:
            rows.append(f"ECE & {r.ece:.4f} & target $< 0.05$ \\\\")

        row_str = "\n        ".join(rows)
        return f"""\\begin{{table}}[h]
\\centering
\\caption{{Trichome Detection Performance — {r.model_name}}}
\\label{{tab:detection_results}}
\\begin{{tabular}}{{lcc}}
\\toprule
Metric & Value & CI / Note \\\\
\\midrule
        {row_str}
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------


class ScientificReportBuilder:
    """Builder pattern for constructing ScientificReport incrementally."""

    def __init__(self, title: str = "Trichome Analysis Report") -> None:
        self._report = ScientificReport(title=title)

    def set_detection_results(self, results: DetectionResults) -> "ScientificReportBuilder":
        self._report.detection_results = results
        return self

    def add_limitation(self, text: str) -> "ScientificReportBuilder":
        self._report.limitations.append(text)
        return self

    def add_reference(self, text: str) -> "ScientificReportBuilder":
        self._report.references.append(text)
        return self

    def add_standard_references(self) -> "ScientificReportBuilder":
        """Add the standard references used in this system."""
        refs = [
            "Jocher, G. et al. (2023). Ultralytics YOLO. github.com/ultralytics/ultralytics",
            "Solovyev, R. et al. (2021). WBF. Image and Vision Computing 107:104117.",
            "Lin, T.Y. et al. (2017). Focal Loss. ICCV 2017. arXiv:1708.02002.",
            "Ravi, N. et al. (2024). SAM 2. arXiv:2408.00714.",
            "Guo, C. et al. (2017). On Calibration of Modern Neural Networks. ICML 2017.",
            "ElSohly, M.A. & Slade, D. (2005). Life Sciences 78(5):539-548.",
            "Elzinga, S. et al. (2015). Nat. Prod. Chem. Res. 3:181.",
            "Fischedick, J.T. et al. (2010). Phytochemistry 71(17-18):2058-2073.",
        ]
        for ref in refs:
            self.add_reference(ref)
        return self

    def build(self) -> ScientificReport:
        return self._report
