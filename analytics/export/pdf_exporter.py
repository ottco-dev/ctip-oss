"""
analytics/export/pdf_exporter.py — ReportLab PDF report generation.

Generates a professionally formatted PDF report for trichome analysis sessions.

Features:
  - Cover page with session metadata
  - Executive summary with key findings
  - Maturity distribution table + embedded chart
  - Per-class detection results table
  - Mandatory scientific caveats section (always included)
  - Page numbers and headers
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Optional

try:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
    from reportlab.lib.pagesizes import A4, letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm, inch
    from reportlab.platypus import (
        BaseDocTemplate,
        Frame,
        Image,
        PageBreak,
        PageTemplate,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )
    from reportlab.platypus.flowables import HRFlowable

    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BRAND_GREEN = "#2e7d32"
BRAND_DARK = "#1a237e"
WARNING_RED = "#c62828"
TABLE_HEADER_BG = "#263238"
TABLE_ALT_BG = "#eceff1"

SCIENTIFIC_CAVEAT = (
    "IMPORTANT SCIENTIFIC DISCLAIMER: Visual trichome maturity analysis provides a "
    "qualitative proxy for harvest timing guidance ONLY. It does NOT allow quantitative "
    "determination of THC, CBD, or any other cannabinoid concentrations. Identical "
    "trichome appearance may correspond to 10-20% difference in THC content between "
    "cannabis strains (Elzinga et al., 2015, Nat. Prod. Chem. Res. 3:181). Amber "
    "coloration indicates photo-oxidative THC→CBN degradation, NOT high THC content "
    "(ElSohly & Slade, 2005, Life Sciences 78(5):539-548). All conclusions from "
    "visual analysis should be validated with analytical chemistry (HPLC, GC-MS)."
)


# ---------------------------------------------------------------------------
# PDF builder
# ---------------------------------------------------------------------------


def export_calibration_pdf(
    calibration: dict,
    output_path: str | Path,
    model_id: str = "unknown",
    run_id: str | None = None,
) -> Path:
    """
    Generate a standalone PDF calibration report.

    Args:
        calibration: Dict matching ``CalibrationResponse`` schema, e.g.:
            {
              "ece": 0.034,
              "mce": 0.072,
              "num_bins": 10,
              "total_samples": 3200,
              "bins": [{"mean_confidence": 0.05, "accuracy": 0.06, ...}, ...],
              "interpretation": "...",
              "is_overconfident": False,
            }
        output_path: Output .pdf path.
        model_id: Model identifier for the report title.
        run_id: Optional MLflow / experiment run ID.

    Returns:
        Path to the generated PDF.

    Raises:
        ImportError: If reportlab is not installed.
    """
    if not REPORTLAB_AVAILABLE:
        raise ImportError("reportlab is not installed. Run: pip install reportlab")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2.5 * cm,
        bottomMargin=2 * cm,
        title=f"Calibration Report — {model_id}",
        author="Trichome Analysis System",
    )

    styles = getSampleStyleSheet()
    story = []

    # Cover
    story.append(Spacer(1, 2 * cm))
    story.append(
        Paragraph(
            "Model Calibration Report",
            ParagraphStyle(
                "cal_title",
                fontSize=24,
                textColor=colors.HexColor(BRAND_DARK),
                alignment=TA_CENTER,
                spaceAfter=0.5 * cm,
                fontName="Helvetica-Bold",
            ),
        )
    )
    story.append(
        Paragraph(
            model_id,
            ParagraphStyle(
                "cal_subtitle",
                fontSize=14,
                textColor=colors.HexColor(BRAND_GREEN),
                alignment=TA_CENTER,
                spaceAfter=0.5 * cm,
            ),
        )
    )
    if run_id:
        story.append(
            Paragraph(
                f"Run ID: {run_id}",
                ParagraphStyle("runid", fontSize=9, alignment=TA_CENTER, textColor=colors.gray),
            )
        )
    story.append(Spacer(1, 0.5 * cm))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor(BRAND_DARK)))
    story.append(Spacer(1, 1 * cm))

    # Calibration section
    story.extend(_build_calibration_section(calibration, styles, include_chart=True))

    # References
    story.append(Spacer(1, 1 * cm))
    story.append(Paragraph("Methodology", styles["Heading2"]))
    story.append(
        Paragraph(
            "Expected Calibration Error (ECE) computed via equal-width binning "
            "(Guo et al., 2017). ECE = Σ |B_b|/n × |acc(B_b) − conf(B_b)|. "
            "Maximum Calibration Error (MCE) = max_b |acc(B_b) − conf(B_b)|. "
            "IoU matching follows the COCO protocol (class-aware, greedy "
            "confidence-descending, each GT matched at most once, IoU ≥ 0.5).",
            styles["Normal"],
        )
    )
    story.append(Spacer(1, 0.5 * cm))
    story.append(
        Paragraph(
            "Guo C, Pleiss G, Sun Y, Weinberger KQ (2017). On Calibration of Modern "
            "Neural Networks. ICML 2017. arXiv:1706.04599",
            styles["Normal"],
        )
    )

    doc.build(story)
    return output_path


def export_session_pdf(
    session_report,  # analytics.reporting.session_report.SessionReport
    output_path: str | Path,
    include_charts: bool = True,
    page_size=None,
    calibration: dict | None = None,
) -> Path:
    """
    Generate a PDF report from a SessionReport.

    Args:
        session_report: Populated SessionReport object.
        output_path: Output .pdf file path.
        include_charts: Embed matplotlib charts in PDF.
        page_size: Defaults to A4.
        calibration: Optional ``CalibrationResponse``-compatible dict.
            If provided, a calibration section (reliability diagram + ECE/MCE table)
            is appended before the References. Obtain from
            ``POST /api/v1/analytics/calibration`` or
            ``GET /api/v1/analytics/calibration/run/{run_id}``.

    Returns:
        Path to the generated PDF.

    Raises:
        ImportError: If reportlab is not installed.
    """
    if not REPORTLAB_AVAILABLE:
        raise ImportError(
            "reportlab is not installed. Run: pip install reportlab"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    page_size = page_size or A4

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=page_size,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2.5 * cm,
        bottomMargin=2 * cm,
        title=f"Trichome Analysis — {session_report.session_name}",
        author="Trichome Analysis System",
    )

    styles = getSampleStyleSheet()
    story = []

    # ------------------------------------------------------------------
    # Cover page
    # ------------------------------------------------------------------
    story.append(Spacer(1, 3 * cm))
    story.append(
        Paragraph(
            "Trichome Analysis Report",
            ParagraphStyle(
                "cover_title",
                fontSize=28,
                textColor=colors.HexColor(BRAND_DARK),
                alignment=TA_CENTER,
                spaceAfter=0.5 * cm,
                fontName="Helvetica-Bold",
            ),
        )
    )
    story.append(
        Paragraph(
            session_report.session_name,
            ParagraphStyle(
                "cover_subtitle",
                fontSize=16,
                textColor=colors.HexColor(BRAND_GREEN),
                alignment=TA_CENTER,
                spaceAfter=2 * cm,
            ),
        )
    )

    meta_style = ParagraphStyle(
        "meta", fontSize=11, alignment=TA_CENTER, textColor=colors.gray
    )
    story.append(Paragraph(f"Generated: {session_report.created_at.strftime('%Y-%m-%d %H:%M UTC')}", meta_style))
    story.append(Paragraph(f"Session ID: {session_report.session_id}", meta_style))
    story.append(Spacer(1, 1 * cm))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor(BRAND_DARK)))
    story.append(PageBreak())

    # ------------------------------------------------------------------
    # Scientific caveat (mandatory — page 2)
    # ------------------------------------------------------------------
    caveat_style = ParagraphStyle(
        "caveat",
        fontSize=10,
        textColor=colors.HexColor(WARNING_RED),
        backColor=colors.HexColor("#fff8e1"),
        borderPadding=(10, 10, 10, 10),
        spaceAfter=1 * cm,
        alignment=TA_JUSTIFY,
    )
    story.append(
        Paragraph("⚠ Scientific Disclaimer", styles["Heading1"])
    )
    story.append(Paragraph(SCIENTIFIC_CAVEAT, caveat_style))
    story.append(Spacer(1, 0.5 * cm))

    # ------------------------------------------------------------------
    # Executive summary
    # ------------------------------------------------------------------
    story.append(Paragraph("Executive Summary", styles["Heading1"]))

    summary_data = [
        ["Metric", "Value"],
        ["Total images analyzed", str(session_report.total_images)],
        ["Successful", str(session_report.successful_images)],
        ["Total trichomes detected", str(session_report.total_trichomes)],
        ["Avg trichomes / image", f"{session_report.mean_trichomes_per_image:.1f}"],
        ["Mean detection confidence", f"{session_report.mean_confidence:.2%}"],
        ["Mean focus score", f"{session_report.mean_focus_score:.1f}"],
        ["Low-focus images", str(session_report.low_focus_images)],
        ["Processing time", f"{session_report.processing_time_s:.1f}s"],
        ["Detection model", session_report.detection_model or "—"],
        ["Hardware", session_report.hardware],
    ]

    summary_table = Table(summary_data, colWidths=[8 * cm, 6 * cm])
    summary_table.setStyle(_standard_table_style())
    story.append(summary_table)
    story.append(Spacer(1, 0.8 * cm))

    # ------------------------------------------------------------------
    # Maturity distribution
    # ------------------------------------------------------------------
    if session_report.maturity_summary:
        story.append(Paragraph("Maturity Distribution", styles["Heading1"]))
        m = session_report.maturity_summary

        maturity_data = [
            ["Stage", "Fraction", "Interpretation"],
            ["Clear", f"{m.clear_fraction:.1%}", "Early development"],
            ["Cloudy", f"{m.cloudy_fraction:.1%}", "Peak resin density (visual)"],
            ["Amber", f"{m.amber_fraction:.1%}", "Oxidative degradation signal"],
            ["Mixed", f"{m.mixed_fraction:.1%}", "Heterogeneous population"],
        ]

        mat_table = Table(maturity_data, colWidths=[4 * cm, 4 * cm, 9 * cm])
        mat_table.setStyle(_standard_table_style())
        story.append(mat_table)
        story.append(Spacer(1, 0.4 * cm))

        story.append(
            Paragraph(
                f"<b>Dominant stage:</b> {m.dominant_stage.capitalize()}",
                styles["Normal"],
            )
        )
        story.append(Spacer(1, 0.3 * cm))
        story.append(
            Paragraph(
                f"<b>Harvest guidance:</b> {m.harvest_recommendation}",
                styles["Normal"],
            )
        )
        story.append(Spacer(1, 0.5 * cm))

        # Embed chart if available (BytesIO — no temp files, no lifecycle issues)
        if include_charts:
            try:
                import io as _io
                import matplotlib.pyplot as _plt
                from analytics.visualization.plotter import plot_maturity_distribution

                fig = plot_maturity_distribution(
                    {
                        "clear": m.clear_fraction,
                        "cloudy": m.cloudy_fraction,
                        "amber": m.amber_fraction,
                        "mixed": m.mixed_fraction,
                    },
                    scientific_caveat=False,  # Already in caveat section
                )
                buf = _io.BytesIO()
                fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
                _plt.close(fig)
                buf.seek(0)
                story.append(Image(buf, width=14 * cm, height=4 * cm))
            except Exception:
                pass

    story.append(PageBreak())

    # ------------------------------------------------------------------
    # Calibration section (optional)
    # ------------------------------------------------------------------
    if calibration:
        story.extend(
            _build_calibration_section(
                calibration,
                styles,
                include_chart=include_charts,
            )
        )
        story.append(PageBreak())

    # ------------------------------------------------------------------
    # References
    # ------------------------------------------------------------------
    story.append(Paragraph("References", styles["Heading1"]))
    refs = [
        "Elzinga S. et al. (2015). Nat. Prod. Chem. Res. 3:181. DOI:10.4172/2329-6836.1000181",
        "ElSohly MA, Slade D. (2005). Life Sciences 78(5):539-548. DOI:10.1016/j.lsc.2005.09.011",
        "Fischedick JT et al. (2010). Phytochemistry 71(17-18):2058-2073.",
        "Jocher G et al. (2023). Ultralytics YOLO. github.com/ultralytics/ultralytics",
        "Guo C et al. (2017). On Calibration of Modern Neural Networks. ICML 2017.",
    ]
    for ref in refs:
        story.append(Paragraph(f"• {ref}", styles["Normal"]))
        story.append(Spacer(1, 0.2 * cm))

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------
    doc.build(story)
    return output_path


def _build_calibration_section(
    calibration: dict,
    styles,
    include_chart: bool = True,
) -> list:
    """
    Build a list of ReportLab flowables for the calibration section.

    Args:
        calibration: CalibrationResponse-compatible dict (from analytics API).
        styles: getSampleStyleSheet() result.
        include_chart: Embed reliability diagram chart.

    Returns:
        List of flowables to extend into a PDF story.
    """
    story: list = []

    ece = calibration.get("ece", 0.0)
    mce = calibration.get("mce", 0.0)
    total_samples = calibration.get("total_samples", 0)
    num_bins = calibration.get("num_bins", 10)
    is_overconfident = calibration.get("is_overconfident", False)
    interpretation = calibration.get("interpretation", "")
    bins = calibration.get("bins", [])
    run_id = calibration.get("mlflow_run_id") or calibration.get("run_id")

    # Section header
    story.append(Paragraph("Confidence Calibration Analysis", styles["Heading1"]))

    # ECE quality classification
    ece_quality = (
        "Excellent (ECE < 0.02)" if ece < 0.02 else
        "Good (ECE < 0.05)" if ece < 0.05 else
        "Moderate (ECE < 0.10)" if ece < 0.10 else
        "Poor (ECE ≥ 0.10)"
    )
    quality_color = (
        "#2e7d32" if ece < 0.02 else
        "#1565C0" if ece < 0.05 else
        "#E65100" if ece < 0.10 else
        "#c62828"
    )

    story.append(
        Paragraph(
            f"Calibration quality: <font color='{quality_color}'><b>{ece_quality}</b></font>",
            ParagraphStyle("cal_quality", fontSize=12, spaceAfter=0.3 * cm),
        )
    )

    # Summary metrics table
    direction = "overconfident" if is_overconfident else "underconfident"
    summary_data = [
        ["Metric", "Value"],
        ["ECE (Expected Calibration Error)", f"{ece:.4f}"],
        ["MCE (Maximum Calibration Error)", f"{mce:.4f}"],
        ["Calibration quality", ece_quality],
        ["Direction", direction.capitalize()],
        ["Total predictions", f"{total_samples:,}"],
        ["Bins", str(num_bins)],
    ]
    if run_id:
        run_id_str = str(run_id)
        summary_data.append([
            "MLflow Run ID",
            (run_id_str[:16] + "…") if len(run_id_str) > 16 else run_id_str,
        ])

    cal_table = Table(summary_data, colWidths=[9 * cm, 7 * cm])
    cal_table.setStyle(_standard_table_style())
    story.append(cal_table)
    story.append(Spacer(1, 0.5 * cm))

    # Interpretation text
    if interpretation:
        story.append(
            Paragraph(
                f"<i>{interpretation}</i>",
                ParagraphStyle(
                    "cal_interp",
                    fontSize=9,
                    textColor=colors.HexColor("#37474F"),
                    leftIndent=0.5 * cm,
                    rightIndent=0.5 * cm,
                    spaceAfter=0.5 * cm,
                ),
            )
        )

    # Reliability diagram chart (embedded PNG via in-memory BytesIO — no temp files)
    if include_chart and bins:
        try:
            import io as _io
            import matplotlib.pyplot as _plt
            from analytics.visualization.plotter import (  # noqa: PLC0415
                plot_reliability_diagram_from_bins,
            )

            fig = plot_reliability_diagram_from_bins(
                bins=bins,
                ece=ece,
                mce=mce,
                total_samples=total_samples,
                title="Reliability Diagram",
            )

            # Render to BytesIO so ReportLab reads it eagerly (no temp file lifecycle issue)
            buf = _io.BytesIO()
            fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
            _plt.close(fig)
            buf.seek(0)

            story.append(Image(buf, width=16 * cm, height=7 * cm))

        except Exception:
            story.append(
                Paragraph(
                    "[Reliability diagram unavailable — ensure matplotlib is installed]",
                    styles["Normal"],
                )
            )

        story.append(Spacer(1, 0.5 * cm))

    # Per-bin detail table
    non_empty_bins = [b for b in bins if not b.get("is_empty", True)]
    if non_empty_bins:
        story.append(Paragraph("Per-bin Calibration Detail", styles["Heading2"]))

        bin_data = [
            ["Bin", "Conf Range", "Mean Conf", "Accuracy", "Count", "Gap", "Status"]
        ]
        for i, b in enumerate(non_empty_bins, 1):
            lo = b.get("confidence_lower", 0.0)
            hi = b.get("confidence_upper", 1.0)
            conf = b.get("mean_confidence", 0.0)
            acc = b.get("accuracy", 0.0)
            count = b.get("count", 0)
            gap = b.get("gap", conf - acc)
            overconf = b.get("is_overconfident", False)

            bin_data.append([
                str(i),
                f"{lo:.2f}–{hi:.2f}",
                f"{conf:.3f}",
                f"{acc:.3f}",
                str(count),
                f"{gap:+.3f}",
                "Over" if overconf else "Under",
            ])

        bin_table = Table(
            bin_data,
            colWidths=[1.5 * cm, 3 * cm, 2.5 * cm, 2.5 * cm, 2 * cm, 2 * cm, 2.5 * cm],
        )
        bin_style = _standard_table_style()
        # Colour-code overconfident/underconfident status column
        for row_idx, b in enumerate(non_empty_bins, 1):
            text_color = (
                colors.HexColor("#BF360C") if b.get("is_overconfident")
                else colors.HexColor("#0D47A1")
            )
            bin_style.add("TEXTCOLOR", (6, row_idx), (6, row_idx), text_color)
            bin_style.add("FONTNAME", (6, row_idx), (6, row_idx), "Helvetica-Bold")

        bin_table.setStyle(bin_style)
        story.append(bin_table)

    story.append(Spacer(1, 0.5 * cm))
    return story


def _standard_table_style() -> TableStyle:
    """Return a standard table style with header + alternating rows."""
    return TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(TABLE_HEADER_BG)),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 10),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
            ("TOPPADDING", (0, 0), (-1, 0), 8),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor(TABLE_ALT_BG)]),
            ("FONTSIZE", (0, 1), (-1, -1), 9),
            ("TOPPADDING", (0, 1), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 1), (-1, -1), 5),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#b0bec5")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]
    )


def reportlab_available() -> bool:
    """Check if reportlab is installed."""
    return REPORTLAB_AVAILABLE
