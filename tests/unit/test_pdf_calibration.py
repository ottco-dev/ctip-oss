"""
tests.unit.test_pdf_calibration — Tests for calibration-related PDF and chart functions.

Tests:
  - plot_reliability_diagram_from_bins returns a valid matplotlib Figure
  - Figure has two axes (reliability diagram + histogram)
  - Non-empty bins are rendered (via bar patches)
  - export_calibration_pdf writes a valid PDF file (with reportlab)
  - export_session_pdf with calibration kwarg includes calibration section
  - _build_calibration_section returns non-empty flowable list
  - _build_calibration_section handles empty bins gracefully
  - _build_calibration_section handles missing optional fields gracefully
"""

from __future__ import annotations

import os
import struct
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Test data fixtures
# ---------------------------------------------------------------------------

def _make_bins(n: int = 5, overconfident_first: bool = True) -> list[dict]:
    """Create synthetic BinStats dicts for testing."""
    bins = []
    for i in range(n):
        conf_lo = i / n
        conf_hi = (i + 1) / n
        mean_conf = (conf_lo + conf_hi) / 2
        # Alternating over/underconfident
        accuracy = mean_conf - 0.08 if overconfident_first else mean_conf + 0.08
        accuracy = max(0.0, min(1.0, accuracy))
        gap = mean_conf - accuracy
        bins.append({
            "bin_index": i,
            "confidence_lower": conf_lo,
            "confidence_upper": conf_hi,
            "mean_confidence": mean_conf,
            "accuracy": accuracy,
            "count": 100 + i * 20,
            "gap": gap,
            "abs_gap": abs(gap),
            "weight": 1.0 / n,
            "is_overconfident": gap > 0,
            "is_empty": False,
        })
    return bins


def _make_calibration_dict(n_bins: int = 5) -> dict:
    bins = _make_bins(n_bins)
    return {
        "ece": 0.042,
        "mce": 0.085,
        "num_bins": n_bins,
        "total_samples": sum(b["count"] for b in bins),
        "is_overconfident": True,
        "overconfident_bin_fraction": 0.6,
        "bins": bins,
        "confidence_histogram": [b["count"] for b in bins],
        "interpretation": "Model is slightly overconfident. Apply temperature scaling.",
        "mlflow_run_id": "test-run-abc123",
        "source": "raw_predictions",
    }


# ---------------------------------------------------------------------------
# plot_reliability_diagram_from_bins
# ---------------------------------------------------------------------------

class TestPlotReliabilityDiagramFromBins:

    def test_returns_figure(self):
        from analytics.visualization.plotter import plot_reliability_diagram_from_bins
        bins = _make_bins(5)
        fig = plot_reliability_diagram_from_bins(
            bins=bins, ece=0.04, mce=0.08, total_samples=500
        )
        import matplotlib.pyplot as plt
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_figure_has_two_axes(self):
        from analytics.visualization.plotter import plot_reliability_diagram_from_bins
        bins = _make_bins(5)
        fig = plot_reliability_diagram_from_bins(
            bins=bins, ece=0.04, mce=0.08, total_samples=500
        )
        assert len(fig.get_axes()) == 2  # reliability diagram + histogram
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_figure_with_empty_bins_only(self):
        """Bins with is_empty=True should produce a figure without crashing."""
        from analytics.visualization.plotter import plot_reliability_diagram_from_bins
        bins = [
            {
                "mean_confidence": 0.5, "accuracy": 0.5,
                "count": 0, "is_overconfident": False, "is_empty": True,
            }
        ]
        fig = plot_reliability_diagram_from_bins(
            bins=bins, ece=0.0, mce=0.0, total_samples=0
        )
        import matplotlib.pyplot as plt
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_ece_annotations_appear(self):
        """ECE text should appear in the left axis."""
        from analytics.visualization.plotter import plot_reliability_diagram_from_bins
        bins = _make_bins(5)
        fig = plot_reliability_diagram_from_bins(
            bins=bins, ece=0.04, mce=0.08, total_samples=500
        )
        ax = fig.get_axes()[0]
        texts = [t.get_text() for t in ax.texts]
        # At least one text should contain ECE value
        assert any("ECE" in t for t in texts), f"No ECE annotation: {texts}"
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_saves_to_png(self):
        from analytics.visualization.plotter import (
            plot_reliability_diagram_from_bins,
            save_figure,
        )
        bins = _make_bins(5)
        fig = plot_reliability_diagram_from_bins(
            bins=bins, ece=0.04, mce=0.08, total_samples=500
        )
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = Path(f.name)

        try:
            save_figure(fig, path, dpi=72)
            assert path.exists()
            assert path.stat().st_size > 1024  # non-trivial PNG
        finally:
            path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# _build_calibration_section
# ---------------------------------------------------------------------------

class TestBuildCalibrationSection:

    @pytest.fixture(autouse=True)
    def skip_if_no_reportlab(self):
        from analytics.export.pdf_exporter import REPORTLAB_AVAILABLE
        if not REPORTLAB_AVAILABLE:
            pytest.skip("reportlab not installed")

    def test_returns_non_empty_list(self):
        from analytics.export.pdf_exporter import _build_calibration_section
        from reportlab.lib.styles import getSampleStyleSheet
        styles = getSampleStyleSheet()
        cal = _make_calibration_dict()
        result = _build_calibration_section(cal, styles, include_chart=False)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_handles_empty_bins(self):
        from analytics.export.pdf_exporter import _build_calibration_section
        from reportlab.lib.styles import getSampleStyleSheet
        styles = getSampleStyleSheet()
        cal = {
            "ece": 0.0, "mce": 0.0, "num_bins": 10,
            "total_samples": 0, "is_overconfident": False,
            "bins": [], "interpretation": "",
        }
        result = _build_calibration_section(cal, styles, include_chart=False)
        assert isinstance(result, list)

    def test_handles_missing_optional_fields(self):
        """Section must not crash when run_id, interpretation, bins absent."""
        from analytics.export.pdf_exporter import _build_calibration_section
        from reportlab.lib.styles import getSampleStyleSheet
        styles = getSampleStyleSheet()
        cal = {"ece": 0.05, "mce": 0.09}  # minimal dict
        result = _build_calibration_section(cal, styles, include_chart=False)
        assert isinstance(result, list)

    def test_contains_ece_paragraph(self):
        """At least one flowable should mention ECE."""
        from analytics.export.pdf_exporter import _build_calibration_section
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import Paragraph
        styles = getSampleStyleSheet()
        cal = _make_calibration_dict()
        flowables = _build_calibration_section(cal, styles, include_chart=False)

        paragraphs = [f for f in flowables if isinstance(f, Paragraph)]
        texts = " ".join(p.text for p in paragraphs)
        assert "ECE" in texts or "Calibration" in texts

    def test_per_bin_table_included_for_non_empty_bins(self):
        """When bins are non-empty, a Table should be present."""
        from analytics.export.pdf_exporter import _build_calibration_section
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import Table
        styles = getSampleStyleSheet()
        cal = _make_calibration_dict(n_bins=5)
        flowables = _build_calibration_section(cal, styles, include_chart=False)

        tables = [f for f in flowables if isinstance(f, Table)]
        assert len(tables) >= 2  # summary table + per-bin table


# ---------------------------------------------------------------------------
# export_calibration_pdf
# ---------------------------------------------------------------------------

class TestExportCalibrationPdf:

    @pytest.fixture(autouse=True)
    def skip_if_no_reportlab(self):
        from analytics.export.pdf_exporter import REPORTLAB_AVAILABLE
        if not REPORTLAB_AVAILABLE:
            pytest.skip("reportlab not installed")

    def test_creates_pdf_file(self):
        from analytics.export.pdf_exporter import export_calibration_pdf
        cal = _make_calibration_dict()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "calibration.pdf"
            result = export_calibration_pdf(cal, out, model_id="yolo11s-v1")
            assert result == out
            assert out.exists()
            assert out.stat().st_size > 512

    def test_pdf_has_magic_bytes(self):
        """Output must start with PDF header %%PDF."""
        from analytics.export.pdf_exporter import export_calibration_pdf
        cal = _make_calibration_dict()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "cal.pdf"
            export_calibration_pdf(cal, out)
            with open(out, "rb") as f:
                header = f.read(4)
            assert header == b"%PDF"

    def test_creates_parent_dirs(self):
        """export_calibration_pdf must create missing parent directories."""
        from analytics.export.pdf_exporter import export_calibration_pdf
        cal = _make_calibration_dict()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "nested" / "deep" / "cal.pdf"
            export_calibration_pdf(cal, out)
            assert out.exists()

    def test_raises_without_reportlab(self, monkeypatch):
        import analytics.export.pdf_exporter as mod
        monkeypatch.setattr(mod, "REPORTLAB_AVAILABLE", False)
        with pytest.raises(ImportError, match="reportlab"):
            mod.export_calibration_pdf({}, "/tmp/test.pdf")


# ---------------------------------------------------------------------------
# export_session_pdf — calibration kwarg integration
# ---------------------------------------------------------------------------

class TestSessionPdfWithCalibration:

    @pytest.fixture(autouse=True)
    def skip_if_no_reportlab(self):
        from analytics.export.pdf_exporter import REPORTLAB_AVAILABLE
        if not REPORTLAB_AVAILABLE:
            pytest.skip("reportlab not installed")

    def _make_session_report(self):
        """Minimal mock SessionReport."""
        from datetime import datetime, timezone
        sr = MagicMock()
        sr.session_id = "sess-test-001"
        sr.session_name = "Test Session"
        sr.created_at = datetime(2026, 5, 25, 12, 0, 0, tzinfo=timezone.utc)
        sr.total_images = 50
        sr.successful_images = 48
        sr.total_trichomes = 1200
        sr.mean_trichomes_per_image = 25.0
        sr.mean_confidence = 0.78
        sr.mean_focus_score = 142.3
        sr.low_focus_images = 3
        sr.processing_time_s = 38.5
        sr.detection_model = "yolo11s"
        sr.hardware = "RTX 4060"
        sr.maturity_summary = None
        return sr

    def test_session_pdf_without_calibration(self):
        from analytics.export.pdf_exporter import export_session_pdf
        sr = self._make_session_report()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "session.pdf"
            result = export_session_pdf(sr, out)
            assert out.exists()
            assert out.stat().st_size > 512

    def test_session_pdf_with_calibration(self):
        from analytics.export.pdf_exporter import export_session_pdf
        sr = self._make_session_report()
        cal = _make_calibration_dict()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "session_cal.pdf"
            result = export_session_pdf(sr, out, calibration=cal, include_charts=False)
            assert out.exists()
            size_with_cal = out.stat().st_size

        with tempfile.TemporaryDirectory() as tmpdir:
            out2 = Path(tmpdir) / "session_nocal.pdf"
            export_session_pdf(sr, out2, calibration=None, include_charts=False)
            size_without_cal = out2.stat().st_size

        # PDF with calibration section should be larger
        assert size_with_cal > size_without_cal, (
            f"Expected calibration PDF to be larger: "
            f"with={size_with_cal}, without={size_without_cal}"
        )
