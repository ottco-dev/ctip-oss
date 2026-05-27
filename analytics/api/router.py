"""
analytics.api.router — FastAPI router for model analytics and calibration.

Endpoints:
  POST /analytics/calibration
    Compute ECE / MCE / reliability-diagram data from raw predictions or a MLflow run.

  POST /analytics/confidence/histogram
    Compute a confidence distribution histogram from a raw list of scores.

  GET  /analytics/calibration/run/{run_id}
    Compute calibration for a specific training run (looks up MLflow or local store).

  POST /analytics/calibration/report
    Generate a standalone PDF calibration report from a CalibrationResponse.

Scientific basis:
  Guo, C. et al. (2017). On Calibration of Modern Neural Networks.
  ICML 2017. arXiv:1706.04599

  Naeini, M.P. et al. (2015). Obtaining Well Calibrated Probabilities
  Using Bayesian Binning. AAAI 2015.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException, status

from analytics.api.schemas import (
    BinStats,
    CalibrationRequest,
    CalibrationResponse,
    ConfidenceBin,
    ConfidenceHistogramRequest,
    ConfidenceHistogramResponse,
)
from fastapi.responses import Response
from shared.metrics.calibration_metrics import CalibrationResult, compute_calibration

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analytics", tags=["analytics"])


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _interpret_ece(ece: float, is_overconfident: bool) -> str:
    """
    Produce a human-readable calibration quality interpretation.

    Thresholds follow common CV research practice:
      ECE < 0.02   → excellent
      ECE < 0.05   → good
      ECE < 0.10   → moderate
      ECE ≥ 0.10   → poor
    """
    direction = "overconfident" if is_overconfident else "underconfident"
    if ece < 0.02:
        quality = "Excellent calibration"
        advice = "Model confidence closely matches observed accuracy."
    elif ece < 0.05:
        quality = "Good calibration"
        advice = (
            f"Model is slightly {direction}. "
            "Consider temperature scaling for marginal improvement."
        )
    elif ece < 0.10:
        quality = "Moderate calibration"
        advice = (
            f"Model is {direction} (ECE={ece:.3f}). "
            "Apply Platt scaling or temperature scaling before deployment."
        )
    else:
        quality = "Poor calibration"
        advice = (
            f"Model is significantly {direction} (ECE={ece:.3f}). "
            "Strong post-hoc calibration (isotonic regression, temperature scaling) required. "
            "Raw confidence scores should NOT be reported to end users."
        )
    return f"{quality}. {advice}"


def _build_bin_stats(
    result: CalibrationResult,
    total_n: int,
) -> tuple:
    """Convert CalibrationResult per-bin arrays → BinStats list."""
    bins: List[BinStats] = []
    num_bins = result.num_bins
    bin_width = 1.0 / num_bins

    overconfident_non_empty = 0
    non_empty_count = 0

    for i in range(num_bins):
        count = int(result.bin_counts[i])
        is_empty = count == 0
        conf = float(result.bin_confidences[i])
        acc = float(result.bin_accuracies[i])
        gap = conf - acc if not is_empty else 0.0
        abs_gap = abs(gap) if not is_empty else 0.0
        weight = count / max(total_n, 1)
        over = (conf > acc) if not is_empty else False

        if not is_empty:
            non_empty_count += 1
            if over:
                overconfident_non_empty += 1

        bins.append(BinStats(
            bin_index=i,
            confidence_lower=round(i * bin_width, 6),
            confidence_upper=round((i + 1) * bin_width, 6),
            mean_confidence=round(conf, 6),
            accuracy=round(acc, 6),
            count=count,
            gap=round(gap, 6),
            abs_gap=round(abs_gap, 6),
            weight=round(weight, 6),
            is_overconfident=over,
            is_empty=is_empty,
        ))

    return bins, overconfident_non_empty, non_empty_count


def _calibration_result_to_response(
    result: CalibrationResult,
    total_n: int,
    source: str = "direct",
    mlflow_run_id: Optional[str] = None,
) -> CalibrationResponse:
    """Translate a CalibrationResult into the API response schema."""
    bins, overconfident_non_empty, non_empty_count = _build_bin_stats(result, total_n)
    overconfident_frac = (
        overconfident_non_empty / non_empty_count if non_empty_count > 0 else 0.0
    )
    interpretation = _interpret_ece(result.ece, result.is_overconfident)

    return CalibrationResponse(
        ece=round(result.ece, 6),
        mce=round(result.mce, 6),
        num_bins=result.num_bins,
        total_samples=total_n,
        is_overconfident=result.is_overconfident,
        overconfident_bin_fraction=round(overconfident_frac, 4),
        bins=bins,
        confidence_histogram=[b.count for b in bins],
        interpretation=interpretation,
        mlflow_run_id=mlflow_run_id,
        source=source,
    )


# ---------------------------------------------------------------------------
# POST /analytics/calibration
# ---------------------------------------------------------------------------

@router.post(
    "/calibration",
    response_model=CalibrationResponse,
    summary="Compute ECE / reliability diagram",
    description="""
Compute Expected Calibration Error (ECE), Maximum Calibration Error (MCE),
and per-bin reliability diagram data from raw model predictions.

**Input options:**
- Provide `confidences` + `is_correct` directly, OR
- Provide `mlflow_run_id` to load prediction artifacts from MLflow.

**Scientific basis:** Guo et al. (2017). On Calibration of Modern Neural Networks.
ICML 2017. arXiv:1706.04599
""",
    status_code=status.HTTP_200_OK,
)
async def compute_calibration_endpoint(request: CalibrationRequest) -> CalibrationResponse:
    """
    Compute ECE / reliability diagram from raw predictions or a MLflow run.
    """
    # ── Resolve inputs ───────────────────────────────────────────────────────
    if request.mlflow_run_id is not None:
        confidences, is_correct = await _load_mlflow_predictions(request.mlflow_run_id)
        source = "mlflow"
    elif request.confidences is not None and request.is_correct is not None:
        confidences = request.confidences
        is_correct = request.is_correct
        source = "direct"
    else:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Provide either (confidences + is_correct) or mlflow_run_id.",
        )

    # ── Validate ─────────────────────────────────────────────────────────────
    if len(confidences) != len(is_correct):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Length mismatch: confidences={len(confidences)}, is_correct={len(is_correct)}.",
        )
    if len(confidences) == 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Empty input: at least one prediction is required.",
        )

    out_of_range = [c for c in confidences if not (0.0 <= c <= 1.0)]
    if out_of_range:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"{len(out_of_range)} confidence values outside [0, 1]. First offender: {out_of_range[0]:.4f}.",
        )

    # ── Compute ───────────────────────────────────────────────────────────────
    try:
        result = compute_calibration(
            confidences=confidences,
            correct=is_correct,
            num_bins=request.num_bins,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )
    except Exception as exc:
        logger.exception("Unexpected error in calibration computation")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Calibration computation failed: {exc}",
        )

    return _calibration_result_to_response(
        result=result,
        total_n=len(confidences),
        source=source,
        mlflow_run_id=request.mlflow_run_id,
    )


# ---------------------------------------------------------------------------
# GET /analytics/calibration/run/{run_id}
# ---------------------------------------------------------------------------

@router.get(
    "/calibration/run/{run_id}",
    response_model=CalibrationResponse,
    summary="Calibration for a specific training run",
    description=(
        "Load confidence predictions logged by a completed training run "
        "and return ECE / reliability diagram data. The run must have logged "
        "'confidence_scores' and 'is_correct' artifacts to MLflow."
    ),
    status_code=status.HTTP_200_OK,
)
async def calibration_for_run(
    run_id: str,
    num_bins: int = 10,
) -> CalibrationResponse:
    """
    Compute calibration metrics for a specific training run by MLflow run ID.
    """
    if num_bins < 5 or num_bins > 50:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="num_bins must be in [5, 50].",
        )

    try:
        confidences, is_correct = await _load_mlflow_predictions(run_id)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load predictions for run {run_id}: {exc}",
        )

    try:
        result = compute_calibration(confidences, correct=is_correct, num_bins=num_bins)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )

    return _calibration_result_to_response(
        result=result,
        total_n=len(confidences),
        source="mlflow",
        mlflow_run_id=run_id,
    )


# ---------------------------------------------------------------------------
# POST /analytics/confidence/histogram
# ---------------------------------------------------------------------------

@router.post(
    "/confidence/histogram",
    response_model=ConfidenceHistogramResponse,
    summary="Confidence score distribution histogram",
    description=(
        "Compute a histogram of confidence scores. Useful for inspecting how "
        "predictions distribute across the confidence range before calibration."
    ),
    status_code=status.HTTP_200_OK,
)
async def confidence_histogram(
    request: ConfidenceHistogramRequest,
) -> ConfidenceHistogramResponse:
    """
    Return a confidence histogram with descriptive statistics.
    """
    confs = np.array(request.confidences, dtype=np.float64)

    if np.any((confs < 0) | (confs > 1)):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="All confidence values must be in [0, 1].",
        )

    num_bins = request.num_bins
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    counts, _ = np.histogram(confs, bins=bin_edges)
    total = len(confs)
    fractions = counts / max(total, 1)

    bins = [
        ConfidenceBin(
            bin_index=i,
            lower=round(float(bin_edges[i]), 6),
            upper=round(float(bin_edges[i + 1]), 6),
            count=int(counts[i]),
            fraction=round(float(fractions[i]), 6),
        )
        for i in range(num_bins)
    ]

    high_conf_thresh = 0.70
    fraction_high = float(np.mean(confs >= high_conf_thresh))

    return ConfidenceHistogramResponse(
        bins=bins,
        total=total,
        mean_confidence=round(float(np.mean(confs)), 6),
        median_confidence=round(float(np.median(confs)), 6),
        std_confidence=round(float(np.std(confs)), 6),
        fraction_high_confidence=round(fraction_high, 4),
    )


# ---------------------------------------------------------------------------
# Internal: MLflow prediction artifact loader
# ---------------------------------------------------------------------------

async def _load_mlflow_predictions(
    run_id: str,
) -> tuple[list[float], list[bool]]:
    """
    Load confidence scores and correctness flags from a MLflow run.

    Expects two logged metrics arrays:
      - artifact: "predictions/confidence_scores.npy"  (float array)
      - artifact: "predictions/is_correct.npy"         (bool array)

    Raises HTTPException(404) if run not found or artifacts missing.
    """
    try:
        import mlflow  # type: ignore[import]
    except ImportError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="MLflow is not installed. Install with: pip install mlflow",
        )

    try:
        client = mlflow.tracking.MlflowClient()
        # Validate run exists
        try:
            client.get_run(run_id)
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"MLflow run '{run_id}' not found.",
            )

        # Try to load numpy artifacts from the run
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            # Attempt to download artifacts
            try:
                conf_path = client.download_artifacts(
                    run_id, "predictions/confidence_scores.npy", tmpdir
                )
                correct_path = client.download_artifacts(
                    run_id, "predictions/is_correct.npy", tmpdir
                )
            except Exception:
                # Fall back to per-epoch mAP50 confidence proxy if raw not available
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=(
                        f"Run '{run_id}' has no calibration artifacts. "
                        "Log 'predictions/confidence_scores.npy' and "
                        "'predictions/is_correct.npy' during evaluation to enable this endpoint."
                    ),
                )

            confidences = np.load(conf_path).tolist()
            is_correct = np.load(correct_path).astype(bool).tolist()

        return confidences, is_correct

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception(f"Failed to load MLflow predictions for run {run_id}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error loading MLflow run: {exc}",
        )

# ---------------------------------------------------------------------------
# POST /analytics/calibration/report  — standalone PDF calibration report
# ---------------------------------------------------------------------------


@router.post(
    "/calibration/report",
    response_class=Response,
    summary="Generate a standalone PDF calibration report",
    description=(
        "Accepts a CalibrationResponse dict and generates a downloadable PDF report "
        "containing the reliability diagram, ECE / MCE table, per-bin accuracy breakdown, "
        "and scientific methodology section. "
        "Requires `reportlab` and `matplotlib` to be installed. "
        "Typical use: call `POST /analytics/calibration`, "
        "then pass the JSON response body here to get the PDF."
    ),
    responses={
        200: {
            "content": {"application/pdf": {}},
            "description": "Calibration report PDF.",
        },
        503: {"description": "reportlab or matplotlib not installed."},
    },
)
async def calibration_report(
    calibration: CalibrationResponse,
    model_id: str = "model",
) -> Response:
    """
    Generate a downloadable PDF calibration report.

    Send the full ``CalibrationResponse`` object from ``POST /analytics/calibration``
    (or ``GET /analytics/calibration/run/{run_id}``) as the request body.

    Returns a PDF file with:
    - ECE / MCE summary table with quality classification
    - Reliability diagram (PNG embedded in PDF)
    - Per-bin accuracy / confidence breakdown table
    - Calibration methodology reference (Guo et al., 2017)

    Query args:
        model_id: Model identifier shown in the report title (default: "model").
    """
    try:
        from analytics.export.pdf_exporter import export_calibration_pdf, REPORTLAB_AVAILABLE
        if not REPORTLAB_AVAILABLE:
            raise ImportError("reportlab not installed")
    except ImportError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "PDF generation requires reportlab. "
                "Install: pip install reportlab"
            ),
        )

    import tempfile
    from pathlib import Path

    cal_dict = calibration.model_dump()

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "calibration_report.pdf"
            export_calibration_pdf(
                calibration=cal_dict,
                output_path=out,
                model_id=model_id,
                run_id=calibration.mlflow_run_id,
            )
            pdf_bytes = out.read_bytes()

    except Exception as exc:
        logger.exception("PDF calibration report generation failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Report generation failed: {exc}",
        )

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="calibration_{model_id}.pdf"',
        },
    )
