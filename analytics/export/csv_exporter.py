"""
analytics.export.csv_exporter — CSV export for trichome analysis results.

Exports:
1. Detection results: per-image bounding box table
2. Maturity analysis: per-session maturity distribution
3. Morphology: per-instance type and size measurements
4. Training metrics: epoch-by-epoch mAP50, loss curves
5. Dataset statistics: class balance, quality distribution
"""

from __future__ import annotations

import csv
import io
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

logger = logging.getLogger(__name__)


def _write_csv(rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> str:
    """Write list of dicts to CSV string."""
    if not rows:
        return ""

    if fieldnames is None:
        fieldnames = list(rows[0].keys())

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Detection export
# ---------------------------------------------------------------------------

def export_detections_csv(
    detections: list[dict[str, Any]],
    output_path: str | Path | None = None,
) -> str:
    """
    Export detection results to CSV.

    Each row = one bounding box detection.

    Columns: image_path, x1, y1, x2, y2, confidence, class_id, class_name,
             width_px, height_px, area_px, cx, cy

    Args:
        detections: List of detection dicts (from DetectionPipeline).
        output_path: If provided, also write to file.

    Returns:
        CSV string.
    """
    rows = []
    fieldnames = [
        "image_path", "detection_idx",
        "x1", "y1", "x2", "y2",
        "width_px", "height_px", "area_px",
        "cx", "cy",
        "confidence", "class_id", "class_name",
    ]

    for det in detections:
        x1 = float(det.get("x1", 0))
        y1 = float(det.get("y1", 0))
        x2 = float(det.get("x2", 0))
        y2 = float(det.get("y2", 0))
        w = x2 - x1
        h = y2 - y1

        rows.append({
            "image_path": str(det.get("image_path", "")),
            "detection_idx": int(det.get("detection_idx", 0)),
            "x1": round(x1, 2),
            "y1": round(y1, 2),
            "x2": round(x2, 2),
            "y2": round(y2, 2),
            "width_px": round(w, 2),
            "height_px": round(h, 2),
            "area_px": round(w * h, 2),
            "cx": round(x1 + w / 2, 2),
            "cy": round(y1 + h / 2, 2),
            "confidence": round(float(det.get("confidence", 0)), 4),
            "class_id": int(det.get("class_id", 0)),
            "class_name": str(det.get("class_name", "")),
        })

    csv_str = _write_csv(rows, fieldnames=fieldnames)
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(csv_str)
        logger.info("Detections CSV written to %s (%d rows)", output_path, len(rows))

    return csv_str


# ---------------------------------------------------------------------------
# Maturity export
# ---------------------------------------------------------------------------

def export_maturity_csv(
    maturity_results: list[dict[str, Any]],
    output_path: str | Path | None = None,
) -> str:
    """
    Export maturity analysis results to CSV.

    Columns: image_path, maturity_stage, clear_fraction, cloudy_fraction,
             amber_fraction, confidence, backend, scientific_caveat

    Scientific note is included in every row as a reminder.
    """
    CAVEAT = (
        "Maturity stage is an observable optical property. "
        "No inference about cannabinoid content can be made from visual appearance."
    )

    rows = []
    fieldnames = [
        "image_path", "maturity_stage",
        "clear_fraction", "cloudy_fraction", "amber_fraction",
        "confidence", "backend", "scientific_caveat",
    ]

    for result in maturity_results:
        rows.append({
            "image_path": str(result.get("image_path", "")),
            "maturity_stage": str(result.get("maturity_stage", "unknown")),
            "clear_fraction": round(float(result.get("clear_fraction", 0)), 4),
            "cloudy_fraction": round(float(result.get("cloudy_fraction", 0)), 4),
            "amber_fraction": round(float(result.get("amber_fraction", 0)), 4),
            "confidence": round(float(result.get("confidence", 0)), 4),
            "backend": str(result.get("backend", "color_rules")),
            "scientific_caveat": CAVEAT,
        })

    csv_str = _write_csv(rows, fieldnames=fieldnames)
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(csv_str)
        logger.info("Maturity CSV written to %s (%d rows)", output_path, len(rows))

    return csv_str


# ---------------------------------------------------------------------------
# Morphology / Measurement export
# ---------------------------------------------------------------------------

def export_morphology_csv(
    instances: list[dict[str, Any]],
    output_path: str | Path | None = None,
) -> str:
    """
    Export per-instance morphology and measurement results.

    Columns: image_path, instance_id, class_name, area_px, area_um2,
             diameter_um, circularity, elongation, cx, cy, mask_score
    """
    rows = []
    fieldnames = [
        "image_path", "instance_id", "class_name",
        "area_px", "area_um2", "diameter_um",
        "circularity", "elongation",
        "cx", "cy",
        "mask_score", "detection_confidence",
    ]

    for inst in instances:
        row: dict[str, Any] = {
            "image_path": str(inst.get("image_path", "")),
            "instance_id": int(inst.get("instance_id", 0)),
            "class_name": str(inst.get("detection_class_name", "")),
            "area_px": round(float(inst.get("area_px", 0)), 2),
            "area_um2": round(float(inst.get("area_um2", 0) or 0), 4),
            "diameter_um": round(float(inst.get("diameter_um", 0) or 0), 4),
            "circularity": round(float(inst.get("circularity", 0)), 4),
            "elongation": round(float(inst.get("elongation", 0)), 4),
            "cx": round(float(inst.get("centroid_x", 0)), 2),
            "cy": round(float(inst.get("centroid_y", 0)), 2),
            "mask_score": round(float(inst.get("mask_score", 0)), 4),
            "detection_confidence": round(float(inst.get("detection_confidence", 0)), 4),
        }
        rows.append(row)

    csv_str = _write_csv(rows, fieldnames=fieldnames)
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(csv_str)
        logger.info("Morphology CSV written to %s (%d instances)", output_path, len(rows))

    return csv_str


# ---------------------------------------------------------------------------
# Training metrics export
# ---------------------------------------------------------------------------

def export_training_metrics_csv(
    metrics_history: list[dict[str, Any]],
    output_path: str | Path | None = None,
) -> str:
    """
    Export training metrics history to CSV.

    Columns: epoch, mAP50, mAP50_95, precision, recall,
             box_loss, cls_loss, dfl_loss, lr, elapsed_s
    """
    if not metrics_history:
        return ""

    # Collect all keys across all epochs
    all_keys: set[str] = set()
    for record in metrics_history:
        all_keys.update(record.keys())

    # Always put epoch first
    fieldnames = ["epoch"] + sorted(all_keys - {"epoch"})

    rows = []
    for record in metrics_history:
        row = {}
        for key in fieldnames:
            val = record.get(key, "")
            if isinstance(val, float):
                row[key] = round(val, 6)
            else:
                row[key] = val
        rows.append(row)

    csv_str = _write_csv(rows, fieldnames=fieldnames)
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(csv_str)
        logger.info("Training metrics CSV written to %s (%d epochs)", output_path, len(rows))

    return csv_str


# ---------------------------------------------------------------------------
# Dataset statistics export
# ---------------------------------------------------------------------------

def export_dataset_stats_csv(
    dataset_stats: dict[str, Any],
    output_path: str | Path | None = None,
) -> str:
    """
    Export dataset statistics to CSV (class distribution, quality histogram).

    Args:
        dataset_stats: Dict from GET /datasets/{id}/stats endpoint.
        output_path: Optional output file path.
    """
    rows = []

    # Class distribution
    class_dist = dataset_stats.get("class_distribution", {})
    total_annotations = sum(class_dist.values()) or 1

    for class_name, count in class_dist.items():
        rows.append({
            "section": "class_distribution",
            "key": class_name,
            "count": count,
            "fraction": round(count / total_annotations, 4),
            "percent": round(count / total_annotations * 100, 2),
        })

    # Quality histogram
    quality_hist = dataset_stats.get("quality_histogram", {})
    for bucket, count in quality_hist.items():
        rows.append({
            "section": "quality_histogram",
            "key": bucket,
            "count": count,
            "fraction": "",
            "percent": "",
        })

    # Split distribution
    split_dist = dataset_stats.get("split_distribution", {})
    total_images = sum(split_dist.values()) or 1
    for split, count in split_dist.items():
        rows.append({
            "section": "split_distribution",
            "key": split,
            "count": count,
            "fraction": round(count / total_images, 4),
            "percent": round(count / total_images * 100, 2),
        })

    fieldnames = ["section", "key", "count", "fraction", "percent"]
    csv_str = _write_csv(rows, fieldnames=fieldnames)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(csv_str)
        logger.info("Dataset stats CSV written to %s", output_path)

    return csv_str
