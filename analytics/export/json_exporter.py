"""
analytics.export.json_exporter — Structured JSON export for trichome analysis.

Produces:
1. Session exports: full analysis session (detections + maturity + morphology)
2. Dataset exports: dataset metadata + annotation summary
3. Benchmark exports: mAP, precision, recall, calibration in JSON
4. COCO-format exports: for interoperability with standard tools

All exports include version markers and scientific caveats where applicable.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema version
# ---------------------------------------------------------------------------

EXPORT_SCHEMA_VERSION = "1.0"

SCIENTIFIC_CAVEATS = {
    "maturity": (
        "Maturity stage (clear/cloudy/amber) reflects observable optical properties. "
        "No cannabinoid content (THC, CBD, CBN) can be inferred from visual appearance. "
        "Reference: Elzinga et al. (2015). Natural Products Chemistry & Research 3:181."
    ),
    "thc_proxy": "THC concentration CANNOT be determined from trichome visual appearance.",
    "density": (
        "Trichome density measurements require calibrated microscopy (px→µm). "
        "Results without calibration are in pixel units only."
    ),
}


# ---------------------------------------------------------------------------
# Helper: JSON-safe serialization
# ---------------------------------------------------------------------------

def _json_safe(obj: Any) -> Any:
    """Convert numpy/torch types to JSON-serializable Python types."""
    try:
        import numpy as np

        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
    except ImportError:
        pass

    try:
        import torch

        if isinstance(obj, torch.Tensor):
            return obj.tolist()
    except ImportError:
        pass

    if hasattr(obj, "__dict__"):
        return {k: _json_safe(v) for k, v in obj.__dict__.items() if not k.startswith("_")}

    return obj


def _to_json(data: Any, indent: int = 2) -> str:
    """Serialize to JSON string with numpy/torch safe conversion."""

    class SafeEncoder(json.JSONEncoder):
        def default(self, o: Any) -> Any:
            safe = _json_safe(o)
            if safe is not o:
                return safe
            return super().default(o)

    return json.dumps(data, cls=SafeEncoder, indent=indent, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Session export
# ---------------------------------------------------------------------------

def export_session_json(
    session_data: dict[str, Any],
    output_path: str | Path | None = None,
) -> str:
    """
    Export a complete analysis session to JSON.

    Args:
        session_data: Session data dict with keys:
            - session_id, input_path, timestamp
            - detections: list of detection dicts
            - maturity: maturity analysis results
            - morphology: morphology results (optional)
            - measurements: calibrated measurements (optional)
        output_path: Optional file to write.

    Returns:
        JSON string.
    """
    export = {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "export_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "generator": "Trichome Analysis System",
        "scientific_caveats": SCIENTIFIC_CAVEATS,
        "session": session_data,
    }

    json_str = _to_json(export)

    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json_str, encoding="utf-8")
        logger.info("Session JSON exported to %s", path)

    return json_str


# ---------------------------------------------------------------------------
# Detection export
# ---------------------------------------------------------------------------

def export_detections_json(
    image_path: str,
    detections: list[dict[str, Any]],
    image_width: int,
    image_height: int,
    model_variant: str = "unknown",
    conf_threshold: float = 0.35,
    inference_time_ms: float | None = None,
    output_path: str | Path | None = None,
) -> str:
    """
    Export detection results in structured JSON.

    Args:
        image_path: Source image path.
        detections: List of detection dicts (x1, y1, x2, y2, confidence, class_id, class_name).
        image_width, image_height: Image dimensions.
        model_variant: Model used (e.g. 'yolo11s').
        conf_threshold: Confidence threshold used.
        inference_time_ms: Inference time.
        output_path: Optional file output.

    Returns:
        JSON string.
    """
    export = {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "type": "detection_result",
        "image": {
            "path": image_path,
            "width": image_width,
            "height": image_height,
        },
        "model": {
            "variant": model_variant,
            "conf_threshold": conf_threshold,
        },
        "inference_time_ms": inference_time_ms,
        "num_detections": len(detections),
        "detections": [
            {
                "id": i,
                "x1": round(float(d.get("x1", 0)), 2),
                "y1": round(float(d.get("y1", 0)), 2),
                "x2": round(float(d.get("x2", 0)), 2),
                "y2": round(float(d.get("y2", 0)), 2),
                "confidence": round(float(d.get("confidence", 0)), 4),
                "class_id": int(d.get("class_id", 0)),
                "class_name": str(d.get("class_name", "")),
            }
            for i, d in enumerate(detections)
        ],
    }

    json_str = _to_json(export)
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(json_str, encoding="utf-8")

    return json_str


# ---------------------------------------------------------------------------
# COCO format export
# ---------------------------------------------------------------------------

def export_coco_json(
    samples: list[dict[str, Any]],
    categories: list[dict[str, Any]] | None = None,
    output_path: str | Path | None = None,
) -> str:
    """
    Export annotations in COCO detection format.

    Compatible with:
    - CVAT import
    - Label Studio import
    - pycocotools evaluation
    - Most standard CV frameworks

    Args:
        samples: List of sample dicts with keys:
                 - id, file_name, width, height
                 - annotations: list of dicts with bbox [x,y,w,h], category_id
        categories: List of {id, name, supercategory} dicts.
                    Defaults to standard 4 trichome classes.
        output_path: Optional file output.

    Returns:
        JSON string in COCO format.
    """
    if categories is None:
        categories = [
            {"id": 0, "name": "capitate_stalked", "supercategory": "trichome"},
            {"id": 1, "name": "capitate_sessile", "supercategory": "trichome"},
            {"id": 2, "name": "bulbous", "supercategory": "trichome"},
            {"id": 3, "name": "non_glandular", "supercategory": "trichome"},
        ]

    images = []
    annotations = []
    ann_id = 0

    for sample in samples:
        img_id = sample.get("id", 0)
        images.append({
            "id": img_id,
            "file_name": sample.get("file_name", ""),
            "width": sample.get("width", 0),
            "height": sample.get("height", 0),
        })

        for ann in sample.get("annotations", []):
            bbox = ann.get("bbox", [0, 0, 0, 0])  # [x, y, w, h] COCO format
            if len(bbox) == 4 and all(len(str(v)) > 0 for v in bbox):
                x, y, w, h = [float(v) for v in bbox]
            else:
                continue

            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": int(ann.get("category_id", 0)),
                "bbox": [round(x, 2), round(y, 2), round(w, 2), round(h, 2)],
                "area": round(w * h, 2),
                "segmentation": ann.get("segmentation", []),
                "iscrowd": 0,
            })
            ann_id += 1

    coco = {
        "info": {
            "description": "Trichome Analysis Dataset",
            "version": EXPORT_SCHEMA_VERSION,
            "contributor": "Trichome Analysis System",
            "date_created": time.strftime("%Y-%m-%d", time.gmtime()),
        },
        "licenses": [],
        "categories": categories,
        "images": images,
        "annotations": annotations,
    }

    json_str = _to_json(coco)
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(json_str, encoding="utf-8")
        logger.info("COCO JSON exported: %d images, %d annotations", len(images), len(annotations))

    return json_str


# ---------------------------------------------------------------------------
# Benchmark export
# ---------------------------------------------------------------------------

def export_benchmark_json(
    model_name: str,
    metrics: dict[str, Any],
    per_class_metrics: dict[str, dict[str, float]] | None = None,
    calibration: dict[str, Any] | None = None,
    dataset_info: dict[str, Any] | None = None,
    output_path: str | Path | None = None,
) -> str:
    """
    Export benchmark evaluation results to structured JSON.

    Args:
        model_name: Model identifier (e.g. 'yolo11s_trichome_v1').
        metrics: Overall metrics dict: {mAP50, mAP50_95, precision, recall, f1}.
        per_class_metrics: Per-class breakdown.
        calibration: Calibration metrics (ECE, MCE) from compute_calibration().
        dataset_info: Dataset metadata.
        output_path: Optional file output.
    """
    export = {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "type": "benchmark_result",
        "model": model_name,
        "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": dataset_info or {},
        "metrics": {k: round(float(v), 6) if isinstance(v, float) else v for k, v in metrics.items()},
        "per_class_metrics": per_class_metrics or {},
        "calibration": calibration or {},
        "methodology": {
            "framework": "Ultralytics + custom evaluation",
            "iou_threshold": 0.50,
            "conf_threshold": 0.35,
        },
    }

    json_str = _to_json(export)
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(json_str, encoding="utf-8")
        logger.info("Benchmark JSON exported to %s", output_path)

    return json_str
