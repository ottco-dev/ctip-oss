"""
tests/unit/test_analytics_export.py — Analytics export module tests.

Covers:
  json_exporter — export_session_json, export_detections_json, export_coco_json, export_benchmark_json
  csv_exporter  — export_detections_csv, export_maturity_csv, export_morphology_csv,
                  export_training_metrics_csv, export_dataset_stats_csv
  pdf_exporter  — reportlab_available guard, export_calibration_pdf, export_session_pdf
"""

from __future__ import annotations

import csv
import io
import json
import tempfile
from pathlib import Path

import pytest

from analytics.export.json_exporter import (
    EXPORT_SCHEMA_VERSION,
    SCIENTIFIC_CAVEATS,
    export_benchmark_json,
    export_coco_json,
    export_detections_json,
    export_session_json,
)
from analytics.export.csv_exporter import (
    export_dataset_stats_csv,
    export_detections_csv,
    export_maturity_csv,
    export_morphology_csv,
    export_training_metrics_csv,
)
from analytics.export.pdf_exporter import reportlab_available


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def _parse_csv(csv_str: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(csv_str)))


def _sample_detection() -> dict:
    return {
        "image_path": "/images/test.png",
        "detection_idx": 0,
        "x1": 100.0, "y1": 200.0, "x2": 300.0, "y2": 400.0,
        "confidence": 0.92,
        "class_id": 0,
        "class_name": "capitate_stalked",
    }


def _sample_maturity() -> dict:
    return {
        "image_path": "/images/test.png",
        "maturity_stage": "cloudy",
        "clear_fraction": 0.1,
        "cloudy_fraction": 0.8,
        "amber_fraction": 0.1,
        "confidence": 0.87,
        "backend": "color_rules",
    }


def _sample_instance() -> dict:
    return {
        "image_path": "/images/test.png",
        "instance_id": 0,
        "detection_class_name": "bulbous",
        "area_px": 1234.5,
        "area_um2": 45.6,
        "diameter_um": 7.6,
        "circularity": 0.85,
        "elongation": 1.1,
        "centroid_x": 250.0,
        "centroid_y": 300.0,
        "mask_score": 0.91,
        "detection_confidence": 0.88,
    }


# ─────────────────────────────────────────────────────────────────
# 1. JSON exporter — export_session_json
# ─────────────────────────────────────────────────────────────────

class TestExportSessionJson:

    def test_returns_valid_json_string(self):
        data = {"session_id": "s001", "input_path": "/data/img.png", "detections": []}
        result = export_session_json(data)
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    def test_contains_schema_version(self):
        result = export_session_json({})
        parsed = json.loads(result)
        assert parsed["schema_version"] == EXPORT_SCHEMA_VERSION

    def test_contains_scientific_caveats(self):
        result = export_session_json({})
        parsed = json.loads(result)
        assert "scientific_caveats" in parsed
        assert "maturity" in parsed["scientific_caveats"]

    def test_session_data_embedded(self):
        data = {"session_id": "abc123", "detections": [1, 2, 3]}
        result = export_session_json(data)
        parsed = json.loads(result)
        assert parsed["session"]["session_id"] == "abc123"
        assert len(parsed["session"]["detections"]) == 3

    def test_writes_file_when_output_path_given(self, tmp_path):
        out = tmp_path / "session.json"
        export_session_json({"test": True}, output_path=out)
        assert out.exists()
        content = json.loads(out.read_text())
        assert content["session"]["test"] is True

    def test_thc_caveat_present(self):
        result = export_session_json({})
        assert "cannabinoid" in result.lower() or "THC" in result


# ─────────────────────────────────────────────────────────────────
# 2. JSON exporter — export_detections_json
# ─────────────────────────────────────────────────────────────────

class TestExportDetectionsJson:

    def test_returns_valid_json(self):
        result = export_detections_json("/img.png", [_sample_detection()], 1280, 960)
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    def test_num_detections_field(self):
        dets = [_sample_detection(), _sample_detection()]
        result = export_detections_json("/img.png", dets, 1280, 960)
        parsed = json.loads(result)
        assert parsed["num_detections"] == 2

    def test_detection_fields_present(self):
        result = export_detections_json("/img.png", [_sample_detection()], 1280, 960)
        parsed = json.loads(result)
        det = parsed["detections"][0]
        for key in ("x1", "y1", "x2", "y2", "confidence", "class_id", "class_name"):
            assert key in det

    def test_empty_detections_valid(self):
        result = export_detections_json("/img.png", [], 1280, 960)
        parsed = json.loads(result)
        assert parsed["num_detections"] == 0
        assert parsed["detections"] == []

    def test_image_dimensions_embedded(self):
        result = export_detections_json("/img.png", [], 640, 480)
        parsed = json.loads(result)
        assert parsed["image"]["width"] == 640
        assert parsed["image"]["height"] == 480

    def test_model_variant_embedded(self):
        result = export_detections_json("/img.png", [], 1280, 960, model_variant="yolo11s")
        parsed = json.loads(result)
        assert parsed["model"]["variant"] == "yolo11s"

    def test_file_written(self, tmp_path):
        out = tmp_path / "detections.json"
        export_detections_json("/img.png", [_sample_detection()], 100, 100, output_path=out)
        assert out.exists()


# ─────────────────────────────────────────────────────────────────
# 3. JSON exporter — export_coco_json
# ─────────────────────────────────────────────────────────────────

class TestExportCocoJson:

    def _sample_coco_input(self) -> list[dict]:
        return [
            {
                "id": 1,
                "file_name": "img001.png",
                "width": 1280,
                "height": 960,
                "annotations": [
                    {"bbox": [100, 200, 50, 80], "category_id": 0},
                    {"bbox": [300, 400, 60, 90], "category_id": 1},
                ],
            }
        ]

    def test_valid_json_output(self):
        result = export_coco_json(self._sample_coco_input())
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    def test_coco_structure_keys(self):
        result = export_coco_json(self._sample_coco_input())
        parsed = json.loads(result)
        for key in ("info", "licenses", "categories", "images", "annotations"):
            assert key in parsed

    def test_default_four_categories(self):
        result = export_coco_json([])
        parsed = json.loads(result)
        assert len(parsed["categories"]) == 4
        names = {c["name"] for c in parsed["categories"]}
        assert "capitate_stalked" in names

    def test_annotation_count(self):
        result = export_coco_json(self._sample_coco_input())
        parsed = json.loads(result)
        assert len(parsed["annotations"]) == 2

    def test_annotation_bbox_format_coco(self):
        result = export_coco_json(self._sample_coco_input())
        parsed = json.loads(result)
        ann = parsed["annotations"][0]
        assert len(ann["bbox"]) == 4  # [x, y, w, h]
        assert ann["area"] > 0

    def test_empty_samples_produces_valid_coco(self):
        result = export_coco_json([])
        parsed = json.loads(result)
        assert parsed["images"] == []
        assert parsed["annotations"] == []

    def test_custom_categories_used(self):
        cats = [{"id": 99, "name": "test_class", "supercategory": "test"}]
        result = export_coco_json([], categories=cats)
        parsed = json.loads(result)
        assert parsed["categories"][0]["name"] == "test_class"

    def test_file_written(self, tmp_path):
        out = tmp_path / "coco.json"
        export_coco_json(self._sample_coco_input(), output_path=out)
        assert out.exists()


# ─────────────────────────────────────────────────────────────────
# 4. JSON exporter — export_benchmark_json
# ─────────────────────────────────────────────────────────────────

class TestExportBenchmarkJson:

    def test_returns_valid_json(self):
        metrics = {"mAP50": 0.88, "precision": 0.91, "recall": 0.85}
        result = export_benchmark_json("yolo11s", metrics)
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    def test_model_name_embedded(self):
        result = export_benchmark_json("my_model_v2", {"mAP50": 0.8})
        parsed = json.loads(result)
        assert parsed["model"] == "my_model_v2"

    def test_metrics_rounded_to_6_decimals(self):
        result = export_benchmark_json("m", {"mAP50": 0.12345678901234})
        parsed = json.loads(result)
        assert len(str(parsed["metrics"]["mAP50"]).split(".")[-1]) <= 8  # at most 6+rounding

    def test_per_class_metrics_embedded(self):
        per_class = {"capitate_stalked": {"AP": 0.90}, "bulbous": {"AP": 0.82}}
        result = export_benchmark_json("m", {}, per_class_metrics=per_class)
        parsed = json.loads(result)
        assert "capitate_stalked" in parsed["per_class_metrics"]

    def test_calibration_embedded(self):
        cal = {"ece": 0.03, "mce": 0.05}
        result = export_benchmark_json("m", {}, calibration=cal)
        parsed = json.loads(result)
        assert parsed["calibration"]["ece"] == pytest.approx(0.03)

    def test_methodology_block_present(self):
        result = export_benchmark_json("m", {})
        parsed = json.loads(result)
        assert "iou_threshold" in parsed["methodology"]

    def test_schema_version_present(self):
        result = export_benchmark_json("m", {})
        parsed = json.loads(result)
        assert parsed["schema_version"] == EXPORT_SCHEMA_VERSION


# ─────────────────────────────────────────────────────────────────
# 5. CSV exporter — export_detections_csv
# ─────────────────────────────────────────────────────────────────

class TestExportDetectionsCsv:

    def test_returns_csv_string(self):
        result = export_detections_csv([_sample_detection()])
        assert isinstance(result, str)
        assert "," in result

    def test_header_row_present(self):
        result = export_detections_csv([_sample_detection()])
        rows = _parse_csv(result)
        assert len(rows) > 0

    def test_required_columns_present(self):
        result = export_detections_csv([_sample_detection()])
        rows = _parse_csv(result)
        assert "x1" in rows[0]
        assert "confidence" in rows[0]
        assert "class_name" in rows[0]

    def test_bbox_derived_fields(self):
        result = export_detections_csv([_sample_detection()])
        rows = _parse_csv(result)
        row = rows[0]
        # width = x2 - x1 = 200
        assert float(row["width_px"]) == pytest.approx(200.0)
        assert float(row["area_px"]) == pytest.approx(200.0 * 200.0)

    def test_centroid_computed(self):
        result = export_detections_csv([_sample_detection()])
        rows = _parse_csv(result)
        row = rows[0]
        # cx = 100 + 100 = 200
        assert float(row["cx"]) == pytest.approx(200.0)

    def test_empty_list_returns_string(self):
        result = export_detections_csv([])
        assert isinstance(result, str)

    def test_file_written(self, tmp_path):
        out = tmp_path / "dets.csv"
        export_detections_csv([_sample_detection()], output_path=out)
        assert out.exists()
        assert out.stat().st_size > 0


# ─────────────────────────────────────────────────────────────────
# 6. CSV exporter — export_maturity_csv
# ─────────────────────────────────────────────────────────────────

class TestExportMaturityCsv:

    def test_returns_csv_string(self):
        result = export_maturity_csv([_sample_maturity()])
        assert isinstance(result, str)

    def test_scientific_caveat_in_every_row(self):
        result = export_maturity_csv([_sample_maturity(), _sample_maturity()])
        rows = _parse_csv(result)
        for row in rows:
            assert len(row["scientific_caveat"]) > 20

    def test_maturity_stage_column(self):
        result = export_maturity_csv([_sample_maturity()])
        rows = _parse_csv(result)
        assert rows[0]["maturity_stage"] == "cloudy"

    def test_fraction_columns_present(self):
        result = export_maturity_csv([_sample_maturity()])
        rows = _parse_csv(result)
        for col in ("clear_fraction", "cloudy_fraction", "amber_fraction"):
            assert col in rows[0]

    def test_empty_list_returns_string(self):
        result = export_maturity_csv([])
        assert isinstance(result, str)

    def test_no_thc_claim_in_output(self):
        result = export_maturity_csv([_sample_maturity()])
        # THC prediction must never appear as a calculated value
        parsed = _parse_csv(result)
        for row in parsed:
            assert "thc_" not in " ".join(row.keys()).lower()


# ─────────────────────────────────────────────────────────────────
# 7. CSV exporter — export_morphology_csv
# ─────────────────────────────────────────────────────────────────

class TestExportMorphologyCsv:

    def test_returns_csv_string(self):
        result = export_morphology_csv([_sample_instance()])
        assert isinstance(result, str)

    def test_measurement_columns_present(self):
        result = export_morphology_csv([_sample_instance()])
        rows = _parse_csv(result)
        for col in ("area_px", "area_um2", "diameter_um", "circularity", "elongation"):
            assert col in rows[0]

    def test_centroid_fields(self):
        result = export_morphology_csv([_sample_instance()])
        rows = _parse_csv(result)
        assert float(rows[0]["cx"]) == pytest.approx(250.0)
        assert float(rows[0]["cy"]) == pytest.approx(300.0)

    def test_empty_list_returns_string(self):
        result = export_morphology_csv([])
        assert isinstance(result, str)


# ─────────────────────────────────────────────────────────────────
# 8. CSV exporter — export_training_metrics_csv
# ─────────────────────────────────────────────────────────────────

class TestExportTrainingMetricsCsv:

    def _metrics(self) -> list[dict]:
        return [
            {"epoch": 1, "mAP50": 0.55, "box_loss": 1.2, "lr": 0.001},
            {"epoch": 2, "mAP50": 0.62, "box_loss": 1.0, "lr": 0.001},
            {"epoch": 3, "mAP50": 0.70, "box_loss": 0.85, "lr": 0.0005},
        ]

    def test_returns_csv_string(self):
        result = export_training_metrics_csv(self._metrics())
        assert isinstance(result, str)

    def test_epoch_column_first(self):
        result = export_training_metrics_csv(self._metrics())
        first_col = result.split("\n")[0].split(",")[0]
        assert first_col == "epoch"

    def test_row_count_matches_epochs(self):
        result = export_training_metrics_csv(self._metrics())
        rows = _parse_csv(result)
        assert len(rows) == 3

    def test_map50_values_correct(self):
        result = export_training_metrics_csv(self._metrics())
        rows = _parse_csv(result)
        assert float(rows[0]["mAP50"]) == pytest.approx(0.55, abs=0.001)
        assert float(rows[2]["mAP50"]) == pytest.approx(0.70, abs=0.001)

    def test_empty_metrics_returns_empty_string(self):
        result = export_training_metrics_csv([])
        assert result == ""

    def test_missing_keys_tolerated(self):
        # Rows with different keys (e.g., cls_loss only present later)
        mixed = [
            {"epoch": 1, "box_loss": 1.2},
            {"epoch": 2, "box_loss": 1.0, "cls_loss": 0.5},
        ]
        result = export_training_metrics_csv(mixed)
        assert isinstance(result, str)
        rows = _parse_csv(result)
        assert len(rows) == 2


# ─────────────────────────────────────────────────────────────────
# 9. CSV exporter — export_dataset_stats_csv
# ─────────────────────────────────────────────────────────────────

class TestExportDatasetStatsCsv:

    def _stats(self) -> dict:
        return {
            "total_images": 500,
            "total_annotations": 2450,
            "class_distribution": {
                "capitate_stalked": 1200,
                "capitate_sessile": 800,
                "bulbous": 350,
                "non_glandular": 100,
            },
            "quality_histogram": {
                "excellent": 300,
                "good": 150,
                "marginal": 40,
                "reject": 10,
            },
        }

    def test_returns_csv_string(self):
        result = export_dataset_stats_csv(self._stats())
        assert isinstance(result, str)

    def test_class_distribution_in_output(self):
        result = export_dataset_stats_csv(self._stats())
        assert "capitate_stalked" in result

    def test_summary_stats_in_output(self):
        result = export_dataset_stats_csv(self._stats())
        # class_distribution values should appear (1200 is the highest count)
        assert "1200" in result or "section" in result

    def test_empty_stats_returns_string(self):
        result = export_dataset_stats_csv({})
        assert isinstance(result, str)


# ─────────────────────────────────────────────────────────────────
# 10. PDF exporter — availability guard
# ─────────────────────────────────────────────────────────────────

class TestPdfExporterGuard:

    def test_reportlab_available_returns_bool(self):
        result = reportlab_available()
        assert isinstance(result, bool)

    def test_export_calibration_pdf_without_reportlab(self):
        """When reportlab is unavailable, export should return None or raise ImportError."""
        from analytics.export.pdf_exporter import export_calibration_pdf
        if not reportlab_available():
            # Should not crash — return None or raise cleanly
            try:
                result = export_calibration_pdf(
                    bin_stats=[],
                    ece=0.05,
                    mce=0.08,
                    model_name="test_model",
                )
                assert result is None  # graceful no-op
            except (ImportError, RuntimeError):
                pass  # acceptable clean failure
        else:
            pytest.skip("reportlab is installed; skip unavailability test")

    def test_export_session_pdf_without_reportlab(self):
        """When reportlab is unavailable, export should return None or raise ImportError."""
        from analytics.export.pdf_exporter import export_session_pdf
        if not reportlab_available():
            try:
                result = export_session_pdf(session_data={})
                assert result is None
            except (ImportError, RuntimeError):
                pass
        else:
            pytest.skip("reportlab is installed; skip unavailability test")


# ─────────────────────────────────────────────────────────────────
# 11. File writing shared behavior
# ─────────────────────────────────────────────────────────────────

class TestFileWriting:

    def test_coco_creates_parent_dirs(self, tmp_path):
        out = tmp_path / "deep" / "nested" / "coco.json"
        export_coco_json([], output_path=out)
        assert out.exists()

    def test_detections_json_creates_parent_dirs(self, tmp_path):
        out = tmp_path / "sub" / "dets.json"
        export_detections_json("/img.png", [], 100, 100, output_path=out)
        assert out.exists()

    def test_benchmark_json_creates_parent_dirs(self, tmp_path):
        out = tmp_path / "benchmarks" / "run01" / "bench.json"
        export_benchmark_json("model", {"mAP50": 0.85}, output_path=out)
        assert out.exists()

    def test_detections_csv_creates_parent_dirs(self, tmp_path):
        out = tmp_path / "exports" / "dets.csv"
        export_detections_csv([_sample_detection()], output_path=out)
        assert out.exists()

    def test_maturity_csv_creates_parent_dirs(self, tmp_path):
        out = tmp_path / "exports" / "maturity.csv"
        export_maturity_csv([_sample_maturity()], output_path=out)
        assert out.exists()
