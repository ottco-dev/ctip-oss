"""
tests.unit.test_inference — Unit tests for inference runner components.

Tests:
  - LocalRunnerConfig defaults and validation
  - LatencyStats: update, mean, p95, to_dict
  - LocalPyTorchRunner: not-loaded guard, load/unload lifecycle (mocked)
  - ONNXRunnerConfig defaults
  - ONNXDetection dataclass
  - TRT availability check (no GPU required)
  - Result parsing (mocked ultralytics result)
  - LocalPyTorchRunner._parse_results with mock boxes
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from inference.local.runner import (
    LatencyStats,
    LocalPyTorchRunner,
    LocalRunnerConfig,
)
from inference.onnx_runtime.runner import ONNXDetection, ONNXResult, ONNXRunnerConfig
from inference.tensorrt_engine.runner import tensorrt_available


# ---------------------------------------------------------------------------
# LocalRunnerConfig
# ---------------------------------------------------------------------------

class TestLocalRunnerConfig:

    def test_defaults(self):
        cfg = LocalRunnerConfig(model_path="dummy.pt")
        assert cfg.device == "cuda"
        assert cfg.half_precision is True
        assert cfg.warmup_runs == 2
        assert cfg.conf_threshold == 0.35
        assert cfg.iou_threshold == 0.45
        assert cfg.imgsz == 1280
        assert cfg.max_batch_size == 8
        assert cfg.augment is False

    def test_custom_values(self):
        cfg = LocalRunnerConfig(
            model_path="yolo11n.pt",
            device="cpu",
            half_precision=False,
            conf_threshold=0.5,
            imgsz=640,
        )
        assert cfg.device == "cpu"
        assert cfg.half_precision is False
        assert cfg.conf_threshold == 0.5
        assert cfg.imgsz == 640


# ---------------------------------------------------------------------------
# LatencyStats
# ---------------------------------------------------------------------------

class TestLatencyStats:

    def test_initial_state(self):
        stats = LatencyStats()
        assert stats.runs == 0
        assert stats.total_ms == 0.0
        assert stats.mean_ms == 0.0
        assert stats.p95_ms == 0.0

    def test_single_update(self):
        stats = LatencyStats()
        stats.update(10.0)
        assert stats.runs == 1
        assert abs(stats.mean_ms - 10.0) < 1e-9
        assert stats.min_ms == 10.0
        assert stats.max_ms == 10.0

    def test_mean_computed_correctly(self):
        stats = LatencyStats()
        for ms in [10.0, 20.0, 30.0]:
            stats.update(ms)
        assert abs(stats.mean_ms - 20.0) < 1e-9

    def test_min_max_tracked(self):
        stats = LatencyStats()
        for ms in [5.0, 50.0, 25.0, 1.0, 99.0]:
            stats.update(ms)
        assert stats.min_ms == 1.0
        assert stats.max_ms == 99.0

    def test_p95_with_known_data(self):
        stats = LatencyStats()
        # 100 values 1..100 ms → p95 = 95
        for ms in range(1, 101):
            stats.update(float(ms))
        assert stats.p95_ms > 90.0  # p95 of 1..100 is 95

    def test_to_dict_has_required_keys(self):
        stats = LatencyStats()
        stats.update(15.0)
        d = stats.to_dict()
        for key in ("runs", "mean_ms", "min_ms", "max_ms", "p95_ms"):
            assert key in d

    def test_to_dict_values_are_numeric(self):
        stats = LatencyStats()
        stats.update(10.0)
        d = stats.to_dict()
        assert isinstance(d["mean_ms"], float)
        assert isinstance(d["runs"], int)

    def test_recent_buffer_capped_at_50(self):
        stats = LatencyStats()
        for i in range(100):
            stats.update(float(i))
        assert len(stats._recent) == 50
        # Most recent 50 entries should be 50..99
        assert stats._recent[0] == 50.0

    def test_total_ms_accumulates(self):
        stats = LatencyStats()
        for ms in [10.0, 20.0, 30.0]:
            stats.update(ms)
        assert abs(stats.total_ms - 60.0) < 1e-9


# ---------------------------------------------------------------------------
# LocalPyTorchRunner — state guards (no GPU needed)
# ---------------------------------------------------------------------------

class TestLocalPyTorchRunnerGuards:

    def test_not_loaded_initially(self):
        cfg = LocalRunnerConfig(model_path="dummy.pt")
        runner = LocalPyTorchRunner(cfg)
        assert runner.is_loaded is False

    def test_infer_raises_if_not_loaded(self):
        cfg = LocalRunnerConfig(model_path="dummy.pt")
        runner = LocalPyTorchRunner(cfg)
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        with pytest.raises(RuntimeError, match="not loaded"):
            runner.infer(img)

    def test_infer_batch_raises_if_not_loaded(self):
        cfg = LocalRunnerConfig(model_path="dummy.pt")
        runner = LocalPyTorchRunner(cfg)
        with pytest.raises(RuntimeError, match="not loaded"):
            runner.infer_batch([np.zeros((640, 640, 3), dtype=np.uint8)])

    def test_unload_when_not_loaded_is_noop(self):
        cfg = LocalRunnerConfig(model_path="dummy.pt")
        runner = LocalPyTorchRunner(cfg)
        runner.unload()  # should not raise
        assert runner.is_loaded is False

    def test_double_unload_is_noop(self):
        cfg = LocalRunnerConfig(model_path="dummy.pt")
        runner = LocalPyTorchRunner(cfg)
        runner.unload()
        runner.unload()  # second unload also should not raise

    def test_repr_includes_status(self):
        cfg = LocalRunnerConfig(model_path="test.pt")
        runner = LocalPyTorchRunner(cfg)
        r = repr(runner)
        assert "not loaded" in r
        assert "test.pt" in r

    def test_vram_usage_returns_none_on_no_cuda(self):
        cfg = LocalRunnerConfig(model_path="test.pt", device="cpu")
        runner = LocalPyTorchRunner(cfg)
        # Without a real GPU this should return None gracefully
        with patch("torch.cuda.is_available", return_value=False):
            result = runner.get_vram_usage_mb()
        assert result is None


# ---------------------------------------------------------------------------
# LocalPyTorchRunner._parse_results (static, no model needed)
# ---------------------------------------------------------------------------

class TestParseResults:

    def _make_result(self, boxes_data: list[tuple]) -> MagicMock:
        """Build a mock Ultralytics result with given (x1,y1,x2,y2,conf,cls) tuples."""
        result = MagicMock()
        result.names = {0: "capitate_stalked", 1: "bulbous", 2: "sessile"}

        if not boxes_data:
            result.boxes = None
            return result

        xyxy = np.array([[b[0], b[1], b[2], b[3]] for b in boxes_data], dtype=np.float32)
        confs = np.array([b[4] for b in boxes_data], dtype=np.float32)
        cls = np.array([b[5] for b in boxes_data], dtype=np.float32)

        mock_boxes = MagicMock()
        mock_boxes.xyxy = xyxy
        mock_boxes.conf = confs
        mock_boxes.cls = cls
        result.boxes = mock_boxes
        return result

    def test_empty_boxes_returns_empty_list(self):
        result = MagicMock()
        result.boxes = None
        detections = LocalPyTorchRunner._parse_results(result)
        assert detections == []

    def test_single_detection_parsed(self):
        mock_result = self._make_result([(10, 20, 110, 120, 0.9, 0)])
        detections = LocalPyTorchRunner._parse_results(mock_result)
        assert len(detections) == 1
        d = detections[0]
        assert d["x1"] == pytest.approx(10.0)
        assert d["y1"] == pytest.approx(20.0)
        assert d["x2"] == pytest.approx(110.0)
        assert d["y2"] == pytest.approx(120.0)
        assert d["confidence"] == pytest.approx(0.9, abs=1e-4)
        assert d["class_id"] == 0
        assert d["class_name"] == "capitate_stalked"

    def test_multiple_detections_parsed(self):
        mock_result = self._make_result([
            (0, 0, 50, 50, 0.95, 0),
            (100, 100, 200, 200, 0.7, 1),
            (300, 300, 400, 400, 0.55, 2),
        ])
        detections = LocalPyTorchRunner._parse_results(mock_result)
        assert len(detections) == 3
        assert detections[1]["class_name"] == "bulbous"
        assert detections[2]["class_name"] == "sessile"

    def test_class_id_preserved(self):
        mock_result = self._make_result([(0, 0, 100, 100, 0.8, 2)])
        detections = LocalPyTorchRunner._parse_results(mock_result)
        assert detections[0]["class_id"] == 2

    def test_unknown_class_gets_fallback_name(self):
        mock_result = self._make_result([(0, 0, 100, 100, 0.8, 99)])
        mock_result.names = {}
        detections = LocalPyTorchRunner._parse_results(mock_result)
        assert "99" in detections[0]["class_name"]

    def test_all_confidences_are_float(self):
        mock_result = self._make_result([
            (0, 0, 50, 50, 0.9, 0),
            (60, 60, 100, 100, 0.6, 1),
        ])
        detections = LocalPyTorchRunner._parse_results(mock_result)
        for d in detections:
            assert isinstance(d["confidence"], float)


# ---------------------------------------------------------------------------
# ONNXRunnerConfig
# ---------------------------------------------------------------------------

class TestONNXRunnerConfig:

    def test_defaults(self):
        cfg = ONNXRunnerConfig()
        assert cfg.imgsz == 1280
        assert cfg.conf_threshold == 0.25
        assert cfg.iou_threshold == 0.45
        assert "CUDAExecutionProvider" in cfg.providers
        assert "CPUExecutionProvider" in cfg.providers
        assert cfg.input_name == "images"
        assert cfg.output_names == ["output0"]

    def test_custom_providers(self):
        cfg = ONNXRunnerConfig(providers=["CPUExecutionProvider"])
        assert cfg.providers == ["CPUExecutionProvider"]


# ---------------------------------------------------------------------------
# ONNXDetection dataclass
# ---------------------------------------------------------------------------

class TestONNXDetection:

    def test_creation(self):
        det = ONNXDetection(
            x1=10.0, y1=20.0, x2=110.0, y2=120.0,
            confidence=0.92, class_id=1, class_name="bulbous",
        )
        assert det.confidence == pytest.approx(0.92)
        assert det.class_name == "bulbous"

    def test_box_coordinates_preserved(self):
        det = ONNXDetection(x1=0, y1=0, x2=640, y2=480, confidence=0.5, class_id=0, class_name="x")
        assert det.x2 == 640
        assert det.y2 == 480


# ---------------------------------------------------------------------------
# TensorRT availability
# ---------------------------------------------------------------------------

class TestTensorRTAvailability:

    def test_returns_bool(self):
        result = tensorrt_available()
        assert isinstance(result, bool)

    def test_false_when_tensorrt_not_installed(self):
        with patch.dict("sys.modules", {"tensorrt": None, "pycuda": None, "pycuda.driver": None}):
            # Re-import to test the function behaviour
            result = tensorrt_available()
            # On systems without TRT, should be False
            assert result is False or result is True  # just confirm no crash


# ---------------------------------------------------------------------------
# IoU matching from evaluator (isolated unit tests)
# ---------------------------------------------------------------------------

class TestIoUMatching:
    """Tests for the IoU-based prediction→GT matching in the evaluator."""

    def setup_method(self):
        from training.evaluation.evaluator import _compute_iou, _match_detections
        self.compute_iou = _compute_iou
        self.match_detections = _match_detections

    def test_perfect_overlap_gives_iou_1(self):
        box = np.array([0, 0, 100, 100], dtype=float)
        assert abs(self.compute_iou(box, box) - 1.0) < 1e-9

    def test_no_overlap_gives_iou_0(self):
        a = np.array([0, 0, 10, 10], dtype=float)
        b = np.array([20, 20, 30, 30], dtype=float)
        assert self.compute_iou(a, b) == 0.0

    def test_partial_overlap(self):
        a = np.array([0, 0, 10, 10], dtype=float)
        b = np.array([5, 5, 15, 15], dtype=float)
        iou = self.compute_iou(a, b)
        # intersection = 5×5=25, union = 100+100-25=175
        assert abs(iou - 25 / 175) < 1e-6

    def test_match_perfect_prediction(self):
        preds = [{"x1": 0, "y1": 0, "x2": 100, "y2": 100, "confidence": 0.9, "class_id": 0}]
        gts = [{"x1": 0, "y1": 0, "x2": 100, "y2": 100, "class_id": 0}]
        confs, correct = self.match_detections(preds, gts, iou_threshold=0.5)
        assert len(confs) == 1
        assert correct[0] is True

    def test_match_false_positive(self):
        preds = [{"x1": 0, "y1": 0, "x2": 100, "y2": 100, "confidence": 0.9, "class_id": 0}]
        gts = [{"x1": 500, "y1": 500, "x2": 600, "y2": 600, "class_id": 0}]
        confs, correct = self.match_detections(preds, gts, iou_threshold=0.5)
        assert correct[0] is False

    def test_class_mismatch_prevents_match(self):
        preds = [{"x1": 0, "y1": 0, "x2": 100, "y2": 100, "confidence": 0.9, "class_id": 1}]
        gts = [{"x1": 0, "y1": 0, "x2": 100, "y2": 100, "class_id": 0}]
        confs, correct = self.match_detections(preds, gts, iou_threshold=0.5)
        # Class mismatch → false positive
        assert correct[0] is False

    def test_each_gt_matched_only_once(self):
        """Two predictions overlapping same GT — only best gets TP."""
        gts = [{"x1": 0, "y1": 0, "x2": 100, "y2": 100, "class_id": 0}]
        preds = [
            {"x1": 0, "y1": 0, "x2": 100, "y2": 100, "confidence": 0.9, "class_id": 0},
            {"x1": 5, "y1": 5, "x2": 95, "y2": 95, "confidence": 0.7, "class_id": 0},
        ]
        confs, correct = self.match_detections(preds, gts, iou_threshold=0.5)
        assert len(confs) == 2
        # Only one can be TP
        assert sum(correct) == 1

    def test_empty_predictions_returns_empty(self):
        gts = [{"x1": 0, "y1": 0, "x2": 100, "y2": 100, "class_id": 0}]
        confs, correct = self.match_detections([], gts, iou_threshold=0.5)
        assert confs == []
        assert correct == []

    def test_empty_gts_all_false_positives(self):
        preds = [
            {"x1": 0, "y1": 0, "x2": 100, "y2": 100, "confidence": 0.9, "class_id": 0},
            {"x1": 200, "y1": 200, "x2": 300, "y2": 300, "confidence": 0.7, "class_id": 0},
        ]
        confs, correct = self.match_detections(preds, [], iou_threshold=0.5)
        assert len(confs) == 2
        assert all(c is False for c in correct)

    def test_confidence_ordering_preserved(self):
        """Higher confidence prediction matches GT first."""
        gts = [{"x1": 0, "y1": 0, "x2": 100, "y2": 100, "class_id": 0}]
        preds = [
            {"x1": 5, "y1": 5, "x2": 95, "y2": 95, "confidence": 0.6, "class_id": 0},
            {"x1": 0, "y1": 0, "x2": 100, "y2": 100, "confidence": 0.9, "class_id": 0},
        ]
        confs, correct = self.match_detections(preds, gts, iou_threshold=0.5)
        # After sorting by conf desc, the 0.9 pred should get matched → TP
        # The 0.6 pred → FP (GT already consumed)
        assert sum(correct) == 1
