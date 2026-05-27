"""detection.tests.test_detector — Unit tests for detection domain. No GPU required."""
from __future__ import annotations
import pytest
import numpy as np
from shared.core.value_objects import BoundingBox, Confidence
from shared.core.entities import Detection
from shared.core.enums import TrichomeType
from detection.domain.tiled_inference import TiledInferenceEngine, TileConfig
from detection.domain.confidence_calibrator import TemperatureCalibrator, compute_ece
from shared.metrics.detection_metrics import compute_iou_matrix, evaluate_detection


class TestBoundingBox:
    def test_area(self):
        bbox = BoundingBox(100, 100, 200, 200)
        assert bbox.area == pytest.approx(10000.0)

    def test_iou_self(self):
        bbox = BoundingBox(0, 0, 100, 100)
        assert bbox.iou(bbox) == pytest.approx(1.0)

    def test_iou_no_overlap(self):
        a = BoundingBox(0, 0, 50, 50)
        b = BoundingBox(100, 100, 200, 200)
        assert a.iou(b) == pytest.approx(0.0)

    def test_iou_partial(self):
        a = BoundingBox(0, 0, 100, 100)
        b = BoundingBox(50, 50, 150, 150)
        assert a.iou(b) == pytest.approx(2500/17500, rel=1e-4)

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            BoundingBox(x_min=100, y_min=100, x_max=50, y_max=200)

    def test_clip_to_image(self):
        bbox = BoundingBox(x_min=0.1, y_min=0.1, x_max=200, y_max=200)
        clipped = bbox.clip_to_image(150, 150)
        assert clipped.x_max == pytest.approx(150.0)


class TestTiledInference:
    def _engine(self, tile_size=640, overlap=0.2):
        class MockDet:
            model_id = "mock"; is_loaded = True
            def load(self): pass
            def unload(self): pass
            def detect(self, image, config=None):
                from detection.domain.detector import DetectionResult
                return DetectionResult(detections=[], image_id="", model_id="mock",
                    inference_time_ms=1.0, image_shape=image.shape)
            def detect_batch(self, images, config=None): return []
        return TiledInferenceEngine(detector=MockDet(),
            tile_config=TileConfig(tile_size=tile_size, overlap_fraction=overlap))

    def test_single_tile_small_image(self):
        tiles = self._engine(tile_size=1280).compute_tiles(640, 640)
        assert len(tiles) == 1

    def test_multi_tile_large_image(self):
        tiles = self._engine(tile_size=1280, overlap=0.2).compute_tiles(2560, 2560)
        assert len(tiles) >= 4

    def test_coordinate_translation(self):
        from detection.domain.tiled_inference import TileInfo
        tile = TileInfo(0, x_start=500, y_start=300, x_end=1140, y_end=940)
        local = BoundingBox(50, 60, 100, 120)
        g = tile.to_global_bbox(local)
        assert g.x_min == pytest.approx(550.0)
        assert g.y_min == pytest.approx(360.0)

    def test_coverage_complete(self):
        engine = self._engine(tile_size=640, overlap=0.2)
        cov = engine.get_tile_coverage_map(1280, 1920)
        assert int(cov.min()) >= 1


class TestCalibration:
    def test_temperature_reduces_confidence(self):
        cal = TemperatureCalibrator(); cal.temperature = 2.0
        logit = 2.0
        p1 = 1/(1+np.exp(-2.0))
        p2 = cal.calibrate(logit)
        assert p2 < p1

    def test_ece_overconfident(self):
        confs = np.full(500, 0.95, dtype=np.float32)
        accs = np.random.binomial(1, 0.6, 500).astype(np.float32)
        result = compute_ece(confs, accs)
        assert result.ece > 0.20


class TestDetectionMetrics:
    def test_perfect(self):
        gt = [{"image_id":"i1","boxes":[[0,0,100,100]],"labels":[0]}]
        pr = [{"image_id":"i1","boxes":[[0,0,100,100]],"labels":[0],"scores":[0.99]}]
        r = evaluate_detection(gt, pr, ["cls"])
        assert r.map50 == pytest.approx(1.0, abs=0.01)

    def test_no_preds(self):
        gt = [{"image_id":"i1","boxes":[[0,0,100,100]],"labels":[0]}]
        pr = [{"image_id":"i1","boxes":[],"labels":[],"scores":[]}]
        r = evaluate_detection(gt, pr, ["cls"])
        assert r.recall == pytest.approx(0.0)
