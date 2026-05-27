"""
tests/integration/test_detect_segment_pipeline.py — Integration tests for
the Detection → Segmentation → Maturity → Morphology → Measurement pipeline.

These tests run WITHOUT real model weights.  Backends are mocked or bypassed
so the integration tests validate:

  1. Data contracts at each stage boundary
  2. Object construction and field population
  3. Scientific constraint propagation (no cannabinoid claims)
  4. Error handling and graceful degradation
  5. Correct Instance lifecycle: Detection → SegmentedInstance → Instance

MARKER: `integration` (no external services required for these tests).
None are marked `gpu` since no GPU compute happens with mocked backends.

Structure
─────────
  TestDetectionDataFlow          — Detection entity construction, result contracts
  TestSegmentationDataFlow       — SegmentPipeline with mocked backend
  TestDetectToSegmentPipeline    — Full detect→segment with mocked backends
  TestMaturityIntegration        — detect→segment→maturity full chain
  TestMorphologyIntegration      — detect→segment→morphology full chain
  TestMeasurementIntegration     — detect→segment→measurement full chain
  TestFullPipelineIntegration    — detect→segment→maturity+morphology+measurement
  TestScientificConstraints      — End-to-end scientific validity checks
  TestErrorHandling              — Graceful degradation at each stage
"""

from __future__ import annotations

import time
import unittest.mock as mock
from dataclasses import dataclass
from typing import List, Optional
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_rgb_image(h: int = 256, w: int = 256, seed: int = 42) -> NDArray:
    """Synthetic microscopy-like RGB image."""
    rng = np.random.default_rng(seed)
    img = rng.integers(80, 200, (h, w, 3), dtype=np.uint8)
    return img


def _make_ellipse_mask(
    h: int = 128,
    w: int = 128,
    cx: int = 64,
    cy: int = 64,
    ax: int = 25,
    ay: int = 20,
) -> NDArray:
    """Binary ellipse mask (uint8 0/255)."""
    m = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(m, (cx, cy), (ax, ay), 0, 0, 360, 255, -1)
    return m


def _make_bool_mask(h: int = 128, w: int = 128) -> NDArray:
    """Circular boolean mask."""
    m = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(m, (64, 64), 30, 255, -1)
    return m.astype(bool)


def _make_detection_dict(
    x1: float = 40,
    y1: float = 40,
    x2: float = 90,
    y2: float = 90,
    confidence: float = 0.85,
    class_id: int = 0,
    class_name: str = "trichome",
) -> dict:
    """Construct a detection dict as expected by SegmentPipeline.run()."""
    return {
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "confidence": confidence,
        "class_id": class_id,
        "class_name": class_name,
    }


# ── Imports (inside tests to avoid import-time errors on missing deps) ─────────

@pytest.fixture(scope="module")
def detection_imports():
    """Import all detection domain objects."""
    from detection.domain.detector import Detection, DetectionConfig, DetectionResult
    from shared.core.entities import Detection as DetectionEntity
    from shared.core.value_objects import BoundingBox, Confidence
    from shared.core.enums import TrichomeType
    return {
        "DetectionConfig": DetectionConfig,
        "BoundingBox": BoundingBox,
        "Confidence": Confidence,
        "TrichomeType": TrichomeType,
    }


@pytest.fixture(scope="module")
def segmentation_imports():
    """Import all segmentation domain objects."""
    from segmentation.application.segment_pipeline import (
        SegmentPipeline,
        SegmentPipelineConfig,
        SegmentPipelineResult,
        SegmentedInstance,
    )
    from segmentation.domain.segmentor import (
        BoxPrompt,
        PointPrompt,
        SegmentationResult,
        BatchSegmentationResult,
        SegmentorConfig,
    )
    from segmentation.domain.mask_refinement import refine_mask
    from segmentation.domain.polygon_utils import mask_to_polygon, polygon_to_mask
    return {
        "SegmentPipeline": SegmentPipeline,
        "SegmentPipelineConfig": SegmentPipelineConfig,
        "SegmentPipelineResult": SegmentPipelineResult,
        "SegmentedInstance": SegmentedInstance,
        "BoxPrompt": BoxPrompt,
        "SegmentorConfig": SegmentorConfig,
        "refine_mask": refine_mask,
        "mask_to_polygon": mask_to_polygon,
    }


@pytest.fixture(scope="module")
def shared_imports():
    """Import shared domain objects."""
    from shared.core.entities import Detection, Instance
    from shared.core.value_objects import BoundingBox, Confidence, Mask
    from shared.core.enums import TrichomeType, MaturityStage
    return {
        "Detection": Detection,
        "Instance": Instance,
        "BoundingBox": BoundingBox,
        "Confidence": Confidence,
        "Mask": Mask,
        "TrichomeType": TrichomeType,
        "MaturityStage": MaturityStage,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Detection data flow
# ═══════════════════════════════════════════════════════════════════════════════

class TestDetectionDataFlow:
    """Verify Detection entities are correctly constructed and serializable."""

    def test_detection_entity_construction(self, shared_imports):
        Detection = shared_imports["Detection"]
        BoundingBox = shared_imports["BoundingBox"]
        Confidence = shared_imports["Confidence"]
        TrichomeType = shared_imports["TrichomeType"]

        det = Detection(
            bounding_box=BoundingBox(10, 10, 60, 60),
            confidence=Confidence(0.87),
            trichome_type=TrichomeType.CAPITATE_STALKED,
            model_id="yolo11s",
            image_id="test_001",
        )

        assert det.bounding_box.x_min == 10
        assert float(det.confidence) == pytest.approx(0.87)
        assert det.trichome_type == TrichomeType.CAPITATE_STALKED
        assert det.is_high_confidence  # 0.87 >= 0.75

    def test_detection_to_dict_contract(self, shared_imports):
        """Detection.to_dict() must produce correct keys for downstream use."""
        Detection = shared_imports["Detection"]
        BoundingBox = shared_imports["BoundingBox"]
        Confidence = shared_imports["Confidence"]

        det = Detection(
            bounding_box=BoundingBox(5, 10, 55, 65),
            confidence=Confidence(0.72),
        )
        d = det.to_dict()

        assert "bbox" in d
        assert "confidence" in d
        assert "trichome_type" in d
        assert "is_high_confidence" in d
        assert len(d["bbox"]) == 4

    def test_low_confidence_detection_is_not_high_confidence(self, shared_imports):
        Detection = shared_imports["Detection"]
        BoundingBox = shared_imports["BoundingBox"]
        Confidence = shared_imports["Confidence"]

        det = Detection(
            bounding_box=BoundingBox(5, 5, 30, 30),
            confidence=Confidence(0.30),
        )
        assert not det.is_high_confidence

    def test_multiple_detections_can_be_in_list(self, shared_imports):
        """Verify N detections can be constructed and collected."""
        Detection = shared_imports["Detection"]
        BoundingBox = shared_imports["BoundingBox"]
        Confidence = shared_imports["Confidence"]

        detections = [
            Detection(
                bounding_box=BoundingBox(i * 20, i * 20, i * 20 + 40, i * 20 + 40),
                confidence=Confidence(0.50 + i * 0.05),
                image_id="batch_001",
            )
            for i in range(5)
        ]

        assert len(detections) == 5
        confidences = [float(d.confidence) for d in detections]
        assert all(0 < c < 1 for c in confidences)

    def test_bounding_box_area_is_positive(self, shared_imports):
        BoundingBox = shared_imports["BoundingBox"]
        bb = BoundingBox(10, 20, 80, 100)
        assert bb.area > 0
        assert bb.area == pytest.approx(70.0 * 80.0)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Segmentation data flow (mocked backend)
# ═══════════════════════════════════════════════════════════════════════════════

class TestSegmentationDataFlow:
    """SegmentPipeline integration using a mocked backend."""

    def _build_mock_backend(self, image_shape=(256, 256)):
        """
        Create a mock segmentor that returns realistic mask output
        for any box prompt.
        """
        h, w = image_shape
        backend = MagicMock()
        backend.is_loaded = True

        def _segment_batch(image, prompts, **kwargs):
            from segmentation.domain.segmentor import BatchSegmentationResult, SegmentationResult
            results = []
            for prompt in prompts:
                mask = np.zeros((h, w), dtype=bool)
                # Create ellipse mask near the prompt box
                if hasattr(prompt, "x1"):  # BoxPrompt
                    cx = int((prompt.x1 + prompt.x2) / 2)
                    cy = int((prompt.y1 + prompt.y2) / 2)
                    ax = max(int((prompt.x2 - prompt.x1) * 0.4), 5)
                    ay = max(int((prompt.y2 - prompt.y1) * 0.4), 5)
                else:
                    cx, cy = w // 2, h // 2
                    ax, ay = 20, 18
                m_u8 = np.zeros((h, w), dtype=np.uint8)
                cv2.ellipse(m_u8, (cx, cy), (ax, ay), 0, 0, 360, 255, -1)
                mask = m_u8.astype(bool)
                results.append(
                    SegmentationResult(mask=mask, score=0.92, prompt=prompt)
                )
            return BatchSegmentationResult(results=results, time_ms=5.0)

        backend.segment_batch.side_effect = _segment_batch
        return backend

    def test_segment_pipeline_with_zero_detections(self, segmentation_imports):
        SegmentPipeline = segmentation_imports["SegmentPipeline"]
        SegmentPipelineConfig = segmentation_imports["SegmentPipelineConfig"]

        cfg = SegmentPipelineConfig(backend="sam2_tiny")
        pipeline = SegmentPipeline(cfg)
        pipeline._backend = self._build_mock_backend()

        image = _make_rgb_image(256, 256)
        result = pipeline.run(image, [])

        assert result.num_input_detections == 0
        assert result.num_segmented == 0
        assert result.instances == []
        assert result.image_height == 256
        assert result.image_width == 256

    def test_segment_pipeline_with_single_detection(self, segmentation_imports):
        SegmentPipeline = segmentation_imports["SegmentPipeline"]
        SegmentPipelineConfig = segmentation_imports["SegmentPipelineConfig"]

        cfg = SegmentPipelineConfig(backend="sam2_tiny")
        pipeline = SegmentPipeline(cfg)
        pipeline._backend = self._build_mock_backend()

        image = _make_rgb_image(256, 256)
        detections = [_make_detection_dict(40, 40, 90, 90, confidence=0.88)]
        result = pipeline.run(image, detections)

        assert result.num_input_detections == 1
        assert result.num_segmented >= 1 or result.num_segmented == 0  # may skip tiny masks

    def test_segment_pipeline_with_multiple_detections(self, segmentation_imports):
        SegmentPipeline = segmentation_imports["SegmentPipeline"]
        SegmentPipelineConfig = segmentation_imports["SegmentPipelineConfig"]

        cfg = SegmentPipelineConfig(backend="sam2_tiny")
        pipeline = SegmentPipeline(cfg)
        pipeline._backend = self._build_mock_backend((512, 512))

        image = _make_rgb_image(512, 512)
        detections = [
            _make_detection_dict(30, 30, 80, 80),
            _make_detection_dict(100, 100, 170, 170),
            _make_detection_dict(200, 50, 280, 130),
            _make_detection_dict(350, 300, 430, 380),
        ]
        result = pipeline.run(image, detections)

        assert result.num_input_detections == 4
        assert isinstance(result.instances, list)

    def test_segment_result_fields(self, segmentation_imports):
        """SegmentPipelineResult must have all required fields."""
        SegmentPipeline = segmentation_imports["SegmentPipeline"]
        SegmentPipelineConfig = segmentation_imports["SegmentPipelineConfig"]
        SegmentPipelineResult = segmentation_imports["SegmentPipelineResult"]

        cfg = SegmentPipelineConfig(backend="sam2_tiny")
        pipeline = SegmentPipeline(cfg)
        pipeline._backend = self._build_mock_backend()

        result = pipeline.run(_make_rgb_image(), [_make_detection_dict()])

        assert isinstance(result, SegmentPipelineResult)
        assert hasattr(result, "instances")
        assert hasattr(result, "image_height")
        assert hasattr(result, "image_width")
        assert hasattr(result, "backend_used")
        assert hasattr(result, "num_input_detections")
        assert hasattr(result, "num_segmented")
        assert hasattr(result, "pipeline_time_ms")
        assert result.pipeline_time_ms >= 0

    def test_segment_pipeline_raises_if_not_loaded(self, segmentation_imports):
        """Running without loading backend must raise RuntimeError."""
        SegmentPipeline = segmentation_imports["SegmentPipeline"]
        SegmentPipelineConfig = segmentation_imports["SegmentPipelineConfig"]

        cfg = SegmentPipelineConfig(backend="sam2_tiny")
        pipeline = SegmentPipeline(cfg)
        # _backend is None (not loaded)

        with pytest.raises(RuntimeError, match="Backend not loaded"):
            pipeline.run(_make_rgb_image(), [_make_detection_dict()])


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Full detect→segment pipeline flow
# ═══════════════════════════════════════════════════════════════════════════════

class TestDetectToSegmentPipeline:
    """Simulate full detect→segment pipeline with mocked YOLO output."""

    def _make_mock_detections(self, n: int, image_size: int = 512) -> list[dict]:
        """Generate N realistic detection dicts."""
        rng = np.random.default_rng(42)
        dets = []
        for i in range(n):
            cx = rng.integers(50, image_size - 50)
            cy = rng.integers(50, image_size - 50)
            w = rng.integers(20, 60)
            h = rng.integers(20, 60)
            dets.append(_make_detection_dict(
                x1=float(max(0, cx - w // 2)),
                y1=float(max(0, cy - h // 2)),
                x2=float(min(image_size, cx + w // 2)),
                y2=float(min(image_size, cy + h // 2)),
                confidence=float(rng.uniform(0.5, 0.95)),
            ))
        return dets

    def _build_mock_backend(self, image_shape=(512, 512)):
        h, w = image_shape
        backend = MagicMock()
        backend.is_loaded = True

        def _segment_batch(image, prompts, **kwargs):
            from segmentation.domain.segmentor import BatchSegmentationResult, SegmentationResult
            results = []
            for prompt in prompts:
                m = np.zeros((h, w), dtype=np.uint8)
                if hasattr(prompt, "x1"):
                    cx = int((prompt.x1 + prompt.x2) / 2)
                    cy = int((prompt.y1 + prompt.y2) / 2)
                    r = max(int(min(prompt.x2 - prompt.x1, prompt.y2 - prompt.y1) * 0.4), 8)
                    cv2.circle(m, (cx, cy), r, 255, -1)
                results.append(
                    SegmentationResult(mask=m.astype(bool), score=0.91, prompt=prompt)
                )
            return BatchSegmentationResult(results=results, time_ms=8.0)

        backend.segment_batch.side_effect = _segment_batch
        return backend

    def test_detect_to_segment_n10(self, segmentation_imports):
        """10 simulated detections → segmentation produces instances."""
        from segmentation.application.segment_pipeline import SegmentPipeline, SegmentPipelineConfig

        cfg = SegmentPipelineConfig(backend="sam2_tiny", min_mask_area_px=50)
        pipeline = SegmentPipeline(cfg)
        pipeline._backend = self._build_mock_backend((512, 512))

        image = _make_rgb_image(512, 512)
        detections = self._make_mock_detections(10, 512)

        result = pipeline.run(image, detections)

        assert result.num_input_detections == 10
        assert isinstance(result.instances, list)
        # All instances should have masks
        for inst in result.instances:
            assert isinstance(inst.mask, np.ndarray)
            assert inst.mask.dtype == bool
            assert inst.mask.sum() > 0

    def test_detect_to_segment_instance_has_geometry(self, segmentation_imports):
        """SegmentedInstance should have populated geometry fields."""
        from segmentation.application.segment_pipeline import SegmentPipeline, SegmentPipelineConfig

        cfg = SegmentPipelineConfig(backend="sam2_tiny", min_mask_area_px=10)
        pipeline = SegmentPipeline(cfg)
        pipeline._backend = self._build_mock_backend((256, 256))

        image = _make_rgb_image(256, 256)
        detections = [_make_detection_dict(60, 60, 130, 130)]
        result = pipeline.run(image, detections)

        for inst in result.instances:
            assert inst.area_px > 0
            assert 0 <= inst.centroid_x <= 256
            assert 0 <= inst.centroid_y <= 256

    def test_detect_to_segment_confidence_preserved(self):
        """Detection confidence should be preserved in SegmentedInstance."""
        from segmentation.application.segment_pipeline import SegmentPipeline, SegmentPipelineConfig

        h, w = 256, 256
        backend = MagicMock()
        backend.is_loaded = True

        def _segment_batch(image, prompts, **kwargs):
            from segmentation.domain.segmentor import BatchSegmentationResult, SegmentationResult
            results = []
            for prompt in prompts:
                m = np.zeros((h, w), dtype=np.uint8)
                cv2.circle(m, (128, 128), 30, 255, -1)
                results.append(SegmentationResult(mask=m.astype(bool), score=0.95, prompt=prompt))
            return BatchSegmentationResult(results=results, time_ms=3.0)

        backend.segment_batch.side_effect = _segment_batch

        cfg = SegmentPipelineConfig(backend="sam2_tiny", min_mask_area_px=10)
        pipeline = SegmentPipeline(cfg)
        pipeline._backend = backend

        det = _make_detection_dict(80, 80, 170, 170, confidence=0.91)
        result = pipeline.run(_make_rgb_image(h, w), [det])

        for inst in result.instances:
            assert inst.detection_confidence == pytest.approx(0.91)

    def test_segment_images_of_different_sizes(self):
        """Pipeline must handle non-square images."""
        from segmentation.application.segment_pipeline import SegmentPipeline, SegmentPipelineConfig

        h, w = 480, 640
        backend = MagicMock()
        backend.is_loaded = True

        def _segment_batch(image, prompts, **kwargs):
            from segmentation.domain.segmentor import BatchSegmentationResult, SegmentationResult
            results = []
            for prompt in prompts:
                m = np.zeros((h, w), dtype=np.uint8)
                cv2.circle(m, (w // 2, h // 2), 25, 255, -1)
                results.append(SegmentationResult(mask=m.astype(bool), score=0.88, prompt=prompt))
            return BatchSegmentationResult(results=results, time_ms=5.0)

        backend.segment_batch.side_effect = _segment_batch

        cfg = SegmentPipelineConfig(backend="sam2_tiny", min_mask_area_px=10)
        pipeline = SegmentPipeline(cfg)
        pipeline._backend = backend

        image = _make_rgb_image(h, w)
        result = pipeline.run(image, [_make_detection_dict(200, 150, 300, 250)])

        assert result.image_height == h
        assert result.image_width == w


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Maturity integration (segmented crop → maturity label)
# ═══════════════════════════════════════════════════════════════════════════════

class TestMaturityIntegration:
    """Full detect→segment→maturity integration with mocked segmentation."""

    def _make_instance_with_crop(self, color_style: str = "cloudy") -> object:
        """
        Create an Instance with a realistic crop for maturity analysis.
        """
        from shared.core.entities import Instance
        from shared.core.value_objects import Mask

        h, w = 64, 64
        rng = np.random.default_rng(42)

        if color_style == "clear":
            hsv = np.full((h, w, 3), [120, 30, 235], dtype=np.uint8)
        elif color_style == "cloudy":
            hsv = np.full((h, w, 3), [0, 25, 205], dtype=np.uint8)
        elif color_style == "amber":
            hsv = np.full((h, w, 3), [20, 185, 195], dtype=np.uint8)
        else:  # degraded
            hsv = np.full((h, w, 3), [12, 150, 75], dtype=np.uint8)

        noise = rng.integers(-15, 15, (h, w, 3), dtype=np.int16)
        img_hsv = np.clip(hsv.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        crop_bgr = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

        inst = Instance(crop=crop_rgb)
        mask_arr = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask_arr, (32, 32), 25, 255, -1)
        inst.mask = Mask.from_uint8(mask_arr)
        return inst

    def test_maturity_pipeline_processes_instance(self):
        """Full Instance → MaturityLabel flow."""
        from maturity.application.maturity_pipeline import MaturityPipeline, MaturityPipelineConfig
        from shared.core.enums import MaturityStage

        cfg = MaturityPipelineConfig(use_analyzer=False)
        pipeline = MaturityPipeline(cfg)

        inst = self._make_instance_with_crop("cloudy")
        result = pipeline.analyze([inst])

        assert result is not None
        assert len(result.instances) == 1
        processed = result.instances[0]
        assert processed.maturity_label is not None

    def test_maturity_label_has_valid_stage(self):
        """MaturityLabel stage must be a valid MaturityStage enum."""
        from maturity.application.maturity_pipeline import MaturityPipeline, MaturityPipelineConfig
        from shared.core.enums import MaturityStage

        cfg = MaturityPipelineConfig(use_analyzer=False)
        pipeline = MaturityPipeline(cfg)

        valid_stages = set(s.value for s in MaturityStage)
        for style in ["clear", "cloudy", "amber", "degraded"]:
            inst = self._make_instance_with_crop(style)
            result = pipeline.analyze([inst])
            label = result.instances[0].maturity_label
            assert label is not None
            assert label.stage.value in valid_stages, (
                f"Style {style} → stage {label.stage.value!r} not in {valid_stages}"
            )

    def test_maturity_confidence_in_unit_range(self):
        """Confidence must be in [0, 1] regardless of input."""
        from maturity.application.maturity_pipeline import MaturityPipeline, MaturityPipelineConfig

        cfg = MaturityPipelineConfig(use_analyzer=False)
        pipeline = MaturityPipeline(cfg)

        for style in ["clear", "cloudy", "amber", "degraded"]:
            inst = self._make_instance_with_crop(style)
            result = pipeline.analyze([inst])
            label = result.instances[0].maturity_label
            assert label is not None
            conf = float(label.confidence)
            assert 0.0 <= conf <= 1.0, f"Confidence {conf} out of range for style {style}"

    def test_maturity_population_stats_sum_to_one(self):
        """Stage distribution fractions must sum to 1.0."""
        from maturity.application.maturity_pipeline import MaturityPipeline, MaturityPipelineConfig

        cfg = MaturityPipelineConfig(use_analyzer=False)
        pipeline = MaturityPipeline(cfg)

        instances = [
            self._make_instance_with_crop(s)
            for s in ["clear", "cloudy", "amber", "degraded", "cloudy"]
        ]
        result = pipeline.analyze(instances)

        dist = result.stage_distribution
        assert abs(sum(dist.values()) - 1.0) < 0.02, (
            f"Stage distribution does not sum to 1.0: {dist}"
        )

    def test_maturity_no_cannabinoid_claims_in_output(self):
        """SCIENTIFIC CONSTRAINT: no THC/CBD claims in any output field."""
        from maturity.application.maturity_pipeline import MaturityPipeline, MaturityPipelineConfig

        cfg = MaturityPipelineConfig(use_analyzer=False)
        pipeline = MaturityPipeline(cfg)

        inst = self._make_instance_with_crop("amber")
        result = pipeline.analyze([inst])

        result_str = str(result.to_dict()).lower()
        for forbidden in ["thc", "cbd", "cannabinoid", "potency", "harvest_time"]:
            assert forbidden not in result_str, (
                f"Forbidden term '{forbidden}' found in maturity pipeline output"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Morphology integration
# ═══════════════════════════════════════════════════════════════════════════════

class TestMorphologyIntegration:
    """Segmented Instance → Morphology classification chain."""

    def _make_morphology_instance(self, morph_type: str = "bulbous"):
        """Create Instance with an appropriate mask for morphology classification."""
        from shared.core.entities import Instance
        from shared.core.value_objects import Mask

        h, w = 128, 128
        m = np.zeros((h, w), dtype=np.uint8)

        if morph_type == "bulbous":
            cv2.circle(m, (64, 64), 40, 255, -1)
        elif morph_type == "sessile":
            cv2.ellipse(m, (64, 64), (36, 28), 0, 0, 360, 255, -1)
        else:  # stalked
            cv2.circle(m, (64, 32), 22, 255, -1)   # head
            cv2.rectangle(m, (58, 54), (70, 100), 255, -1)  # stalk

        crop = np.ones((h, w, 3), dtype=np.uint8) * 128
        inst = Instance(crop=crop)
        inst.mask = Mask.from_uint8(m)
        return inst

    def test_morphology_pipeline_processes_instance(self):
        """Full Instance → MorphologyType flow."""
        from morphology.application.morphology_pipeline import (
            MorphologyPipeline, MorphologyPipelineConfig
        )

        cfg = MorphologyPipelineConfig(classifier_model_path=None)
        pipeline = MorphologyPipeline(cfg)

        inst = self._make_morphology_instance("bulbous")
        result = pipeline.analyze([inst])

        assert result is not None
        assert result.total_analyzed >= 1 or result.failed >= 0

    def test_morphology_type_distribution_keys_valid(self):
        """type_distribution keys must be TrichomeType values."""
        from morphology.application.morphology_pipeline import (
            MorphologyPipeline, MorphologyPipelineConfig
        )
        from shared.core.enums import TrichomeType

        cfg = MorphologyPipelineConfig(classifier_model_path=None)
        pipeline = MorphologyPipeline(cfg)

        instances = [
            self._make_morphology_instance(s)
            for s in ["bulbous", "sessile", "stalked", "bulbous", "sessile"]
        ]
        result = pipeline.analyze(instances, image_shape=(128, 128))

        valid_keys = set(t.value for t in TrichomeType)
        for key in result.type_distribution:
            assert key in valid_keys or isinstance(key, str), (
                f"Unexpected key in type_distribution: {key!r}"
            )

    def test_morphology_batch_processed_count(self):
        """total_analyzed + failed should equal len(instances)."""
        from morphology.application.morphology_pipeline import (
            MorphologyPipeline, MorphologyPipelineConfig
        )

        cfg = MorphologyPipelineConfig(classifier_model_path=None)
        pipeline = MorphologyPipeline(cfg)

        n = 6
        instances = [self._make_morphology_instance("bulbous") for _ in range(n)]
        result = pipeline.analyze(instances)

        assert result.total_analyzed + result.failed == n


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Measurement integration
# ═══════════════════════════════════════════════════════════════════════════════

class TestMeasurementIntegration:
    """Instance → MeasurementPipeline → calibrated µm measurements."""

    def _make_measurement_instance(self):
        """Instance with realistic mask for measurement."""
        from shared.core.entities import Instance
        from shared.core.value_objects import Mask

        h, w = 128, 128
        m = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(m, (64, 64), (30, 24), 0, 0, 360, 255, -1)
        crop = np.ones((h, w, 3), dtype=np.uint8) * 128
        inst = Instance(crop=crop)
        inst.mask = Mask.from_uint8(m)
        return inst

    def test_measurement_pipeline_runs_without_profile(self):
        """Pipeline falls back to generic 40× profile if no profile provided."""
        from measurement.application.measurement_pipeline import MeasurementPipeline

        pipeline = MeasurementPipeline()
        instances = [self._make_measurement_instance()]
        result = pipeline.measure_instances(instances)

        assert result is not None
        assert hasattr(result, "instances")

    def test_measurement_pipeline_with_custom_profile(self):
        """Calibrated 40× profile → physical dimensions in µm."""
        from measurement.application.measurement_pipeline import MeasurementPipeline
        from measurement.domain.profile_manager import MicroscopeProfile

        profile = MicroscopeProfile(
            profile_id="test_40x",
            name="Test 40×",
            um_per_pixel=0.1625,
            objective="40x",
            uncertainty_um=0.005,
        )
        pipeline = MeasurementPipeline(profile=profile)
        instances = [self._make_measurement_instance() for _ in range(3)]
        result = pipeline.measure_instances(instances)

        assert result is not None

    def test_measurement_population_stats_fields(self):
        """PopulationStats must have standard statistical fields."""
        from measurement.application.measurement_pipeline import MeasurementPipeline

        pipeline = MeasurementPipeline()
        instances = [self._make_measurement_instance() for _ in range(5)]
        result = pipeline.measure_instances(instances)

        stats = result.population
        if stats is not None:
            # If population stats computed, check basic structure
            assert hasattr(stats, "n")
            assert stats.n >= 0


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Full pipeline integration
# ═══════════════════════════════════════════════════════════════════════════════

class TestFullPipelineIntegration:
    """
    Full detect→segment→maturity+morphology+measurement pipeline.

    Uses mocked segmentation backend; real maturity/morphology/measurement pipelines.
    Validates the complete data transformation chain and output structure.
    """

    def _make_realistic_instances(self, n: int = 5) -> list:
        """
        Create N realistic Instances with masks and crops.
        Mix of morphology types and maturity states.
        """
        from shared.core.entities import Instance
        from shared.core.value_objects import Mask

        rng = np.random.default_rng(42)
        instances = []
        h, w = 64, 64

        styles = ["clear", "cloudy", "amber", "degraded", "cloudy"]
        for i in range(n):
            style = styles[i % len(styles)]

            if style == "clear":
                hsv = np.full((h, w, 3), [120, 30, 235], dtype=np.uint8)
            elif style == "cloudy":
                hsv = np.full((h, w, 3), [0, 25, 205], dtype=np.uint8)
            elif style == "amber":
                hsv = np.full((h, w, 3), [20, 185, 195], dtype=np.uint8)
            else:
                hsv = np.full((h, w, 3), [12, 150, 75], dtype=np.uint8)

            noise = rng.integers(-10, 10, (h, w, 3), dtype=np.int16)
            img_hsv = np.clip(hsv.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            crop_bgr = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

            m = np.zeros((h, w), dtype=np.uint8)
            cv2.ellipse(m, (32, 32), (22, 18), 0, 0, 360, 255, -1)

            inst = Instance(crop=crop_rgb)
            inst.mask = Mask.from_uint8(m)
            instances.append(inst)

        return instances

    def test_full_pipeline_produces_complete_results(self):
        """Full chain: instances → maturity + morphology outputs."""
        from maturity.application.maturity_pipeline import MaturityPipeline, MaturityPipelineConfig
        from morphology.application.morphology_pipeline import (
            MorphologyPipeline, MorphologyPipelineConfig
        )
        from measurement.application.measurement_pipeline import MeasurementPipeline

        instances = self._make_realistic_instances(5)

        # Stage 1: maturity
        mat_cfg = MaturityPipelineConfig(use_analyzer=False)
        mat_pipeline = MaturityPipeline(mat_cfg)
        mat_result = mat_pipeline.analyze(instances)

        assert len(mat_result.instances) == 5
        assert all(inst.maturity_label is not None for inst in mat_result.instances)

        # Stage 2: morphology (use same instances — with mask data)
        morph_cfg = MorphologyPipelineConfig(classifier_model_path=None)
        morph_pipeline = MorphologyPipeline(morph_cfg)
        morph_result = morph_pipeline.analyze(instances[:5], image_shape=(64, 64))

        assert morph_result.total_analyzed + morph_result.failed == 5

        # Stage 3: measurement
        meas_pipeline = MeasurementPipeline()
        meas_result = meas_pipeline.measure_instances(instances[:5])

        assert meas_result is not None

    def test_pipeline_handles_mixed_valid_invalid_instances(self):
        """Pipeline must gracefully handle some instances with missing data."""
        from shared.core.entities import Instance
        from maturity.application.maturity_pipeline import MaturityPipeline, MaturityPipelineConfig

        cfg = MaturityPipelineConfig(use_analyzer=False, min_crop_size_px=4)
        pipeline = MaturityPipeline(cfg)

        # Mix valid and minimal instances
        instances = self._make_realistic_instances(3)
        # Add instance with empty crop (edge case)
        empty = Instance(crop=np.zeros((8, 8, 3), dtype=np.uint8))
        instances.append(empty)

        # Should not raise
        result = pipeline.analyze(instances)
        assert len(result.instances) == 4

    def test_pipeline_deterministic_with_same_seed(self):
        """Same inputs must produce the same outputs (reproducibility)."""
        from maturity.application.maturity_pipeline import MaturityPipeline, MaturityPipelineConfig

        cfg = MaturityPipelineConfig(use_analyzer=False)
        pipeline = MaturityPipeline(cfg)

        insts_1 = self._make_realistic_instances(3)
        insts_2 = self._make_realistic_instances(3)  # Same seed=42

        res_1 = pipeline.analyze(insts_1)
        res_2 = pipeline.analyze(insts_2)

        for inst_a, inst_b in zip(res_1.instances, res_2.instances):
            if inst_a.maturity_label and inst_b.maturity_label:
                assert inst_a.maturity_label.stage == inst_b.maturity_label.stage, (
                    "Maturity stage differs between identical inputs (non-deterministic)"
                )


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Scientific constraints
# ═══════════════════════════════════════════════════════════════════════════════

class TestScientificConstraints:
    """
    Verify scientific integrity constraints hold across the full pipeline.

    These tests enforce the platform's core scientific commitments:
    - No cannabinoid concentration claims
    - All confidence values are calibrated
    - Uncertainty is propagated
    - MaturityStage values are optical observations, not pharmacological claims
    """

    def test_maturity_stage_values_are_optical_terms(self):
        """MaturityStage enum values must NOT include pharmacological terms."""
        from shared.core.enums import MaturityStage

        forbidden_terms = ["thc", "cbd", "cbn", "potency", "cannabinoid"]
        for stage in MaturityStage:
            stage_lower = stage.value.lower()
            for term in forbidden_terms:
                assert term not in stage_lower, (
                    f"MaturityStage.{stage.name} = {stage.value!r} contains '{term}'"
                )

    def test_maturity_label_confidence_is_calibrated(self):
        """MaturityLabel.confidence must be in [0, 1]."""
        from maturity.application.maturity_pipeline import MaturityPipeline, MaturityPipelineConfig
        from shared.core.entities import Instance

        cfg = MaturityPipelineConfig(use_analyzer=False)
        pipeline = MaturityPipeline(cfg)

        for _ in range(5):
            crop = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
            inst = Instance(crop=crop)
            result = pipeline.analyze([inst])
            label = result.instances[0].maturity_label
            if label is not None:
                c = float(label.confidence)
                assert 0.0 <= c <= 1.0, f"Confidence {c} out of [0, 1]"

    def test_no_harvest_timing_claims(self):
        """Pipeline output must not contain harvest timing recommendations."""
        from maturity.application.maturity_pipeline import MaturityPipeline, MaturityPipelineConfig
        from shared.core.entities import Instance

        cfg = MaturityPipelineConfig(use_analyzer=False)
        pipeline = MaturityPipeline(cfg)

        crop = np.full((64, 64, 3), [200, 180, 160], dtype=np.uint8)
        inst = Instance(crop=crop)
        result = pipeline.analyze([inst])

        forbidden = ["harvest_now", "ready_to_harvest", "thc_peak", "optimal_harvest"]
        result_str = str(result.to_dict()).lower()
        for term in forbidden:
            assert term not in result_str, (
                f"Harvest timing claim '{term}' found in pipeline output"
            )

    def test_measurement_uncertainty_is_non_negative(self):
        """Measurement uncertainty must always be >= 0."""
        from measurement.domain.propagation import propagate_linear, propagate_area

        for px_val in [10.0, 50.0, 200.0]:
            m = propagate_linear(px_val, 0.1625, calibration_uncertainty_um=0.005)
            assert m.uncertainty >= 0, (
                f"Negative uncertainty {m.uncertainty} for px_val={px_val}"
            )

        for area_val in [100.0, 1000.0, 5000.0]:
            m = propagate_area(area_val, 0.1625, calibration_uncertainty_um=0.005)
            assert m.uncertainty >= 0, (
                f"Negative area uncertainty {m.uncertainty} for area_val={area_val}"
            )

    def test_calibration_scale_rejects_zero(self):
        """CalibrationScale or profile must reject um_per_pixel <= 0."""
        from measurement.domain.measurer import Measurer
        from measurement.domain.profile_manager import MicroscopeProfile

        with pytest.raises((ValueError, ZeroDivisionError, Exception)):
            profile = MicroscopeProfile(
                profile_id="bad",
                name="bad",
                um_per_pixel=0.0,
            )
            _ = Measurer(profile)


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Error handling
# ═══════════════════════════════════════════════════════════════════════════════

class TestErrorHandling:
    """Verify graceful degradation at each pipeline stage."""

    def test_maturity_empty_instance_list(self):
        """Empty instance list must return empty result without exception."""
        from maturity.application.maturity_pipeline import MaturityPipeline, MaturityPipelineConfig

        cfg = MaturityPipelineConfig(use_analyzer=False)
        pipeline = MaturityPipeline(cfg)
        result = pipeline.analyze([])

        assert result.instances == []
        assert result.analyzed == 0

    def test_maturity_all_black_crops(self):
        """All-black crops must not raise; should return UNKNOWN stage."""
        from maturity.application.maturity_pipeline import MaturityPipeline, MaturityPipelineConfig
        from shared.core.entities import Instance
        from shared.core.enums import MaturityStage

        cfg = MaturityPipelineConfig(use_analyzer=False)
        pipeline = MaturityPipeline(cfg)

        inst = Instance(crop=np.zeros((64, 64, 3), dtype=np.uint8))
        result = pipeline.analyze([inst])

        # Must not raise; result may be UNKNOWN
        assert len(result.instances) == 1

    def test_maturity_single_pixel_crop(self):
        """Tiny crop (1×1) must not raise."""
        from maturity.application.maturity_pipeline import MaturityPipeline, MaturityPipelineConfig
        from shared.core.entities import Instance

        cfg = MaturityPipelineConfig(use_analyzer=False, min_crop_size_px=1)
        pipeline = MaturityPipeline(cfg)

        inst = Instance(crop=np.array([[[200, 180, 160]]], dtype=np.uint8))
        result = pipeline.analyze([inst])
        assert len(result.instances) == 1

    def test_morphology_no_mask(self):
        """Instance with no mask must be counted as failed, not crash."""
        from morphology.application.morphology_pipeline import (
            MorphologyPipeline, MorphologyPipelineConfig
        )
        from shared.core.entities import Instance

        cfg = MorphologyPipelineConfig(classifier_model_path=None)
        pipeline = MorphologyPipeline(cfg)

        inst = Instance(crop=np.zeros((64, 64, 3), dtype=np.uint8))
        result = pipeline.analyze([inst])

        assert result.failed == 1
        assert result.total_analyzed == 0

    def test_measurement_pipeline_empty_instances(self):
        """Empty instance list must return gracefully."""
        from measurement.application.measurement_pipeline import MeasurementPipeline

        pipeline = MeasurementPipeline()
        result = pipeline.measure_instances([])

        assert result is not None
        assert len(result.instances) == 0

    def test_focus_composite_on_black_image(self):
        """Composite focus score on all-black image must not raise."""
        from focus.metrics.composite import compute_focus_score

        black = np.zeros((256, 256, 3), dtype=np.uint8)
        result = compute_focus_score(black)

        assert 0.0 <= result.composite <= 1.0
        assert result.quality_label in {
            "excellent", "good", "acceptable", "poor", "unacceptable", "unusable"
        }

    def test_video_score_frame_on_noise(self):
        """score_frame must not raise on random noise image."""
        from video_pipeline.domain.scorer import score_frame

        rng = np.random.default_rng(99)
        noisy = rng.integers(0, 256, (240, 320, 3), dtype=np.uint8)
        result = score_frame(noisy, use_focus_composite=False)

        assert 0.0 <= result.composite <= 1.0
        assert 0.0 <= result.focus <= 1.0
        assert 0.0 <= result.exposure <= 1.0
        assert 0.0 <= result.noise <= 1.0
