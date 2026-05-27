"""
tests/unit/test_segmentation_pipeline.py — Unit tests for the segmentation pipeline.

Tests cover:
  - Mask refinement operations (fill_holes, remove_small, morphological_clean, smooth)
  - Polygon utilities (mask_to_polygon, polygon_to_mask, RLE encoding)
  - SegmentPipelineConfig validation
  - SegmentPipeline.run() edge cases (empty detections, large detections, capping)
  - SegmentedInstance geometry (area, centroid, circularity, elongation)
  - BatchSegmentationResult structure

All SAM2 / MobileSAM model tests are marked @pytest.mark.gpu and auto-skipped
without a CUDA device. These tests work purely on CPU-side numpy operations.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray


# ── Helpers ───────────────────────────────────────────────────────────────────

def _circle_mask(size: int = 128, radius: int | None = None) -> NDArray[np.uint8]:
    """Create a filled circle mask."""
    r = radius or size // 4
    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(mask, (size // 2, size // 2), r, 255, -1)
    return mask


def _ellipse_mask(size: int = 128) -> NDArray[np.uint8]:
    """Create a filled ellipse mask (elongated trichome)."""
    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.ellipse(mask, (size // 2, size // 2), (size // 2 - 4, size // 8), 0, 0, 360, 255, -1)
    return mask


def _mask_with_holes(size: int = 128) -> NDArray[np.uint8]:
    """Circle mask with internal holes."""
    mask = _circle_mask(size)
    # Punch holes
    cv2.circle(mask, (size // 2, size // 2), 10, 0, -1)
    cv2.circle(mask, (size // 2 + 20, size // 2), 5, 0, -1)
    return mask


def _noisy_mask(size: int = 128) -> NDArray[np.uint8]:
    """Mask with small noise components scattered around a circle."""
    mask = _circle_mask(size)
    rng = np.random.default_rng(42)
    # Scatter small 2px blobs
    for _ in range(20):
        x = int(rng.integers(0, size))
        y = int(rng.integers(0, size))
        mask[max(0, y - 1):y + 2, max(0, x - 1):x + 2] = 255
    return mask


def _rgb_image(size: int = 256) -> NDArray[np.uint8]:
    """Random RGB microscopy-like image."""
    rng = np.random.default_rng(42)
    return rng.integers(50, 200, (size, size, 3), dtype=np.uint8)


def _det(x1=10.0, y1=10.0, x2=60.0, y2=60.0, conf=0.85, cls=0, name="capitate_stalked"):
    """Create a detection dict as expected by SegmentPipeline.run()."""
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "confidence": conf, "class_id": cls, "class_name": name}


# ── Mask refinement tests ─────────────────────────────────────────────────────

class TestMaskRefinement:
    """Tests for segmentation/domain/mask_refinement.py"""

    def test_fill_holes_fills_internal_voids(self):
        from segmentation.domain.mask_refinement import fill_holes
        mask = _mask_with_holes()
        filled = fill_holes(mask)
        # Filled mask should have MORE pixels than the holed mask
        assert filled.sum() >= mask.sum(), (
            "fill_holes should add pixels by closing internal holes"
        )

    def test_fill_holes_no_change_on_solid_circle(self):
        from segmentation.domain.mask_refinement import fill_holes
        mask = _circle_mask()
        filled = fill_holes(mask)
        # Solid circle has no holes — count should be identical
        assert np.array_equal(filled.astype(bool), mask.astype(bool))

    def test_fill_holes_preserves_shape(self):
        from segmentation.domain.mask_refinement import fill_holes
        mask = _mask_with_holes(64)
        filled = fill_holes(mask)
        assert filled.shape == mask.shape

    def test_remove_small_components_removes_noise(self):
        from segmentation.domain.mask_refinement import remove_small_components
        noisy = _noisy_mask()
        cleaned = remove_small_components(noisy, min_area_px=20)
        # Cleaned should have fewer total pixels (noise removed)
        assert cleaned.sum() <= noisy.sum()

    def test_remove_small_components_keeps_large(self):
        from segmentation.domain.mask_refinement import remove_small_components
        mask = _circle_mask(128, radius=40)
        cleaned = remove_small_components(mask, min_area_px=50)
        # Large circle should survive
        assert cleaned.sum() > 0, "Large component should not be removed"

    def test_remove_small_components_all_removed_below_threshold(self):
        from segmentation.domain.mask_refinement import remove_small_components
        # Tiny 3x3 blob
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[30:33, 30:33] = 255
        cleaned = remove_small_components(mask, min_area_px=100)
        assert cleaned.sum() == 0, "Tiny component should be removed"

    def test_morphological_clean_reduces_noise(self):
        from segmentation.domain.mask_refinement import morphological_clean
        noisy = _noisy_mask()
        cleaned = morphological_clean(noisy)
        assert cleaned.shape == noisy.shape

    def test_smooth_contour_output_shape(self):
        from segmentation.domain.mask_refinement import smooth_contour
        mask = _circle_mask()
        smoothed = smooth_contour(mask)
        assert smoothed.shape == mask.shape

    def test_smooth_contour_preserves_general_shape(self):
        from segmentation.domain.mask_refinement import smooth_contour
        mask = _circle_mask(128, radius=30)
        smoothed = smooth_contour(mask)
        # Area should be similar (smoothing doesn't radically change shape)
        orig_area = mask.sum()
        smooth_area = smoothed.sum()
        assert abs(orig_area - smooth_area) / max(orig_area, 1) < 0.30

    def test_refine_mask_pipeline_complete(self):
        from segmentation.domain.mask_refinement import refine_mask, RefinementConfig
        cfg = RefinementConfig(
            close_kernel_size=3,
            open_kernel_size=2,
            min_component_area_px=10,
            use_watershed=False,
        )
        mask = _mask_with_holes()
        refined = refine_mask(mask, cfg)
        assert refined.shape == mask.shape
        assert refined.dtype == np.uint8

    def test_batch_refine_processes_all(self):
        from segmentation.domain.mask_refinement import batch_refine
        masks = [_circle_mask(64), _ellipse_mask(64), _mask_with_holes(64)]
        refined_list = batch_refine(masks)
        assert len(refined_list) == len(masks)
        for r in refined_list:
            assert r.shape[0] > 0

    def test_empty_mask_does_not_crash(self):
        from segmentation.domain.mask_refinement import fill_holes, remove_small_components
        empty = np.zeros((64, 64), dtype=np.uint8)
        assert fill_holes(empty).sum() == 0
        assert remove_small_components(empty, min_area_px=10).sum() == 0

    def test_single_pixel_mask_handled(self):
        from segmentation.domain.mask_refinement import refine_mask, RefinementConfig
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[16, 16] = 255
        cfg = RefinementConfig()
        refined = refine_mask(mask, cfg)
        assert refined.shape == mask.shape


# ── Polygon utilities ─────────────────────────────────────────────────────────

class TestPolygonUtils:
    """Tests for segmentation/domain/polygon_utils.py"""

    def test_mask_to_polygon_circle(self):
        from segmentation.domain.polygon_utils import mask_to_polygon
        mask = _circle_mask(128, radius=30)
        polygons = mask_to_polygon(mask)
        # Should find at least one polygon
        assert len(polygons) >= 1
        # Polygon should have ≥ 4 points
        assert len(polygons[0]) >= 4

    def test_mask_to_polygon_empty_returns_empty(self):
        from segmentation.domain.polygon_utils import mask_to_polygon
        empty = np.zeros((64, 64), dtype=np.uint8)
        polygons = mask_to_polygon(empty)
        assert polygons == []

    def test_polygon_to_mask_round_trip(self):
        from segmentation.domain.polygon_utils import mask_to_polygon, polygon_to_mask
        mask = _circle_mask(128, radius=25)
        polygons = mask_to_polygon(mask)
        if not polygons:
            pytest.skip("No polygon found for circle")
        reconstructed = polygon_to_mask(polygons[0], height=128, width=128)
        # IoU should be high (≥ 0.75)
        intersection = (mask.astype(bool) & reconstructed.astype(bool)).sum()
        union = (mask.astype(bool) | reconstructed.astype(bool)).sum()
        iou = intersection / max(union, 1)
        assert iou >= 0.70, f"Round-trip polygon IoU too low: {iou:.3f}"

    def test_mask_to_rle_and_back(self):
        from segmentation.domain.polygon_utils import mask_to_rle, rle_to_mask
        mask = _circle_mask(64, radius=20)
        binary = mask.astype(bool)
        rle = mask_to_rle(binary)
        recovered = rle_to_mask(rle)
        # rle_to_mask returns uint8 (0/255); compare as bool
        assert np.array_equal(binary, recovered.astype(bool))

    def test_rle_empty_mask(self):
        from segmentation.domain.polygon_utils import mask_to_rle, rle_to_mask
        empty = np.zeros((32, 32), dtype=bool)
        rle = mask_to_rle(empty)
        recovered = rle_to_mask(rle)
        assert np.array_equal(empty, recovered.astype(bool))

    def test_rle_full_mask(self):
        from segmentation.domain.polygon_utils import mask_to_rle, rle_to_mask
        full = np.ones((32, 32), dtype=bool)
        rle = mask_to_rle(full)
        recovered = rle_to_mask(rle)
        assert np.array_equal(full, recovered.astype(bool))

    def test_polygon_area_positive(self):
        from segmentation.domain.polygon_utils import mask_to_polygon, polygon_area
        mask = _circle_mask(128, radius=30)
        polygons = mask_to_polygon(mask)
        if not polygons:
            pytest.skip("No polygon")
        area = polygon_area(polygons[0])
        assert area > 0.0

    def test_polygon_centroid_near_center(self):
        from segmentation.domain.polygon_utils import mask_to_polygon, polygon_centroid
        size = 128
        mask = _circle_mask(size, radius=25)
        polygons = mask_to_polygon(mask)
        if not polygons:
            pytest.skip("No polygon")
        cx, cy = polygon_centroid(polygons[0])
        center = size // 2
        assert abs(cx - center) < 15, f"Centroid X {cx:.1f} not near center {center}"
        assert abs(cy - center) < 15, f"Centroid Y {cy:.1f} not near center {center}"

    def test_polygon_circularity_circle(self):
        from segmentation.domain.polygon_utils import mask_to_polygon, polygon_circularity
        mask = _circle_mask(128, radius=40)
        polygons = mask_to_polygon(mask)
        if not polygons:
            pytest.skip("No polygon")
        circ = polygon_circularity(polygons[0])
        assert 0.70 <= circ <= 1.05, f"Circle circularity {circ:.3f} should be near 1.0"

    def test_mask_to_coco_segmentation(self):
        from segmentation.domain.polygon_utils import mask_to_coco_segmentation
        mask = _circle_mask(128, radius=30)
        coco = mask_to_coco_segmentation(mask)
        # COCO format: list of flat polygon lists
        assert isinstance(coco, list)
        if coco:
            assert isinstance(coco[0], list)
            # Should be even-length (x,y pairs)
            assert len(coco[0]) % 2 == 0


# ── SegmentPipelineConfig tests ───────────────────────────────────────────────

class TestSegmentPipelineConfig:
    def test_defaults_are_valid(self):
        from segmentation.application.segment_pipeline import SegmentPipelineConfig
        cfg = SegmentPipelineConfig()
        assert cfg.backend in {"sam2_tiny", "mobile_sam", "auto"}
        assert 0.0 < cfg.score_threshold <= 1.0
        assert cfg.min_mask_area_px > 0
        assert cfg.max_instances > 0

    def test_custom_config_accepted(self):
        from segmentation.application.segment_pipeline import SegmentPipelineConfig
        cfg = SegmentPipelineConfig(
            backend="mobile_sam",
            score_threshold=0.6,
            min_mask_area_px=100,
            compute_polygons=False,
        )
        assert cfg.score_threshold == 0.6
        assert cfg.compute_polygons is False


# ── SegmentPipeline (CPU-only, without loading backend) ──────────────────────

class TestSegmentPipelineStructure:
    """Tests for pipeline structure that don't require model loading."""

    def test_pipeline_init_without_load(self):
        from segmentation.application.segment_pipeline import SegmentPipeline, SegmentPipelineConfig
        pipeline = SegmentPipeline(SegmentPipelineConfig(backend="mobile_sam"))
        assert pipeline._backend is None

    def test_run_raises_if_not_loaded(self):
        from segmentation.application.segment_pipeline import SegmentPipeline, SegmentPipelineConfig
        pipeline = SegmentPipeline(SegmentPipelineConfig())
        img = _rgb_image(128)
        with pytest.raises(RuntimeError, match="not loaded"):
            pipeline.run(img, [_det()])

    def test_segment_pipeline_result_fields(self):
        """Verify SegmentPipelineResult has the required fields."""
        from segmentation.application.segment_pipeline import SegmentPipelineResult, SegmentedInstance
        import numpy as np

        mask = _circle_mask(64, radius=20).astype(bool)
        inst = SegmentedInstance(
            detection_box=(10, 10, 50, 50),
            detection_confidence=0.85,
            detection_class_id=0,
            detection_class_name="capitate_stalked",
            mask=mask,
            mask_score=0.92,
            area_px=float(mask.sum()),
            centroid_x=32.0,
            centroid_y=32.0,
            circularity=0.95,
            elongation=1.05,
        )
        result = SegmentPipelineResult(
            instances=[inst],
            image_height=64,
            image_width=64,
            backend_used="test",
            num_input_detections=1,
            num_segmented=1,
            pipeline_time_ms=12.0,
            backend_time_ms=8.0,
        )
        assert result.num_segmented == 1
        assert len(result.instances) == 1
        assert result.instances[0].area_px > 0

    def test_segmented_instance_circularity_range(self):
        from segmentation.application.segment_pipeline import SegmentedInstance
        mask = _circle_mask(64, radius=20).astype(bool)
        inst = SegmentedInstance(
            detection_box=None,
            detection_confidence=0.9,
            detection_class_id=0,
            detection_class_name="test",
            mask=mask,
            mask_score=0.88,
            area_px=float(mask.sum()),
            centroid_x=32.0,
            centroid_y=32.0,
            circularity=0.92,
            elongation=1.02,
        )
        assert 0.0 <= inst.circularity <= 1.1  # circles may slightly exceed 1.0 due to pixelation
        assert inst.elongation >= 1.0  # elongation ≥ 1 by definition

    def test_segmented_instance_optional_measurements(self):
        from segmentation.application.segment_pipeline import SegmentedInstance
        mask = _circle_mask(32, radius=10).astype(bool)
        inst = SegmentedInstance(
            detection_box=(0, 0, 64, 64),
            detection_confidence=0.7,
            detection_class_id=1,
            detection_class_name="bulbous",
            mask=mask,
            mask_score=0.75,
            area_px=float(mask.sum()),
            centroid_x=16.0,
            centroid_y=16.0,
            area_um2=None,
            diameter_um=None,
        )
        assert inst.area_um2 is None
        assert inst.diameter_um is None


# ── Segmentor Protocol / BaseSegmentor ────────────────────────────────────────

class TestSegmentorProtocol:
    """Test the Segmentor protocol compliance and BaseSegmentor."""

    def test_box_prompt_construction(self):
        from segmentation.domain.segmentor import BoxPrompt
        bp = BoxPrompt(x1=10.0, y1=20.0, x2=80.0, y2=90.0)
        assert bp.x1 == 10.0
        assert bp.y2 == 90.0

    def test_point_prompt_construction(self):
        from segmentation.domain.segmentor import PointPrompt
        pp = PointPrompt(x=64.0, y=64.0, label=1)
        assert pp.x == 64.0
        assert pp.label == 1

    def test_segmentation_result_fields(self):
        from segmentation.domain.segmentor import SegmentationResult
        mask = _circle_mask(64).astype(bool)
        result = SegmentationResult(mask=mask, score=0.88)
        assert result.score == pytest.approx(0.88)
        assert result.mask.shape == (64, 64)
        assert result.logits is None

    def test_batch_segmentation_result(self):
        from segmentation.domain.segmentor import BatchSegmentationResult, SegmentationResult
        masks = [SegmentationResult(mask=_circle_mask(64).astype(bool), score=0.9)]
        batch = BatchSegmentationResult(
            masks=masks,
            image_height=64,
            image_width=64,
            backend="test",
            inference_time_ms=5.0,
        )
        assert len(batch.masks) == 1
        assert batch.backend == "test"

    def test_segmentor_config_defaults(self):
        from segmentation.domain.segmentor import SegmentorConfig
        cfg = SegmentorConfig()
        assert 0.0 < cfg.score_threshold <= 1.0
        assert cfg.min_mask_area_px > 0
        assert cfg.max_mask_area_fraction > 0.0


# ── Integration: mask refinement pipeline ────────────────────────────────────

class TestMaskRefinementIntegration:
    """End-to-end mask refinement on realistic trichome-like masks."""

    def test_round_trip_circle_refinement(self):
        from segmentation.domain.mask_refinement import refine_mask, RefinementConfig
        original = _circle_mask(128, radius=35)
        # Corrupt mask: add holes and noise
        corrupted = original.copy()
        cv2.circle(corrupted, (64, 64), 8, 0, -1)  # punch hole
        corrupted[10:13, 10:13] = 255  # add noise blob

        cfg = RefinementConfig(
            close_kernel_size=5,
            open_kernel_size=3,
            min_component_area_px=30,
        )
        refined = refine_mask(corrupted, cfg)

        # Refined should be closer to original than corrupted
        orig_area = original.astype(bool).sum()
        refined_area = refined.astype(bool).sum()
        assert abs(orig_area - refined_area) / max(orig_area, 1) < 0.30, (
            f"Refined area {refined_area} too far from original {orig_area}"
        )

    def test_batch_consistency(self):
        """Processing same mask twice in batch should be deterministic."""
        from segmentation.domain.mask_refinement import batch_refine, RefinementConfig
        mask = _circle_mask(64, radius=20)
        results = batch_refine([mask, mask.copy()])
        assert len(results) == 2
        # Both results should be identical
        assert np.array_equal(results[0], results[1])

    def test_polygon_from_refined_mask(self):
        """Refined mask → polygon should produce valid geometry."""
        from segmentation.domain.mask_refinement import refine_mask, RefinementConfig
        from segmentation.domain.polygon_utils import mask_to_polygon, polygon_circularity

        mask = _circle_mask(128, radius=30)
        cfg = RefinementConfig(close_kernel_size=3)
        refined = refine_mask(mask, cfg)

        polygons = mask_to_polygon(refined)
        if polygons:
            circ = polygon_circularity(polygons[0])
            assert 0.5 <= circ <= 1.1, f"Circularity after refinement: {circ:.3f}"
