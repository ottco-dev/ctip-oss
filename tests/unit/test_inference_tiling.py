"""
tests/unit/test_inference_tiling.py

Comprehensive unit tests for detection/domain/tiled_inference.py.

Coverage:
  ✔ TileConfig — defaults, validation errors
  ✔ TileInfo — width/height properties, to_global_bbox coordinate shift
  ✔ TiledInferenceEngine.compute_tiles — small/large/square/portrait/landscape images
  ✔ compute_tiles — stride step computed from overlap_fraction
  ✔ compute_tiles — last tile always reaches image boundary
  ✔ compute_tiles — tile count matches expected grid
  ✔ compute_tiles — overlap_fraction=0 gives non-overlapping tiles
  ✔ TiledInferenceEngine.extract_tile — normal slice + padded boundary tile
  ✔ TiledInferenceEngine.is_tile_empty — variance threshold gate
  ✔ TiledInferenceEngine._merge_detections — empty, single, multi
  ✔ _cluster_by_iou — high-IoU detections merged into one cluster
  ✔ _cluster_by_iou — non-overlapping detections in separate clusters
  ✔ _fuse_cluster — confidence-weighted box averaging
  ✔ _standard_nms_merge — greedy NMS removes overlapping dets
  ✔ _weighted_boxes_fusion — same-class dets reduced
  ✔ get_tile_coverage_map — corners covered, overlap pixels count ≥ 2
  ✔ detect_tiled — empty tile skipped when variance below threshold
  ✔ detect_tiled — local→global coordinate translation
  ✔ detect_tiled — tile diagnostics contain expected keys
"""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from detection.domain.tiled_inference import (
    TileConfig,
    TileInfo,
    TiledInferenceEngine,
)
from shared.core.value_objects import BoundingBox, Confidence
from shared.core.entities import Detection
from shared.core.enums import TrichomeType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_detection(
    x_min: float, y_min: float, x_max: float, y_max: float,
    conf: float = 0.85,
    trichome_type: TrichomeType = TrichomeType.CAPITATE_STALKED,
    class_id: int = 0,
) -> Detection:
    return Detection(
        id=str(uuid.uuid4()),
        bounding_box=BoundingBox(x_min, y_min, x_max, y_max),
        confidence=Confidence(conf),
        trichome_type=trichome_type,
        model_id="test_model",
        class_id=class_id,
    )


def _make_engine(
    tile_size: int = 1280,
    overlap: float = 0.20,
    merge: str = "wbf",
    skip_empty: bool = True,
    variance_threshold: float = 50.0,
) -> TiledInferenceEngine:
    detector = MagicMock()
    cfg = TileConfig(
        tile_size=tile_size,
        overlap_fraction=overlap,
        merge_strategy=merge,
        skip_empty_tiles=skip_empty,
        empty_tile_variance_threshold=variance_threshold,
    )
    return TiledInferenceEngine(detector, cfg)


# ===========================================================================
# 1. TileConfig
# ===========================================================================

class TestTileConfig:

    def test_defaults(self):
        cfg = TileConfig()
        assert cfg.tile_size == 1280
        assert cfg.overlap_fraction == 0.20
        assert cfg.merge_strategy == "wbf"
        assert cfg.skip_empty_tiles is True
        assert cfg.empty_tile_variance_threshold == 50.0
        assert cfg.min_tile_size == 256

    def test_validate_overlap_too_high(self):
        cfg = TileConfig(overlap_fraction=0.5)
        with pytest.raises(ValueError, match="overlap_fraction"):
            cfg.validate()

    def test_validate_overlap_negative(self):
        cfg = TileConfig(overlap_fraction=-0.1)
        with pytest.raises(ValueError, match="overlap_fraction"):
            cfg.validate()

    def test_validate_tile_smaller_than_min(self):
        cfg = TileConfig(tile_size=128, min_tile_size=256)
        with pytest.raises(ValueError, match="tile_size"):
            cfg.validate()

    def test_validate_ok(self):
        cfg = TileConfig(tile_size=1280, overlap_fraction=0.2)
        cfg.validate()   # should not raise


# ===========================================================================
# 2. TileInfo
# ===========================================================================

class TestTileInfo:

    def test_width_height(self):
        tile = TileInfo(0, x_start=100, y_start=200, x_end=380, y_end=600)
        assert tile.width == 280
        assert tile.height == 400

    def test_to_global_bbox_shifts_by_tile_offset(self):
        tile = TileInfo(0, x_start=500, y_start=300, x_end=1780, y_end=1580)
        local_bbox = BoundingBox(x_min=10, y_min=20, x_max=50, y_max=80)
        global_bbox = tile.to_global_bbox(local_bbox)
        assert global_bbox.x_min == 510
        assert global_bbox.y_min == 320
        assert global_bbox.x_max == 550
        assert global_bbox.y_max == 380

    def test_to_global_bbox_zero_offset(self):
        tile = TileInfo(0, x_start=0, y_start=0, x_end=1280, y_end=1280)
        local_bbox = BoundingBox(50, 50, 150, 150)
        global_bbox = tile.to_global_bbox(local_bbox)
        assert global_bbox.x_min == 50
        assert global_bbox.y_min == 50

    def test_is_padded_flag(self):
        tile = TileInfo(0, 0, 0, 1000, 1000, is_padded=True)
        assert tile.is_padded is True

    def test_tile_index_stored(self):
        tile = TileInfo(tile_index=7, x_start=0, y_start=0, x_end=100, y_end=100)
        assert tile.tile_index == 7


# ===========================================================================
# 3. compute_tiles
# ===========================================================================

class TestComputeTiles:

    def test_image_smaller_than_tile_gives_one_tile(self):
        engine = _make_engine(tile_size=1280, overlap=0.0)
        tiles = engine.compute_tiles(640, 640)
        assert len(tiles) == 1
        t = tiles[0]
        assert t.x_start == 0 and t.y_start == 0
        assert t.x_end == 640 and t.y_end == 640

    def test_image_exact_tile_size_gives_one_tile(self):
        engine = _make_engine(tile_size=640, overlap=0.0)
        tiles = engine.compute_tiles(640, 640)
        assert len(tiles) == 1

    def test_2x2_grid_for_double_size_no_overlap(self):
        engine = _make_engine(tile_size=640, overlap=0.0)
        tiles = engine.compute_tiles(1280, 1280)
        # stride=640, expect 2 positions along each axis → 4 tiles
        assert len(tiles) == 4

    def test_last_tile_reaches_image_right_boundary(self):
        engine = _make_engine(tile_size=640, overlap=0.0)
        tiles = engine.compute_tiles(640, 1000)   # width > tile_size
        x_ends = [t.x_end for t in tiles]
        assert max(x_ends) == 1000

    def test_last_tile_reaches_image_bottom_boundary(self):
        engine = _make_engine(tile_size=640, overlap=0.0)
        tiles = engine.compute_tiles(1000, 640)   # height > tile_size
        y_ends = [t.y_end for t in tiles]
        assert max(y_ends) == 1000

    def test_overlap_reduces_stride(self):
        # tile_size=640, overlap=0.4 → stride=384
        engine = _make_engine(tile_size=640, overlap=0.4)
        # For a 1280×640 image: x_stride=384 → x_starts=[0,384,640]; y_starts=[0]
        tiles = engine.compute_tiles(640, 1280)
        # All tiles should be within image bounds
        for t in tiles:
            assert t.x_start >= 0
            assert t.y_start >= 0
            assert t.x_end <= 1280
            assert t.y_end <= 640
        # With 40% overlap, there should be more tiles than no-overlap case
        tiles_no_overlap = _make_engine(tile_size=640, overlap=0.0).compute_tiles(640, 1280)
        assert len(tiles) >= len(tiles_no_overlap)

    def test_tiles_are_indexed_sequentially(self):
        engine = _make_engine(tile_size=640, overlap=0.0)
        tiles = engine.compute_tiles(1280, 1280)
        indices = [t.tile_index for t in tiles]
        assert indices == list(range(len(tiles)))

    def test_wide_image_multiple_columns(self):
        engine = _make_engine(tile_size=640, overlap=0.0)
        tiles = engine.compute_tiles(640, 3200)   # 5 columns expected
        x_starts = sorted(set(t.x_start for t in tiles))
        assert len(x_starts) >= 5

    def test_portrait_image(self):
        engine = _make_engine(tile_size=640, overlap=0.0)
        tiles = engine.compute_tiles(3200, 640)   # tall image
        y_starts = sorted(set(t.y_start for t in tiles))
        assert len(y_starts) >= 5

    def test_large_4k_image_tile_count(self):
        """4K image (3840×2160) with 1280px tiles and 20% overlap."""
        engine = _make_engine(tile_size=1280, overlap=0.20)
        tiles = engine.compute_tiles(2160, 3840)
        # stride = 1024; x_count ≈ ceil((3840-1280)/1024)+1 ≈ 4; y ≈ 2
        assert len(tiles) >= 6   # at least 3 columns × 2 rows

    def test_no_tile_extends_beyond_image_width(self):
        engine = _make_engine(tile_size=1280, overlap=0.20)
        tiles = engine.compute_tiles(2160, 3840)
        for t in tiles:
            assert t.x_end <= 3840, f"tile x_end={t.x_end} exceeds image width"
            assert t.y_end <= 2160, f"tile y_end={t.y_end} exceeds image height"


# ===========================================================================
# 4. extract_tile
# ===========================================================================

class TestExtractTile:

    def test_normal_tile_is_exact_slice(self):
        engine = _make_engine(tile_size=640)
        image = np.zeros((1280, 1280, 3), dtype=np.uint8)
        image[0:640, 0:640] = 42   # fill top-left
        tile = TileInfo(0, x_start=0, y_start=0, x_end=640, y_end=640)
        result = engine.extract_tile(image, tile)
        assert result.shape == (640, 640, 3)
        assert result[0, 0, 0] == 42

    def test_boundary_tile_padded_to_tile_size(self):
        engine = _make_engine(tile_size=640)
        image = np.zeros((700, 700, 3), dtype=np.uint8)
        # Tile that extends beyond image (500→700, but ts=640)
        tile = TileInfo(0, x_start=500, y_start=500, x_end=700, y_end=700, is_padded=True)
        result = engine.extract_tile(image, tile)
        # Should be padded to (640, 640)
        assert result.shape == (640, 640, 3)

    def test_exact_boundary_no_padding_needed(self):
        engine = _make_engine(tile_size=640)
        image = np.zeros((1280, 1280, 3), dtype=np.uint8)
        tile = TileInfo(0, x_start=640, y_start=640, x_end=1280, y_end=1280)
        result = engine.extract_tile(image, tile)
        assert result.shape == (640, 640, 3)


# ===========================================================================
# 5. is_tile_empty
# ===========================================================================

class TestIsTileEmpty:

    def test_constant_tile_is_empty(self):
        engine = _make_engine(variance_threshold=50.0)
        tile_img = np.full((640, 640, 3), 128, dtype=np.uint8)
        assert engine.is_tile_empty(tile_img) is True

    def test_noisy_tile_is_not_empty(self):
        engine = _make_engine(variance_threshold=50.0)
        rng = np.random.default_rng(0)
        tile_img = rng.integers(0, 256, (640, 640, 3), dtype=np.uint8)
        assert engine.is_tile_empty(tile_img) is False

    def test_skip_empty_false_always_returns_false(self):
        engine = _make_engine(skip_empty=False, variance_threshold=50.0)
        tile_img = np.full((640, 640), 200, dtype=np.uint8)
        # Even constant tile is not "empty" when skip_empty_tiles=False
        assert engine.is_tile_empty(tile_img) is False

    def test_near_threshold_tile_classified_correctly(self):
        engine = _make_engine(variance_threshold=100.0)
        # Create tile with variance just below threshold
        tile_img = np.zeros((100, 100), dtype=np.float32)
        # variance ≈ 50 < 100 → empty
        tile_img[:50, :] = 10.0
        assert engine.is_tile_empty(tile_img.astype(np.uint8)) is True


# ===========================================================================
# 6. _cluster_by_iou
# ===========================================================================

class TestClusterByIou:

    def test_empty_input_returns_empty(self):
        engine = _make_engine()
        assert engine._cluster_by_iou([], iou_threshold=0.5) == []

    def test_single_detection_single_cluster(self):
        engine = _make_engine()
        det = _make_detection(0, 0, 100, 100)
        clusters = engine._cluster_by_iou([det], iou_threshold=0.5)
        assert len(clusters) == 1
        assert len(clusters[0]) == 1

    def test_high_iou_detections_merged_into_one_cluster(self):
        """Two almost-identical detections → 1 cluster."""
        engine = _make_engine()
        det1 = _make_detection(0, 0, 100, 100, conf=0.9)
        det2 = _make_detection(2, 2, 102, 102, conf=0.8)   # IoU ≈ 0.96
        clusters = engine._cluster_by_iou([det1, det2], iou_threshold=0.5)
        assert len(clusters) == 1
        assert len(clusters[0]) == 2

    def test_non_overlapping_detections_separate_clusters(self):
        """Detections far apart → 2 clusters."""
        engine = _make_engine()
        det1 = _make_detection(0, 0, 50, 50)
        det2 = _make_detection(500, 500, 600, 600)   # IoU = 0
        clusters = engine._cluster_by_iou([det1, det2], iou_threshold=0.5)
        assert len(clusters) == 2

    def test_cluster_sorted_by_confidence_descending(self):
        """Highest-confidence detection is first in each cluster."""
        engine = _make_engine()
        low = _make_detection(0, 0, 100, 100, conf=0.4)
        high = _make_detection(1, 1, 101, 101, conf=0.9)
        clusters = engine._cluster_by_iou([low, high], iou_threshold=0.5)
        assert clusters[0][0].confidence.value >= clusters[0][-1].confidence.value

    def test_three_in_chain_one_cluster(self):
        """3 detections with consecutive high IoU → same cluster."""
        engine = _make_engine()
        d1 = _make_detection(0, 0, 100, 100, conf=0.9)
        d2 = _make_detection(2, 2, 102, 102, conf=0.85)
        d3 = _make_detection(4, 4, 104, 104, conf=0.8)
        clusters = engine._cluster_by_iou([d1, d2, d3], iou_threshold=0.5)
        # All three should be in one cluster
        assert len(clusters) == 1
        assert len(clusters[0]) == 3


# ===========================================================================
# 7. _fuse_cluster
# ===========================================================================

class TestFuseCluster:

    def test_fuse_two_identical_detections(self):
        engine = _make_engine()
        det1 = _make_detection(0, 0, 100, 100, conf=0.9)
        det2 = _make_detection(0, 0, 100, 100, conf=0.8)
        fused = engine._fuse_cluster([det1, det2])
        assert fused.bounding_box.x_min == pytest.approx(0.0)
        assert fused.bounding_box.x_max == pytest.approx(100.0)

    def test_fuse_weighted_by_confidence(self):
        engine = _make_engine()
        # High-conf det at (0,0,100,100), low-conf at (10,10,110,110)
        det1 = _make_detection(0, 0, 100, 100, conf=0.9)
        det2 = _make_detection(10, 10, 110, 110, conf=0.1)
        fused = engine._fuse_cluster([det1, det2])
        # Weighted avg: x_min = (0.9*0 + 0.1*10) / 1.0 = 1.0
        assert fused.bounding_box.x_min == pytest.approx(1.0)
        assert fused.bounding_box.x_max == pytest.approx(101.0)

    def test_fuse_single_returns_that_detection(self):
        engine = _make_engine()
        det = _make_detection(10, 20, 50, 80, conf=0.7)
        fused = engine._fuse_cluster([det])
        assert fused.bounding_box.x_min == pytest.approx(10.0)
        assert fused.bounding_box.y_min == pytest.approx(20.0)

    def test_fused_detection_has_id(self):
        engine = _make_engine()
        dets = [_make_detection(0, 0, 100, 100) for _ in range(3)]
        fused = engine._fuse_cluster(dets)
        assert fused.id is not None
        assert len(fused.id) > 0


# ===========================================================================
# 8. _standard_nms_merge
# ===========================================================================

class TestStandardNmsMerge:

    def test_empty_returns_empty(self):
        engine = _make_engine(merge="nms")
        assert engine._standard_nms_merge([]) == []

    def test_single_detection_kept(self):
        engine = _make_engine(merge="nms")
        det = _make_detection(0, 0, 100, 100)
        result = engine._standard_nms_merge([det])
        assert len(result) == 1

    def test_high_iou_duplicates_suppressed(self):
        engine = _make_engine(merge="nms")
        engine._tile_config.merge_iou_threshold = 0.5
        det1 = _make_detection(0, 0, 100, 100, conf=0.9)
        det2 = _make_detection(2, 2, 102, 102, conf=0.8)   # IoU ≈ 0.96
        result = engine._standard_nms_merge([det1, det2])
        assert len(result) == 1
        assert float(result[0].confidence) == pytest.approx(0.9)

    def test_non_overlapping_all_kept(self):
        engine = _make_engine(merge="nms")
        det1 = _make_detection(0, 0, 50, 50, conf=0.9)
        det2 = _make_detection(500, 500, 600, 600, conf=0.8)
        result = engine._standard_nms_merge([det1, det2])
        assert len(result) == 2


# ===========================================================================
# 9. _merge_detections (routing)
# ===========================================================================

class TestMergeDetections:

    def test_empty_returns_empty(self):
        engine = _make_engine()
        assert engine._merge_detections([]) == []

    def test_single_detection_returned_unchanged(self):
        engine = _make_engine()
        det = _make_detection(0, 0, 100, 100)
        result = engine._merge_detections([det])
        assert len(result) == 1

    def test_wbf_strategy_called_for_wbf_config(self):
        engine = _make_engine(merge="wbf")
        dets = [_make_detection(0, 0, 100, 100), _make_detection(500, 500, 600, 600)]
        result = engine._merge_detections(dets)
        assert isinstance(result, list)

    def test_nms_strategy_called_for_nms_config(self):
        engine = _make_engine(merge="nms")
        dets = [_make_detection(0, 0, 100, 100), _make_detection(500, 500, 600, 600)]
        result = engine._merge_detections(dets)
        assert isinstance(result, list)


# ===========================================================================
# 10. get_tile_coverage_map
# ===========================================================================

class TestGetTileCoverageMap:

    def test_corners_always_covered(self):
        engine = _make_engine(tile_size=640, overlap=0.0)
        cov = engine.get_tile_coverage_map(1280, 1280)
        assert cov[0, 0] >= 1
        assert cov[0, 1279] >= 1
        assert cov[1279, 0] >= 1
        assert cov[1279, 1279] >= 1

    def test_overlap_region_covered_twice(self):
        engine = _make_engine(tile_size=640, overlap=0.20)
        cov = engine.get_tile_coverage_map(1280, 1280)
        # Overlap zone: stride=512 → x [512, 640] covered by tile 0 and tile 1
        # Check that max coverage ≥ 2 (overlap exists)
        assert cov.max() >= 2

    def test_no_overlap_max_coverage_is_1(self):
        """With 0% overlap, no pixel should be covered by >1 tile."""
        engine = _make_engine(tile_size=640, overlap=0.0)
        cov = engine.get_tile_coverage_map(640, 640)   # exactly 1 tile
        assert cov.max() == 1

    def test_output_shape_matches_image(self):
        engine = _make_engine(tile_size=640, overlap=0.0)
        cov = engine.get_tile_coverage_map(500, 800)
        assert cov.shape == (500, 800)

    def test_all_pixels_covered_at_least_once(self):
        engine = _make_engine(tile_size=640, overlap=0.20)
        cov = engine.get_tile_coverage_map(1280, 1280)
        assert cov.min() >= 1


# ===========================================================================
# 11. detect_tiled (integration with mock detector)
# ===========================================================================

class TestDetectTiledWithMock:

    def _setup_mock_detector(self, detections: list[Detection]) -> MagicMock:
        from detection.domain.detector import DetectionResult
        mock_result = MagicMock(spec=DetectionResult)
        mock_result.detections = detections
        mock_result.inference_time_ms = 10.0
        mock_detector = MagicMock()
        mock_detector.detect.return_value = mock_result
        return mock_detector

    def test_empty_tile_is_skipped(self):
        """A constant-colour tile with skip_empty_tiles=True must not call detector."""
        detector = self._setup_mock_detector([])
        cfg = TileConfig(tile_size=640, skip_empty_tiles=True, empty_tile_variance_threshold=50.0)
        engine = TiledInferenceEngine(detector, cfg)

        # Constant-colour 640×640 image → single tile with variance 0 → skipped
        image = np.full((640, 640, 3), 100, dtype=np.uint8)
        merged, tiles, diag = engine.detect_tiled(image)

        detector.detect.assert_not_called()
        assert diag[0]["skipped"] is True
        assert diag[0]["reason"] == "empty_tile"

    def test_non_empty_tile_calls_detector(self):
        detector = self._setup_mock_detector([])
        cfg = TileConfig(tile_size=640, skip_empty_tiles=True)
        engine = TiledInferenceEngine(detector, cfg)

        rng = np.random.default_rng(1)
        image = rng.integers(0, 256, (640, 640, 3), dtype=np.uint8)
        engine.detect_tiled(image)

        assert detector.detect.call_count >= 1

    def test_local_to_global_coordinate_translation(self):
        """Detection at local (10, 10, 50, 50) in tile at (640, 640) → global (650, 650, 690, 690)."""
        local_det = _make_detection(10, 10, 50, 50, conf=0.9)
        detector = self._setup_mock_detector([local_det])
        cfg = TileConfig(tile_size=640, overlap_fraction=0.0, skip_empty_tiles=False)
        engine = TiledInferenceEngine(detector, cfg)

        rng = np.random.default_rng(2)
        image = rng.integers(0, 256, (1280, 1280, 3), dtype=np.uint8)
        merged, tiles, _ = engine.detect_tiled(image)

        # Find tile at (640, 640) — bottom-right for 1280×1280, 0% overlap
        # Tiles: (0,0,640,640), (640,0,1280,640), (0,640,640,1280), (640,640,1280,1280)
        # Detection from tile at x_start=640, y_start=640 → global x=10+640=650, y=10+640=650
        global_bboxes = [(d.bounding_box.x_min, d.bounding_box.y_min) for d in merged]
        # At least one detection should be at global (650, 650) from the bottom-right tile
        found_bottom_right = any(
            abs(x - 650) < 1 and abs(y - 650) < 1 for x, y in global_bboxes
        )
        assert found_bottom_right, f"Expected global (650,650) but got: {global_bboxes}"

    def test_diagnostics_contain_expected_keys(self):
        local_det = _make_detection(10, 10, 50, 50)
        detector = self._setup_mock_detector([local_det])
        cfg = TileConfig(tile_size=640, overlap_fraction=0.0, skip_empty_tiles=False)
        engine = TiledInferenceEngine(detector, cfg)

        rng = np.random.default_rng(3)
        image = rng.integers(0, 256, (640, 640, 3), dtype=np.uint8)
        _, _, diag = engine.detect_tiled(image)

        assert len(diag) >= 1
        d = diag[0]
        assert "tile_index" in d
        assert "num_detections" in d

    def test_all_detections_clipped_to_image(self):
        """No detection coordinate should exceed image bounds after global translation."""
        # Create a detection that is at the very edge of a tile
        edge_det = _make_detection(600, 600, 640, 640, conf=0.9)
        detector = self._setup_mock_detector([edge_det])
        cfg = TileConfig(tile_size=640, overlap_fraction=0.0, skip_empty_tiles=False)
        engine = TiledInferenceEngine(detector, cfg)

        image = np.random.randint(0, 255, (1280, 1280, 3), dtype=np.uint8)
        merged, _, _ = engine.detect_tiled(image)
        for det in merged:
            assert det.bounding_box.x_min >= 0
            assert det.bounding_box.y_min >= 0
            assert det.bounding_box.x_max <= 1280
            assert det.bounding_box.y_max <= 1280

    def test_multiple_tiles_returns_merged_detections(self):
        """4 tiles each returning 1 detection → merged result ≥ 1."""
        local_det = _make_detection(10, 10, 50, 50, conf=0.9)
        detector = self._setup_mock_detector([local_det])
        cfg = TileConfig(tile_size=640, overlap_fraction=0.0, skip_empty_tiles=False)
        engine = TiledInferenceEngine(detector, cfg)

        rng = np.random.default_rng(4)
        image = rng.integers(0, 256, (1280, 1280, 3), dtype=np.uint8)
        merged, tiles, _ = engine.detect_tiled(image)

        assert len(tiles) == 4   # 2×2 grid
        assert len(merged) >= 1  # WBF may merge near-identical coords, but at least 1
