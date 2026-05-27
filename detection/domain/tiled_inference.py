"""
detection.domain.tiled_inference — Tiled/Sliding-Window Inference Engine.

WHY TILED INFERENCE?
━━━━━━━━━━━━━━━━━━━━
Standard YOLO operates at 640×640 or 1280×1280 input resolution.
High-resolution microscopy images are often 4K (3840×2160) or larger.
Downsampling a 4K image to 1280px makes trichomes ~4px in size —
well below reliable detection thresholds.

Solution: Divide the image into overlapping tiles, run inference on each,
then merge detections back to original coordinates.

OVERLAP HANDLING:
━━━━━━━━━━━━━━━━
Overlap is essential to prevent:
1. Trichomes at tile boundaries being cut in half
2. NMS not seeing context around boundary objects
3. False positives from partial trichomes at edges

Tile merging uses Weighted Boxes Fusion (WBF) for ensemble-like results,
or standard NMS as fallback.

PERFORMANCE:
━━━━━━━━━━━
- 4K image with 1280px tiles (20% overlap): ~12 tiles
- Each tile: ~50ms inference on RTX 4090
- Total: ~600ms + WBF overhead
- Parallelism: tiles can be batched and sent to GPU in parallel

Reference:
  Zhu, H. et al. (2019). "Object Detection based on Fast/Faster RCNN
  Employing Fully Convolutional Architectures." arXiv:1904.01939.

  Solovyev, R. et al. (2021). "Weighted Boxes Fusion: Ensembling Boxes
  for Object Detection Models." Image and Vision Computing 107, 104117.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from shared.core.entities import Detection
from shared.core.value_objects import BoundingBox, Confidence

if TYPE_CHECKING:
    from detection.domain.detector import DetectionConfig, DetectionResult, TrichomeDetector


@dataclass
class TileConfig:
    """Configuration for tiled inference."""

    tile_size: int = 1280
    """Size of each square tile in pixels."""

    overlap_fraction: float = 0.20
    """
    Fractional overlap between adjacent tiles.
    0.20 = 20% overlap.

    Tradeoff:
    - Higher overlap → fewer missed boundary objects, more compute
    - Lower overlap → faster, more boundary artifacts
    """

    min_tile_size: int = 256
    """
    Minimum tile dimension. If image is smaller in one dimension,
    don't create unnecessary small tiles.
    """

    merge_strategy: str = "wbf"
    """
    How to merge overlapping detections from multiple tiles.
    Options: "wbf" (Weighted Boxes Fusion), "nms" (standard NMS)

    WBF is preferred as it produces more stable box coordinates
    when the same object appears in multiple tiles.
    """

    merge_iou_threshold: float = 0.45
    """IoU threshold for post-merge NMS or WBF confidence threshold."""

    skip_empty_tiles: bool = True
    """Skip tiles with no foreground content (based on variance threshold)."""

    empty_tile_variance_threshold: float = 50.0
    """
    Tiles with intensity variance below this are considered empty.
    Background microscopy fields have very low variance.
    Tune based on your specific microscopy setup.
    """

    def validate(self) -> None:
        if not (0.0 <= self.overlap_fraction < 0.5):
            raise ValueError(f"overlap_fraction must be in [0, 0.5), got {self.overlap_fraction}")
        if self.tile_size < self.min_tile_size:
            raise ValueError(
                f"tile_size ({self.tile_size}) < min_tile_size ({self.min_tile_size})"
            )


@dataclass
class TileInfo:
    """Metadata for a single tile."""

    tile_index: int
    x_start: int
    y_start: int
    x_end: int
    y_end: int
    is_padded: bool = False

    @property
    def width(self) -> int:
        return self.x_end - self.x_start

    @property
    def height(self) -> int:
        return self.y_end - self.y_start

    def to_global_bbox(self, local_bbox: BoundingBox) -> BoundingBox:
        """
        Convert tile-local bounding box coordinates to global image coordinates.

        This is the critical step that makes tiled inference work.
        Every detection from a tile must be shifted by the tile's offset.
        """
        return BoundingBox(
            x_min=local_bbox.x_min + self.x_start,
            y_min=local_bbox.y_min + self.y_start,
            x_max=local_bbox.x_max + self.x_start,
            y_max=local_bbox.y_max + self.y_start,
        )


class TiledInferenceEngine:
    """
    Sliding window inference engine for high-resolution microscopy images.

    Wraps any TrichomeDetector and applies tiled inference transparently.

    Usage:
        base_detector = YOLODetector(model_id="yolo11x", ...)
        tiled = TiledInferenceEngine(base_detector, TileConfig(tile_size=1280))
        result = tiled.detect_tiled(large_image)
    """

    def __init__(
        self,
        detector: "TrichomeDetector",
        tile_config: TileConfig | None = None,
    ) -> None:
        self._detector = detector
        self._tile_config = tile_config or TileConfig()
        self._tile_config.validate()

    def compute_tiles(
        self,
        image_height: int,
        image_width: int,
    ) -> list[TileInfo]:
        """
        Compute tile positions for a given image size.

        Algorithm:
        1. Start at (0, 0)
        2. Step by tile_size * (1 - overlap_fraction)
        3. Ensure last tile always reaches image boundary (partial tile)
        4. Return list of TileInfo objects
        """
        ts = self._tile_config.tile_size
        stride = int(ts * (1.0 - self._tile_config.overlap_fraction))

        tiles: list[TileInfo] = []
        tile_idx = 0

        # Compute y positions
        y_starts = list(range(0, max(1, image_height - ts + 1), stride))
        if not y_starts or (y_starts and y_starts[-1] + ts < image_height):
            y_starts.append(max(0, image_height - ts))

        # Compute x positions
        x_starts = list(range(0, max(1, image_width - ts + 1), stride))
        if not x_starts or (x_starts and x_starts[-1] + ts < image_width):
            x_starts.append(max(0, image_width - ts))

        for y_start in y_starts:
            for x_start in x_starts:
                x_end = min(x_start + ts, image_width)
                y_end = min(y_start + ts, image_height)

                is_padded = (x_end - x_start) < ts or (y_end - y_start) < ts

                tiles.append(
                    TileInfo(
                        tile_index=tile_idx,
                        x_start=x_start,
                        y_start=y_start,
                        x_end=x_end,
                        y_end=y_end,
                        is_padded=is_padded,
                    )
                )
                tile_idx += 1

        return tiles

    def extract_tile(
        self,
        image: NDArray[np.uint8],
        tile: TileInfo,
    ) -> NDArray[np.uint8]:
        """
        Extract and optionally pad a tile from the full image.

        Padding with border replication (cv2.BORDER_REFLECT_101) instead of zeros
        prevents the model from seeing artificial black edges at tile boundaries.
        """
        import cv2

        tile_img = image[tile.y_start:tile.y_end, tile.x_start:tile.x_end]

        ts = self._tile_config.tile_size
        h, w = tile_img.shape[:2]

        if h < ts or w < ts:
            pad_h = ts - h
            pad_w = ts - w
            tile_img = cv2.copyMakeBorder(
                tile_img,
                top=0, bottom=pad_h,
                left=0, right=pad_w,
                borderType=cv2.BORDER_REFLECT_101,
            )

        return tile_img

    def is_tile_empty(self, tile_img: NDArray[np.uint8]) -> bool:
        """
        Heuristic check for empty/background-only tiles.

        Saves computation by skipping tiles with no interesting content.
        Background microscopy fields have very low intensity variance.

        Note: Be conservative — better to process empty tiles than miss trichomes.
        """
        if not self._tile_config.skip_empty_tiles:
            return False
        return float(np.var(tile_img.astype(np.float32))) < self._tile_config.empty_tile_variance_threshold

    def detect_tiled(
        self,
        image: NDArray[np.uint8],
        config: "DetectionConfig | None" = None,
    ) -> tuple[list[Detection], list[TileInfo], list[dict[str, Any]]]:
        """
        Run tiled detection on a full high-resolution image.

        Returns:
            - Merged list of global-coordinate detections
            - List of tile metadata
            - Per-tile result dicts (for diagnostics)
        """
        from detection.domain.detector import DetectionConfig

        cfg = config or DetectionConfig()
        h, w = image.shape[:2]

        tiles = self.compute_tiles(h, w)
        all_detections: list[Detection] = []
        tile_diagnostics: list[dict[str, Any]] = []

        for tile in tiles:
            tile_img = self.extract_tile(image, tile)

            if self.is_tile_empty(tile_img):
                tile_diagnostics.append({
                    "tile_index": tile.tile_index,
                    "skipped": True,
                    "reason": "empty_tile",
                    "num_detections": 0,
                })
                continue

            result = self._detector.detect(tile_img, cfg)

            # CRITICAL: Translate local tile coordinates → global image coordinates
            # NOTE: Do NOT mutate det.bounding_box — create a new Detection so the
            # same Detection object returned by the detector for multiple tiles does
            # not accumulate offset shifts across calls.
            import uuid as _uuid
            global_detections = []
            for det in result.detections:
                global_bbox = tile.to_global_bbox(det.bounding_box)
                # Clip to image boundaries — skip degenerate boxes
                try:
                    global_bbox = global_bbox.clip_to_image(w, h)
                except ValueError:
                    continue  # bbox fell entirely outside image after shift; drop it
                global_det = Detection(
                    id=str(_uuid.uuid4()),
                    bounding_box=global_bbox,
                    confidence=det.confidence,
                    trichome_type=det.trichome_type,
                    model_id=det.model_id,
                    class_id=det.class_id,
                )
                global_detections.append(global_det)

            all_detections.extend(global_detections)

            tile_diagnostics.append({
                "tile_index": tile.tile_index,
                "skipped": False,
                "x_start": tile.x_start,
                "y_start": tile.y_start,
                "x_end": tile.x_end,
                "y_end": tile.y_end,
                "num_detections": len(global_detections),
                "inference_ms": result.inference_time_ms,
            })

        # Merge overlapping detections from tile boundaries
        merged = self._merge_detections(all_detections)
        return merged, tiles, tile_diagnostics

    def _merge_detections(self, detections: list[Detection]) -> list[Detection]:
        """
        Merge detections from multiple tiles using WBF or NMS.

        The same trichome will typically appear in 1-4 tiles depending
        on overlap settings. WBF produces the best final box coordinates.
        """
        if len(detections) <= 1:
            return detections

        if self._tile_config.merge_strategy == "wbf":
            return self._weighted_boxes_fusion(detections)
        else:
            return self._standard_nms_merge(detections)

    def _weighted_boxes_fusion(self, detections: list[Detection]) -> list[Detection]:
        """
        Weighted Boxes Fusion (WBF) for tile merging.

        Produces averaged, stable box coordinates when the same object
        is detected in multiple overlapping tiles.

        Reference:
          Solovyev, R. et al. (2021). "Weighted Boxes Fusion."
          Image and Vision Computing 107, 104117.
        """
        # Group detections by class
        from collections import defaultdict

        class_detections: dict[str, list[Detection]] = defaultdict(list)
        for det in detections:
            class_detections[det.trichome_type.value].append(det)

        merged: list[Detection] = []

        for class_name, class_dets in class_detections.items():
            # Simple IoU-based clustering as WBF approximation
            # Full WBF implementation would normalize coordinates
            clustered = self._cluster_by_iou(
                class_dets,
                iou_threshold=self._tile_config.merge_iou_threshold,
            )
            for cluster in clustered:
                if len(cluster) == 1:
                    merged.append(cluster[0])
                else:
                    merged.append(self._fuse_cluster(cluster))

        return merged

    def _cluster_by_iou(
        self,
        detections: list[Detection],
        iou_threshold: float,
    ) -> list[list[Detection]]:
        """Group detections by IoU overlap into clusters."""
        if not detections:
            return []

        # Sort by confidence (descending)
        sorted_dets = sorted(detections, key=lambda d: float(d.confidence), reverse=True)
        used = [False] * len(sorted_dets)
        clusters: list[list[Detection]] = []

        for i, det_i in enumerate(sorted_dets):
            if used[i]:
                continue
            cluster = [det_i]
            used[i] = True
            for j, det_j in enumerate(sorted_dets[i + 1:], start=i + 1):
                if used[j]:
                    continue
                iou = det_i.bounding_box.iou(det_j.bounding_box)
                if iou > iou_threshold:
                    cluster.append(det_j)
                    used[j] = True
            clusters.append(cluster)

        return clusters

    def _fuse_cluster(self, cluster: list[Detection]) -> Detection:
        """
        Fuse a cluster of overlapping detections into one.

        Weighted average of box coordinates, confidence-weighted.
        This is the core of WBF.
        """
        import uuid

        confidences = np.array([float(d.confidence) for d in cluster])
        weights = confidences / confidences.sum()

        # Weighted average of box coordinates
        x_mins = np.array([d.bounding_box.x_min for d in cluster])
        y_mins = np.array([d.bounding_box.y_min for d in cluster])
        x_maxs = np.array([d.bounding_box.x_max for d in cluster])
        y_maxs = np.array([d.bounding_box.y_max for d in cluster])

        fused_box = BoundingBox(
            x_min=float(np.dot(weights, x_mins)),
            y_min=float(np.dot(weights, y_mins)),
            x_max=float(np.dot(weights, x_maxs)),
            y_max=float(np.dot(weights, y_maxs)),
        )

        # Average confidence (WBF semantics)
        fused_confidence = Confidence(float(np.mean(confidences)))

        # Use highest-confidence detection's metadata
        best = cluster[np.argmax(confidences)]

        return Detection(
            id=str(uuid.uuid4()),
            bounding_box=fused_box,
            confidence=fused_confidence,
            trichome_type=best.trichome_type,
            model_id=best.model_id,
            class_id=best.class_id,
        )

    def _standard_nms_merge(self, detections: list[Detection]) -> list[Detection]:
        """Standard NMS as fallback merge strategy."""
        if not detections:
            return []

        # Sort by confidence
        sorted_dets = sorted(detections, key=lambda d: float(d.confidence), reverse=True)
        kept: list[Detection] = []

        while sorted_dets:
            best = sorted_dets.pop(0)
            kept.append(best)
            sorted_dets = [
                d for d in sorted_dets
                if best.bounding_box.iou(d.bounding_box) < self._tile_config.merge_iou_threshold
            ]

        return kept

    def get_tile_coverage_map(
        self,
        image_height: int,
        image_width: int,
    ) -> NDArray[np.uint8]:
        """
        Generate a visualization of tile coverage.

        Returns a heatmap showing how many tiles cover each pixel.
        Useful for verifying overlap settings and diagnosing edge cases.
        """
        coverage = np.zeros((image_height, image_width), dtype=np.uint8)
        tiles = self.compute_tiles(image_height, image_width)

        for tile in tiles:
            coverage[tile.y_start:tile.y_end, tile.x_start:tile.x_end] += 1

        return coverage
