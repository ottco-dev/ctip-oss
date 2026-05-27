"""
detection.application.detect_pipeline — Full detection pipeline orchestrator.

Coordinates: preprocessing → tiled detection → NMS → calibration →
uncertainty estimation → postprocessing → output.

This is the primary entry point for running detection. It wires together
all domain components and handles the business logic of when to use tiling,
whether to apply TTA, and how to structure results for downstream services.

Hardware-aware design (RTX 4060, 8 GB VRAM):
- Default: YOLO11s (1.2 GB VRAM) + tiled inference
- SAM2 segmentation only when explicitly requested (adds 3.8 GB VRAM)
- Ensemble mode disabled by default (doubles inference time)
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from detection.domain.detector import BaseDetector, DetectionConfig, DetectionResult
from detection.domain.tiled_inference import TiledInferenceEngine, TileConfig
from shared.core.entities import Detection, TrichomeRegion
from shared.core.enums import TrichomeType
from shared.logging.logger import get_logger
from shared.utils.image_utils import apply_clahe, compute_image_stats

logger = get_logger(__name__)


@dataclass
class PipelineConfig:
    """
    High-level configuration for the full detection pipeline.

    Separates pipeline orchestration concerns from model-level config.
    """

    # Detection model config
    detection: DetectionConfig = field(default_factory=DetectionConfig)

    # Tiling
    use_tiling: bool = True
    """
    Enable tiled inference for images wider/taller than tile_size.
    RECOMMENDED for all microscopy images > 1280px on any side.
    """
    tile_size: int = 1280
    tile_overlap: float = 0.2

    # Preprocessing
    apply_clahe: bool = True
    """Contrast enhancement before detection. Helps with low-contrast trichomes."""

    denoise: bool = True
    """Mild bilateral denoising. Reduces false positives from sensor noise."""

    # Output control
    export_crops: bool = False
    """Save individual trichome crops. Required for downstream maturity analysis."""
    crop_margin_px: int = 10
    """Expand crop by this many pixels on each side."""

    min_area_px: int = 25
    """
    Minimum bounding box area in pixels². Smaller detections are discarded.
    Prevents noise detections that are too small to be real trichomes.
    Set based on calibration: at 0.43 µm/px, smallest bulbous trichome
    is ~10µm = ~23px diameter → area ~415px². Use 25 for safety margin.
    """

    max_aspect_ratio: float = 10.0
    """
    Discard detections with extreme aspect ratios (likely artifacts).
    Capitate stalked have stalk aspect ratio ~5-8 max.
    """


@dataclass
class PipelineResult:
    """Complete output from the detection pipeline."""

    detections: list[Detection]
    region: TrichomeRegion
    inference_time_ms: float
    total_time_ms: float
    image_stats: dict[str, float]
    was_tiled: bool
    num_tiles: int | None
    crops: dict[str, NDArray[np.uint8]] = field(default_factory=dict)
    """Map of detection_id → crop image (only populated if export_crops=True)"""

    @property
    def num_detections(self) -> int:
        return len(self.detections)

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_detections": self.num_detections,
            "inference_time_ms": self.inference_time_ms,
            "total_time_ms": self.total_time_ms,
            "was_tiled": self.was_tiled,
            "num_tiles": self.num_tiles,
            "image_stats": self.image_stats,
            "detections": [d.to_dict() for d in self.detections],
            "region_summary": self.region.to_summary_dict(),
        }


class DetectionPipeline:
    """
    Orchestrates the complete trichome detection pipeline.

    Usage:
        pipeline = DetectionPipeline(detector=yolo_detector, config=PipelineConfig())
        result = pipeline.run(image_array, image_id="sample_001")
    """

    def __init__(
        self,
        detector: BaseDetector,
        config: PipelineConfig | None = None,
    ) -> None:
        self._detector = detector
        self._config = config or PipelineConfig()

        # Initialize tiled inference engine
        self._tiled_engine = TiledInferenceEngine(
            detector=detector,
            tile_config=TileConfig(
                tile_size=self._config.tile_size,
                overlap_fraction=self._config.tile_overlap,
            ),
        )

    def run(
        self,
        image: NDArray[np.uint8],
        image_id: str = "",
        image_path: Path | None = None,
    ) -> PipelineResult:
        """
        Run full detection pipeline on a single image.

        Args:
            image: RGB image array (H, W, 3) uint8
            image_id: Unique identifier for this image
            image_path: Optional source file path

        Returns:
            PipelineResult with all detections and metadata.
        """
        t_start = time.perf_counter()

        if not image_id:
            image_id = str(uuid.uuid4())

        logger.info(
            "Running detection pipeline",
            image_id=image_id,
            shape=image.shape,
            tiled=self._config.use_tiling,
        )

        # Step 1: Compute image statistics for quality assessment
        img_stats = compute_image_stats(image)
        logger.debug("Image stats computed", **img_stats)

        # Step 2: Preprocessing
        preprocessed = self._preprocess(image)

        # Step 3: Detection (tiled or full-image)
        h, w = preprocessed.shape[:2]
        use_tiling = (
            self._config.use_tiling
            and (h > self._config.tile_size or w > self._config.tile_size)
        )

        t_inf_start = time.perf_counter()

        if use_tiling:
            detections, tiles, tile_diagnostics = self._tiled_engine.detect_tiled(
                preprocessed, self._config.detection
            )
            num_tiles = len(tiles)
            logger.info(
                "Tiled inference complete",
                num_tiles=num_tiles,
                num_detections=len(detections),
            )
        else:
            result = self._detector.detect(preprocessed, self._config.detection)
            detections = result.detections
            num_tiles = None

        t_inf_end = time.perf_counter()
        inference_ms = (t_inf_end - t_inf_start) * 1000

        # Step 4: Post-filtering
        detections = self._filter_detections(detections, w, h)

        # Step 5: Assign image_id to all detections
        for det in detections:
            det.image_id = image_id

        logger.info(
            "Detection complete",
            image_id=image_id,
            num_detections=len(detections),
            inference_ms=f"{inference_ms:.1f}",
        )

        # Step 6: Extract crops (optional)
        crops: dict[str, NDArray[np.uint8]] = {}
        if self._config.export_crops:
            crops = self._extract_crops(image, detections)

        # Step 7: Build TrichomeRegion
        region = TrichomeRegion(
            image_id=image_id,
            image_path=image_path,
            focus_score=img_stats.get("contrast"),
            image_quality_score=self._estimate_quality_score(img_stats),
        )

        t_end = time.perf_counter()
        total_ms = (t_end - t_start) * 1000

        return PipelineResult(
            detections=detections,
            region=region,
            inference_time_ms=inference_ms,
            total_time_ms=total_ms,
            image_stats=img_stats,
            was_tiled=use_tiling,
            num_tiles=num_tiles,
            crops=crops,
        )

    def run_batch(
        self,
        images: list[NDArray[np.uint8]],
        image_ids: list[str] | None = None,
    ) -> list[PipelineResult]:
        """
        Run pipeline on a batch of images.

        Sequential currently; batch GPU inference optimization planned.
        """
        if image_ids is None:
            image_ids = [str(uuid.uuid4()) for _ in images]

        results = []
        for image, img_id in zip(images, image_ids):
            result = self.run(image, image_id=img_id)
            results.append(result)
        return results

    def _preprocess(self, image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Apply preprocessing pipeline."""
        import cv2

        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

        if self._config.denoise:
            image = cv2.bilateralFilter(image, d=5, sigmaColor=20, sigmaSpace=5)

        if self._config.apply_clahe:
            image = apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8))

        return image

    def _filter_detections(
        self,
        detections: list[Detection],
        img_w: int,
        img_h: int,
    ) -> list[Detection]:
        """
        Post-hoc filtering of detections.

        Removes:
        1. Boxes below minimum area threshold
        2. Boxes with extreme aspect ratios
        3. Boxes outside image boundaries

        This filtering is separate from NMS and handles biological plausibility.
        """
        filtered: list[Detection] = []

        for det in detections:
            bbox = det.bounding_box

            # Area filter
            if bbox.area < self._config.min_area_px:
                continue

            # Aspect ratio filter (both orientations: horizontal and vertical stalks)
            ar = bbox.aspect_ratio
            if ar > self._config.max_aspect_ratio or ar < (1.0 / self._config.max_aspect_ratio):
                continue

            # Boundary check (clip to image)
            try:
                det.bounding_box = bbox.clip_to_image(img_w, img_h)
            except ValueError:
                continue

            filtered.append(det)

        return filtered

    def _extract_crops(
        self,
        image: NDArray[np.uint8],
        detections: list[Detection],
    ) -> dict[str, NDArray[np.uint8]]:
        """Extract individual trichome crops for downstream analysis."""
        from shared.utils.image_utils import crop_region

        crops: dict[str, NDArray[np.uint8]] = {}
        h, w = image.shape[:2]

        for det in detections:
            bbox = det.bounding_box
            crop = crop_region(
                image,
                x_min=int(bbox.x_min),
                y_min=int(bbox.y_min),
                x_max=int(bbox.x_max),
                y_max=int(bbox.y_max),
                margin_px=self._config.crop_margin_px,
            )
            if crop.size > 0:
                crops[det.id] = crop

        return crops

    @staticmethod
    def _estimate_quality_score(stats: dict[str, float]) -> float:
        """
        Heuristic image quality score [0, 1] from image statistics.

        Combines contrast (sharpness proxy) and dynamic range.
        """
        contrast = min(stats.get("contrast", 0.0), 1.0)
        dynamic_range = stats.get("dynamic_range", 0.0)
        return float((contrast * 0.6 + dynamic_range * 0.4))
