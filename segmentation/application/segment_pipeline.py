"""
segmentation.application.segment_pipeline — Trichome instance segmentation pipeline.

Full pipeline:
  1. Receive detection results (bounding boxes)
  2. Select segmentation backend (SAM2-tiny or MobileSAM fallback)
  3. Segment each detected trichome instance
  4. Post-process: refine masks, compute polygon approximations
  5. Compute per-instance features (area, centroid, circularity)
  6. Return enriched InstanceSegmentation results

VRAM planning for RTX 4060:
  - Detection (YOLO11s): ~1.2 GB
  - Segmentation (SAM2-tiny): ~3.8 GB
  - Total: ~5.0 GB (leaves ~3 GB headroom for other tasks)
  - VLM (Moondream): only run separately via unload/load cycle
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from segmentation.domain.segmentor import (
    BaseSegmentor,
    BatchSegmentationResult,
    BoxPrompt,
    SegmentationResult,
    SegmentorConfig,
)
from shared.utils.geometry import (
    mask_to_polygon,
    polygon_area,
    polygon_centroid,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SegmentPipelineConfig:
    """Configuration for the segmentation pipeline."""

    backend: Literal["sam2_tiny", "mobile_sam", "auto"] = "auto"
    """Segmentation backend. 'auto' selects SAM2-tiny if GPU available."""

    device: str = "cuda"

    # SAM2 settings
    sam2_variant: str = "tiny"
    sam2_checkpoint: str | None = None

    # MobileSAM settings
    mobile_sam_checkpoint: str = "weights/mobile_sam.pt"

    # Mask quality
    score_threshold: float = 0.50
    min_mask_area_px: int = 50
    max_mask_area_fraction: float = 0.4

    # Polygon output
    compute_polygons: bool = True
    polygon_simplify_tolerance: float = 2.0

    # Feature extraction
    compute_circularity: bool = True
    compute_elongation: bool = True

    # Output
    max_instances: int = 500  # maximum segmented instances per image


# ---------------------------------------------------------------------------
# Instance data
# ---------------------------------------------------------------------------

@dataclass
class SegmentedInstance:
    """Full description of a single segmented trichome."""

    # Detection data
    detection_box: tuple[float, float, float, float] | None
    detection_confidence: float
    detection_class_id: int
    detection_class_name: str

    # Segmentation
    mask: NDArray[np.bool_]
    mask_score: float

    # Geometry
    area_px: float
    centroid_x: float
    centroid_y: float
    polygon: NDArray[np.float64] | None = None  # (N, 2) points

    # Shape descriptors
    circularity: float = 0.0
    """4π·area/perimeter². Circle = 1.0, elongated < 1.0."""

    elongation: float = 0.0
    """Bounding box aspect ratio. 1.0 = square, > 1.0 = elongated."""

    # Calibrated measurements (requires MicroscopeProfile)
    area_um2: float | None = None
    diameter_um: float | None = None

    instance_id: int = -1


@dataclass
class SegmentPipelineResult:
    """Full segmentation pipeline output."""

    instances: list[SegmentedInstance]
    image_height: int
    image_width: int
    backend_used: str
    num_input_detections: int
    num_segmented: int
    pipeline_time_ms: float
    backend_time_ms: float


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class SegmentPipeline:
    """
    Trichome instance segmentation pipeline.

    Takes detection results as input, returns fully segmented instances.

    Usage::

        pipeline = SegmentPipeline(config)
        pipeline.load_backend()

        result = pipeline.run(image, detections)
        # result.instances: list of SegmentedInstance

        pipeline.unload_backend()
    """

    def __init__(self, config: SegmentPipelineConfig | None = None) -> None:
        self.config = config or SegmentPipelineConfig()
        self._backend: BaseSegmentor | None = None

    # ------------------------------------------------------------------
    # Backend lifecycle
    # ------------------------------------------------------------------

    def load_backend(self) -> None:
        """Load the configured segmentation backend."""
        backend_name = self._resolve_backend()

        seg_config = SegmentorConfig(
            device=self.config.device,
            score_threshold=self.config.score_threshold,
            min_mask_area_px=self.config.min_mask_area_px,
            max_mask_area_fraction=self.config.max_mask_area_fraction,
        )

        if backend_name == "sam2_tiny":
            from segmentation.infrastructure.sam2_backend import SAM2TinyBackend

            self._backend = SAM2TinyBackend(
                config=seg_config,
                model_variant=self.config.sam2_variant,
                checkpoint_path=self.config.sam2_checkpoint,
            )
        elif backend_name == "mobile_sam":
            from segmentation.infrastructure.mobile_sam import MobileSAMBackend

            self._backend = MobileSAMBackend(
                config=seg_config,
                checkpoint_path=self.config.mobile_sam_checkpoint,
            )
        else:
            raise ValueError(f"Unknown backend: {backend_name}")

        self._backend.load()
        logger.info("Loaded backend: %s", backend_name)

    def unload_backend(self) -> None:
        """Unload backend and free memory."""
        if self._backend is not None:
            self._backend.unload()

    def _resolve_backend(self) -> str:
        """Resolve 'auto' to actual backend based on available resources."""
        if self.config.backend != "auto":
            return self.config.backend

        try:
            import torch
            cuda_available = torch.cuda.is_available()
            if cuda_available:
                free_mb = torch.cuda.mem_get_info()[0] / 1e6
                # SAM2-tiny needs ~3800 MB
                if free_mb >= 3800:
                    return "sam2_tiny"
        except Exception:
            pass

        return "mobile_sam"

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def run(
        self,
        image: NDArray[np.uint8],
        detections: list[dict],
        pixels_per_um: float | None = None,
    ) -> SegmentPipelineResult:
        """
        Run full segmentation pipeline.

        Args:
            image: HWC uint8 RGB numpy array.
            detections: List of detection dicts with keys:
                        x1, y1, x2, y2, confidence, class_id, class_name
            pixels_per_um: Calibration scale (px/µm) for metric measurements.

        Returns:
            SegmentPipelineResult with all segmented instances.
        """
        if not self._backend or not self._backend.is_loaded:
            raise RuntimeError("Backend not loaded. Call load_backend() first.")

        t_start = time.perf_counter()
        h, w = image.shape[:2]

        # Cap instances
        if len(detections) > self.config.max_instances:
            logger.warning(
                "Capping detections from %d to %d (max_instances limit)",
                len(detections),
                self.config.max_instances,
            )
            detections = detections[: self.config.max_instances]

        num_input = len(detections)

        if num_input == 0:
            return SegmentPipelineResult(
                instances=[],
                image_height=h,
                image_width=w,
                backend_used=type(self._backend).__name__,
                num_input_detections=0,
                num_segmented=0,
                pipeline_time_ms=0.0,
                backend_time_ms=0.0,
            )

        # Build box prompts
        boxes = [
            BoxPrompt(
                x1=float(d.get("x1", 0)),
                y1=float(d.get("y1", 0)),
                x2=float(d.get("x2", 1)),
                y2=float(d.get("y2", 1)),
            )
            for d in detections
        ]

        # Run segmentation
        t_backend = time.perf_counter()
        batch_result: BatchSegmentationResult = self._backend.segment_with_boxes(
            image, boxes
        )
        backend_ms = (time.perf_counter() - t_backend) * 1000

        # Build instances
        instances: list[SegmentedInstance] = []

        for idx, (det, seg_result) in enumerate(
            zip(detections, batch_result.masks)
        ):
            if not seg_result.mask.any():
                # Empty mask — skip
                continue

            instance = self._build_instance(
                idx=idx,
                det=det,
                seg_result=seg_result,
                pixels_per_um=pixels_per_um,
            )
            instances.append(instance)

        total_ms = (time.perf_counter() - t_start) * 1000

        return SegmentPipelineResult(
            instances=instances,
            image_height=h,
            image_width=w,
            backend_used=batch_result.backend,
            num_input_detections=num_input,
            num_segmented=len(instances),
            pipeline_time_ms=total_ms,
            backend_time_ms=backend_ms,
        )

    # ------------------------------------------------------------------
    # Instance building
    # ------------------------------------------------------------------

    def _build_instance(
        self,
        idx: int,
        det: dict,
        seg_result: SegmentationResult,
        pixels_per_um: float | None,
    ) -> SegmentedInstance:
        """Build a SegmentedInstance from detection + segmentation result."""
        mask = seg_result.mask

        # Basic geometry
        area_px = float(mask.sum())
        ys, xs = np.nonzero(mask)
        cx = float(xs.mean()) if len(xs) > 0 else 0.0
        cy = float(ys.mean()) if len(ys) > 0 else 0.0

        # Polygon
        polygon = None
        if self.config.compute_polygons:
            polygon = mask_to_polygon(
                mask,
                simplify_tolerance=self.config.polygon_simplify_tolerance,
            )

        # Shape descriptors
        circularity = 0.0
        elongation = 0.0

        if self.config.compute_circularity and polygon is not None and len(polygon) >= 3:
            try:
                import cv2

                contour = polygon.astype(np.float32).reshape(-1, 1, 2)
                perimeter = cv2.arcLength(contour, closed=True)
                if perimeter > 0:
                    circularity = (4 * np.pi * area_px) / (perimeter ** 2)
            except Exception:
                pass

        if self.config.compute_elongation:
            x1 = float(det.get("x1", 0))
            y1 = float(det.get("y1", 0))
            x2 = float(det.get("x2", 1))
            y2 = float(det.get("y2", 1))
            bw = x2 - x1
            bh = y2 - y1
            if bw > 0:
                elongation = bh / bw

        # Calibrated measurements
        area_um2 = None
        diameter_um = None
        if pixels_per_um is not None and pixels_per_um > 0:
            area_um2 = area_px / (pixels_per_um ** 2)
            # Equivalent diameter of circle with same area
            diameter_um = 2.0 * np.sqrt(area_um2 / np.pi)

        return SegmentedInstance(
            instance_id=idx,
            detection_box=(
                float(det.get("x1", 0)),
                float(det.get("y1", 0)),
                float(det.get("x2", 0)),
                float(det.get("y2", 0)),
            ),
            detection_confidence=float(det.get("confidence", 0)),
            detection_class_id=int(det.get("class_id", 0)),
            detection_class_name=str(det.get("class_name", "")),
            mask=mask,
            mask_score=float(seg_result.score),
            area_px=area_px,
            centroid_x=cx,
            centroid_y=cy,
            polygon=polygon,
            circularity=float(circularity),
            elongation=float(elongation),
            area_um2=area_um2,
            diameter_um=diameter_um,
        )
