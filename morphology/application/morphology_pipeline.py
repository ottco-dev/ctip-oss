"""
morphology.application.morphology_pipeline — End-to-end morphology analysis pipeline.

Orchestrates:
  1. Geometric feature extraction (from mask)
  2. Stalk/head detection
  3. Type classification (rule-based → CNN if available)
  4. Density map generation (population-level)

INPUT:  List of Instance objects (detection + mask)
OUTPUT: Instance objects with MorphologyType populated
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from numpy.typing import NDArray

from shared.core.entities import Instance, MorphologyType
from shared.core.enums import TrichomeType
from shared.core.value_objects import Confidence
from morphology.domain.geometric import extract_geometric_descriptors
from morphology.domain.stalk_detector import detect_stalk_and_head
from morphology.domain.density_map import (
    TrichomeCentroid,
    DensityMapResult,
    compute_density_map,
)
from morphology.classification.classifier import MorphologyClassifier

logger = logging.getLogger(__name__)


@dataclass
class MorphologyPipelineConfig:
    """Configuration for the morphology analysis pipeline."""

    min_mask_area_px: float = 25.0
    """Minimum mask area (px²) to attempt morphology analysis."""

    use_geometric_fallback: bool = True
    """If CNN classifier fails, fall back to geometric rule-based classifier."""

    density_grid_rows: int = 8
    density_grid_cols: int = 8
    """Grid dimensions for density map generation."""

    kde_bandwidth_px: float = 30.0
    """Gaussian KDE bandwidth for density estimation."""

    classifier_model_path: Optional[str] = None
    """Path to ONNX classifier model. None = rule-based only."""


@dataclass
class MorphologyPipelineResult:
    """Complete result of the morphology pipeline."""

    instances: List[Instance]
    """Instances with MorphologyType populated."""

    density_map: Optional[DensityMapResult] = None
    """Population density map (None if fewer than 2 instances)."""

    type_distribution: dict = field(default_factory=dict)
    """Counts of each TrichomeType in this analysis."""

    total_analyzed: int = 0
    failed: int = 0
    used_geometric: int = 0
    used_cnn: int = 0


class MorphologyPipeline:
    """
    Production-grade morphology analysis pipeline.

    Thread-safe. GPU-free (rule-based classifier). CNN optional.
    """

    def __init__(self, config: Optional[MorphologyPipelineConfig] = None) -> None:
        self.config = config or MorphologyPipelineConfig()
        self._classifier = MorphologyClassifier(
            model_path=self.config.classifier_model_path
        )
        logger.info(
            "MorphologyPipeline initialized",
            extra={
                "classifier": "CNN" if self.config.classifier_model_path else "rule-based",
                "grid": f"{self.config.density_grid_rows}×{self.config.density_grid_cols}",
            },
        )

    def analyze(
        self,
        instances: List[Instance],
        image_shape: Optional[tuple] = None,
        um_per_pixel: Optional[float] = None,
    ) -> MorphologyPipelineResult:
        """
        Run morphology analysis on a list of Instance objects.

        Args:
            instances:    Detected/segmented trichome instances (with masks).
            image_shape:  (H, W) of the source image. Required for density map.
            um_per_pixel: Calibration factor for physical density calculation.

        Returns:
            MorphologyPipelineResult with all morphology data.
        """
        result = MorphologyPipelineResult(instances=list(instances))
        centroids: List[TrichomeCentroid] = []

        for inst in instances:
            if inst.mask is None:
                result.failed += 1
                continue

            mask_array: NDArray[np.uint8] = (
                inst.mask.data if hasattr(inst.mask, "data") else np.array(inst.mask)
            )

            if mask_array.ndim != 2:
                result.failed += 1
                continue

            if mask_array.sum() < self.config.min_mask_area_px:
                result.failed += 1
                continue

            try:
                morph_type = self._classify_instance(mask_array, inst)
                inst.morphology_type = morph_type
                result.total_analyzed += 1

                if morph_type.model_id == "geometric":
                    result.used_geometric += 1
                else:
                    result.used_cnn += 1

                # Collect centroid for density map (import already at module level)
                geo = extract_geometric_descriptors(mask_array)
                centroids.append(
                    TrichomeCentroid(
                        x=geo.centroid_x,
                        y=geo.centroid_y,
                        trichome_type=morph_type.primary_type.value,
                        confidence=float(morph_type.confidence),
                    )
                )

            except Exception as e:
                logger.warning(f"Morphology failed for instance {inst.id[:8]}: {e}")
                result.failed += 1

        # Type distribution
        from collections import Counter
        type_counts = Counter(
            inst.morphology_type.primary_type.value
            for inst in instances
            if inst.morphology_type is not None
        )
        result.type_distribution = dict(type_counts)

        # Density map
        if image_shape and len(centroids) >= 2:
            h, w = image_shape[:2]
            try:
                result.density_map = compute_density_map(
                    centroids=centroids,
                    image_height=h,
                    image_width=w,
                    grid_rows=self.config.density_grid_rows,
                    grid_cols=self.config.density_grid_cols,
                    kde_bandwidth=self.config.kde_bandwidth_px,
                    um_per_pixel=um_per_pixel,
                )
            except Exception as e:
                logger.warning(f"Density map failed: {e}")

        logger.info(
            f"Morphology complete: {result.total_analyzed} analyzed, "
            f"{result.failed} failed, dist={result.type_distribution}"
        )
        return result

    def _classify_instance(
        self, mask: NDArray[np.uint8], inst: Instance
    ) -> MorphologyType:
        """Classify a single instance. CNN first, geometric fallback."""
        # Geometric features (always computed)
        geo = extract_geometric_descriptors(mask)
        stalk, head = detect_stalk_and_head(mask)

        # Try CNN first
        if self._classifier.has_model:
            try:
                crop = self._get_crop(inst)
                if crop is not None:
                    cnn_result = self._classifier.predict_from_crop(crop)
                    # Augment with geometric measurements
                    cnn_result.head_diameter_px = head.head_diameter_px if head else None
                    cnn_result.stalk_length_px = stalk.stalk_length_px
                    cnn_result.head_circularity = head.head_circularity if head else None
                    cnn_result.elongation = geo.elongation
                    return cnn_result
            except Exception as e:
                logger.debug(f"CNN classifier failed, using geometric: {e}")

        # Geometric classification
        return self._classifier.predict_geometric(
            geo=geo,
            stalk=stalk,
            head=head,
        )

    def _get_crop(self, inst: Instance) -> Optional[NDArray[np.uint8]]:
        """Extract instance crop if available."""
        if inst.crop is not None:
            return inst.crop
        return None
