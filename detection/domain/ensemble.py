"""
detection.domain.ensemble — Multi-model detection ensemble.

WHY ENSEMBLES?
━━━━━━━━━━━━━
Single models have systematic blind spots. A YOLO model and an RTMDet model
trained on the same data with different architectures will have different
failure modes. Combining their outputs via WBF typically improves:
- Recall: +3-8% (catch what the other missed)
- Precision: +1-4% (boxes are more precisely localized)
- mAP: +3-7% depending on data distribution

Cost: 2-3× inference time. Acceptable for offline batch analysis.
Not suitable for real-time video at 30fps.

UNCERTAINTY FROM ENSEMBLE:
━━━━━━━━━━━━━━━━━━━━━━━━━
Ensemble disagreement is a proxy for epistemic uncertainty:
- All models agree → high confidence, low uncertainty
- Models disagree → uncertain, flag for human review

This is the foundation of the active learning pipeline.

Reference:
  Lakshminarayanan, B. et al. (2017). "Simple and Scalable Predictive
  Uncertainty Estimation using Deep Ensembles." NeurIPS 2017.

  Solovyev, R. et al. (2021). "Weighted Boxes Fusion." Image and Vision
  Computing 107, 104117. https://doi.org/10.1016/j.imavis.2021.104117
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from detection.domain.detector import (
    BaseDetector,
    DetectionConfig,
    DetectionResult,
    TrichomeDetector,
)
from shared.core.entities import Detection
from shared.core.value_objects import BoundingBox, Confidence


@dataclass
class EnsembleConfig:
    """Configuration for ensemble detection."""

    iou_threshold_wbf: float = 0.45
    """WBF IoU threshold for merging cross-model detections."""

    skip_box_threshold: float = 0.20
    """
    WBF: boxes with weighted confidence below this are discarded.
    Lower = more inclusive. Start at 0.20 and tune on validation set.
    """

    confidence_weights: list[float] | None = None
    """
    Per-model confidence weights for WBF.
    None = equal weights.
    Larger weight = this model's boxes pull merged box more strongly.
    """

    uncertainty_method: str = "disagreement"
    """
    Method for estimating epistemic uncertainty.
    Options: "disagreement" (variance across models), "entropy"
    """

    uncertainty_threshold_flag: float = 0.15
    """Flag detections with uncertainty above this for human review."""

    parallel: bool = False
    """
    Run models in parallel (requires multiple GPUs or CPU fallback).
    Currently: sequential only. Parallel support coming.
    """


class DetectionEnsemble:
    """
    Ensemble of multiple trichome detectors.

    Combines predictions using Weighted Boxes Fusion (WBF) and provides
    epistemic uncertainty estimates from inter-model disagreement.

    Usage:
        yolo = YOLODetector(...)
        rtmdet = RTMDetDetector(...)

        ensemble = DetectionEnsemble(
            detectors=[yolo, rtmdet],
            config=EnsembleConfig(confidence_weights=[1.2, 1.0])
        )
        result = ensemble.detect(image)
    """

    def __init__(
        self,
        detectors: list[TrichomeDetector],
        config: EnsembleConfig | None = None,
    ) -> None:
        if len(detectors) < 2:
            raise ValueError(
                f"Ensemble requires at least 2 detectors, got {len(detectors)}. "
                f"Use a single detector directly for single-model inference."
            )
        self._detectors = detectors
        self._config = config or EnsembleConfig()

        if self._config.confidence_weights is not None:
            if len(self._config.confidence_weights) != len(detectors):
                raise ValueError(
                    f"confidence_weights length ({len(self._config.confidence_weights)}) "
                    f"must match number of detectors ({len(detectors)})"
                )

    def detect(
        self,
        image: NDArray[np.uint8],
        config: DetectionConfig | None = None,
    ) -> DetectionResult:
        """
        Run ensemble detection.

        Steps:
        1. Run each detector independently
        2. Collect all detections from all models
        3. Apply WBF to merge overlapping detections across models
        4. Compute per-detection uncertainty from inter-model disagreement
        5. Return merged result with uncertainty annotations
        """
        t_start = time.perf_counter()

        # Step 1: Run all detectors
        per_model_results: list[DetectionResult] = []
        for detector in self._detectors:
            result = detector.detect(image, config)
            per_model_results.append(result)

        # Step 2: WBF merge
        h, w = image.shape[:2]
        merged_detections = self._weighted_boxes_fusion(
            per_model_results, image_width=w, image_height=h
        )

        # Step 3: Compute uncertainty
        merged_detections = self._estimate_uncertainty(
            merged_detections, per_model_results, image_width=w, image_height=h
        )

        t_end = time.perf_counter()

        return DetectionResult(
            detections=merged_detections,
            image_id="",
            model_id=f"ensemble[{'+'.join(d.model_id for d in self._detectors)}]",
            inference_time_ms=(t_end - t_start) * 1000,
            image_shape=image.shape,
            num_raw_detections=sum(r.num_raw_detections for r in per_model_results),
            was_augmented=any(r.was_augmented for r in per_model_results),
        )

    def _weighted_boxes_fusion(
        self,
        per_model_results: list[DetectionResult],
        image_width: int,
        image_height: int,
    ) -> list[Detection]:
        """
        WBF implementation.

        Normalizes box coordinates to [0,1] range, applies WBF,
        then denormalizes back to pixel coordinates.

        Implementation follows original WBF paper algorithm.
        """
        if not any(r.detections for r in per_model_results):
            return []

        # Normalize boxes per model
        boxes_list: list[list[list[float]]] = []
        scores_list: list[list[float]] = []
        labels_list: list[list[int]] = []

        weights = self._config.confidence_weights
        if weights is None:
            weights = [1.0] * len(per_model_results)

        for model_result in per_model_results:
            model_boxes = []
            model_scores = []
            model_labels = []

            for det in model_result.detections:
                # Normalize to [0, 1]
                norm_box = [
                    det.bounding_box.x_min / image_width,
                    det.bounding_box.y_min / image_height,
                    det.bounding_box.x_max / image_width,
                    det.bounding_box.y_max / image_height,
                ]
                model_boxes.append(norm_box)
                model_scores.append(float(det.confidence))
                model_labels.append(det.class_id)

            boxes_list.append(model_boxes)
            scores_list.append(model_scores)
            labels_list.append(model_labels)

        # Apply WBF
        try:
            from ensemble_boxes import weighted_boxes_fusion

            fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
                boxes_list,
                scores_list,
                labels_list,
                weights=weights,
                iou_thr=self._config.iou_threshold_wbf,
                skip_box_thr=self._config.skip_box_threshold,
            )
        except ImportError:
            # Fallback: simple NMS across all detections
            import warnings
            warnings.warn(
                "ensemble-boxes package not installed. Falling back to NMS merge. "
                "Install with: uv pip install ensemble-boxes",
                stacklevel=2,
            )
            fused_boxes, fused_scores, fused_labels = self._fallback_nms_merge(
                boxes_list, scores_list, labels_list
            )

        # Denormalize and create Detection objects
        from shared.core.enums import TrichomeType
        from detection.infrastructure.yolo_backend import YOLO_CLASS_MAP

        merged: list[Detection] = []
        for box, score, label in zip(fused_boxes, fused_scores, fused_labels):
            label_int = int(label)
            trichome_type = YOLO_CLASS_MAP.get(label_int, TrichomeType.UNKNOWN)

            try:
                bbox = BoundingBox(
                    x_min=max(0.0, box[0] * image_width),
                    y_min=max(0.0, box[1] * image_height),
                    x_max=min(float(image_width), box[2] * image_width),
                    y_max=min(float(image_height), box[3] * image_height),
                )
            except ValueError:
                continue

            det = Detection(
                id=str(uuid.uuid4()),
                bounding_box=bbox,
                confidence=Confidence(min(float(score), 1.0)),
                trichome_type=trichome_type,
                model_id=f"ensemble[{len(self._detectors)}]",
                class_id=label_int,
            )
            merged.append(det)

        return merged

    def _estimate_uncertainty(
        self,
        merged_detections: list[Detection],
        per_model_results: list[DetectionResult],
        image_width: int,
        image_height: int,
    ) -> list[Detection]:
        """
        Estimate epistemic uncertainty via inter-model disagreement.

        For each merged detection, check how many models detected it
        and with what confidence.

        High uncertainty = large variance in per-model confidence scores
        OR detection only found by a subset of models.

        This is a simplified version. For production, use MC Dropout
        or full deep ensemble uncertainty decomposition.
        """
        all_single_model_dets: list[Detection] = []
        for r in per_model_results:
            all_single_model_dets.extend(r.detections)

        for merged_det in merged_detections:
            # Find which single-model detections overlap with this merged box
            overlapping_confs: list[float] = []

            for single_det in all_single_model_dets:
                iou = merged_det.bounding_box.iou(single_det.bounding_box)
                if iou > 0.3:
                    overlapping_confs.append(float(single_det.confidence))

            if len(overlapping_confs) >= 2:
                # Uncertainty = variance across overlapping detections
                merged_det.uncertainty = float(np.var(overlapping_confs))
            elif len(overlapping_confs) == 1:
                # Detected by only one model = inherently uncertain
                merged_det.uncertainty = 0.25
            else:
                merged_det.uncertainty = 0.30  # No overlap found = edge case

        return merged_detections

    def _fallback_nms_merge(
        self,
        boxes_list: list[list[list[float]]],
        scores_list: list[list[float]],
        labels_list: list[list[int]],
    ) -> tuple[list[list[float]], list[float], list[int]]:
        """Simple NMS fallback when ensemble-boxes not installed."""
        # Flatten all boxes
        all_boxes: list[list[float]] = []
        all_scores: list[float] = []
        all_labels: list[int] = []

        for boxes, scores, labels in zip(boxes_list, scores_list, labels_list):
            all_boxes.extend(boxes)
            all_scores.extend(scores)
            all_labels.extend(labels)

        if not all_boxes:
            return [], [], []

        # Sort by score
        order = sorted(range(len(all_scores)), key=lambda i: all_scores[i], reverse=True)

        kept_boxes: list[list[float]] = []
        kept_scores: list[float] = []
        kept_labels: list[int] = []

        def iou(b1: list[float], b2: list[float]) -> float:
            ix1 = max(b1[0], b2[0])
            iy1 = max(b1[1], b2[1])
            ix2 = min(b1[2], b2[2])
            iy2 = min(b1[3], b2[3])
            iw = max(0, ix2 - ix1)
            ih = max(0, iy2 - iy1)
            inter = iw * ih
            area1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
            area2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
            union = area1 + area2 - inter
            return inter / union if union > 0 else 0.0

        used = set()
        for i in order:
            if i in used:
                continue
            kept_boxes.append(all_boxes[i])
            kept_scores.append(all_scores[i])
            kept_labels.append(all_labels[i])
            for j in order:
                if j != i and j not in used:
                    if iou(all_boxes[i], all_boxes[j]) > self._config.iou_threshold_wbf:
                        used.add(j)
            used.add(i)

        return kept_boxes, kept_scores, kept_labels

    def __repr__(self) -> str:
        model_ids = [d.model_id for d in self._detectors]
        return f"DetectionEnsemble(models={model_ids})"
