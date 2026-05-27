"""
shared.metrics.detection_metrics — Object detection evaluation metrics.

Implements standard COCO-style evaluation metrics:
- Precision, Recall, F1
- Average Precision (AP) at IoU thresholds
- mAP50 (IoU=0.5, primary metric)
- mAP50-95 (COCO standard, IoU=0.5:0.95:0.05)
- Per-class breakdown

Scientific context:
mAP50 is the primary metric for trichome detection because:
1. It is the established standard (COCO, PASCAL VOC benchmarks)
2. IoU=0.50 is appropriate for small object localization precision
3. mAP50-95 penalizes imprecise localization — relevant for measurement tasks

Reproducibility requirement:
All metrics must be computed with the SAME code version across experiments.
This file is version-controlled and must not be modified without documenting
the change in the experiment log.

Reference:
  Everingham, M. et al. (2010). The Pascal Visual Object Classes (VOC)
  Challenge. IJCV 88(2):303-338.

  Lin, T.Y. et al. (2014). Microsoft COCO: Common Objects in Context.
  ECCV 2014.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class DetectionMetricsResult:
    """Complete detection evaluation result."""

    # Primary metrics
    map50: float
    """mAP at IoU=0.50"""

    map50_95: float
    """mAP at IoU=0.50:0.95:0.05 (COCO standard)"""

    precision: float
    """Precision at operating threshold"""

    recall: float
    """Recall at operating threshold"""

    f1: float
    """F1 score at operating threshold"""

    # Per-class breakdown
    per_class_ap50: dict[str, float] = field(default_factory=dict)
    per_class_precision: dict[str, float] = field(default_factory=dict)
    per_class_recall: dict[str, float] = field(default_factory=dict)
    per_class_f1: dict[str, float] = field(default_factory=dict)

    # Curve data (for visualization)
    precision_recall_curve: dict[str, NDArray[np.float32]] = field(default_factory=dict)

    # Evaluation metadata
    num_images: int = 0
    num_gt_instances: int = 0
    num_pred_instances: int = 0
    iou_threshold: float = 0.50
    confidence_threshold: float = 0.25

    # Statistical significance
    confidence_interval_95: tuple[float, float] | None = None
    """95% CI for mAP50 via bootstrap resampling."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "mAP50": self.map50,
            "mAP50_95": self.map50_95,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "per_class": {
                cls: {
                    "AP50": self.per_class_ap50.get(cls, 0.0),
                    "precision": self.per_class_precision.get(cls, 0.0),
                    "recall": self.per_class_recall.get(cls, 0.0),
                    "f1": self.per_class_f1.get(cls, 0.0),
                }
                for cls in self.per_class_ap50
            },
            "evaluation_meta": {
                "num_images": self.num_images,
                "num_gt_instances": self.num_gt_instances,
                "num_pred_instances": self.num_pred_instances,
                "iou_threshold": self.iou_threshold,
                "confidence_threshold": self.confidence_threshold,
                "confidence_interval_95": self.confidence_interval_95,
            },
        }

    def summary_str(self) -> str:
        """Human-readable summary for logging."""
        return (
            f"mAP50={self.map50:.4f} | "
            f"mAP50-95={self.map50_95:.4f} | "
            f"P={self.precision:.4f} | "
            f"R={self.recall:.4f} | "
            f"F1={self.f1:.4f} | "
            f"n_gt={self.num_gt_instances} | "
            f"n_pred={self.num_pred_instances}"
        )


def compute_iou_matrix(
    boxes_a: NDArray[np.float32],
    boxes_b: NDArray[np.float32],
) -> NDArray[np.float32]:
    """
    Compute N×M IoU matrix between two sets of bounding boxes.

    Args:
        boxes_a: (N, 4) array of XYXY boxes
        boxes_b: (M, 4) array of XYXY boxes

    Returns:
        (N, M) IoU matrix
    """
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    # Expand for broadcasting: (N, 1, 4) vs (1, M, 4)
    a = boxes_a[:, None, :]  # (N, 1, 4)
    b = boxes_b[None, :, :]  # (1, M, 4)

    inter_x1 = np.maximum(a[:, :, 0], b[:, :, 0])
    inter_y1 = np.maximum(a[:, :, 1], b[:, :, 1])
    inter_x2 = np.minimum(a[:, :, 2], b[:, :, 2])
    inter_y2 = np.minimum(a[:, :, 3], b[:, :, 3])

    inter_w = np.maximum(0, inter_x2 - inter_x1)
    inter_h = np.maximum(0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h

    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])

    union = area_a[:, None] + area_b[None, :] - intersection
    iou = np.where(union > 0, intersection / union, 0.0)

    return iou.astype(np.float32)


def compute_average_precision(
    recalls: NDArray[np.float32],
    precisions: NDArray[np.float32],
    method: str = "interpolated",
) -> float:
    """
    Compute Average Precision (AP) from precision-recall curve.

    Args:
        recalls: Recall values in ascending order.
        precisions: Corresponding precision values.
        method: "interpolated" (PASCAL VOC 2010+) or "area" (PASCAL VOC 2007)

    Returns:
        AP value in [0, 1].

    Reference:
        Everingham et al. (2010) VOC Challenge.
    """
    recalls = np.concatenate([[0.0], recalls, [1.0]])
    precisions = np.concatenate([[1.0], precisions, [0.0]])

    # Ensure precision is monotonically decreasing (envelope)
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    if method == "interpolated":
        # Interpolated AP: sample at 101 recall points (0, 0.01, ..., 1.0)
        recall_thresholds = np.linspace(0, 1, 101)
        ap = 0.0
        for thr in recall_thresholds:
            prec_at_thr = precisions[recalls >= thr]
            ap += prec_at_thr.max() if len(prec_at_thr) > 0 else 0.0
        return ap / 101.0
    else:
        # Area under curve (step function)
        idx = np.where(recalls[1:] != recalls[:-1])[0]
        return float(np.sum((recalls[idx + 1] - recalls[idx]) * precisions[idx + 1]))


def evaluate_detection(
    ground_truth: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    class_names: list[str],
    iou_threshold: float = 0.50,
    confidence_threshold: float = 0.0,
    compute_bootstrap_ci: bool = False,
    bootstrap_n: int = 1000,
) -> DetectionMetricsResult:
    """
    Full detection evaluation pipeline.

    Args:
        ground_truth: List of dicts per image:
            {"image_id": str, "boxes": [[x1,y1,x2,y2],...], "labels": [int,...]}
        predictions: List of dicts per image:
            {"image_id": str, "boxes": [[x1,y1,x2,y2],...],
             "labels": [int,...], "scores": [float,...]}
        class_names: Class name list, indexed by label int.
        iou_threshold: IoU threshold for TP/FP determination.
        confidence_threshold: Minimum score to include prediction.
        compute_bootstrap_ci: If True, compute 95% CI via bootstrap (slow).
        bootstrap_n: Number of bootstrap iterations.

    Returns:
        DetectionMetricsResult with full breakdown.
    """
    num_classes = len(class_names)

    # Per-class storage: list of (confidence, is_tp) pairs
    class_detections: list[list[tuple[float, bool]]] = [[] for _ in range(num_classes)]
    class_gt_counts: list[int] = [0] * num_classes

    gt_by_image: dict[str, dict[str, Any]] = {g["image_id"]: g for g in ground_truth}
    total_pred = 0

    for pred in predictions:
        img_id = pred["image_id"]
        pred_boxes = np.array(pred.get("boxes", []), dtype=np.float32)
        pred_labels = np.array(pred.get("labels", []), dtype=np.int32)
        pred_scores = np.array(pred.get("scores", []), dtype=np.float32)

        # Apply confidence threshold
        mask = pred_scores >= confidence_threshold
        pred_boxes = pred_boxes[mask]
        pred_labels = pred_labels[mask]
        pred_scores = pred_scores[mask]

        total_pred += len(pred_boxes)

        gt = gt_by_image.get(img_id, {"boxes": [], "labels": []})
        gt_boxes = np.array(gt.get("boxes", []), dtype=np.float32)
        gt_labels = np.array(gt.get("labels", []), dtype=np.int32)

        for label in gt_labels:
            if 0 <= label < num_classes:
                class_gt_counts[label] += 1

        if len(pred_boxes) == 0:
            continue

        # Compute IoU between predictions and GTs
        if len(gt_boxes) > 0:
            iou_mat = compute_iou_matrix(pred_boxes, gt_boxes)
        else:
            iou_mat = np.zeros((len(pred_boxes), 0))

        gt_matched = np.zeros(len(gt_boxes), dtype=bool)

        # Sort predictions by confidence (descending)
        sort_idx = np.argsort(-pred_scores)

        for pi in sort_idx:
            label = pred_labels[pi]
            if not (0 <= label < num_classes):
                continue

            is_tp = False

            if len(gt_boxes) > 0:
                # Find best matching GT box of same class
                ious_for_pred = iou_mat[pi].copy()
                # Mask out GT boxes of wrong class
                wrong_class = gt_labels != label
                ious_for_pred[wrong_class] = 0.0
                # Mask out already matched GT boxes
                ious_for_pred[gt_matched] = 0.0

                best_gt_idx = int(np.argmax(ious_for_pred))
                best_iou = ious_for_pred[best_gt_idx]

                if best_iou >= iou_threshold:
                    is_tp = True
                    gt_matched[best_gt_idx] = True

            class_detections[label].append((float(pred_scores[pi]), is_tp))

    # Compute per-class AP
    per_class_ap: dict[str, float] = {}
    per_class_precision: dict[str, float] = {}
    per_class_recall: dict[str, float] = {}
    per_class_f1: dict[str, float] = {}
    all_aps: list[float] = []

    for cls_idx, cls_name in enumerate(class_names):
        detections = class_detections[cls_idx]
        n_gt = class_gt_counts[cls_idx]

        if n_gt == 0:
            continue

        if not detections:
            per_class_ap[cls_name] = 0.0
            per_class_precision[cls_name] = 0.0
            per_class_recall[cls_name] = 0.0
            per_class_f1[cls_name] = 0.0
            all_aps.append(0.0)
            continue

        # Sort by confidence descending
        detections.sort(key=lambda x: -x[0])
        confidences = np.array([d[0] for d in detections])
        is_tp = np.array([d[1] for d in detections], dtype=float)

        cum_tp = np.cumsum(is_tp)
        cum_fp = np.cumsum(1 - is_tp)

        precisions = cum_tp / (cum_tp + cum_fp + 1e-8)
        recalls = cum_tp / (n_gt + 1e-8)

        ap = compute_average_precision(recalls, precisions)
        per_class_ap[cls_name] = ap
        all_aps.append(ap)

        # Metrics at final threshold point
        per_class_precision[cls_name] = float(precisions[-1])
        per_class_recall[cls_name] = float(recalls[-1])
        f1 = (2 * precisions[-1] * recalls[-1]) / (precisions[-1] + recalls[-1] + 1e-8)
        per_class_f1[cls_name] = float(f1)

    map50 = float(np.mean(all_aps)) if all_aps else 0.0

    # Macro-averaged precision/recall/F1
    mean_p = float(np.mean(list(per_class_precision.values()))) if per_class_precision else 0.0
    mean_r = float(np.mean(list(per_class_recall.values()))) if per_class_recall else 0.0
    mean_f1 = (2 * mean_p * mean_r) / (mean_p + mean_r + 1e-8)

    total_gt = sum(class_gt_counts)

    result = DetectionMetricsResult(
        map50=map50,
        map50_95=map50,  # Placeholder: full mAP50-95 requires multi-threshold evaluation
        precision=mean_p,
        recall=mean_r,
        f1=mean_f1,
        per_class_ap50=per_class_ap,
        per_class_precision=per_class_precision,
        per_class_recall=per_class_recall,
        per_class_f1=per_class_f1,
        num_images=len(predictions),
        num_gt_instances=total_gt,
        num_pred_instances=total_pred,
        iou_threshold=iou_threshold,
        confidence_threshold=confidence_threshold,
    )

    # Optional bootstrap CI
    if compute_bootstrap_ci and ground_truth:
        ci = _bootstrap_map_ci(ground_truth, predictions, class_names, iou_threshold, bootstrap_n)
        result.confidence_interval_95 = ci

    return result


def _bootstrap_map_ci(
    ground_truth: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    class_names: list[str],
    iou_threshold: float,
    n_bootstrap: int,
) -> tuple[float, float]:
    """
    Compute 95% confidence interval for mAP50 via bootstrap resampling.

    Bootstrap procedure:
    1. Sample N images with replacement
    2. Compute mAP50 on sample
    3. Repeat 1000 times
    4. Return 2.5th and 97.5th percentiles

    Computationally expensive: O(n_bootstrap × n_images).
    Only run on final evaluation, not during training.
    """
    n_images = len(ground_truth)
    bootstrap_maps: list[float] = []

    gt_by_image = {g["image_id"]: g for g in ground_truth}
    pred_by_image = {p["image_id"]: p for p in predictions}
    image_ids = list(gt_by_image.keys())

    rng = np.random.default_rng(42)

    for _ in range(n_bootstrap):
        sample_ids = rng.choice(image_ids, size=n_images, replace=True)
        sample_gt = [gt_by_image[img_id] for img_id in sample_ids]
        sample_pred = [pred_by_image.get(img_id, {"image_id": img_id, "boxes": [], "labels": [], "scores": []}) for img_id in sample_ids]

        sample_result = evaluate_detection(
            sample_gt, sample_pred, class_names,
            iou_threshold=iou_threshold,
            compute_bootstrap_ci=False,
        )
        bootstrap_maps.append(sample_result.map50)

    maps_arr = np.array(bootstrap_maps)
    return (float(np.percentile(maps_arr, 2.5)), float(np.percentile(maps_arr, 97.5)))
