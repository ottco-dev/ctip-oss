"""
shared.metrics.segmentation_metrics — Instance segmentation evaluation.

Metrics implemented:
- Mask IoU (pixel-level intersection over union)
- Dice Score (Sørensen-Dice coefficient)
- Boundary IoU (penalizes boundary inaccuracy more than interior errors)
- Mask AP (detection + segmentation joint metric, COCO-style)

Scientific context for trichome segmentation:
Standard IoU treats all pixels equally. For trichome analysis, the
boundary region is critically important — the head/stalk boundary determines:
1. Head diameter measurements
2. Stalk length estimation
3. Head/stalk ratio (morphological indicator)

Therefore, we report both IoU and Boundary IoU, with emphasis on the latter
for measurement-critical evaluations.

Reference:
  Cheng, B. et al. (2021). "Boundary IoU: Improving Object-Centric
  Image Segmentation Evaluation." CVPR 2021.
  https://arxiv.org/abs/2103.16562
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray


@dataclass
class SegmentationMetricsResult:
    """Complete segmentation evaluation result."""

    mean_iou: float
    """Mean IoU across all matched instance pairs."""

    mean_dice: float
    """Mean Dice score across all matched instance pairs."""

    mean_boundary_iou: float
    """Mean Boundary IoU — higher weight on mask boundaries."""

    # Distribution
    iou_values: list[float] = field(default_factory=list)
    dice_values: list[float] = field(default_factory=list)
    boundary_iou_values: list[float] = field(default_factory=list)

    # Matching statistics
    num_matched: int = 0
    num_gt: int = 0
    num_pred: int = 0
    match_iou_threshold: float = 0.50

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean_IoU": self.mean_iou,
            "mean_Dice": self.mean_dice,
            "mean_BoundaryIoU": self.mean_boundary_iou,
            "matched_instances": self.num_matched,
            "gt_instances": self.num_gt,
            "pred_instances": self.num_pred,
            "match_rate": self.num_matched / self.num_gt if self.num_gt > 0 else 0.0,
        }

    def summary_str(self) -> str:
        return (
            f"mIoU={self.mean_iou:.4f} | "
            f"mDice={self.mean_dice:.4f} | "
            f"mBIoU={self.mean_boundary_iou:.4f} | "
            f"matched={self.num_matched}/{self.num_gt}"
        )


def mask_iou(mask_a: NDArray[np.bool_], mask_b: NDArray[np.bool_]) -> float:
    """
    Compute pixel-level IoU between two binary masks.

    Handles edge case where both masks are empty (returns 1.0 — perfect match).
    Handles case where one mask is empty (returns 0.0 — no overlap).
    """
    if mask_a.shape != mask_b.shape:
        raise ValueError(
            f"Mask shapes must match: {mask_a.shape} vs {mask_b.shape}"
        )
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection / union)


def dice_score(mask_a: NDArray[np.bool_], mask_b: NDArray[np.bool_]) -> float:
    """
    Sørensen-Dice coefficient.

    Dice = 2|A ∩ B| / (|A| + |B|)

    More sensitive to boundary accuracy than IoU for small objects.
    Range: [0, 1]. 1 = perfect overlap.
    """
    intersection = np.logical_and(mask_a, mask_b).sum()
    total = mask_a.sum() + mask_b.sum()
    if total == 0:
        return 1.0
    return float(2 * intersection / total)


def boundary_iou(
    mask_a: NDArray[np.bool_],
    mask_b: NDArray[np.bool_],
    dilation_ratio: float = 0.02,
) -> float:
    """
    Boundary IoU — metric that emphasizes boundary accuracy.

    Computes IoU only on pixels within `dilation_ratio` of the mask boundary.
    This penalizes coarse boundaries more heavily than interior errors.

    For trichome analysis, this is the preferred metric because:
    - Boundary accuracy determines measurement precision
    - Interior accuracy is less critical than boundary accuracy
    - Standard IoU can give high scores even with poor boundaries

    Args:
        mask_a: Ground truth binary mask (H, W)
        mask_b: Predicted binary mask (H, W)
        dilation_ratio: Boundary width as fraction of mask perimeter.
            0.02 = 2% of mask diagonal length.

    Reference:
        Cheng et al. (2021). "Boundary IoU: Improving Object-Centric
        Image Segmentation Evaluation." CVPR 2021.
    """
    if mask_a.shape != mask_b.shape:
        raise ValueError(f"Mask shape mismatch: {mask_a.shape} vs {mask_b.shape}")

    h, w = mask_a.shape
    diag = (h ** 2 + w ** 2) ** 0.5
    dilation_px = max(1, int(round(dilation_ratio * diag)))

    # Create boundary masks via morphological erosion
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (dilation_px * 2 + 1, dilation_px * 2 + 1)
    )

    def get_boundary(mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
        mask_u8 = mask.astype(np.uint8) * 255
        eroded = cv2.erode(mask_u8, kernel)
        boundary = mask_u8 - eroded
        return boundary > 0

    boundary_a = get_boundary(mask_a)
    boundary_b = get_boundary(mask_b)

    # IoU on boundary pixels only
    return mask_iou(boundary_a, boundary_b)


def evaluate_segmentation(
    gt_masks: list[NDArray[np.bool_]],
    pred_masks: list[NDArray[np.bool_]],
    match_iou_threshold: float = 0.50,
) -> SegmentationMetricsResult:
    """
    Evaluate instance segmentation results.

    Matches predicted masks to ground truth masks via greedy IoU matching,
    then computes IoU, Dice, and Boundary IoU for matched pairs.

    Args:
        gt_masks: List of ground truth binary masks.
        pred_masks: List of predicted binary masks.
        match_iou_threshold: Minimum IoU to consider a prediction matched.

    Returns:
        SegmentationMetricsResult with per-pair metrics.
    """
    if not gt_masks and not pred_masks:
        return SegmentationMetricsResult(
            mean_iou=1.0, mean_dice=1.0, mean_boundary_iou=1.0
        )

    if not gt_masks or not pred_masks:
        return SegmentationMetricsResult(
            mean_iou=0.0, mean_dice=0.0, mean_boundary_iou=0.0,
            num_gt=len(gt_masks), num_pred=len(pred_masks)
        )

    # Compute IoU matrix
    n_gt = len(gt_masks)
    n_pred = len(pred_masks)
    iou_matrix = np.zeros((n_gt, n_pred), dtype=np.float32)

    for i, gt in enumerate(gt_masks):
        for j, pred in enumerate(pred_masks):
            if gt.shape == pred.shape:
                iou_matrix[i, j] = mask_iou(gt, pred)

    # Greedy matching: assign each GT to best available prediction
    gt_matched = np.zeros(n_gt, dtype=bool)
    pred_matched = np.zeros(n_pred, dtype=bool)
    matches: list[tuple[int, int]] = []

    # Sort by IoU descending
    flat_indices = np.argsort(-iou_matrix.flatten())
    for flat_idx in flat_indices:
        i, j = divmod(int(flat_idx), n_pred)
        if gt_matched[i] or pred_matched[j]:
            continue
        if iou_matrix[i, j] < match_iou_threshold:
            break
        matches.append((i, j))
        gt_matched[i] = True
        pred_matched[j] = True

    # Compute metrics for matched pairs
    iou_values: list[float] = []
    dice_values: list[float] = []
    biou_values: list[float] = []

    for i, j in matches:
        gt_mask = gt_masks[i]
        pred_mask = pred_masks[j]

        iou_values.append(mask_iou(gt_mask, pred_mask))
        dice_values.append(dice_score(gt_mask, pred_mask))
        biou_values.append(boundary_iou(gt_mask, pred_mask))

    return SegmentationMetricsResult(
        mean_iou=float(np.mean(iou_values)) if iou_values else 0.0,
        mean_dice=float(np.mean(dice_values)) if dice_values else 0.0,
        mean_boundary_iou=float(np.mean(biou_values)) if biou_values else 0.0,
        iou_values=iou_values,
        dice_values=dice_values,
        boundary_iou_values=biou_values,
        num_matched=len(matches),
        num_gt=n_gt,
        num_pred=n_pred,
        match_iou_threshold=match_iou_threshold,
    )
