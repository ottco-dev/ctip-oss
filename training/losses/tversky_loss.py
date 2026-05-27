"""
training.losses.tversky_loss — Tversky and Focal Tversky losses for segmentation.

Tversky loss generalizes Dice loss by separately weighting false positives (FP)
and false negatives (FN). For trichome segmentation:
  - FN (missed trichome pixels) is usually more costly than FP
  - Small trichome heads → set β > α to penalize FN more

Tversky index:
    TI(α, β) = TP / (TP + α·FP + β·FN)

When α = β = 0.5: Tversky index = Dice coefficient
When α = 0, β = 1: Tversky index = Recall

Focal Tversky Loss:
    FTL = (1 - TI)^γ  (γ > 1 focuses on hard examples with low TI)

Reference:
    Salehi, S.S.M. et al. (2017). Tversky Loss Function for Image Segmentation
    Using 3D Fully Convolutional Deep Networks. MLMI 2017. arXiv:1706.05721.

    Abraham, N. & Khan, N. (2019). A Novel Focal Tversky Loss Function With
    Improved Attention U-Net for Lesion Segmentation.
    ISBI 2019. arXiv:1810.07842.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TverskyLoss(nn.Module):
    """
    Tversky Loss for binary segmentation masks.

    Args:
        alpha: Weight for false positives (FP). Higher = penalizes FP more.
        beta: Weight for false negatives (FN). Higher = penalizes FN more.
        smooth: Laplace smoothing to avoid division by zero.
        reduction: 'mean' | 'sum' | 'none'.
        from_logits: If True, applies sigmoid to inputs first.

    Recommended settings for trichome segmentation:
        alpha=0.3, beta=0.7 — penalizes missed trichomes more than false alarms
        alpha=0.5, beta=0.5 — equivalent to Dice loss
    """

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        smooth: float = 1e-6,
        reduction: str = "mean",
        from_logits: bool = True,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.reduction = reduction
        self.from_logits = from_logits

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Tversky loss.

        Args:
            inputs: Predicted logits or probabilities (B, 1, H, W) or (B, H, W).
            targets: Binary ground truth masks (B, 1, H, W) or (B, H, W).

        Returns:
            Scalar loss.
        """
        if self.from_logits:
            inputs = torch.sigmoid(inputs)

        # Flatten spatial dims
        inputs_flat = inputs.reshape(inputs.shape[0], -1)
        targets_flat = targets.reshape(targets.shape[0], -1).float()

        tp = (inputs_flat * targets_flat).sum(dim=1)
        fp = ((1 - targets_flat) * inputs_flat).sum(dim=1)
        fn = (targets_flat * (1 - inputs_flat)).sum(dim=1)

        tversky_index = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )
        loss = 1.0 - tversky_index

        return self._reduce(loss)

    def _reduce(self, loss: torch.Tensor) -> torch.Tensor:
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss: applies focusing to Tversky index.

    FTL = (1 - TI(α, β))^γ

    When TI is high (easy, well-segmented): (1 - TI)^γ is very small → low loss
    When TI is low (hard, missed trichomes): (1 - TI)^γ ≈ (1 - TI) → full loss

    Recommended γ = 4/3 for medical image segmentation (Abraham & Khan 2019).
    For trichomes: γ = 1.0–2.0 depending on difficulty.
    """

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        gamma: float = 1.33,
        smooth: float = 1e-6,
        reduction: str = "mean",
        from_logits: bool = True,
    ) -> None:
        super().__init__()
        self._tversky = TverskyLoss(
            alpha=alpha,
            beta=beta,
            smooth=smooth,
            reduction="none",  # reduce after applying focal
            from_logits=from_logits,
        )
        self.gamma = gamma
        self.reduction = reduction

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Focal Tversky loss.

        Args:
            inputs: Predicted logits or probabilities.
            targets: Binary ground truth masks.
        """
        tversky_loss = self._tversky(inputs, targets)  # (B,)
        focal_tversky_loss = tversky_loss ** self.gamma

        if self.reduction == "mean":
            return focal_tversky_loss.mean()
        elif self.reduction == "sum":
            return focal_tversky_loss.sum()
        return focal_tversky_loss


class MultiClassTverskyLoss(nn.Module):
    """
    Multi-class Tversky loss via one-vs-rest decomposition.

    Computes Tversky loss for each class separately and averages
    (with optional per-class weighting).

    Usage::

        # For 4 trichome classes
        loss_fn = MultiClassTverskyLoss(num_classes=4, alpha=0.3, beta=0.7)
        # inputs: (B, C, H, W) logits, targets: (B, H, W) class indices
        loss = loss_fn(inputs, targets)
    """

    def __init__(
        self,
        num_classes: int = 4,
        alpha: float = 0.3,
        beta: float = 0.7,
        smooth: float = 1e-6,
        class_weights: list[float] | None = None,
        from_logits: bool = True,
        focal_gamma: float | None = None,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.from_logits = from_logits
        self.focal_gamma = focal_gamma

        if class_weights is not None:
            self.register_buffer(
                "class_weights",
                torch.tensor(class_weights, dtype=torch.float32),
            )
        else:
            self.class_weights = None

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            inputs: (B, C, H, W) raw logits.
            targets: (B, H, W) ground truth class indices [int64].
        """
        if self.from_logits:
            probs = F.softmax(inputs, dim=1)  # (B, C, H, W)
        else:
            probs = inputs

        # One-hot encode targets
        b, c, h, w = probs.shape
        targets_one_hot = F.one_hot(targets.long(), num_classes=c)  # (B, H, W, C)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()  # (B, C, H, W)

        per_class_losses = []

        for cls in range(self.num_classes):
            pred_cls = probs[:, cls]  # (B, H, W)
            tgt_cls = targets_one_hot[:, cls]  # (B, H, W)

            pred_flat = pred_cls.reshape(b, -1)
            tgt_flat = tgt_cls.reshape(b, -1)

            tp = (pred_flat * tgt_flat).sum(dim=1)
            fp = ((1 - tgt_flat) * pred_flat).sum(dim=1)
            fn = (tgt_flat * (1 - pred_flat)).sum(dim=1)

            tversky = (tp + self.smooth) / (
                tp + self.alpha * fp + self.beta * fn + self.smooth
            )
            loss = (1 - tversky).mean()

            if self.focal_gamma is not None:
                loss = loss ** self.focal_gamma

            per_class_losses.append(loss)

        losses = torch.stack(per_class_losses)

        if self.class_weights is not None:
            losses = losses * self.class_weights.to(losses.device)
            return losses.sum() / self.class_weights.sum()

        return losses.mean()


class CombinedSegmentationLoss(nn.Module):
    """
    Combined BCE + Tversky loss for balanced segmentation training.

    Loss = λ_bce · BCE + λ_tversky · Tversky

    Rationale:
    - BCE: pixel-level accuracy, stable gradients
    - Tversky: shape-level overlap, robust to imbalance

    Typical weights for trichome segmentation:
        λ_bce = 0.5, λ_tversky = 0.5 (equal weighting)
    """

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        gamma: float | None = 1.33,
        lambda_bce: float = 0.5,
        lambda_tversky: float = 0.5,
        from_logits: bool = True,
    ) -> None:
        super().__init__()
        self.lambda_bce = lambda_bce
        self.lambda_tversky = lambda_tversky

        if gamma is not None:
            self._tversky_loss = FocalTverskyLoss(
                alpha=alpha, beta=beta, gamma=gamma, from_logits=from_logits
            )
        else:
            self._tversky_loss = TverskyLoss(
                alpha=alpha, beta=beta, from_logits=from_logits
            )

        self.from_logits = from_logits

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(inputs, targets.float())
        tversky = self._tversky_loss(inputs, targets)
        return self.lambda_bce * bce + self.lambda_tversky * tversky
