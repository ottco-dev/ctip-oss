"""
training.losses.focal_loss — Focal loss for class imbalance in trichome detection.

Trichome class distribution (typical):
  - capitate-stalked: ~60%  (most common, easy examples)
  - capitate-sessile: ~25%
  - bulbous: ~10%
  - non-glandular: ~5%  (hardest, rarest)

Standard BCE with class weights handles imbalance moderately.
Focal loss additionally down-weights easy examples, forcing the model
to focus on hard negatives (small bulbous trichomes, blurry stalks).

Reference:
    Lin, T.Y. et al. (2017). Focal Loss for Dense Object Detection.
    ICCV 2017. arXiv:1708.02002.
    DOI: 10.1109/ICCV.2017.324
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss implementation for binary/multi-class classification.

    Formula:
        FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)

    where p_t is the model's estimated probability for the correct class,
    α_t is the class weighting factor, and γ is the focusing parameter.

    Hyperparameter guidelines for trichome detection:
    - γ = 2.0: standard; reduces loss contribution of easy examples by ~4×
    - γ = 0.5: mild focusing (good starting point)
    - α = 0.25: compensate for class imbalance in one-vs-rest setting
    - alpha_weights: per-class [stalked=1.0, sessile=2.0, bulbous=5.0, ngl=8.0]

    Usage::

        loss_fn = FocalLoss(gamma=2.0, alpha=0.25, reduction="mean")
        loss = loss_fn(predictions, targets)

        # With per-class weights:
        weights = torch.tensor([1.0, 2.0, 5.0, 8.0])
        loss_fn = FocalLoss(gamma=2.0, alpha=weights, reduction="mean")
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float | list[float] | torch.Tensor | None = None,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

        if alpha is None:
            self.alpha = None
        elif isinstance(alpha, (int, float)):
            self.register_buffer("alpha", torch.tensor(float(alpha)))
        elif isinstance(alpha, (list, tuple)):
            self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))
        elif isinstance(alpha, torch.Tensor):
            self.register_buffer("alpha", alpha.float())
        else:
            raise ValueError(f"Invalid alpha type: {type(alpha)}")

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            inputs: Raw logits (B, C) for multi-class or (B,) for binary.
                    For YOLO classification head: (N, num_classes).
            targets: Ground truth class indices (B,) [int64] for multi-class,
                     or binary labels (B,) [float] for binary.

        Returns:
            Scalar loss value (or unreduced tensor if reduction='none').
        """
        if inputs.dim() == 1 or (inputs.dim() == 2 and inputs.shape[1] == 1):
            return self._binary_focal(inputs.squeeze(-1), targets.float())
        else:
            return self._multiclass_focal(inputs, targets)

    def _binary_focal(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Binary focal loss via sigmoid."""
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = torch.sigmoid(inputs) * targets + (1 - torch.sigmoid(inputs)) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        loss = focal_weight * bce

        if self.alpha is not None and self.alpha.dim() == 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        return self._reduce(loss)

    def _multiclass_focal(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Multi-class focal loss via softmax."""
        if self.label_smoothing > 0:
            num_classes = inputs.shape[1]
            smooth_targets = torch.full_like(inputs, self.label_smoothing / num_classes)
            smooth_targets.scatter_(1, targets.unsqueeze(1).long(), 1 - self.label_smoothing + self.label_smoothing / num_classes)

            log_probs = F.log_softmax(inputs, dim=1)
            loss = -(smooth_targets * log_probs).sum(dim=1)

            probs = torch.softmax(inputs, dim=1)
            p_t = (probs * smooth_targets).sum(dim=1)
        else:
            log_probs = F.log_softmax(inputs, dim=1)
            target_log_probs = log_probs.gather(1, targets.unsqueeze(1).long()).squeeze(1)

            probs = torch.softmax(inputs, dim=1)
            p_t = probs.gather(1, targets.unsqueeze(1).long()).squeeze(1)

            loss = -target_log_probs

        focal_weight = (1 - p_t) ** self.gamma
        loss = focal_weight * loss

        if self.alpha is not None and self.alpha.dim() > 0:
            # Per-class alpha weighting
            alpha_t = self.alpha[targets.long()]
            loss = alpha_t * loss

        return self._reduce(loss)

    def _reduce(self, loss: torch.Tensor) -> torch.Tensor:
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class SigmoidFocalLoss(nn.Module):
    """
    Sigmoid focal loss for YOLO-style multi-label classification.

    Treats each class as independent binary classification.
    More appropriate when a trichome can have multiple class attributes.

    Usage::

        loss_fn = SigmoidFocalLoss(gamma=2.0, alpha=0.25)
        # inputs: (B, C) logits, targets: (B, C) binary labels
        loss = loss_fn(inputs, targets)
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.25,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            inputs: (B, C) raw logits.
            targets: (B, C) binary labels in {0, 1}.
        """
        prob = torch.sigmoid(inputs)
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

        p_t = prob * targets + (1 - prob) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        loss = alpha_t * focal_weight * bce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_focal_loss(
    num_classes: int = 4,
    gamma: float = 2.0,
    class_weights: list[float] | None = None,
    reduction: str = "mean",
) -> FocalLoss:
    """
    Build focal loss with optional per-class weights for trichome detection.

    Default class weights tuned for typical trichome class distribution:
    [stalked=1.0, sessile=2.4, bulbous=6.0, non-glandular=12.0]

    Args:
        num_classes: Number of trichome classes (default 4).
        gamma: Focusing parameter (2.0 standard, 0.5 mild).
        class_weights: Per-class alpha weights. If None, uses inverse frequency defaults.
        reduction: 'mean' | 'sum' | 'none'.
    """
    if class_weights is None:
        # Derived from typical class distribution: [60%, 25%, 10%, 5%]
        # Weight = 1 / relative_frequency (normalized to min=1.0)
        class_weights = [1.0, 2.4, 6.0, 12.0][:num_classes]

    alpha = torch.tensor(class_weights, dtype=torch.float32)
    return FocalLoss(gamma=gamma, alpha=alpha, reduction=reduction)
