"""
training/samplers/hard_example_sampler.py — Hard example mining sampler.

Implements Online Hard Example Mining (OHEM) and offline hard example selection
for trichome detection training.

Online OHEM: during a forward pass, keep only the top-K% highest-loss samples
for the backward pass, forcing the model to focus on hard examples.

Offline mining: score the dataset by prediction difficulty, return a sampler
that oversamples hard examples.

References:
  Shrivastava et al. (2016). Training Region-based Object Detectors with Online
  Hard Example Mining. CVPR 2016. arXiv:1604.03540.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Optional, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import Sampler


# ---------------------------------------------------------------------------
# Offline hard example index
# ---------------------------------------------------------------------------


@dataclass
class SampleDifficulty:
    """Difficulty record for one training sample."""

    index: int          # index in the dataset
    loss: float         # training loss at last evaluation
    iou: float          # best IoU achieved
    confidence: float   # max detection confidence
    correct: bool       # correctly classified
    epoch: int = 0      # epoch when this was computed


def compute_hardness_score(
    loss: float,
    iou: float,
    confidence: float,
    correct: bool,
) -> float:
    """
    Composite hardness score (higher = harder example).

    Combines:
      - loss (primary signal)
      - IoU miss (low IoU = hard localization)
      - misclassification penalty
      - high confidence + wrong = very hard (adversarial)
    """
    iou_miss = max(0.0, 1.0 - iou)
    incorrect_penalty = 1.0 if not correct else 0.0

    # High confidence + wrong = extra penalty (confident mistakes)
    confident_wrong = confidence * (1.0 - float(correct))

    score = (
        0.5 * min(10.0, loss) / 10.0
        + 0.2 * iou_miss
        + 0.2 * incorrect_penalty
        + 0.1 * confident_wrong
    )
    return round(min(1.0, score), 6)


# ---------------------------------------------------------------------------
# Hard example dataset
# ---------------------------------------------------------------------------


class HardExampleRegistry:
    """
    Tracks per-sample difficulty across training epochs.

    Updated after each epoch via update_batch(). Provides sorted indices
    for oversampling.
    """

    def __init__(self, dataset_size: int) -> None:
        self.dataset_size = dataset_size
        self._records: dict[int, SampleDifficulty] = {}
        self._current_epoch = 0

    def update_batch(
        self,
        indices: list[int],
        losses: list[float],
        ious: list[float],
        confidences: list[float],
        corrects: list[bool],
    ) -> None:
        """Update difficulty records for a batch of samples."""
        for idx, loss, iou, conf, correct in zip(
            indices, losses, ious, confidences, corrects
        ):
            self._records[idx] = SampleDifficulty(
                index=idx,
                loss=loss,
                iou=iou,
                confidence=conf,
                correct=correct,
                epoch=self._current_epoch,
            )

    def advance_epoch(self) -> None:
        self._current_epoch += 1

    def get_hardness_scores(self) -> dict[int, float]:
        """Return {index: hardness_score} for all tracked samples."""
        return {
            idx: compute_hardness_score(
                loss=rec.loss,
                iou=rec.iou,
                confidence=rec.confidence,
                correct=rec.correct,
            )
            for idx, rec in self._records.items()
        }

    def get_sorted_indices(
        self,
        top_fraction: float = 0.5,
    ) -> list[int]:
        """Return indices sorted by hardness (hardest first)."""
        scores = self.get_hardness_scores()
        sorted_indices = sorted(scores.keys(), key=lambda i: scores[i], reverse=True)

        # Include all untracked samples (they haven't been seen yet → treat as hard)
        untracked = [i for i in range(self.dataset_size) if i not in scores]

        # Top fraction of tracked hard + all untracked
        n_hard = max(1, int(len(sorted_indices) * top_fraction))
        return sorted_indices[:n_hard] + untracked


# ---------------------------------------------------------------------------
# OHEM loss reduction
# ---------------------------------------------------------------------------


def ohem_loss(
    losses: torch.Tensor,
    keep_fraction: float = 0.5,
    min_kept: int = 32,
) -> torch.Tensor:
    """
    Online Hard Example Mining: keep top-K% highest-loss samples.

    Args:
        losses: Per-sample loss tensor, shape (N,).
        keep_fraction: Fraction of samples to keep (0-1).
        min_kept: Minimum number of samples to keep (prevents degenerate batches).

    Returns:
        Mean loss over kept samples.
    """
    if losses.numel() == 0:
        return losses.mean()

    n_keep = max(min_kept, int(keep_fraction * losses.numel()))
    n_keep = min(n_keep, losses.numel())

    # Sort losses descending and keep top-n
    sorted_losses, _ = torch.sort(losses, descending=True)
    threshold = sorted_losses[n_keep - 1]

    # Keep samples above threshold
    mask = losses >= threshold
    kept = losses[mask]

    return kept.mean()


# ---------------------------------------------------------------------------
# Hard example sampler (PyTorch Sampler)
# ---------------------------------------------------------------------------


class HardExampleSampler(Sampler):
    """
    PyTorch Sampler that oversamples hard examples.

    In early epochs (warmup_epochs): random sampling (uniform, like default).
    After warmup: mix hard examples (hard_fraction) + random (1 - hard_fraction).

    Usage:
        registry = HardExampleRegistry(len(dataset))
        sampler = HardExampleSampler(registry, dataset_size=len(dataset))
        loader = DataLoader(dataset, sampler=sampler, batch_size=4)

        # After each epoch:
        registry.update_batch(...)
        registry.advance_epoch()
        sampler.advance_epoch()
    """

    def __init__(
        self,
        registry: HardExampleRegistry,
        dataset_size: int,
        hard_fraction: float = 0.5,
        warmup_epochs: int = 5,
        num_samples: Optional[int] = None,
    ) -> None:
        self.registry = registry
        self.dataset_size = dataset_size
        self.hard_fraction = hard_fraction
        self.warmup_epochs = warmup_epochs
        self._epoch = 0
        self._num_samples = num_samples or dataset_size

    def advance_epoch(self) -> None:
        self._epoch += 1

    def __len__(self) -> int:
        return self._num_samples

    def __iter__(self) -> Iterator[int]:
        import random

        if self._epoch < self.warmup_epochs:
            # Warmup: uniform random
            indices = list(range(self.dataset_size))
            random.shuffle(indices)
            return iter(indices[: self._num_samples])

        # Mining phase: mix hard + random
        hard_indices = self.registry.get_sorted_indices(top_fraction=0.5)
        n_hard = int(self.hard_fraction * self._num_samples)
        n_random = self._num_samples - n_hard

        sampled_hard = (hard_indices * (n_hard // max(1, len(hard_indices)) + 1))[:n_hard]
        all_indices = list(range(self.dataset_size))
        random.shuffle(all_indices)
        sampled_random = all_indices[:n_random]

        combined = sampled_hard + sampled_random
        random.shuffle(combined)
        return iter(combined[: self._num_samples])


# ---------------------------------------------------------------------------
# OHEM focal loss wrapper
# ---------------------------------------------------------------------------


class OHEMFocalLoss(torch.nn.Module):
    """
    Combines Focal Loss with OHEM selection.

    For each batch: compute per-sample focal loss, then OHEM-select hardest
    samples before averaging.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[float] = 0.25,
        keep_fraction: float = 0.5,
        min_kept: int = 32,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.keep_fraction = keep_fraction
        self.min_kept = min_kept

    def forward(
        self,
        logits: torch.Tensor,  # (N, C) for multi-class or (N,) for binary
        targets: torch.Tensor,  # (N,) long for multi-class or (N,) float for binary
    ) -> torch.Tensor:
        if logits.dim() == 1 or (logits.dim() == 2 and logits.shape[1] == 1):
            # Binary
            logits = logits.view(-1)
            bce = F.binary_cross_entropy_with_logits(
                logits, targets.float(), reduction="none"
            )
            probs = torch.sigmoid(logits)
            p_t = probs * targets + (1 - probs) * (1 - targets)
            per_sample_loss = bce * (1 - p_t) ** self.gamma
        else:
            # Multi-class
            log_probs = F.log_softmax(logits, dim=1)
            probs = torch.exp(log_probs)
            targets_long = targets.long()
            ce = F.nll_loss(log_probs, targets_long, reduction="none")
            p_t = probs.gather(1, targets_long.unsqueeze(1)).squeeze(1)
            per_sample_loss = ce * (1 - p_t) ** self.gamma

        return ohem_loss(per_sample_loss, self.keep_fraction, self.min_kept)
