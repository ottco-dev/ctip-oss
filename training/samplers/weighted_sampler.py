"""
training.samplers.weighted_sampler — Class-balanced weighted sampler for trichome datasets.

Problem: Trichome class distribution is heavily imbalanced:
  - capitate-stalked: ~60%
  - capitate-sessile: ~25%
  - bulbous: ~10%
  - non-glandular: ~5%

Simple random sampling → model sees mostly stalked trichomes, ignores bulbous.

Solution: Weighted sampling where each sample has probability proportional
to 1/(class_frequency) for its most-represented class.

Two strategies:
1. InverseFrequencyWeighting: weight = 1/count (pure balancing)
2. EffectiveSampleWeighting: weight = (1 - β^n) where β controls smoothness
   (Cui et al. 2019, "Class-Balanced Loss Based on Effective Number of Samples")

Reference:
    Cui, Y. et al. (2019). Class-Balanced Loss Based on Effective Number
    of Samples. CVPR 2019. arXiv:1901.05555.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler, Dataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Weight computation utilities
# ---------------------------------------------------------------------------

def compute_inverse_frequency_weights(
    class_counts: dict[int, int],
    num_classes: int,
    power: float = 1.0,
) -> dict[int, float]:
    """
    Compute sampling weights as 1/(count^power) per class.

    Args:
        class_counts: {class_id: count} mapping.
        num_classes: Total number of classes.
        power: Exponent for weight (1.0 = full inverse, 0.5 = square root).

    Returns:
        {class_id: weight} normalized to sum to 1.
    """
    total = sum(class_counts.values())
    weights = {}

    for cls in range(num_classes):
        count = class_counts.get(cls, 1)
        weights[cls] = (total / max(count, 1)) ** power

    # Normalize to sum to num_classes (so weights are meaningful)
    w_sum = sum(weights.values())
    if w_sum > 0:
        weights = {k: v * num_classes / w_sum for k, v in weights.items()}

    return weights


def compute_effective_sample_weights(
    class_counts: dict[int, int],
    num_classes: int,
    beta: float = 0.9999,
) -> dict[int, float]:
    """
    Compute effective-number-based weights (Cui et al. 2019).

    Effective number of samples: E_n = (1 - β^n) / (1 - β)

    More numerically stable than pure inverse frequency for very rare classes.

    Args:
        class_counts: {class_id: count} mapping.
        num_classes: Total number of classes.
        beta: Smoothing factor. 0.9 = mild, 0.9999 = strong (near inverse frequency).

    Returns:
        {class_id: weight} dict.
    """
    weights = {}

    for cls in range(num_classes):
        count = class_counts.get(cls, 1)
        effective_n = (1 - beta ** count) / (1 - beta)
        weights[cls] = 1.0 / max(effective_n, 1e-8)

    # Normalize
    w_sum = sum(weights.values())
    if w_sum > 0:
        weights = {k: v / w_sum for k, v in weights.items()}

    return weights


# ---------------------------------------------------------------------------
# Sample-level weight assignment
# ---------------------------------------------------------------------------

def assign_sample_weights(
    sample_classes: Sequence[int | list[int]],
    class_weights: dict[int, float],
    multi_label_strategy: str = "max",
) -> list[float]:
    """
    Assign a sampling weight to each training sample.

    For single-class samples: weight = class_weights[class_id]
    For multi-class images: aggregate per multi_label_strategy.

    Args:
        sample_classes: List of class labels per sample.
                        Each element is either int (single class)
                        or list[int] (multi-class image).
        class_weights: Per-class sampling weights.
        multi_label_strategy: 'max' | 'mean' | 'sum' for multi-class images.
                               'max' focuses on rarest class present.

    Returns:
        List of float weights, one per sample.
    """
    sample_weights = []

    for cls_labels in sample_classes:
        if isinstance(cls_labels, int):
            cls_labels = [cls_labels]

        per_class = [class_weights.get(c, 1.0) for c in cls_labels]

        if multi_label_strategy == "max":
            w = max(per_class)
        elif multi_label_strategy == "mean":
            w = sum(per_class) / len(per_class)
        elif multi_label_strategy == "sum":
            w = sum(per_class)
        else:
            w = max(per_class)

        sample_weights.append(w)

    return sample_weights


# ---------------------------------------------------------------------------
# WeightedTrichomeSampler
# ---------------------------------------------------------------------------

class WeightedTrichomeSampler:
    """
    Class-balanced sampler for trichome detection datasets.

    Creates a WeightedRandomSampler with:
    - Automatic class count computation from annotations
    - Choice of weight scheme (inverse frequency or effective samples)
    - Support for single-class and multi-class images

    Usage::

        sampler = WeightedTrichomeSampler(
            annotations=label_list,     # list of class_ids per sample
            num_classes=4,
            strategy="inverse_frequency",
            num_samples=len(dataset),   # or 2x for oversampling
        )
        dataloader = DataLoader(dataset, sampler=sampler.get_sampler())
    """

    def __init__(
        self,
        annotations: Sequence[int | list[int]],
        num_classes: int = 4,
        strategy: str = "inverse_frequency",
        beta: float = 0.9999,
        power: float = 1.0,
        num_samples: int | None = None,
        replacement: bool = True,
        multi_label_strategy: str = "max",
    ) -> None:
        """
        Args:
            annotations: Class labels per sample. int for single-class, list for multi.
            num_classes: Total number of trichome classes.
            strategy: 'inverse_frequency' | 'effective_samples'.
            beta: β for effective samples strategy.
            power: Exponent for inverse frequency strategy.
            num_samples: Number of samples per epoch. Defaults to len(annotations).
            replacement: Sample with replacement (required for weighted sampling).
            multi_label_strategy: How to aggregate weights for multi-class images.
        """
        self.num_classes = num_classes
        self.replacement = replacement
        self.num_samples = num_samples or len(annotations)

        # Count classes
        all_classes: list[int] = []
        for ann in annotations:
            if isinstance(ann, int):
                all_classes.append(ann)
            else:
                all_classes.extend(ann)

        class_counts = dict(Counter(all_classes))
        for c in range(num_classes):
            if c not in class_counts:
                logger.warning("Class %d has zero samples in dataset!", c)
                class_counts[c] = 1

        logger.info("Class distribution: %s", class_counts)

        # Compute weights
        if strategy == "effective_samples":
            self.class_weights = compute_effective_sample_weights(
                class_counts, num_classes, beta=beta
            )
        else:
            self.class_weights = compute_inverse_frequency_weights(
                class_counts, num_classes, power=power
            )

        logger.info("Class sampling weights: %s", self.class_weights)

        # Assign to samples
        self.sample_weights = assign_sample_weights(
            annotations,
            self.class_weights,
            multi_label_strategy=multi_label_strategy,
        )

    def get_sampler(self) -> WeightedRandomSampler:
        """Return PyTorch WeightedRandomSampler ready for DataLoader."""
        weights_tensor = torch.tensor(self.sample_weights, dtype=torch.float64)
        return WeightedRandomSampler(
            weights=weights_tensor,
            num_samples=self.num_samples,
            replacement=self.replacement,
        )

    def get_weights(self) -> list[float]:
        """Return raw sample weights list."""
        return self.sample_weights

    def log_statistics(self) -> None:
        """Log sampling statistics for verification."""
        weights = np.array(self.sample_weights)
        logger.info(
            "Sampler statistics: min=%.3f, max=%.3f, mean=%.3f, std=%.3f",
            weights.min(),
            weights.max(),
            weights.mean(),
            weights.std(),
        )
        # Expected class distribution after sampling
        expected = {
            cls: f"{w / sum(self.class_weights.values()) * 100:.1f}%"
            for cls, w in self.class_weights.items()
        }
        logger.info("Expected class sampling distribution: %s", expected)
