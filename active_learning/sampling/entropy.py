"""
active_learning.sampling.entropy — Entropy-based active learning sampling.

Prediction entropy quantifies uncertainty across all classes:
    H(x) = -Σ p(y|x) · log p(y|x)

High entropy → model is uniformly uncertain across classes → valuable for labeling.
Low entropy → model is confident in one class → skip (probably easy).

Advantages over least-confidence:
  - Considers full probability distribution
  - Less sensitive to near-ties in multi-class settings
  - Well-grounded in information theory

Usage::

    sampler = EntropySampler(num_classes=4)
    selected = sampler.select_top_k(predictions, k=50)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class EntropyScore:
    """Entropy-based uncertainty score for a single sample."""

    sample_id: str | int
    entropy: float
    """Shannon entropy H(x) = -Σ p log p. Range [0, log(num_classes)]."""

    normalized_entropy: float
    """Entropy normalized to [0, 1] by dividing by log(num_classes)."""

    probabilities: NDArray[np.float64]
    """Full probability vector P(y|x)."""

    predicted_class: int
    """Argmax class (most likely)."""

    predicted_prob: float
    """Probability of most likely class."""


def compute_entropy(
    probabilities: NDArray[np.float64],
    epsilon: float = 1e-10,
) -> float:
    """
    Compute Shannon entropy from probability vector.

    H(x) = -Σ_i p_i · log(p_i)

    Args:
        probabilities: Probability distribution summing to 1. Shape (C,).
        epsilon: Small value to avoid log(0).

    Returns:
        Entropy in nats.
    """
    probs = np.asarray(probabilities, dtype=np.float64)
    probs = np.clip(probs, epsilon, 1.0)
    return float(-np.sum(probs * np.log(probs)))


def compute_normalized_entropy(
    probabilities: NDArray[np.float64],
    num_classes: int | None = None,
) -> float:
    """
    Compute entropy normalized to [0, 1].

    Max entropy = log(num_classes) for uniform distribution.

    Args:
        probabilities: Probability distribution shape (C,).
        num_classes: Number of classes. If None, inferred from len(probabilities).

    Returns:
        Normalized entropy in [0, 1].
    """
    probs = np.asarray(probabilities, dtype=np.float64)
    c = num_classes or len(probs)
    max_entropy = math.log(max(c, 2))  # max entropy for C-class uniform dist
    h = compute_entropy(probs)
    return min(1.0, h / max_entropy)


class EntropySampler:
    """
    Entropy-based uncertainty sampler for active learning.

    Scores each unlabeled sample by prediction entropy and selects
    the highest-entropy (most uncertain) samples for human labeling.

    Usage::

        sampler = EntropySampler(num_classes=4)

        # Single sample
        score = sampler.score_sample(sample_id="img_001", probabilities=probs)

        # Batch + select top-k
        scores = sampler.score_batch(sample_ids, probability_matrix)
        selected = sampler.select_top_k(scores, k=50)
    """

    def __init__(
        self,
        num_classes: int = 4,
        min_entropy_threshold: float | None = None,
    ) -> None:
        """
        Args:
            num_classes: Number of prediction classes.
            min_entropy_threshold: Minimum normalized entropy to include a sample.
                                   Set to 0.3 to skip very confident predictions.
        """
        self.num_classes = num_classes
        self.min_entropy_threshold = min_entropy_threshold
        self.max_entropy = math.log(max(num_classes, 2))

    def score_sample(
        self,
        sample_id: str | int,
        probabilities: NDArray[np.float64] | list[float],
    ) -> EntropyScore:
        """
        Compute entropy score for a single sample.

        Args:
            sample_id: Identifier for the sample.
            probabilities: Class probability vector shape (C,).

        Returns:
            EntropyScore with entropy, normalized_entropy, and predicted class.
        """
        probs = np.asarray(probabilities, dtype=np.float64)

        # Ensure valid probability distribution
        if abs(probs.sum() - 1.0) > 0.05:
            probs = probs / (probs.sum() + 1e-10)

        h = compute_entropy(probs)
        norm_h = min(1.0, h / self.max_entropy)
        predicted_class = int(np.argmax(probs))

        return EntropyScore(
            sample_id=sample_id,
            entropy=h,
            normalized_entropy=norm_h,
            probabilities=probs,
            predicted_class=predicted_class,
            predicted_prob=float(probs[predicted_class]),
        )

    def score_batch(
        self,
        sample_ids: Sequence[str | int],
        probability_matrix: NDArray[np.float64],
    ) -> list[EntropyScore]:
        """
        Score a batch of samples.

        Args:
            sample_ids: Sample identifiers (length N).
            probability_matrix: (N, C) probability matrix.

        Returns:
            List of EntropyScore objects, one per sample.
        """
        probs = np.asarray(probability_matrix, dtype=np.float64)
        if probs.ndim == 1:
            probs = probs[np.newaxis]

        scores = []
        for sid, prob_row in zip(sample_ids, probs):
            score = self.score_sample(sid, prob_row)
            scores.append(score)

        return scores

    def select_top_k(
        self,
        scores: list[EntropyScore],
        k: int,
        spread_across_quartiles: bool = False,
    ) -> list[EntropyScore]:
        """
        Select top-k most uncertain samples for labeling.

        Args:
            scores: List of EntropyScore objects from score_batch().
            k: Number of samples to select.
            spread_across_quartiles: If True, take 25% from each entropy quartile.
                                     Avoids selecting only the most confused samples.

        Returns:
            Selected EntropyScore objects, sorted descending by entropy.
        """
        # Filter by minimum threshold
        filtered = scores
        if self.min_entropy_threshold is not None:
            filtered = [s for s in scores if s.normalized_entropy >= self.min_entropy_threshold]
            if len(filtered) < k:
                logger.warning(
                    "Only %d samples above min_entropy_threshold=%.2f; "
                    "returning all %d.",
                    len(filtered),
                    self.min_entropy_threshold,
                    len(filtered),
                )

        if not filtered:
            return []

        # Sort by entropy descending
        sorted_scores = sorted(filtered, key=lambda s: s.entropy, reverse=True)

        if not spread_across_quartiles:
            return sorted_scores[:k]

        # Spread across entropy quartiles for diversity
        n = len(sorted_scores)
        q_size = max(1, n // 4)

        quartiles = [
            sorted_scores[: q_size],
            sorted_scores[q_size: 2 * q_size],
            sorted_scores[2 * q_size: 3 * q_size],
            sorted_scores[3 * q_size:],
        ]

        per_quartile = max(1, k // 4)
        selected: list[EntropyScore] = []

        for q in quartiles:
            selected.extend(q[:per_quartile])
            if len(selected) >= k:
                break

        # Fill remaining slots from top
        if len(selected) < k:
            used_ids = {s.sample_id for s in selected}
            remaining = [s for s in sorted_scores if s.sample_id not in used_ids]
            selected.extend(remaining[: k - len(selected)])

        return selected[:k]

    def compute_dataset_entropy_stats(
        self,
        scores: list[EntropyScore],
    ) -> dict[str, float]:
        """
        Compute summary statistics of entropy across a dataset.

        Useful for monitoring model uncertainty trends over training.

        Returns:
            Dict with mean, std, median, p90, p95, p99 entropy values.
        """
        if not scores:
            return {}

        entropies = np.array([s.normalized_entropy for s in scores])
        return {
            "mean": float(entropies.mean()),
            "std": float(entropies.std()),
            "median": float(np.median(entropies)),
            "p25": float(np.percentile(entropies, 25)),
            "p75": float(np.percentile(entropies, 75)),
            "p90": float(np.percentile(entropies, 90)),
            "p95": float(np.percentile(entropies, 95)),
            "p99": float(np.percentile(entropies, 99)),
            "fraction_high_entropy": float((entropies >= 0.7).mean()),
        }
