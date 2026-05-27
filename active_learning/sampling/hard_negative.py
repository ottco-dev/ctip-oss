"""
active_learning.sampling.hard_negative — Hard negative mining for active learning.

Hard negatives: samples that the model confidently misclassifies.
These are the most informative samples for improving decision boundaries.

Hard negative identification:
  1. Model predicts class A with high confidence
  2. Ground truth is class B (if labeled) OR
  3. Prediction disagrees with ensemble/teacher model

In trichome detection:
- Hard negatives often include: blurry stalked/sessile confusion,
  small bulbous trichomes confused with debris, partially focused heads

Two strategies:
1. Confidence-based: high confidence predictions on uncertain/misclassified
2. Prediction disagreement: model disagrees with VLM or ensemble
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class HardNegativeScore:
    """Hard negative mining score for a sample."""

    sample_id: str | int
    confidence: float
    """Model's confidence in its prediction [0, 1]."""

    predicted_class: int
    true_class: int | None
    """Ground truth class, or None if unlabeled."""

    is_confirmed_hard: bool
    """True if ground truth is known and conflicts with prediction."""

    hardness_score: float
    """Combined hardness score [0, 1]. Higher = more valuable to label."""

    disagreement_score: float = 0.0
    """Score from ensemble/VLM disagreement [0, 1]."""

    probabilities: NDArray[np.float64] | None = None


class HardNegativeMiner:
    """
    Hard negative mining for trichome detection active learning.

    Identifies samples where the model makes high-confidence predictions
    that are likely wrong (based on ensemble disagreement or known labels).

    Two modes:
    1. Supervised: compare model predictions against existing labels
    2. Unsupervised: use prediction disagreement between model and ensemble

    Usage::

        miner = HardNegativeMiner(confidence_threshold=0.8)

        # Find high-confidence predictions
        hard_negatives = miner.find_hard_negatives(
            sample_ids, primary_probs, ensemble_probs
        )
        selected = miner.select_top_k(hard_negatives, k=50)
    """

    def __init__(
        self,
        confidence_threshold: float = 0.80,
        disagreement_weight: float = 0.5,
        labeled_only: bool = False,
    ) -> None:
        """
        Args:
            confidence_threshold: Minimum confidence for a prediction to be
                                  considered a potential hard negative.
                                  High confidence wrong predictions are most valuable.
            disagreement_weight: Weight for ensemble disagreement in hardness score.
            labeled_only: If True, only return confirmed hard negatives (requires labels).
        """
        self.confidence_threshold = confidence_threshold
        self.disagreement_weight = disagreement_weight
        self.labeled_only = labeled_only

    def find_hard_negatives(
        self,
        sample_ids: Sequence[str | int],
        primary_probs: NDArray[np.float64],
        ensemble_probs: NDArray[np.float64] | None = None,
        true_classes: Sequence[int | None] | None = None,
    ) -> list[HardNegativeScore]:
        """
        Find hard negatives in a batch of predictions.

        Args:
            sample_ids: Sample identifiers.
            primary_probs: (N, C) primary model probability matrix.
            ensemble_probs: (N, C) ensemble/teacher model probabilities.
                           If None, uses only confidence-based detection.
            true_classes: Ground truth class IDs (or None for unlabeled).

        Returns:
            List of HardNegativeScore objects for all samples.
        """
        primary = np.asarray(primary_probs, dtype=np.float64)
        if primary.ndim == 1:
            primary = primary[np.newaxis]

        n = len(sample_ids)
        scores: list[HardNegativeScore] = []

        for i in range(min(n, len(primary))):
            prob = primary[i]
            predicted_class = int(np.argmax(prob))
            confidence = float(prob[predicted_class])

            true_cls = None
            if true_classes is not None and i < len(true_classes):
                true_cls = true_classes[i]

            # Check if confirmed hard negative (labeled + wrong)
            is_confirmed = (
                true_cls is not None
                and confidence >= self.confidence_threshold
                and predicted_class != true_cls
            )

            # Disagreement score vs ensemble
            disagreement = 0.0
            if ensemble_probs is not None:
                ens = np.asarray(ensemble_probs, dtype=np.float64)
                if i < len(ens):
                    ens_prob = ens[i]
                    ens_class = int(np.argmax(ens_prob))

                    if ens_class != predicted_class:
                        # Different predicted class — high disagreement
                        disagreement = float(confidence * ens_prob[ens_class])
                    else:
                        # Same class but confidence differs
                        ens_conf = float(ens_prob[ens_class])
                        disagreement = abs(confidence - ens_conf)

            # Hardness score
            # - High confidence wrong = high hardness
            # - High ensemble disagreement = high hardness
            # - Confirmed errors score highest
            if is_confirmed:
                hardness = 0.7 + 0.3 * confidence  # 0.7–1.0
            elif confidence >= self.confidence_threshold:
                hardness = (
                    (1 - self.disagreement_weight) * confidence
                    + self.disagreement_weight * disagreement
                )
            else:
                # Low confidence — not a hard negative
                hardness = 0.0

            if self.labeled_only and not is_confirmed:
                continue

            scores.append(
                HardNegativeScore(
                    sample_id=sample_ids[i],
                    confidence=confidence,
                    predicted_class=predicted_class,
                    true_class=true_cls,
                    is_confirmed_hard=is_confirmed,
                    hardness_score=min(1.0, hardness),
                    disagreement_score=float(disagreement),
                    probabilities=prob,
                )
            )

        return scores

    def select_top_k(
        self,
        scores: list[HardNegativeScore],
        k: int,
        prefer_confirmed: bool = True,
    ) -> list[HardNegativeScore]:
        """
        Select top-k hardest negatives for labeling.

        Args:
            scores: HardNegativeScore list from find_hard_negatives().
            k: Number to select.
            prefer_confirmed: If True, always include all confirmed hard negatives first.

        Returns:
            Selected samples sorted by hardness score descending.
        """
        if not scores:
            return []

        if prefer_confirmed:
            confirmed = [s for s in scores if s.is_confirmed_hard]
            unconfirmed = [s for s in scores if not s.is_confirmed_hard]

            # Sort each group by hardness
            confirmed.sort(key=lambda s: s.hardness_score, reverse=True)
            unconfirmed.sort(key=lambda s: s.hardness_score, reverse=True)

            selected = confirmed[:k]
            remaining_k = k - len(selected)
            if remaining_k > 0:
                selected.extend(unconfirmed[:remaining_k])
        else:
            sorted_scores = sorted(scores, key=lambda s: s.hardness_score, reverse=True)
            selected = sorted_scores[:k]

        return selected

    def compute_class_confusion_matrix(
        self,
        scores: list[HardNegativeScore],
        num_classes: int,
    ) -> NDArray[np.int32]:
        """
        Build confusion matrix from confirmed hard negatives.

        Shows which class pairs are most commonly confused.

        Args:
            scores: HardNegativeScore list.
            num_classes: Total number of classes.

        Returns:
            (C, C) confusion matrix where [i, j] = predicted i, true j.
        """
        matrix = np.zeros((num_classes, num_classes), dtype=np.int32)

        for s in scores:
            if s.is_confirmed_hard and s.true_class is not None:
                pred = min(s.predicted_class, num_classes - 1)
                true = min(s.true_class, num_classes - 1)
                matrix[pred, true] += 1

        return matrix

    def get_summary_stats(
        self,
        scores: list[HardNegativeScore],
    ) -> dict[str, float]:
        """Summary statistics for the mined hard negatives."""
        if not scores:
            return {}

        hardness = np.array([s.hardness_score for s in scores])
        confs = np.array([s.confidence for s in scores])
        n_confirmed = sum(1 for s in scores if s.is_confirmed_hard)

        return {
            "total_scored": len(scores),
            "confirmed_hard_negatives": n_confirmed,
            "mean_hardness": float(hardness.mean()),
            "max_hardness": float(hardness.max()),
            "mean_confidence": float(confs.mean()),
            "high_confidence_count": int((confs >= self.confidence_threshold).sum()),
            "confirmation_rate": n_confirmed / len(scores) if scores else 0.0,
        }
