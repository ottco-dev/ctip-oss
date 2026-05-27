"""
active_learning/sampling/disagreement.py — Ensemble disagreement sampling.

Measures disagreement between multiple model predictions on the same sample.
High disagreement → the ensemble is uncertain → good candidate for annotation.

Metrics:
  - Vote entropy: H over class vote distribution
  - KL divergence: average KL from mean prediction
  - Bald score: mutual information I(y; θ | x)
  - Prediction variance: variance of predicted probabilities across ensemble members

Reference: Settles (2009). Active Learning Literature Survey. §3.4
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class EnsemblePrediction:
    """Predictions from one ensemble member for one sample."""

    sample_id: str
    probabilities: list[float]  # per-class softmax probabilities, sum=1
    predicted_class: int
    confidence: float  # max probability


@dataclass
class DisagreementScore:
    """Disagreement score for one sample across multiple ensemble members."""

    sample_id: str
    vote_entropy: float          # H over vote distribution
    mean_entropy: float          # average H(p_i) across members
    bald_score: float            # I(y; θ | x) = H(mean_p) - mean H(p_i)
    kl_divergence: float         # average KL(p_i || mean_p)
    prediction_variance: float   # average variance of probabilities
    num_members: int
    vote_counts: dict[int, int]  # class → vote count
    composite_score: float       # weighted combination for ranking


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def _entropy(probs: np.ndarray, eps: float = 1e-10) -> float:
    """Shannon entropy of a probability vector."""
    probs = np.clip(probs, eps, 1.0)
    return float(-np.sum(probs * np.log(probs)))


def _kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float:
    """KL divergence KL(p || q)."""
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float(np.sum(p * np.log(p / q)))


def compute_disagreement(
    predictions: list[EnsemblePrediction],
) -> DisagreementScore:
    """
    Compute disagreement score for one sample across ensemble members.

    All predictions must share the same sample_id and have the same
    number of classes.

    Args:
        predictions: List of EnsemblePrediction from each ensemble member.

    Returns:
        DisagreementScore with multiple disagreement metrics.
    """
    if not predictions:
        raise ValueError("predictions list is empty")

    sample_id = predictions[0].sample_id
    num_classes = len(predictions[0].probabilities)
    num_members = len(predictions)

    # Stack as (num_members, num_classes)
    prob_matrix = np.array([p.probabilities for p in predictions], dtype=np.float64)

    # Mean prediction (committee prediction)
    mean_probs = prob_matrix.mean(axis=0)

    # Vote counts and vote entropy
    vote_counts: dict[int, int] = {}
    for pred in predictions:
        c = pred.predicted_class
        vote_counts[c] = vote_counts.get(c, 0) + 1

    vote_fracs = np.array(
        [vote_counts.get(c, 0) / num_members for c in range(num_classes)]
    )
    vote_entropy = _entropy(vote_fracs)

    # Mean entropy: average H across members
    member_entropies = [_entropy(prob_matrix[i]) for i in range(num_members)]
    mean_entropy = float(np.mean(member_entropies))

    # BALD: H(mean_p) - mean H(p_i)
    h_mean = _entropy(mean_probs)
    bald_score = max(0.0, h_mean - mean_entropy)

    # KL divergence: average KL(p_i || mean_p) across members
    kl_values = [_kl_divergence(prob_matrix[i], mean_probs) for i in range(num_members)]
    avg_kl = float(np.mean(kl_values))

    # Prediction variance: average std across classes
    variance_per_class = prob_matrix.var(axis=0)
    avg_variance = float(variance_per_class.mean())

    # Composite: weighted sum for ranking
    log_c = math.log(num_classes) if num_classes > 1 else 1.0
    norm_vote_entropy = vote_entropy / log_c
    norm_bald = min(1.0, bald_score / log_c)

    composite = 0.4 * norm_vote_entropy + 0.4 * norm_bald + 0.2 * min(1.0, avg_kl)

    return DisagreementScore(
        sample_id=sample_id,
        vote_entropy=round(vote_entropy, 6),
        mean_entropy=round(mean_entropy, 6),
        bald_score=round(bald_score, 6),
        kl_divergence=round(avg_kl, 6),
        prediction_variance=round(avg_variance, 6),
        num_members=num_members,
        vote_counts=vote_counts,
        composite_score=round(composite, 6),
    )


# ---------------------------------------------------------------------------
# Batch sampler
# ---------------------------------------------------------------------------


@dataclass
class DisagreementSampler:
    """
    Select samples where the ensemble disagrees most.

    Usage:
        sampler = DisagreementSampler(metric="bald")
        scores = sampler.compute_all(all_predictions)
        selected = sampler.select_top_k(scores, k=50)
    """

    metric: str = "composite"  # composite | bald | vote_entropy | kl_divergence | variance
    min_members: int = 2

    def compute_all(
        self,
        predictions_by_sample: dict[str, list[EnsemblePrediction]],
    ) -> list[DisagreementScore]:
        """
        Compute disagreement scores for all samples.

        Args:
            predictions_by_sample: {sample_id: [pred_member_1, pred_member_2, ...]}

        Returns:
            List of DisagreementScore sorted by composite_score desc.
        """
        scores: list[DisagreementScore] = []
        for sample_id, preds in predictions_by_sample.items():
            if len(preds) < self.min_members:
                continue
            try:
                score = compute_disagreement(preds)
                scores.append(score)
            except Exception:  # noqa: BLE001
                continue

        return sorted(scores, key=lambda s: s.composite_score, reverse=True)

    def select_top_k(
        self,
        scores: list[DisagreementScore],
        k: int,
        min_composite: float = 0.05,
    ) -> list[DisagreementScore]:
        """
        Select top-k most disagreed-upon samples.

        Args:
            scores: Pre-computed disagreement scores (from compute_all).
            k: Number of samples to select.
            min_composite: Minimum composite score threshold.

        Returns:
            Top-k samples by selected metric, filtered by threshold.
        """
        metric_key = {
            "composite": "composite_score",
            "bald": "bald_score",
            "vote_entropy": "vote_entropy",
            "kl_divergence": "kl_divergence",
            "variance": "prediction_variance",
        }.get(self.metric, "composite_score")

        filtered = [s for s in scores if s.composite_score >= min_composite]
        ranked = sorted(filtered, key=lambda s: getattr(s, metric_key), reverse=True)
        return ranked[:k]


# ---------------------------------------------------------------------------
# Calibrated disagreement (with temperature scaling)
# ---------------------------------------------------------------------------


def apply_temperature_scaling(
    logits: list[list[float]],
    temperature: float = 1.5,
) -> list[list[float]]:
    """
    Apply temperature scaling to raw logits before softmax.

    Higher temperature → softer distributions → more reliable uncertainty estimates.

    Args:
        logits: Raw model logits, shape (num_members, num_classes).
        temperature: Scaling factor T > 1 broadens distribution.

    Returns:
        Calibrated probabilities.
    """
    import math

    calibrated = []
    for member_logits in logits:
        scaled = [l / temperature for l in member_logits]
        max_val = max(scaled)
        exp_vals = [math.exp(s - max_val) for s in scaled]
        total = sum(exp_vals)
        calibrated.append([e / total for e in exp_vals])
    return calibrated


# ---------------------------------------------------------------------------
# Utility: combine ensemble predictions with detection boxes
# ---------------------------------------------------------------------------


def aggregate_ensemble_boxes(
    detection_lists: list[list[dict]],
    iou_threshold: float = 0.5,
) -> tuple[float, dict[int, int]]:
    """
    Estimate box-level disagreement from multiple detector outputs.

    Returns:
        (avg_iou, vote_distribution) where avg_iou is agreement level and
        vote_distribution maps class_id to vote count.
    """
    if not detection_lists:
        return 0.0, {}

    class_votes: dict[int, int] = {}
    for det_list in detection_lists:
        for det in det_list:
            cls = int(det.get("class_id", 0))
            class_votes[cls] = class_votes.get(cls, 0) + 1

    # Rough agreement measure: fraction of votes for dominant class
    total_votes = sum(class_votes.values())
    max_votes = max(class_votes.values()) if class_votes else 0
    agreement = max_votes / total_votes if total_votes > 0 else 0.0
    disagreement = 1.0 - agreement

    return round(disagreement, 4), class_votes
