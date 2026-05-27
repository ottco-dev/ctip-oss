"""
active_learning.sampling.uncertainty — Uncertainty-based active learning sampling.

STRATEGY:
Select unlabeled samples where the model is most uncertain.
These samples provide the most information gain when labeled.

UNCERTAINTY ESTIMATION METHODS:

1. MC Dropout (Monte Carlo Dropout):
   - Run model N times with dropout enabled at inference time
   - Variance across predictions = epistemic uncertainty
   - No ensemble required — single model with dropout
   - RTX 4060 cost: N=20 passes @ 5ms each = ~100ms/image

2. Entropy Sampling:
   - Use prediction entropy as uncertainty measure
   - H(y|x) = -Σ p(y_k|x) log p(y_k|x)
   - Simple, fast — no repeated inference

3. Ensemble Disagreement:
   - Requires multiple models (expensive on 8GB VRAM)
   - Use WBF confidence variance from detection ensemble

4. Least Confidence:
   - Select samples where max class probability is lowest
   - max_class_prob = max_k P(y_k|x)
   - Sort ascending → most uncertain first

REFERENCES:
  Gal, Y. & Ghahramani, Z. (2016). Dropout as a Bayesian Approximation.
  ICML 2016. arXiv:1506.02142

  Lewis, D.D. & Gale, W.A. (1994). A Sequential Algorithm for Training
  Text Classifiers. SIGIR 1994.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class UncertaintyScore:
    """Uncertainty measurement for a single sample."""

    sample_id: str
    image_path: str

    # Uncertainty metrics
    mc_dropout_variance: float | None = None
    """Epistemic uncertainty from MC Dropout."""

    prediction_entropy: float | None = None
    """Shannon entropy of class probability distribution."""

    least_confidence: float | None = None
    """1 - max class probability."""

    # Combined score (higher = more uncertain = higher priority for labeling)
    combined_uncertainty: float = 0.0

    # Prediction metadata
    mean_prediction: list[float] | None = None
    """Mean class probabilities across MC Dropout runs."""

    num_mc_runs: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "image_path": self.image_path,
            "mc_dropout_variance": self.mc_dropout_variance,
            "prediction_entropy": self.prediction_entropy,
            "least_confidence": self.least_confidence,
            "combined_uncertainty": self.combined_uncertainty,
            "num_mc_runs": self.num_mc_runs,
        }


@dataclass
class SamplingResult:
    """Result from an active learning sampling pass."""

    selected_ids: list[str]
    """Sample IDs selected for annotation."""

    uncertainty_scores: list[UncertaintyScore]
    """All computed uncertainty scores, sorted by uncertainty."""

    strategy: str
    budget: int
    total_scored: int

    @property
    def selected_paths(self) -> list[str]:
        score_map = {s.sample_id: s.image_path for s in self.uncertainty_scores}
        return [score_map[sid] for sid in self.selected_ids]


def compute_entropy(probabilities: NDArray[np.float32]) -> float:
    """
    Compute Shannon entropy of a probability distribution.

    H = -Σ p_i log(p_i), with convention 0 log 0 = 0.

    Args:
        probabilities: Array of class probabilities summing to 1.

    Returns:
        Entropy in nats (natural log).
    """
    probs = np.asarray(probabilities, dtype=np.float64)
    probs = np.clip(probs, 1e-12, 1.0)
    probs = probs / probs.sum()
    return float(-np.sum(probs * np.log(probs)))


def compute_least_confidence(probabilities: NDArray[np.float32]) -> float:
    """
    Compute least confidence uncertainty: 1 - max class probability.

    Args:
        probabilities: Array of class probabilities.

    Returns:
        Least confidence score in [0, 1]. Higher = more uncertain.
    """
    return float(1.0 - np.max(probabilities))


def mc_dropout_uncertainty(
    model: Any,
    image: NDArray[np.uint8],
    n_passes: int = 20,
    device: str = "cuda:0",
) -> tuple[NDArray[np.float32], float]:
    """
    Estimate epistemic uncertainty via Monte Carlo Dropout.

    The model must have dropout layers that are kept active at inference.

    Args:
        model: PyTorch model with dropout layers.
        image: RGB uint8 image array (H, W, 3).
        n_passes: Number of stochastic forward passes (10-50 typical).
        device: CUDA device string.

    Returns:
        (mean_probabilities, variance_score)
        - mean_probabilities: Mean class probs across all passes (n_classes,)
        - variance_score: Scalar epistemic uncertainty estimate
    """
    import torch
    import torch.nn.functional as F

    # Enable dropout at inference (disable eval mode for dropout layers only)
    _enable_mc_dropout(model)

    predictions = []
    img_tensor = _preprocess_for_model(image, device)

    with torch.inference_mode():
        for _ in range(n_passes):
            logits = model(img_tensor)
            probs = F.softmax(logits, dim=-1).cpu().numpy()
            predictions.append(probs[0])  # Remove batch dim

    _disable_mc_dropout(model)

    predictions_array = np.stack(predictions, axis=0)  # (n_passes, n_classes)
    mean_probs = predictions_array.mean(axis=0)
    variance = float(predictions_array.var(axis=0).mean())

    return mean_probs.astype(np.float32), variance


def _enable_mc_dropout(model: Any) -> None:
    """Enable dropout layers for MC Dropout inference."""
    import torch.nn as nn
    for module in model.modules():
        if isinstance(module, nn.Dropout) or isinstance(module, nn.Dropout2d):
            module.train()


def _disable_mc_dropout(model: Any) -> None:
    """Disable dropout (restore eval mode)."""
    import torch.nn as nn
    for module in model.modules():
        if isinstance(module, nn.Dropout) or isinstance(module, nn.Dropout2d):
            module.eval()


def _preprocess_for_model(
    image: NDArray[np.uint8],
    device: str,
) -> Any:
    """Preprocess image for PyTorch model input."""
    import torch
    import cv2

    # Resize to 224x224 (standard CNN input)
    resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LANCZOS4)

    # Normalize (ImageNet stats)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    normalized = (resized.astype(np.float32) / 255.0 - mean) / std

    # HWC → CHW
    tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).unsqueeze(0)
    return tensor.to(device)


class UncertaintySampler:
    """
    Selects the most uncertain unlabeled samples for annotation.

    Usage:
        sampler = UncertaintySampler(strategy="entropy")

        # Score all unlabeled samples
        scores = sampler.score_batch(
            model=maturity_model,
            unlabeled_images=image_arrays,
            sample_ids=image_ids,
        )

        # Select top-K most uncertain
        result = sampler.select_top_k(scores, k=50)

        # result.selected_ids → send to annotation queue
    """

    VALID_STRATEGIES = {"entropy", "least_confidence", "mc_dropout", "combined"}

    def __init__(
        self,
        strategy: str = "entropy",
        mc_n_passes: int = 20,
    ) -> None:
        if strategy not in self.VALID_STRATEGIES:
            raise ValueError(
                f"Unknown strategy '{strategy}'. "
                f"Choose from: {self.VALID_STRATEGIES}"
            )
        self.strategy = strategy
        self.mc_n_passes = mc_n_passes

    def score_single(
        self,
        model: Any,
        image: NDArray[np.uint8],
        sample_id: str,
        image_path: str = "",
        device: str = "cuda:0",
    ) -> UncertaintyScore:
        """
        Compute uncertainty score for a single image.

        Args:
            model: Trained classification model (PyTorch).
            image: RGB uint8 image.
            sample_id: Sample identifier.
            image_path: Path for reference.
            device: CUDA device.
        """
        score = UncertaintyScore(
            sample_id=sample_id,
            image_path=image_path,
        )

        if self.strategy == "mc_dropout" or self.strategy == "combined":
            mean_probs, variance = mc_dropout_uncertainty(
                model, image, self.mc_n_passes, device
            )
            score.mc_dropout_variance = variance
            score.mean_prediction = mean_probs.tolist()
            score.num_mc_runs = self.mc_n_passes

            # Also compute entropy from mean predictions
            score.prediction_entropy = compute_entropy(mean_probs)
            score.least_confidence = compute_least_confidence(mean_probs)

        else:
            # Single-pass inference
            import torch
            import torch.nn.functional as F

            model.eval()
            img_tensor = _preprocess_for_model(image, device)
            with torch.inference_mode():
                logits = model(img_tensor)
                probs = F.softmax(logits, dim=-1).cpu().numpy()[0]

            score.mean_prediction = probs.tolist()
            score.prediction_entropy = compute_entropy(probs)
            score.least_confidence = compute_least_confidence(probs)

        # Compute combined score
        score.combined_uncertainty = self._compute_combined(score)
        return score

    def score_batch(
        self,
        model: Any,
        unlabeled_images: list[NDArray[np.uint8]],
        sample_ids: list[str],
        image_paths: list[str] | None = None,
        device: str = "cuda:0",
    ) -> list[UncertaintyScore]:
        """
        Score a batch of unlabeled images.

        Returns scores sorted by uncertainty descending (most uncertain first).
        """
        if len(unlabeled_images) != len(sample_ids):
            raise ValueError("unlabeled_images and sample_ids must have same length")

        paths = image_paths or [""] * len(sample_ids)
        scores = []

        for i, (image, sid, path) in enumerate(zip(unlabeled_images, sample_ids, paths)):
            score = self.score_single(model, image, sid, path, device)
            scores.append(score)

            if (i + 1) % 100 == 0:
                import logging
                logging.getLogger(__name__).info(
                    f"Scored {i+1}/{len(unlabeled_images)} images"
                )

        # Sort by uncertainty descending
        scores.sort(key=lambda s: s.combined_uncertainty, reverse=True)
        return scores

    def select_top_k(
        self,
        scores: list[UncertaintyScore],
        k: int,
        diversity_filter: bool = False,
    ) -> SamplingResult:
        """
        Select top-K most uncertain samples.

        Args:
            scores: Scored samples (must be sorted by uncertainty desc).
            k: Number of samples to select.
            diversity_filter: If True, apply diversity filtering to avoid
                             selecting very similar samples.

        Returns:
            SamplingResult with selected IDs.
        """
        if diversity_filter:
            selected = self._select_with_diversity(scores, k)
        else:
            selected = scores[:k]

        return SamplingResult(
            selected_ids=[s.sample_id for s in selected],
            uncertainty_scores=scores,
            strategy=self.strategy,
            budget=k,
            total_scored=len(scores),
        )

    def _compute_combined(self, score: UncertaintyScore) -> float:
        """Combine multiple uncertainty signals into a single score."""
        parts = []

        if score.prediction_entropy is not None:
            # Normalize entropy (max entropy for n classes = log(n))
            # For 6 maturity classes: log(6) ≈ 1.79
            max_entropy = np.log(6.0)
            parts.append(score.prediction_entropy / max_entropy)

        if score.least_confidence is not None:
            parts.append(score.least_confidence)

        if score.mc_dropout_variance is not None:
            # Variance in [0, 0.25] for normalized probs
            parts.append(min(score.mc_dropout_variance / 0.25, 1.0))

        if not parts:
            return 0.0

        return float(np.mean(parts))

    def _select_with_diversity(
        self,
        scores: list[UncertaintyScore],
        k: int,
    ) -> list[UncertaintyScore]:
        """
        Select diverse subset of uncertain samples.

        Simple diversity: spread selection across different uncertainty bins
        to avoid clustering in one region.
        """
        if len(scores) <= k:
            return scores

        # Split into 4 quartiles, select proportionally
        n = len(scores)
        selected: list[UncertaintyScore] = []
        per_quartile = k // 4

        for q in range(4):
            start = (q * n) // 4
            end = ((q + 1) * n) // 4
            quartile = scores[start:end]
            selected.extend(quartile[:per_quartile])

        # Fill remaining slots from top
        remaining = k - len(selected)
        added_ids = {s.sample_id for s in selected}
        for s in scores:
            if remaining <= 0:
                break
            if s.sample_id not in added_ids:
                selected.append(s)
                remaining -= 1

        return selected[:k]
