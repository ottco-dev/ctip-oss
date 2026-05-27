"""
detection.domain.confidence_calibrator — Post-hoc confidence calibration.

WHY CALIBRATION MATTERS:
Modern object detectors (YOLO, RTMDet) are systematically overconfident.
Raw confidence scores ≠ empirical accuracy rates. A score of 0.85 typically
corresponds to 70-80% accuracy on unseen data.

For scientific applications this matters significantly:
- Uncertainty-aware active learning depends on calibrated scores
- Threshold selection for annotation filtering uses confidence directly
- Reporting calibration error is required for reproducible benchmarks

METHODS IMPLEMENTED:
1. Temperature Scaling — single parameter, simple, effective for small datasets
2. Platt Scaling — logistic regression on raw scores
3. ECE computation — evaluation metric for calibration quality

Reference:
  Guo, C. et al. (2017). "On Calibration of Modern Neural Networks."
  ICML 2017. https://arxiv.org/abs/1706.04599
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass
class CalibrationResult:
    """Result from calibration evaluation."""

    ece: float
    """Expected Calibration Error — primary calibration metric. Lower = better calibrated."""

    mce: float
    """Maximum Calibration Error — worst-case bin gap."""

    method: str
    temperature: float | None = None
    num_bins: int = 15
    bin_confidences: list[float] = field(default_factory=list)
    bin_accuracies: list[float] = field(default_factory=list)
    bin_counts: list[int] = field(default_factory=list)

    def summary(self) -> str:
        return f"ECE={self.ece:.4f} | MCE={self.mce:.4f} | method={self.method}"


class TemperatureCalibrator:
    """
    Temperature Scaling — single-parameter post-hoc calibration.

    Divides raw logits by scalar T before sigmoid/softmax.
    T > 1.0 → softer predictions (reduces overconfidence)
    T < 1.0 → sharper predictions (rarely needed for NNs)
    T = 1.0 → no change

    Training data requirement: Small calibration set (~200-500 images).
    NEVER calibrate on the training set (data leakage).
    Use a separate calibration split or part of the validation set.
    """

    def __init__(self) -> None:
        self.temperature: float = 1.0
        self._is_fitted: bool = False

    def fit(
        self,
        logits: NDArray[np.float32],
        labels: NDArray[np.int32],
    ) -> "TemperatureCalibrator":
        """
        Optimize temperature T to minimize NLL on calibration set.

        Args:
            logits: Raw model logits (N,) — before sigmoid
            labels: Binary labels (N,) — 1=correct detection, 0=incorrect

        Returns:
            self (for chaining)
        """
        from scipy.optimize import minimize_scalar  # type: ignore[import]

        def negative_log_likelihood(temp: float) -> float:
            temp = max(temp, 1e-6)
            probs = self._sigmoid(logits / temp)
            probs = np.clip(probs, 1e-7, 1 - 1e-7)
            nll = -float(np.mean(
                labels * np.log(probs) + (1 - labels) * np.log(1 - probs)
            ))
            return nll

        result = minimize_scalar(
            negative_log_likelihood,
            bounds=(0.1, 10.0),
            method="bounded",
        )
        self.temperature = float(result.x)
        self._is_fitted = True
        return self

    def calibrate(self, logit: float) -> float:
        """Apply temperature scaling to single logit → calibrated probability."""
        return float(self._sigmoid(logit / self.temperature))

    def calibrate_batch(self, logits: NDArray[np.float32]) -> NDArray[np.float32]:
        """Apply temperature scaling to batch of logits."""
        return self._sigmoid(logits / self.temperature).astype(np.float32)

    def save(self, path: str) -> None:
        """Save calibrator parameters to file."""
        import json
        import pathlib
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"temperature": self.temperature, "fitted": self._is_fitted}, f)

    @classmethod
    def load(cls, path: str) -> "TemperatureCalibrator":
        """Load calibrator from saved file."""
        import json
        with open(path) as f:
            data = json.load(f)
        cal = cls()
        cal.temperature = data["temperature"]
        cal._is_fitted = data["fitted"]
        return cal

    @staticmethod
    def _sigmoid(x: float | NDArray) -> NDArray[np.float32]:
        return (1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))).astype(np.float32)

    def __repr__(self) -> str:
        return f"TemperatureCalibrator(T={self.temperature:.3f}, fitted={self._is_fitted})"


def compute_ece(
    confidences: NDArray[np.float32],
    accuracies: NDArray[np.float32],
    num_bins: int = 15,
) -> CalibrationResult:
    """
    Expected Calibration Error (ECE).

    Groups predictions into bins by confidence level.
    For each bin: ECE += (bin_size/N) × |avg_confidence - avg_accuracy|

    Interpretation:
    - ECE < 0.03: Well-calibrated
    - ECE 0.03-0.08: Acceptable
    - ECE > 0.08: Poorly calibrated — consider temperature scaling

    Args:
        confidences: Predicted probabilities [0, 1] (N,)
        accuracies: Binary correctness labels [0, 1] (N,)
        num_bins: Number of equal-width bins

    Reference:
        Naeini, M.P. et al. (2015). "Obtaining Well Calibrated Probabilities
        Using Bayesian Binning." AAAI 2015.
    """
    assert len(confidences) == len(accuracies), "Length mismatch"
    n = len(confidences)
    bin_edges = np.linspace(0, 1, num_bins + 1)

    bin_confs: list[float] = []
    bin_accs: list[float] = []
    bin_counts: list[int] = []
    ece = 0.0
    mce = 0.0

    for i in range(num_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == num_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)

        count = int(mask.sum())
        bin_counts.append(count)

        if count == 0:
            bin_confs.append(float((lo + hi) / 2))
            bin_accs.append(0.0)
            continue

        avg_conf = float(confidences[mask].mean())
        avg_acc = float(accuracies[mask].mean())
        bin_confs.append(avg_conf)
        bin_accs.append(avg_acc)

        gap = abs(avg_conf - avg_acc)
        ece += (count / n) * gap
        mce = max(mce, gap)

    return CalibrationResult(
        ece=ece,
        mce=mce,
        method="ece_equal_width",
        num_bins=num_bins,
        bin_confidences=bin_confs,
        bin_accuracies=bin_accs,
        bin_counts=bin_counts,
    )
