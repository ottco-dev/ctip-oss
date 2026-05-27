"""
shared.metrics.calibration_metrics — Expected Calibration Error and reliability diagrams.

CALIBRATION:
A model is well-calibrated if its stated confidence matches the actual accuracy.
A model stating 70% confidence should be correct ~70% of the time.

METRICS:
- ECE (Expected Calibration Error): Weighted average bin calibration error
- MCE (Maximum Calibration Error): Worst-case bin error
- Reliability diagram data: for plotting observed vs. expected accuracy

REFERENCE:
  Guo, C. et al. (2017). On Calibration of Modern Neural Networks.
  ICML 2017. arXiv:1706.04599

  Naeini, M.P. et al. (2015). Obtaining Well Calibrated Probabilities
  Using Bayesian Binning. AAAI 2015.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray


@dataclass
class CalibrationResult:
    """Calibration analysis result."""

    ece: float
    """Expected Calibration Error (lower = better calibrated)."""

    mce: float
    """Maximum Calibration Error."""

    # Per-bin data for reliability diagram
    bin_confidences: NDArray[np.float64]
    """Mean confidence per bin."""

    bin_accuracies: NDArray[np.float64]
    """Observed accuracy per bin."""

    bin_counts: NDArray[np.int32]
    """Number of samples per bin."""

    num_bins: int

    @property
    def is_overconfident(self) -> bool:
        """True if model confidence > actual accuracy on average."""
        weighted_conf = np.average(self.bin_confidences, weights=self.bin_counts)
        weighted_acc = np.average(self.bin_accuracies, weights=self.bin_counts)
        return float(weighted_conf) > float(weighted_acc)

    def to_dict(self) -> dict:
        return {
            "ece": self.ece,
            "mce": self.mce,
            "num_bins": self.num_bins,
            "is_overconfident": self.is_overconfident,
            "bin_confidences": self.bin_confidences.tolist(),
            "bin_accuracies": self.bin_accuracies.tolist(),
            "bin_counts": self.bin_counts.tolist(),
        }


def compute_calibration(
    confidences: Sequence[float],
    correct: Sequence[bool],
    num_bins: int = 15,
) -> CalibrationResult:
    """
    Compute Expected Calibration Error.

    Args:
        confidences: Predicted confidence scores (max class probability).
        correct: Whether prediction was correct (True/False).
        num_bins: Number of equally-spaced confidence bins.

    Returns:
        CalibrationResult with ECE, MCE, and reliability diagram data.
    """
    confs = np.asarray(confidences, dtype=np.float64)
    corrects = np.asarray(correct, dtype=bool)

    if len(confs) == 0:
        raise ValueError("Empty confidence/correct arrays")

    if len(confs) != len(corrects):
        raise ValueError(
            f"Length mismatch: confidences={len(confs)}, correct={len(corrects)}"
        )

    # Bin boundaries [0, 1/n, 2/n, ..., 1]
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    bin_indices = np.digitize(confs, bin_edges[:-1]) - 1
    bin_indices = np.clip(bin_indices, 0, num_bins - 1)

    bin_confidences = np.zeros(num_bins, dtype=np.float64)
    bin_accuracies = np.zeros(num_bins, dtype=np.float64)
    bin_counts = np.zeros(num_bins, dtype=np.int32)

    for b in range(num_bins):
        mask = bin_indices == b
        if mask.sum() == 0:
            continue
        bin_counts[b] = int(mask.sum())
        bin_confidences[b] = float(confs[mask].mean())
        bin_accuracies[b] = float(corrects[mask].mean())

    n = len(confs)
    # ECE = Σ |acc(b) - conf(b)| × |B| / n
    ece = float(
        np.sum(bin_counts * np.abs(bin_accuracies - bin_confidences)) / n
    )
    # MCE = max |acc(b) - conf(b)| over non-empty bins
    non_empty = bin_counts > 0
    mce = float(np.max(np.abs(bin_accuracies - bin_confidences)[non_empty])) if non_empty.any() else 0.0

    return CalibrationResult(
        ece=ece,
        mce=mce,
        bin_confidences=bin_confidences,
        bin_accuracies=bin_accuracies,
        bin_counts=bin_counts,
        num_bins=num_bins,
    )
