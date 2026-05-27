"""
tests.unit.test_calibration_metrics — Unit tests for ECE / reliability-diagram metrics.

Tests:
  - compute_calibration: ECE and MCE values on known inputs
  - compute_calibration: edge cases (all correct, all wrong, single sample)
  - CalibrationResult: is_overconfident property
  - CalibrationResult: to_dict serialization
  - Reliability diagram data consistency

Reference:
  Guo et al. (2017). On Calibration of Modern Neural Networks. ICML 2017.
"""

from __future__ import annotations

import numpy as np
import pytest

from shared.metrics.calibration_metrics import CalibrationResult, compute_calibration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _perfect_confidences(n: int = 100, step: float = 0.01) -> tuple[list[float], list[bool]]:
    """
    Generate a perfectly calibrated dataset.
    For confidence c, correct with probability c.
    """
    rng = np.random.default_rng(42)
    confs = np.linspace(0.01, 1.0, n)
    correct = rng.random(n) < confs
    return confs.tolist(), correct.tolist()


# ---------------------------------------------------------------------------
# Basic ECE computation
# ---------------------------------------------------------------------------

class TestComputeCalibration:

    def test_returns_calibration_result(self):
        confs, correct = _perfect_confidences()
        result = compute_calibration(confs, correct)
        assert isinstance(result, CalibrationResult)

    def test_ece_is_non_negative(self):
        confs, correct = _perfect_confidences()
        result = compute_calibration(confs, correct)
        assert result.ece >= 0.0

    def test_ece_bounded_by_one(self):
        confs, correct = _perfect_confidences()
        result = compute_calibration(confs, correct)
        assert result.ece <= 1.0

    def test_perfect_calibration_low_ece(self):
        """A perfectly calibrated model should have ECE close to 0."""
        rng = np.random.default_rng(42)
        n = 10_000
        confs = np.linspace(0.01, 0.99, n)
        correct = rng.random(n) < confs
        result = compute_calibration(confs.tolist(), correct.tolist(), num_bins=10)
        assert result.ece < 0.05

    def test_always_correct_with_high_conf_low_ece(self):
        """All correct, confidence = 1.0 → ECE ≈ 0."""
        n = 100
        confs = [1.0] * n
        correct = [True] * n
        result = compute_calibration(confs, correct)
        assert result.ece < 0.02

    def test_always_wrong_with_high_conf_high_ece(self):
        """Overconfident: conf=0.95, always wrong → ECE ≈ 0.95."""
        n = 200
        confs = [0.95] * n
        correct = [False] * n
        result = compute_calibration(confs, correct)
        assert result.ece > 0.8

    def test_always_correct_with_low_conf_high_ece(self):
        """Underconfident: conf=0.1, always correct → ECE ≈ 0.9."""
        n = 200
        confs = [0.1] * n
        correct = [True] * n
        result = compute_calibration(confs, correct)
        assert result.ece > 0.7

    def test_mce_geq_ece(self):
        """MCE (worst bin) must be >= ECE (weighted average)."""
        confs, correct = _perfect_confidences(n=500)
        result = compute_calibration(confs, correct)
        assert result.mce >= result.ece

    def test_bin_counts_sum_to_n(self):
        n = 200
        confs, correct = _perfect_confidences(n=n)
        result = compute_calibration(confs, correct, num_bins=10)
        assert result.bin_counts.sum() == n

    def test_bin_arrays_have_correct_shape(self):
        confs, correct = _perfect_confidences(n=100)
        result = compute_calibration(confs, correct, num_bins=15)
        assert result.bin_confidences.shape == (15,)
        assert result.bin_accuracies.shape == (15,)
        assert result.bin_counts.shape == (15,)
        assert result.num_bins == 15

    def test_bin_accuracies_in_range(self):
        confs, correct = _perfect_confidences(n=300)
        result = compute_calibration(confs, correct)
        # Only check non-empty bins
        non_empty = result.bin_counts > 0
        assert np.all(result.bin_accuracies[non_empty] >= 0.0)
        assert np.all(result.bin_accuracies[non_empty] <= 1.0)

    def test_bin_confidences_in_range(self):
        confs, correct = _perfect_confidences(n=300)
        result = compute_calibration(confs, correct)
        non_empty = result.bin_counts > 0
        assert np.all(result.bin_confidences[non_empty] >= 0.0)
        assert np.all(result.bin_confidences[non_empty] <= 1.0)

    def test_custom_num_bins(self):
        confs, correct = _perfect_confidences(n=200)
        for num_bins in [5, 10, 20]:
            result = compute_calibration(confs, correct, num_bins=num_bins)
            assert result.num_bins == num_bins
            assert result.bin_counts.shape == (num_bins,)

    def test_empty_arrays_raises(self):
        with pytest.raises(ValueError, match="[Ee]mpty"):
            compute_calibration([], [])

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="[Ll]ength"):
            compute_calibration([0.5, 0.6], [True])

    def test_single_sample(self):
        result = compute_calibration([0.8], [True])
        assert isinstance(result.ece, float)
        assert result.bin_counts.sum() == 1

    def test_reproducible_results(self):
        confs, correct = _perfect_confidences(n=500)
        r1 = compute_calibration(confs, correct)
        r2 = compute_calibration(confs, correct)
        assert r1.ece == r2.ece
        assert r1.mce == r2.mce


# ---------------------------------------------------------------------------
# CalibrationResult — properties
# ---------------------------------------------------------------------------

class TestCalibrationResultProperties:

    def _make_result(self, confs, accs, counts) -> CalibrationResult:
        bin_confidences = np.array(confs)
        bin_accuracies = np.array(accs)
        bin_counts = np.array(counts, dtype=np.int32)
        n = int(bin_counts.sum())
        ece = float(
            np.sum(bin_counts * np.abs(bin_accuracies - bin_confidences)) / max(n, 1)
        )
        non_empty = bin_counts > 0
        mce = float(np.max(np.abs(bin_accuracies - bin_confidences)[non_empty])) \
            if non_empty.any() else 0.0
        return CalibrationResult(
            ece=ece,
            mce=mce,
            bin_confidences=bin_confidences,
            bin_accuracies=bin_accuracies,
            bin_counts=bin_counts,
            num_bins=len(confs),
        )

    def test_is_overconfident_true(self):
        """Model states 0.9 confidence but only 0.5 accurate → overconfident."""
        result = self._make_result(
            confs=[0.9, 0.9],
            accs=[0.5, 0.5],
            counts=[50, 50],
        )
        assert result.is_overconfident is True

    def test_is_overconfident_false(self):
        """Model states 0.3 confidence but 0.8 accurate → underconfident."""
        result = self._make_result(
            confs=[0.3, 0.3],
            accs=[0.8, 0.8],
            counts=[50, 50],
        )
        assert result.is_overconfident is False

    def test_to_dict_has_required_keys(self):
        confs, correct = _perfect_confidences(n=100)
        result = compute_calibration(confs, correct)
        d = result.to_dict()
        for key in ("ece", "mce", "num_bins", "is_overconfident",
                    "bin_confidences", "bin_accuracies", "bin_counts"):
            assert key in d

    def test_to_dict_values_are_serializable(self):
        """All values in to_dict() must be JSON-serializable (list/float/bool)."""
        import json
        confs, correct = _perfect_confidences(n=100)
        result = compute_calibration(confs, correct)
        d = result.to_dict()
        # Should not raise
        json.dumps(d)

    def test_to_dict_bin_lengths_match(self):
        confs, correct = _perfect_confidences(n=100)
        result = compute_calibration(confs, correct, num_bins=10)
        d = result.to_dict()
        assert len(d["bin_confidences"]) == 10
        assert len(d["bin_accuracies"]) == 10
        assert len(d["bin_counts"]) == 10


# ---------------------------------------------------------------------------
# ECE integration: known analytic case
# ---------------------------------------------------------------------------

class TestECEKnownCase:
    """
    Analytic test: two bins, both perfectly miscalibrated.

    Bin 1: conf=0.9, acc=0.0, n=100 → |error|=0.9, weight=100/200=0.5
    Bin 2: conf=0.1, acc=1.0, n=100 → |error|=0.9, weight=100/200=0.5
    ECE = 0.9 × 0.5 + 0.9 × 0.5 = 0.9

    Note: conf=0.1 items land in bin index 1 (acc=1.0 → all correct),
          conf=0.9 items land in bin index 9 (acc=0.0 → all wrong).
          Each bin |error| = 0.9, not 0.8.
    """

    def test_ece_analytic_two_bins(self):
        confs = [0.9] * 100 + [0.1] * 100
        correct = [False] * 100 + [True] * 100
        result = compute_calibration(confs, correct, num_bins=10)
        # ECE should be close to 0.9
        assert abs(result.ece - 0.9) < 0.05
