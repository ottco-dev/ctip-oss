"""
active_learning.analysis.drift — Dataset drift detection for trichome datasets.

Dataset drift occurs when the distribution of new unlabeled images differs
from the training distribution. Common in trichome datasets:
  - New strains with different morphology
  - Different microscope or magnification
  - Different lighting conditions
  - Different growth stage or processing

Drift detection approaches implemented:
1. Feature-based (MMD): Maximum Mean Discrepancy on image statistics
2. Prediction-based: Shift in confidence/class distribution
3. Statistical (KS test): Kolmogorov-Smirnov test on feature distributions
4. Representation drift: Cosine distance in embedding space

If drift is detected → trigger active learning re-annotation of new samples
before including them in training.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class DriftResult:
    """Result of a drift detection test."""

    test_name: str
    """Which drift test was applied."""

    drift_detected: bool
    """True if significant drift is detected."""

    severity: str
    """'none', 'mild', 'moderate', 'severe'."""

    score: float
    """Test-specific drift score. Higher = more drift."""

    threshold: float
    """Threshold used to determine drift_detected."""

    p_value: float | None = None
    """Statistical p-value if applicable."""

    details: dict[str, Any] = field(default_factory=dict)
    """Test-specific details."""


@dataclass
class DriftReport:
    """Combined drift analysis report."""

    results: list[DriftResult]
    overall_drift_detected: bool
    recommendation: str
    num_reference_samples: int
    num_test_samples: int
    analyzed_at: str = ""


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_image_statistics(
    image: NDArray[np.uint8],
) -> NDArray[np.float64]:
    """
    Extract simple statistical features from an image.

    Features (17-dim):
    - RGB channel means (3)
    - RGB channel stds (3)
    - HSV channel means (3)
    - HSV channel stds (3)
    - Laplacian variance (1) — sharpness
    - Histogram entropy (3) — R/G/B histogram entropies
    - Aspect ratio (1)

    Args:
        image: HWC uint8 RGB numpy array.

    Returns:
        17-dimensional feature vector.
    """
    import cv2

    features = []

    # RGB stats
    for c in range(3):
        ch = image[:, :, c].astype(np.float64)
        features.append(ch.mean())
        features.append(ch.std())

    # HSV stats
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    for c in range(3):
        ch = hsv[:, :, c].astype(np.float64)
        features.append(ch.mean())
        features.append(ch.std())

    # Sharpness (Laplacian variance)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    features.append(lap.var())

    # Histogram entropy (R, G, B)
    for c in range(3):
        hist, _ = np.histogram(image[:, :, c], bins=32, range=(0, 256))
        hist = hist / hist.sum()
        # Shannon entropy
        h = -np.sum(hist * np.log(hist + 1e-10))
        features.append(h)

    # Aspect ratio
    h_px, w_px = image.shape[:2]
    features.append(w_px / max(h_px, 1))

    return np.array(features, dtype=np.float64)


# ---------------------------------------------------------------------------
# MMD (Maximum Mean Discrepancy)
# ---------------------------------------------------------------------------

def compute_mmd(
    reference: NDArray[np.float64],
    test: NDArray[np.float64],
    kernel: str = "rbf",
    sigma: float | None = None,
) -> float:
    """
    Compute Maximum Mean Discrepancy between two feature sets.

    MMD ≈ 0: same distribution
    MMD > threshold: distribution shift detected

    Args:
        reference: (N, D) feature matrix from reference (training) data.
        test: (M, D) feature matrix from test (new) data.
        kernel: 'rbf' (Gaussian) or 'linear'.
        sigma: RBF bandwidth. If None, uses median heuristic.

    Returns:
        MMD² estimate (non-negative float).
    """
    def rbf_kernel(x: NDArray, y: NDArray, bw: float) -> NDArray:
        diff = x[:, np.newaxis] - y[np.newaxis]
        dist_sq = (diff ** 2).sum(axis=-1)
        return np.exp(-dist_sq / (2 * bw ** 2))

    def linear_kernel(x: NDArray, y: NDArray) -> NDArray:
        return x @ y.T

    ref = np.asarray(reference, dtype=np.float64)
    tst = np.asarray(test, dtype=np.float64)

    if sigma is None:
        # Median heuristic
        all_data = np.vstack([ref, tst])
        diffs = all_data[:, np.newaxis] - all_data[np.newaxis]
        pairwise = np.sqrt((diffs ** 2).sum(axis=-1))
        sigma = float(np.median(pairwise[pairwise > 0])) + 1e-8

    if kernel == "rbf":
        k_rr = rbf_kernel(ref, ref, sigma).mean()
        k_tt = rbf_kernel(tst, tst, sigma).mean()
        k_rt = rbf_kernel(ref, tst, sigma).mean()
    else:
        k_rr = linear_kernel(ref, ref).mean()
        k_tt = linear_kernel(tst, tst).mean()
        k_rt = linear_kernel(ref, tst).mean()

    mmd_sq = max(0.0, k_rr + k_tt - 2 * k_rt)
    return float(mmd_sq)


# ---------------------------------------------------------------------------
# KS test
# ---------------------------------------------------------------------------

def compute_ks_drift(
    reference_features: NDArray[np.float64],
    test_features: NDArray[np.float64],
    alpha: float = 0.05,
) -> DriftResult:
    """
    Per-feature Kolmogorov-Smirnov test for distribution drift.

    Applies KS test to each feature dimension independently.
    Reports fraction of features with significant p-values.

    Args:
        reference_features: (N, D) feature matrix.
        test_features: (M, D) feature matrix.
        alpha: Significance level.

    Returns:
        DriftResult with per-feature KS statistics.
    """
    ref = np.asarray(reference_features, dtype=np.float64)
    tst = np.asarray(test_features, dtype=np.float64)

    n_features = ref.shape[1]
    ks_stats = []
    p_values = []

    for f in range(n_features):
        ks_stat, p_val = scipy_stats.ks_2samp(ref[:, f], tst[:, f])
        ks_stats.append(float(ks_stat))
        p_values.append(float(p_val))

    n_significant = sum(1 for p in p_values if p < alpha)
    drift_fraction = n_significant / max(n_features, 1)
    mean_ks = float(np.mean(ks_stats))

    drift_detected = drift_fraction > 0.3  # >30% features drifted

    if drift_fraction > 0.6:
        severity = "severe"
    elif drift_fraction > 0.4:
        severity = "moderate"
    elif drift_fraction > 0.2:
        severity = "mild"
    else:
        severity = "none"

    return DriftResult(
        test_name="KolmogorovSmirnov",
        drift_detected=drift_detected,
        severity=severity,
        score=mean_ks,
        threshold=alpha,
        p_value=float(np.mean(p_values)),
        details={
            "n_significant_features": n_significant,
            "drift_fraction": drift_fraction,
            "per_feature_ks": ks_stats[:10],  # first 10 for brevity
            "per_feature_pvalue": p_values[:10],
        },
    )


# ---------------------------------------------------------------------------
# Prediction distribution shift
# ---------------------------------------------------------------------------

def compute_prediction_drift(
    reference_class_dist: dict[int, float],
    test_class_dist: dict[int, float],
    num_classes: int = 4,
    threshold: float = 0.20,
) -> DriftResult:
    """
    Detect drift in class prediction distribution.

    Uses Total Variation distance (TVD) between reference and test
    class probability distributions.

    Args:
        reference_class_dist: {class_id: fraction} for reference data.
        test_class_dist: {class_id: fraction} for new data.
        num_classes: Total class count.
        threshold: TVD threshold for drift detection (0.20 = 20% shift).

    Returns:
        DriftResult with TVD score and per-class comparison.
    """
    ref_vec = np.array([reference_class_dist.get(c, 0.0) for c in range(num_classes)])
    tst_vec = np.array([test_class_dist.get(c, 0.0) for c in range(num_classes)])

    # Normalize to proper distributions
    ref_sum = ref_vec.sum()
    tst_sum = tst_vec.sum()
    if ref_sum > 0:
        ref_vec = ref_vec / ref_sum
    if tst_sum > 0:
        tst_vec = tst_vec / tst_sum

    # Total Variation Distance
    tvd = float(0.5 * np.abs(ref_vec - tst_vec).sum())

    # Per-class shift
    per_class = {c: float(tst_vec[c] - ref_vec[c]) for c in range(num_classes)}

    drift_detected = tvd > threshold
    severity = (
        "severe" if tvd > 0.4
        else "moderate" if tvd > 0.25
        else "mild" if tvd > threshold
        else "none"
    )

    return DriftResult(
        test_name="PredictionDistribution",
        drift_detected=drift_detected,
        severity=severity,
        score=tvd,
        threshold=threshold,
        details={
            "total_variation_distance": tvd,
            "reference_distribution": {c: float(v) for c, v in enumerate(ref_vec)},
            "test_distribution": {c: float(v) for c, v in enumerate(tst_vec)},
            "per_class_shift": per_class,
        },
    )


# ---------------------------------------------------------------------------
# Main drift detector
# ---------------------------------------------------------------------------

class DriftDetector:
    """
    Combined drift detector for trichome datasets.

    Runs multiple drift tests and aggregates results.
    Triggers active learning re-annotation recommendation when drift detected.

    Usage::

        detector = DriftDetector()
        detector.fit_reference(reference_features)

        report = detector.analyze(new_features, new_class_dist)
        if report.overall_drift_detected:
            # Route new images to active learning queue
    """

    def __init__(
        self,
        mmd_threshold: float = 0.05,
        ks_alpha: float = 0.05,
        prediction_tvd_threshold: float = 0.20,
    ) -> None:
        self.mmd_threshold = mmd_threshold
        self.ks_alpha = ks_alpha
        self.prediction_tvd_threshold = prediction_tvd_threshold
        self._reference_features: NDArray[np.float64] | None = None
        self._reference_class_dist: dict[int, float] | None = None

    def fit_reference(
        self,
        features: NDArray[np.float64],
        class_distribution: dict[int, float] | None = None,
    ) -> None:
        """
        Set reference distribution from training data.

        Args:
            features: (N, D) feature matrix from training set.
            class_distribution: {class_id: fraction} for training labels.
        """
        self._reference_features = np.asarray(features, dtype=np.float64)
        self._reference_class_dist = class_distribution
        logger.info(
            "DriftDetector: fitted reference on %d samples (D=%d)",
            len(features),
            features.shape[1] if hasattr(features, "shape") else "?",
        )

    def analyze(
        self,
        test_features: NDArray[np.float64],
        test_class_dist: dict[int, float] | None = None,
    ) -> DriftReport:
        """
        Analyze new data for drift vs. reference.

        Args:
            test_features: (M, D) feature matrix from new images.
            test_class_dist: Class distribution from model predictions on new data.

        Returns:
            DriftReport with all test results and recommendation.
        """
        import time

        if self._reference_features is None:
            raise RuntimeError("Call fit_reference() first")

        ref = self._reference_features
        tst = np.asarray(test_features, dtype=np.float64)

        results: list[DriftResult] = []

        # 1. KS test
        ks_result = compute_ks_drift(ref, tst, alpha=self.ks_alpha)
        results.append(ks_result)

        # 2. MMD
        mmd_score = compute_mmd(ref, tst)
        mmd_result = DriftResult(
            test_name="MMD",
            drift_detected=mmd_score > self.mmd_threshold,
            severity=(
                "severe" if mmd_score > self.mmd_threshold * 4
                else "moderate" if mmd_score > self.mmd_threshold * 2
                else "mild" if mmd_score > self.mmd_threshold
                else "none"
            ),
            score=mmd_score,
            threshold=self.mmd_threshold,
        )
        results.append(mmd_result)

        # 3. Prediction distribution (if available)
        if test_class_dist is not None and self._reference_class_dist is not None:
            pred_result = compute_prediction_drift(
                self._reference_class_dist,
                test_class_dist,
                threshold=self.prediction_tvd_threshold,
            )
            results.append(pred_result)

        # Overall assessment
        n_detected = sum(1 for r in results if r.drift_detected)
        overall_drift = n_detected >= 1  # any test detects drift

        max_severity = "none"
        for r in results:
            if r.severity == "severe":
                max_severity = "severe"
                break
            if r.severity == "moderate":
                max_severity = "moderate"
            elif r.severity == "mild" and max_severity == "none":
                max_severity = "mild"

        if max_severity == "severe":
            rec = (
                "SEVERE DRIFT: New images differ significantly from training distribution. "
                "Recommend labeling a random sample of ≥50 new images before including "
                "them in the training set."
            )
        elif max_severity == "moderate":
            rec = (
                "MODERATE DRIFT: Meaningful distribution shift detected. "
                "Prioritize new images in annotation queue using uncertainty sampling."
            )
        elif max_severity == "mild":
            rec = (
                "MILD DRIFT: Minor distribution shift. "
                "Monitor; include top-uncertainty samples in next annotation batch."
            )
        else:
            rec = "No significant drift detected. New images are consistent with training distribution."

        return DriftReport(
            results=results,
            overall_drift_detected=overall_drift,
            recommendation=rec,
            num_reference_samples=len(ref),
            num_test_samples=len(tst),
            analyzed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
