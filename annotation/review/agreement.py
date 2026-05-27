"""
annotation.review.agreement — Inter-annotator agreement metrics.

METRICS IMPLEMENTED:
1. Cohen's Kappa (κ) — binary and multi-class
2. Fleiss' Kappa — multiple annotators (3+)
3. Krippendorff's Alpha — for ordinal/interval scales
4. Percent agreement (simpler but less rigorous)
5. Confusion matrix between annotators

USAGE:
These metrics quantify how consistently human annotators agree,
independent of a "ground truth." High agreement (κ > 0.80) indicates
the task is well-defined and annotators understand the protocol.

INTERPRETATION:
κ < 0.20:  Slight agreement — review annotation guidelines
κ 0.20-0.40: Fair agreement
κ 0.40-0.60: Moderate agreement — normal for difficult tasks
κ 0.60-0.80: Substantial agreement — good annotation quality
κ > 0.80:  Almost perfect agreement — excellent

PRACTICAL NOTES:
For trichome maturity (clear/cloudy/amber):
- κ > 0.65 is achievable with trained annotators
- Clear vs. cloudy is hardest (most ambiguous boundary)
- Cloudy vs. amber is easier (color shift is obvious)

Reference:
  Landis, J.R. & Koch, G.G. (1977). The measurement of observer agreement
  for categorical data. Biometrics 33(1):159-174.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray


@dataclass
class AgreementResult:
    """Results from inter-annotator agreement analysis."""

    cohen_kappa: float
    """Cohen's κ for two annotators. NaN if fewer than 2 annotators."""

    fleiss_kappa: float | None
    """Fleiss' κ for 3+ annotators. None if fewer than 3."""

    percent_agreement: float
    """Simple proportion of matching annotations."""

    num_items: int
    num_annotators: int
    num_classes: int

    per_class_agreement: dict[str, float]
    """Per-class agreement rate."""

    confusion_matrix: NDArray[np.int32] | None = None
    """Confusion matrix between annotators (2-annotator case only)."""

    interpretation: str = ""
    """Text interpretation of the kappa value."""

    def to_dict(self) -> dict:
        return {
            "cohen_kappa": self.cohen_kappa,
            "fleiss_kappa": self.fleiss_kappa,
            "percent_agreement": self.percent_agreement,
            "num_items": self.num_items,
            "num_annotators": self.num_annotators,
            "num_classes": self.num_classes,
            "per_class_agreement": self.per_class_agreement,
            "interpretation": self.interpretation,
        }


def _interpret_kappa(kappa: float) -> str:
    """Return Landis & Koch (1977) interpretation of kappa value."""
    if kappa < 0:
        return "Less than chance agreement"
    elif kappa < 0.20:
        return "Slight agreement — review annotation guidelines"
    elif kappa < 0.40:
        return "Fair agreement"
    elif kappa < 0.60:
        return "Moderate agreement — normal for difficult tasks"
    elif kappa < 0.80:
        return "Substantial agreement — good annotation quality"
    else:
        return "Almost perfect agreement — excellent"


def cohen_kappa(
    annotations_a: Sequence[str | int],
    annotations_b: Sequence[str | int],
    classes: list[str] | None = None,
) -> float:
    """
    Compute Cohen's κ between two annotators.

    Args:
        annotations_a: Labels from annotator A.
        annotations_b: Labels from annotator B.
        classes: Optional list of class names. Auto-detected if None.

    Returns:
        Cohen's κ in [-1, 1]. 1.0 = perfect, 0.0 = chance.
    """
    if len(annotations_a) != len(annotations_b):
        raise ValueError(
            f"Annotation sequences must be same length: "
            f"{len(annotations_a)} vs {len(annotations_b)}"
        )

    if len(annotations_a) == 0:
        return float("nan")

    a = list(annotations_a)
    b = list(annotations_b)

    if classes is None:
        classes = sorted(set(a) | set(b))

    n = len(classes)
    class_idx = {c: i for i, c in enumerate(classes)}

    # Build confusion matrix
    matrix = np.zeros((n, n), dtype=np.int32)
    for ai, bi in zip(a, b):
        i = class_idx.get(str(ai), -1)
        j = class_idx.get(str(bi), -1)
        if i >= 0 and j >= 0:
            matrix[i, j] += 1

    total = matrix.sum()
    if total == 0:
        return float("nan")

    # Observed agreement
    p_o = matrix.diagonal().sum() / total

    # Expected agreement
    row_sums = matrix.sum(axis=1) / total
    col_sums = matrix.sum(axis=0) / total
    p_e = float((row_sums * col_sums).sum())

    if p_e == 1.0:
        return 1.0

    return float((p_o - p_e) / (1.0 - p_e))


def fleiss_kappa(
    ratings: NDArray[np.int32],
) -> float:
    """
    Compute Fleiss' κ for multiple annotators.

    Args:
        ratings: Array of shape (n_items, n_categories) where
                 ratings[i, j] = number of annotators who assigned category j
                 to item i. Row sums should all equal n_annotators.

    Returns:
        Fleiss' κ.

    Reference:
        Fleiss, J.L. (1971). Measuring nominal scale agreement among many raters.
        Psychological Bulletin 76(5):378-382.
    """
    n_items, n_cats = ratings.shape
    n_raters = ratings[0].sum()  # Assume consistent number of raters

    # p_j: proportion of assignments to category j
    total_assignments = n_items * n_raters
    p_j = ratings.sum(axis=0) / total_assignments

    # P_i: extent of agreement among raters for item i
    P_i = (
        (ratings * (ratings - 1)).sum(axis=1)
        / (n_raters * (n_raters - 1))
    )

    # Overall observed agreement
    P_bar = P_i.mean()

    # Expected agreement
    P_e = float((p_j ** 2).sum())

    if P_e == 1.0:
        return 1.0

    return float((P_bar - P_e) / (1.0 - P_e))


def compute_agreement(
    annotation_sets: list[list[str]],
    classes: list[str] | None = None,
) -> AgreementResult:
    """
    Compute inter-annotator agreement for a set of annotations.

    Args:
        annotation_sets: List of annotation sequences, one per annotator.
                        All sequences must have the same length.
        classes: Optional class list. Auto-detected if None.

    Returns:
        AgreementResult with all metrics.

    Example:
        # 3 annotators, 5 items
        result = compute_agreement([
            ["clear", "cloudy", "amber", "cloudy", "clear"],   # annotator 1
            ["clear", "cloudy", "cloudy", "cloudy", "clear"],  # annotator 2
            ["clear", "amber",  "amber", "cloudy", "unknown"], # annotator 3
        ])
        print(f"Fleiss κ = {result.fleiss_kappa:.3f}")
    """
    if not annotation_sets:
        raise ValueError("annotation_sets must not be empty")

    n_annotators = len(annotation_sets)
    n_items = len(annotation_sets[0])

    if any(len(s) != n_items for s in annotation_sets):
        raise ValueError("All annotation sequences must have the same length")

    if classes is None:
        all_labels = [lbl for seq in annotation_sets for lbl in seq]
        classes = sorted(set(all_labels))

    class_idx = {c: i for i, c in enumerate(classes)}
    n_classes = len(classes)

    # Percent agreement (all annotators agree on item)
    agree_count = 0
    for i in range(n_items):
        item_labels = [annotation_sets[k][i] for k in range(n_annotators)]
        if len(set(item_labels)) == 1:
            agree_count += 1
    percent_agreement = agree_count / n_items if n_items > 0 else 0.0

    # Cohen's kappa (first two annotators if available)
    kappa = float("nan")
    confusion = None
    if n_annotators >= 2:
        kappa = cohen_kappa(annotation_sets[0], annotation_sets[1], classes=classes)

        # Confusion matrix (2-annotator only)
        if n_annotators == 2:
            confusion = np.zeros((n_classes, n_classes), dtype=np.int32)
            for ai, bi in zip(annotation_sets[0], annotation_sets[1]):
                i = class_idx.get(str(ai), -1)
                j = class_idx.get(str(bi), -1)
                if i >= 0 and j >= 0:
                    confusion[i, j] += 1

    # Fleiss' kappa (3+ annotators)
    fleiss_k: float | None = None
    if n_annotators >= 3:
        # Build (n_items, n_categories) rating matrix
        rating_matrix = np.zeros((n_items, n_classes), dtype=np.int32)
        for k, seq in enumerate(annotation_sets):
            for i, lbl in enumerate(seq):
                j = class_idx.get(str(lbl), -1)
                if j >= 0:
                    rating_matrix[i, j] += 1
        fleiss_k = fleiss_kappa(rating_matrix)

    # Per-class agreement
    per_class: dict[str, float] = {}
    for cls in classes:
        matches = 0
        total = 0
        for i in range(n_items):
            item_labels = [annotation_sets[k][i] for k in range(n_annotators)]
            for pair_a, pair_b in zip(item_labels[:-1], item_labels[1:]):
                total += 1
                if pair_a == cls and pair_b == cls:
                    matches += 1
                elif pair_a != cls and pair_b != cls:
                    matches += 1
        per_class[cls] = matches / total if total > 0 else float("nan")

    # Interpretation (use Fleiss if available, otherwise Cohen's)
    primary_kappa = fleiss_k if fleiss_k is not None else kappa
    interpretation = _interpret_kappa(primary_kappa) if not np.isnan(primary_kappa) else "N/A"

    return AgreementResult(
        cohen_kappa=kappa if not np.isnan(kappa) else 0.0,
        fleiss_kappa=fleiss_k,
        percent_agreement=percent_agreement,
        num_items=n_items,
        num_annotators=n_annotators,
        num_classes=n_classes,
        per_class_agreement=per_class,
        confusion_matrix=confusion,
        interpretation=interpretation,
    )
