"""
annotation/statistics/stats.py — Annotation throughput and quality metrics.

Computes:
  - Annotation throughput (annotations/hour, images/hour)
  - Quality metrics (confidence distribution, approval rate)
  - Class distribution and imbalance
  - Inter-annotator agreement (Cohen's kappa)
  - Cumulative annotation curves
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class AnnotationEvent:
    """One annotation event (approve/reject/edit)."""

    annotation_id: str
    sample_id: str
    action: str  # approved | rejected | edited
    timestamp: datetime
    class_id: Optional[int] = None
    confidence: float = 0.0
    annotator_id: str = "annotator_1"
    time_spent_s: float = 0.0


@dataclass
class ThroughputStats:
    """Annotation throughput statistics."""

    total_annotations: int
    approved: int
    rejected: int
    edited: int
    approval_rate: float
    rejection_rate: float
    annotations_per_hour: float
    images_per_hour: float
    mean_time_per_annotation_s: float
    session_duration_h: float
    class_distribution: dict[int, int] = field(default_factory=dict)
    class_imbalance_ratio: float = 1.0


@dataclass
class QualityStats:
    """Annotation quality metrics."""

    mean_confidence: float
    std_confidence: float
    low_confidence_count: int  # confidence < 0.70
    high_confidence_count: int  # confidence >= 0.90
    confidence_histogram: dict[str, int] = field(default_factory=dict)


@dataclass
class AgreementStats:
    """Inter-annotator agreement statistics."""

    cohens_kappa: Optional[float]
    agreement_rate: float
    disagreement_pairs: int
    total_pairs: int
    note: str = ""


# ---------------------------------------------------------------------------
# Statistical functions
# ---------------------------------------------------------------------------


def compute_cohens_kappa(
    annotations_a: list[int],
    annotations_b: list[int],
) -> float:
    """
    Compute Cohen's kappa for two annotators.

    Args:
        annotations_a: Class labels from annotator A.
        annotations_b: Class labels from annotator B (same items in same order).

    Returns:
        Cohen's kappa in [-1, 1]. > 0.6 = substantial agreement.

    Reference: Cohen J. (1960). A coefficient of agreement for nominal scales.
               Educational and Psychological Measurement 20(1):37-46.
    """
    if len(annotations_a) != len(annotations_b):
        raise ValueError("Both annotation lists must have the same length")
    if not annotations_a:
        return 0.0

    n = len(annotations_a)
    classes = list(set(annotations_a + annotations_b))
    k = len(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    # Observed agreement
    agree = sum(1 for a, b in zip(annotations_a, annotations_b) if a == b)
    p_o = agree / n

    # Expected agreement
    p_e = 0.0
    for cls in classes:
        p_a = annotations_a.count(cls) / n
        p_b = annotations_b.count(cls) / n
        p_e += p_a * p_b

    if abs(1.0 - p_e) < 1e-10:
        return 1.0  # Perfect agreement, avoid division by zero

    kappa = (p_o - p_e) / (1.0 - p_e)
    return round(kappa, 4)


def compute_class_imbalance_ratio(class_counts: dict[int, int]) -> float:
    """
    Compute class imbalance ratio (most common / rarest class count).

    1.0 = perfectly balanced, higher = more imbalanced.
    """
    if not class_counts:
        return 1.0
    counts = list(class_counts.values())
    return max(counts) / max(1, min(counts))


def compute_effective_imbalance(class_counts: dict[int, int], beta: float = 0.9999) -> dict[int, float]:
    """
    Compute effective sample weights using Cui et al. (2019) formula.
    Effective number = (1 - beta^n) / (1 - beta)
    """
    weights = {}
    for cls, n in class_counts.items():
        effective_n = (1.0 - beta ** n) / (1.0 - beta)
        weights[cls] = 1.0 / effective_n
    # Normalize so max weight = 1.0
    max_w = max(weights.values()) if weights else 1.0
    return {cls: round(w / max_w, 4) for cls, w in weights.items()}


# ---------------------------------------------------------------------------
# Statistics aggregator
# ---------------------------------------------------------------------------


class AnnotationStatisticsAggregator:
    """
    Aggregates annotation events and computes statistics on demand.

    Usage:
        agg = AnnotationStatisticsAggregator()
        for event in events:
            agg.add_event(event)
        stats = agg.compute_throughput()
    """

    def __init__(self) -> None:
        self._events: list[AnnotationEvent] = []

    def add_event(self, event: AnnotationEvent) -> None:
        """Add one annotation event."""
        self._events.append(event)

    def add_events(self, events: list[AnnotationEvent]) -> None:
        """Add multiple events."""
        self._events.extend(events)

    def compute_throughput(self) -> ThroughputStats:
        """Compute annotation throughput statistics."""
        if not self._events:
            return ThroughputStats(
                total_annotations=0, approved=0, rejected=0, edited=0,
                approval_rate=0.0, rejection_rate=0.0,
                annotations_per_hour=0.0, images_per_hour=0.0,
                mean_time_per_annotation_s=0.0, session_duration_h=0.0,
            )

        approved = sum(1 for e in self._events if e.action == "approved")
        rejected = sum(1 for e in self._events if e.action == "rejected")
        edited = sum(1 for e in self._events if e.action == "edited")
        total = len(self._events)

        # Time range
        timestamps = [e.timestamp for e in self._events]
        start = min(timestamps)
        end = max(timestamps)
        duration_h = max(1.0 / 3600, (end - start).total_seconds() / 3600)

        # Class distribution
        class_dist: dict[int, int] = {}
        for e in self._events:
            if e.class_id is not None and e.action != "rejected":
                class_dist[e.class_id] = class_dist.get(e.class_id, 0) + 1

        # Mean time per annotation
        time_spent = [e.time_spent_s for e in self._events if e.time_spent_s > 0]
        mean_time = sum(time_spent) / len(time_spent) if time_spent else 0.0

        unique_images = len(set(e.sample_id for e in self._events))

        return ThroughputStats(
            total_annotations=total,
            approved=approved,
            rejected=rejected,
            edited=edited,
            approval_rate=round(approved / total, 3) if total > 0 else 0.0,
            rejection_rate=round(rejected / total, 3) if total > 0 else 0.0,
            annotations_per_hour=round(total / duration_h, 1),
            images_per_hour=round(unique_images / duration_h, 1),
            mean_time_per_annotation_s=round(mean_time, 1),
            session_duration_h=round(duration_h, 2),
            class_distribution=class_dist,
            class_imbalance_ratio=round(compute_class_imbalance_ratio(class_dist), 2),
        )

    def compute_quality(self, min_confidence: float = 0.70) -> QualityStats:
        """Compute annotation quality metrics."""
        confidences = [e.confidence for e in self._events if e.confidence > 0]
        if not confidences:
            return QualityStats(
                mean_confidence=0.0,
                std_confidence=0.0,
                low_confidence_count=0,
                high_confidence_count=0,
            )

        mean_c = sum(confidences) / len(confidences)
        variance = sum((c - mean_c) ** 2 for c in confidences) / len(confidences)
        std_c = math.sqrt(variance)

        bins = {"[0.0-0.5)": 0, "[0.5-0.7)": 0, "[0.7-0.9)": 0, "[0.9-1.0]": 0}
        for c in confidences:
            if c < 0.5:
                bins["[0.0-0.5)"] += 1
            elif c < 0.7:
                bins["[0.5-0.7)"] += 1
            elif c < 0.9:
                bins["[0.7-0.9)"] += 1
            else:
                bins["[0.9-1.0]"] += 1

        return QualityStats(
            mean_confidence=round(mean_c, 4),
            std_confidence=round(std_c, 4),
            low_confidence_count=sum(1 for c in confidences if c < min_confidence),
            high_confidence_count=sum(1 for c in confidences if c >= 0.90),
            confidence_histogram=bins,
        )

    def compute_agreement(
        self,
        annotator_a: str = "annotator_1",
        annotator_b: str = "annotator_2",
    ) -> AgreementStats:
        """
        Compute inter-annotator agreement for two annotators.

        Finds samples labeled by both annotators and computes Cohen's kappa.
        """
        events_a = {e.sample_id: e.class_id for e in self._events if e.annotator_id == annotator_a}
        events_b = {e.sample_id: e.class_id for e in self._events if e.annotator_id == annotator_b}

        common = set(events_a.keys()) & set(events_b.keys())
        if len(common) < 5:
            return AgreementStats(
                cohens_kappa=None,
                agreement_rate=0.0,
                disagreement_pairs=0,
                total_pairs=0,
                note="Insufficient overlapping annotations (need ≥ 5)",
            )

        labels_a = [events_a[s] for s in sorted(common) if events_a[s] is not None]
        labels_b = [events_b[s] for s in sorted(common) if events_b[s] is not None]

        if len(labels_a) != len(labels_b):
            labels_a = labels_a[:min(len(labels_a), len(labels_b))]
            labels_b = labels_b[:len(labels_a)]

        if not labels_a:
            return AgreementStats(
                cohens_kappa=None,
                agreement_rate=0.0,
                disagreement_pairs=0,
                total_pairs=0,
            )

        kappa = compute_cohens_kappa(labels_a, labels_b)
        agreement = sum(1 for a, b in zip(labels_a, labels_b) if a == b)
        n = len(labels_a)

        return AgreementStats(
            cohens_kappa=kappa,
            agreement_rate=round(agreement / n, 3),
            disagreement_pairs=n - agreement,
            total_pairs=n,
            note=_kappa_interpretation(kappa),
        )

    def get_cumulative_curve(self) -> list[dict]:
        """Return cumulative annotation count over time for progress chart."""
        sorted_events = sorted(self._events, key=lambda e: e.timestamp)
        curve = []
        for i, event in enumerate(sorted_events):
            curve.append(
                {
                    "timestamp": event.timestamp.isoformat(),
                    "cumulative_count": i + 1,
                    "approved": sum(1 for e in sorted_events[:i+1] if e.action == "approved"),
                }
            )
        return curve


def _kappa_interpretation(kappa: float) -> str:
    if kappa < 0:
        return "Poor (< 0): less agreement than chance"
    elif kappa < 0.20:
        return "Slight (0-0.20)"
    elif kappa < 0.40:
        return "Fair (0.20-0.40)"
    elif kappa < 0.60:
        return "Moderate (0.40-0.60)"
    elif kappa < 0.80:
        return "Substantial (0.60-0.80) — matches human expert level"
    else:
        return "Almost perfect (0.80-1.0)"
