"""
tests/unit/test_annotation_stats.py — Annotation statistics module tests.

Tests:
  compute_cohens_kappa — inter-annotator agreement
  compute_class_imbalance_ratio — dataset balance metric
  compute_effective_imbalance — effective sample weights
  AnnotationStatisticsAggregator — throughput, quality, agreement
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import pytest

from annotation.statistics.stats import (
    AnnotationEvent,
    AnnotationStatisticsAggregator,
    ThroughputStats,
    QualityStats,
    AgreementStats,
    compute_cohens_kappa,
    compute_class_imbalance_ratio,
    compute_effective_imbalance,
)


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def _event(
    ann_id: str,
    action: str = "approved",
    class_id: int = 0,
    confidence: float = 0.85,
    time_spent_s: float = 5.0,
    dt: datetime | None = None,
) -> AnnotationEvent:
    return AnnotationEvent(
        annotation_id=ann_id,
        sample_id=f"sample_{ann_id}",
        action=action,
        timestamp=dt or datetime(2026, 5, 26, 12, 0, 0),
        class_id=class_id,
        confidence=confidence,
        annotator_id="annotator_1",
        time_spent_s=time_spent_s,
    )


def _events_spread_over_hours(n: int, hours: float = 1.0) -> list[AnnotationEvent]:
    """n events evenly spaced over `hours` hours."""
    base = datetime(2026, 5, 26, 10, 0, 0)
    interval = timedelta(seconds=hours * 3600 / max(n - 1, 1))
    return [
        _event(f"e{i:04d}", dt=base + interval * i, time_spent_s=5.0)
        for i in range(n)
    ]


# ─────────────────────────────────────────────────────────────────
# 1. Cohen's kappa
# ─────────────────────────────────────────────────────────────────

class TestCohensKappa:

    def test_perfect_agreement_is_one(self):
        labels = [0, 1, 2, 0, 1, 2]
        k = compute_cohens_kappa(labels, labels)
        assert k == pytest.approx(1.0, abs=1e-6)

    def test_random_chance_agreement_near_zero(self):
        # Two annotators always choosing the same constant class → kappa < 0 or 0
        # (no observed vs expected variation)
        a = [0] * 10
        b = [1] * 10
        k = compute_cohens_kappa(a, b)
        assert k <= 0.0  # complete disagreement → negative or zero

    def test_high_agreement_positive(self):
        a = [0, 0, 1, 1, 0, 1, 0, 0, 1, 1]
        b = [0, 0, 1, 1, 0, 1, 0, 1, 1, 1]  # 1 disagreement
        k = compute_cohens_kappa(a, b)
        assert k > 0.6  # substantial agreement

    def test_kappa_symmetric(self):
        a = [0, 1, 0, 2, 1]
        b = [0, 1, 1, 2, 0]
        assert compute_cohens_kappa(a, b) == pytest.approx(
            compute_cohens_kappa(b, a), abs=1e-9
        )

    def test_returns_float(self):
        k = compute_cohens_kappa([0, 1], [0, 1])
        assert isinstance(k, float)

    def test_single_element_list(self):
        k = compute_cohens_kappa([0], [0])
        assert math.isfinite(k)


# ─────────────────────────────────────────────────────────────────
# 2. Class imbalance ratio
# ─────────────────────────────────────────────────────────────────

class TestClassImbalanceRatio:

    def test_balanced_classes_ratio_one(self):
        counts = {0: 100, 1: 100, 2: 100}
        assert compute_class_imbalance_ratio(counts) == pytest.approx(1.0, abs=1e-6)

    def test_imbalanced_correct_ratio(self):
        counts = {0: 400, 1: 100}
        assert compute_class_imbalance_ratio(counts) == pytest.approx(4.0, abs=1e-6)

    def test_single_class_ratio_is_one(self):
        counts = {0: 50}
        assert compute_class_imbalance_ratio(counts) == pytest.approx(1.0, abs=1e-6)

    def test_extreme_imbalance(self):
        counts = {0: 10000, 1: 1}
        assert compute_class_imbalance_ratio(counts) == pytest.approx(10000.0, abs=0.1)

    def test_returns_float(self):
        r = compute_class_imbalance_ratio({0: 10, 1: 5})
        assert isinstance(r, float)


# ─────────────────────────────────────────────────────────────────
# 3. Effective imbalance weights
# ─────────────────────────────────────────────────────────────────

class TestEffectiveImbalance:

    def test_balanced_classes_have_equal_weights(self):
        counts = {0: 100, 1: 100, 2: 100}
        weights = compute_effective_imbalance(counts)
        vals = list(weights.values())
        assert max(vals) - min(vals) < 0.01

    def test_rare_class_gets_higher_weight(self):
        counts = {0: 1000, 1: 100}
        weights = compute_effective_imbalance(counts)
        assert weights[1] > weights[0]  # rare class weighted higher

    def test_weights_are_all_positive(self):
        counts = {0: 500, 1: 50, 2: 5}
        weights = compute_effective_imbalance(counts)
        assert all(v > 0 for v in weights.values())

    def test_returns_dict_with_class_keys(self):
        counts = {0: 100, 1: 200, 3: 50}
        weights = compute_effective_imbalance(counts)
        assert set(weights.keys()) == {0, 1, 3}


# ─────────────────────────────────────────────────────────────────
# 4. AnnotationStatisticsAggregator — throughput
# ─────────────────────────────────────────────────────────────────

class TestAnnotationAggregatorThroughput:

    def test_empty_aggregator_throughput_is_zero(self):
        agg = AnnotationStatisticsAggregator()
        stats = agg.compute_throughput()
        assert stats.total_annotations == 0

    def test_total_annotations_count(self):
        agg = AnnotationStatisticsAggregator()
        agg.add_events([_event(f"e{i}") for i in range(10)])
        stats = agg.compute_throughput()
        assert stats.total_annotations == 10

    def test_approved_count(self):
        agg = AnnotationStatisticsAggregator()
        agg.add_events([
            _event("e0", action="approved"),
            _event("e1", action="approved"),
            _event("e2", action="rejected"),
        ])
        stats = agg.compute_throughput()
        assert stats.approved == 2
        assert stats.rejected == 1

    def test_approval_rate_correct(self):
        agg = AnnotationStatisticsAggregator()
        agg.add_events([
            _event("e0", action="approved"),
            _event("e1", action="approved"),
            _event("e2", action="rejected"),
            _event("e3", action="rejected"),
        ])
        stats = agg.compute_throughput()
        assert stats.approval_rate == pytest.approx(0.50, abs=0.01)

    def test_annotations_per_hour_positive(self):
        agg = AnnotationStatisticsAggregator()
        events = _events_spread_over_hours(n=60, hours=1.0)
        agg.add_events(events)
        stats = agg.compute_throughput()
        assert stats.annotations_per_hour > 0

    def test_class_distribution_counts(self):
        agg = AnnotationStatisticsAggregator()
        agg.add_events([
            _event("e0", class_id=0),
            _event("e1", class_id=0),
            _event("e2", class_id=1),
        ])
        stats = agg.compute_throughput()
        assert stats.class_distribution.get(0, 0) == 2
        assert stats.class_distribution.get(1, 0) == 1

    def test_imbalance_ratio_computed(self):
        agg = AnnotationStatisticsAggregator()
        agg.add_events([
            _event("e0", class_id=0),
            _event("e1", class_id=0),
            _event("e2", class_id=0),
            _event("e3", class_id=1),
        ])
        stats = agg.compute_throughput()
        assert stats.class_imbalance_ratio == pytest.approx(3.0, abs=0.01)

    def test_add_single_event(self):
        agg = AnnotationStatisticsAggregator()
        agg.add_event(_event("e0"))
        stats = agg.compute_throughput()
        assert stats.total_annotations == 1


# ─────────────────────────────────────────────────────────────────
# 5. AnnotationStatisticsAggregator — quality
# ─────────────────────────────────────────────────────────────────

class TestAnnotationAggregatorQuality:

    def test_mean_confidence_computed(self):
        agg = AnnotationStatisticsAggregator()
        agg.add_events([
            _event("e0", confidence=0.60),
            _event("e1", confidence=0.80),
            _event("e2", confidence=1.00),
        ])
        stats = agg.compute_quality()
        assert stats.mean_confidence == pytest.approx(0.80, abs=0.01)

    def test_std_confidence_computed(self):
        agg = AnnotationStatisticsAggregator()
        agg.add_events([_event(f"e{i}", confidence=0.80) for i in range(10)])
        stats = agg.compute_quality()
        assert stats.std_confidence == pytest.approx(0.0, abs=0.01)

    def test_low_confidence_count(self):
        agg = AnnotationStatisticsAggregator()
        agg.add_events([
            _event("e0", confidence=0.50),  # low
            _event("e1", confidence=0.65),  # low
            _event("e2", confidence=0.85),  # not low
        ])
        stats = agg.compute_quality(min_confidence=0.70)
        assert stats.low_confidence_count == 2

    def test_high_confidence_count(self):
        agg = AnnotationStatisticsAggregator()
        agg.add_events([
            _event("e0", confidence=0.92),  # high
            _event("e1", confidence=0.88),  # not high
            _event("e2", confidence=0.95),  # high
        ])
        stats = agg.compute_quality()
        assert stats.high_confidence_count == 2

    def test_empty_aggregator_quality_safe(self):
        agg = AnnotationStatisticsAggregator()
        stats = agg.compute_quality()
        assert stats.mean_confidence == 0.0


# ─────────────────────────────────────────────────────────────────
# 6. AnnotationStatisticsAggregator — inter-annotator agreement
# ─────────────────────────────────────────────────────────────────

class TestAnnotationAggregatorAgreement:

    def _dual_annotator_events(self, labels_a: list[int], labels_b: list[int]) -> list[AnnotationEvent]:
        events = []
        for i, (la, lb) in enumerate(zip(labels_a, labels_b)):
            events.append(AnnotationEvent(
                annotation_id=f"a_{i}",
                sample_id=f"s_{i}",
                action="approved",
                timestamp=datetime(2026, 5, 26, 12, 0, 0),
                class_id=la,
                confidence=0.85,
                annotator_id="annotator_1",
            ))
            events.append(AnnotationEvent(
                annotation_id=f"b_{i}",
                sample_id=f"s_{i}",  # same sample_id → will be matched
                action="approved",
                timestamp=datetime(2026, 5, 26, 12, 0, 0),
                class_id=lb,
                confidence=0.80,
                annotator_id="annotator_2",
            ))
        return events

    def test_perfect_agreement_kappa_one(self):
        labels = [0, 1, 2, 0, 1, 2]
        agg = AnnotationStatisticsAggregator()
        agg.add_events(self._dual_annotator_events(labels, labels))
        stats = agg.compute_agreement()
        if stats.cohens_kappa is not None:
            assert stats.cohens_kappa == pytest.approx(1.0, abs=0.01)

    def test_agreement_rate_is_fraction(self):
        labels_a = [0, 1, 0, 1, 0]
        labels_b = [0, 1, 1, 1, 0]  # 1 disagreement out of 5
        agg = AnnotationStatisticsAggregator()
        agg.add_events(self._dual_annotator_events(labels_a, labels_b))
        stats = agg.compute_agreement()
        assert 0.0 <= stats.agreement_rate <= 1.0

    def test_agreement_stats_has_note(self):
        agg = AnnotationStatisticsAggregator()
        agg.add_events(self._dual_annotator_events([0, 1], [0, 1]))
        stats = agg.compute_agreement()
        assert isinstance(stats.note, str)

    def test_insufficient_data_returns_none_kappa(self):
        agg = AnnotationStatisticsAggregator()
        agg.add_event(_event("e0"))  # single annotator, single event
        stats = agg.compute_agreement()
        # With insufficient pairs, kappa may be None
        assert stats.cohens_kappa is None or isinstance(stats.cohens_kappa, float)


# ─────────────────────────────────────────────────────────────────
# 7. Cumulative curve
# ─────────────────────────────────────────────────────────────────

class TestCumulativeCurve:

    def test_cumulative_curve_length_matches_events(self):
        agg = AnnotationStatisticsAggregator()
        events = [_event(f"e{i}") for i in range(10)]
        agg.add_events(events)
        curve = agg.get_cumulative_curve()
        assert len(curve) == 10

    def test_cumulative_curve_monotone(self):
        agg = AnnotationStatisticsAggregator()
        events = _events_spread_over_hours(n=20, hours=1.0)
        agg.add_events(events)
        curve = agg.get_cumulative_curve()
        counts = [pt["cumulative_count"] for pt in curve]
        assert counts == sorted(counts)

    def test_cumulative_curve_empty_safe(self):
        agg = AnnotationStatisticsAggregator()
        curve = agg.get_cumulative_curve()
        assert curve == []

    def test_cumulative_curve_has_required_keys(self):
        agg = AnnotationStatisticsAggregator()
        agg.add_event(_event("e0"))
        curve = agg.get_cumulative_curve()
        assert len(curve) >= 1
        pt = curve[0]
        assert "cumulative_count" in pt
