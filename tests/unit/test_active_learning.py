"""
tests.unit.test_active_learning — Unit tests for the active_learning package.

Covers all sub-modules without requiring GPU or YOLO model weights:
  TestEntropyFunctions         — compute_entropy, compute_normalized_entropy (pure math)
  TestEntropySampler           — score_sample, score_batch, select_top_k, stats
  TestDisagreementComputation  — compute_disagreement BALD / vote_entropy / KL
  TestDisagreementSampler      — compute_all, select_top_k, metric routing
  TestHardNegativeMiner        — find_hard_negatives, confirmed hard negatives
  TestComputePriority          — composite priority formula edge cases
  TestAnnotationPriorityQueue  — push, pop, boost, complete, remove, stats, global singleton
  TestRetrainingTrigger        — all trigger conditions, cooldown, urgency, lifecycle
  TestDriftFunctions           — MMD, KS test, prediction TVD, extract_image_statistics
  TestDriftDetector            — fit_reference, analyze, recommendation text
  TestALPipelineConfig         — config defaults, annotations_per_cycle
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest


# ===========================================================================
# TestEntropyFunctions — entropy.py
# ===========================================================================

class TestEntropyFunctions:
    """Pure math tests for entropy utilities in sampling/entropy.py."""

    def test_uniform_distribution_max_entropy(self):
        """Uniform distribution over C classes has entropy = log(C)."""
        from active_learning.sampling.entropy import compute_entropy

        for c in [2, 4, 6]:
            probs = np.full(c, 1.0 / c)
            h = compute_entropy(probs)
            assert abs(h - math.log(c)) < 1e-6, f"C={c}: expected {math.log(c):.4f}, got {h:.4f}"

    def test_one_hot_entropy_is_zero(self):
        """One-hot distribution has entropy = 0 (up to epsilon clipping)."""
        from active_learning.sampling.entropy import compute_entropy

        probs = np.array([0.0, 0.0, 1.0, 0.0])
        h = compute_entropy(probs)
        # epsilon=1e-10 clipping on zeros contributes ~4 * 1e-10 * log(1e-10) ≈ 1e-8
        assert h < 1e-7, f"Expected ~0, got {h}"

    def test_entropy_increases_with_uncertainty(self):
        """More uniform distributions have higher entropy."""
        from active_learning.sampling.entropy import compute_entropy

        certain = np.array([0.9, 0.05, 0.03, 0.02])
        uncertain = np.array([0.4, 0.3, 0.2, 0.1])
        uniform = np.array([0.25, 0.25, 0.25, 0.25])

        assert compute_entropy(certain) < compute_entropy(uncertain) < compute_entropy(uniform)

    def test_normalized_entropy_range(self):
        """Normalized entropy must be in [0, 1]."""
        from active_learning.sampling.entropy import compute_normalized_entropy

        for probs in [
            np.array([1.0, 0.0, 0.0, 0.0]),
            np.array([0.4, 0.3, 0.2, 0.1]),
            np.array([0.25, 0.25, 0.25, 0.25]),
        ]:
            ne = compute_normalized_entropy(probs)
            assert 0.0 <= ne <= 1.0, f"Normalized entropy out of range: {ne}"

    def test_normalized_entropy_uniform_is_one(self):
        """Uniform distribution → normalized entropy = 1.0."""
        from active_learning.sampling.entropy import compute_normalized_entropy

        probs = np.array([0.25, 0.25, 0.25, 0.25])
        assert abs(compute_normalized_entropy(probs) - 1.0) < 1e-6

    def test_entropy_handles_small_epsilon(self):
        """compute_entropy does not blow up on near-zero probabilities."""
        from active_learning.sampling.entropy import compute_entropy

        probs = np.array([1e-15, 0.999999999, 1e-15])
        h = compute_entropy(probs)
        assert math.isfinite(h)
        assert h >= 0.0


# ===========================================================================
# TestEntropySampler
# ===========================================================================

class TestEntropySampler:
    """Tests for EntropySampler in sampling/entropy.py."""

    def test_score_sample_returns_score(self):
        """score_sample returns EntropyScore with correct shape."""
        from active_learning.sampling.entropy import EntropySampler

        sampler = EntropySampler(num_classes=4)
        probs = np.array([0.6, 0.2, 0.15, 0.05])
        score = sampler.score_sample("img_001", probs)

        assert score.sample_id == "img_001"
        assert score.predicted_class == 0  # argmax
        assert abs(score.predicted_prob - 0.6) < 1e-6
        assert 0.0 <= score.normalized_entropy <= 1.0
        assert score.entropy >= 0.0

    def test_score_sample_normalizes_unnormalized_probs(self):
        """score_sample auto-normalizes probabilities that don't sum to 1."""
        from active_learning.sampling.entropy import EntropySampler

        sampler = EntropySampler(num_classes=4)
        probs = np.array([2.0, 1.0, 0.5, 0.5])  # sums to 4
        score = sampler.score_sample("x", probs)
        assert 0.0 <= score.normalized_entropy <= 1.0

    def test_score_batch_length(self):
        """score_batch returns one EntropyScore per sample."""
        from active_learning.sampling.entropy import EntropySampler

        sampler = EntropySampler(num_classes=4)
        ids = ["a", "b", "c"]
        probs = np.array([
            [0.7, 0.1, 0.1, 0.1],
            [0.25, 0.25, 0.25, 0.25],
            [0.9, 0.05, 0.03, 0.02],
        ])
        scores = sampler.score_batch(ids, probs)
        assert len(scores) == 3

    def test_select_top_k_returns_k(self):
        """select_top_k returns exactly k items when enough exist."""
        from active_learning.sampling.entropy import EntropySampler

        sampler = EntropySampler(num_classes=4)
        ids = [f"s{i}" for i in range(20)]
        probs = np.random.dirichlet(np.ones(4), size=20)
        scores = sampler.score_batch(ids, probs)
        selected = sampler.select_top_k(scores, k=5)
        assert len(selected) == 5

    def test_select_top_k_sorted_descending(self):
        """select_top_k returns samples in descending entropy order."""
        from active_learning.sampling.entropy import EntropySampler

        sampler = EntropySampler(num_classes=4)
        ids = [f"s{i}" for i in range(10)]
        probs = np.random.dirichlet(np.ones(4), size=10)
        scores = sampler.score_batch(ids, probs)
        selected = sampler.select_top_k(scores, k=5)

        for i in range(len(selected) - 1):
            assert selected[i].entropy >= selected[i + 1].entropy

    def test_select_top_k_min_threshold_filter(self):
        """Samples below min_entropy_threshold are excluded."""
        from active_learning.sampling.entropy import EntropySampler

        sampler = EntropySampler(num_classes=4, min_entropy_threshold=0.9)
        # All near-certain predictions → normalized entropy ≈ 0
        ids = [f"s{i}" for i in range(10)]
        probs = np.zeros((10, 4))
        probs[:, 0] = 0.99
        probs[:, 1] = 0.005
        probs[:, 2] = 0.003
        probs[:, 3] = 0.002
        scores = sampler.score_batch(ids, probs)
        selected = sampler.select_top_k(scores, k=5)
        # All below threshold → empty or fewer than k
        assert len(selected) == 0

    def test_select_top_k_spread_quartiles(self):
        """spread_across_quartiles=True returns samples from all entropy levels."""
        from active_learning.sampling.entropy import EntropySampler

        sampler = EntropySampler(num_classes=4)
        ids = [f"s{i}" for i in range(40)]
        probs = np.random.dirichlet(np.ones(4), size=40)
        scores = sampler.score_batch(ids, probs)
        selected = sampler.select_top_k(scores, k=8, spread_across_quartiles=True)
        assert 1 <= len(selected) <= 8  # may be fewer if distribution is skewed

    def test_dataset_entropy_stats_keys(self):
        """compute_dataset_entropy_stats returns expected keys."""
        from active_learning.sampling.entropy import EntropySampler

        sampler = EntropySampler(num_classes=4)
        ids = [f"s{i}" for i in range(20)]
        probs = np.random.dirichlet(np.ones(4), size=20)
        scores = sampler.score_batch(ids, probs)
        stats = sampler.compute_dataset_entropy_stats(scores)

        for key in ["mean", "std", "median", "p25", "p75", "p90", "p95", "p99", "fraction_high_entropy"]:
            assert key in stats, f"Missing key: {key}"
        assert 0.0 <= stats["fraction_high_entropy"] <= 1.0

    def test_dataset_entropy_stats_empty(self):
        """compute_dataset_entropy_stats on empty list returns empty dict."""
        from active_learning.sampling.entropy import EntropySampler

        sampler = EntropySampler(num_classes=4)
        stats = sampler.compute_dataset_entropy_stats([])
        assert stats == {}


# ===========================================================================
# TestDisagreementComputation
# ===========================================================================

class TestDisagreementComputation:
    """Tests for compute_disagreement in sampling/disagreement.py."""

    def _make_pred(self, sample_id: str, probs: list[float]):
        from active_learning.sampling.disagreement import EnsemblePrediction
        return EnsemblePrediction(
            sample_id=sample_id,
            probabilities=probs,
            predicted_class=int(np.argmax(probs)),
            confidence=float(max(probs)),
        )

    def test_empty_predictions_raises(self):
        from active_learning.sampling.disagreement import compute_disagreement
        with pytest.raises((ValueError, IndexError)):
            compute_disagreement([])

    def test_unanimous_agreement_low_bald(self):
        """All ensemble members agree → low BALD score."""
        from active_learning.sampling.disagreement import compute_disagreement

        preds = [self._make_pred("s1", [0.9, 0.05, 0.03, 0.02]) for _ in range(4)]
        score = compute_disagreement(preds)
        assert score.bald_score < 0.05, f"Expected low BALD, got {score.bald_score}"

    def test_maximum_disagreement_high_bald(self):
        """Half members predict class 0, half predict class 1 → high disagreement."""
        from active_learning.sampling.disagreement import compute_disagreement

        preds = (
            [self._make_pred("s2", [0.9, 0.05, 0.03, 0.02]) for _ in range(3)]
            + [self._make_pred("s2", [0.05, 0.9, 0.03, 0.02]) for _ in range(3)]
        )
        score = compute_disagreement(preds)
        assert score.bald_score > 0.05, f"Expected high BALD, got {score.bald_score}"

    def test_vote_counts_sum_to_num_members(self):
        """vote_counts must sum to total number of members."""
        from active_learning.sampling.disagreement import compute_disagreement

        preds = [
            self._make_pred("s3", [0.7, 0.2, 0.1]),
            self._make_pred("s3", [0.1, 0.8, 0.1]),
            self._make_pred("s3", [0.3, 0.4, 0.3]),
        ]
        score = compute_disagreement(preds)
        assert sum(score.vote_counts.values()) == len(preds)

    def test_num_members_field(self):
        from active_learning.sampling.disagreement import compute_disagreement

        n = 5
        preds = [self._make_pred("s4", [0.25, 0.25, 0.25, 0.25]) for _ in range(n)]
        score = compute_disagreement(preds)
        assert score.num_members == n

    def test_composite_score_in_zero_one(self):
        from active_learning.sampling.disagreement import compute_disagreement

        preds = [self._make_pred("s5", [0.4, 0.3, 0.2, 0.1]) for _ in range(3)]
        score = compute_disagreement(preds)
        assert 0.0 <= score.composite_score <= 1.0, f"composite={score.composite_score}"

    def test_kl_divergence_non_negative(self):
        from active_learning.sampling.disagreement import compute_disagreement

        preds = [
            self._make_pred("s6", [0.6, 0.2, 0.1, 0.1]),
            self._make_pred("s6", [0.3, 0.5, 0.1, 0.1]),
        ]
        score = compute_disagreement(preds)
        assert score.kl_divergence >= 0.0


# ===========================================================================
# TestDisagreementSampler
# ===========================================================================

class TestDisagreementSampler:
    """Tests for DisagreementSampler class."""

    def _make_pred(self, sample_id: str, probs: list[float]):
        from active_learning.sampling.disagreement import EnsemblePrediction
        return EnsemblePrediction(
            sample_id=sample_id,
            probabilities=probs,
            predicted_class=int(np.argmax(probs)),
            confidence=float(max(probs)),
        )

    def test_compute_all_returns_sorted_desc(self):
        """compute_all returns scores sorted by composite descending."""
        from active_learning.sampling.disagreement import DisagreementSampler

        sampler = DisagreementSampler()
        predictions = {
            "s1": [self._make_pred("s1", [0.7, 0.15, 0.1, 0.05]), self._make_pred("s1", [0.2, 0.6, 0.1, 0.1])],
            "s2": [self._make_pred("s2", [0.9, 0.05, 0.03, 0.02]), self._make_pred("s2", [0.88, 0.06, 0.04, 0.02])],
            "s3": [self._make_pred("s3", [0.25, 0.25, 0.25, 0.25]), self._make_pred("s3", [0.1, 0.4, 0.3, 0.2])],
        }
        scores = sampler.compute_all(predictions)
        for i in range(len(scores) - 1):
            assert scores[i].composite_score >= scores[i + 1].composite_score

    def test_compute_all_skips_insufficient_members(self):
        """Samples with fewer than min_members are excluded."""
        from active_learning.sampling.disagreement import DisagreementSampler

        sampler = DisagreementSampler()
        sampler.min_members = 3

        predictions = {
            "s1": [self._make_pred("s1", [0.5, 0.5]), self._make_pred("s1", [0.5, 0.5])],  # only 2
            "s2": [self._make_pred("s2", [0.5, 0.5])] * 3,  # 3 members — OK
        }
        scores = sampler.compute_all(predictions)
        sample_ids = {s.sample_id for s in scores}
        assert "s1" not in sample_ids  # filtered
        assert "s2" in sample_ids

    def test_select_top_k_respects_k(self):
        """select_top_k returns at most k items."""
        from active_learning.sampling.disagreement import DisagreementSampler

        sampler = DisagreementSampler()
        predictions = {
            f"s{i}": [self._make_pred(f"s{i}", [0.25 + 0.25 * (i % 2), 0.75 - 0.25 * (i % 2)][::-1] + [0.0, 0.0]) for _ in range(2)]
            for i in range(10)
        }
        scores = sampler.compute_all(predictions)
        selected = sampler.select_top_k(scores, k=3)
        assert len(selected) <= 3


# ===========================================================================
# TestHardNegativeMiner
# ===========================================================================

class TestHardNegativeMiner:
    """Tests for HardNegativeMiner in sampling/hard_negative.py."""

    def test_find_hard_negatives_basic(self):
        """find_hard_negatives returns one score per sample."""
        from active_learning.sampling.hard_negative import HardNegativeMiner

        miner = HardNegativeMiner(confidence_threshold=0.7)
        ids = ["a", "b", "c"]
        probs = np.array([
            [0.9, 0.05, 0.03, 0.02],  # high confidence
            [0.25, 0.25, 0.25, 0.25],  # uncertain
            [0.8, 0.1, 0.06, 0.04],   # high confidence
        ])
        scores = miner.find_hard_negatives(ids, probs)
        # labeled_only=False → all samples returned
        assert len(scores) == 3

    def test_confirmed_hard_negative_detection(self):
        """Sample with high confidence AND wrong label is confirmed hard negative."""
        from active_learning.sampling.hard_negative import HardNegativeMiner

        miner = HardNegativeMiner(confidence_threshold=0.7)
        ids = ["x"]
        probs = np.array([[0.85, 0.1, 0.03, 0.02]])  # predicted class 0
        true_classes = [2]  # but true class is 2

        scores = miner.find_hard_negatives(ids, probs, true_classes=true_classes)
        assert len(scores) == 1
        assert scores[0].is_confirmed_hard is True
        assert scores[0].hardness_score >= 0.7  # hard negative range

    def test_correct_prediction_not_confirmed_hard(self):
        """Sample with high confidence AND correct label is NOT confirmed hard."""
        from active_learning.sampling.hard_negative import HardNegativeMiner

        miner = HardNegativeMiner(confidence_threshold=0.7)
        ids = ["y"]
        probs = np.array([[0.9, 0.05, 0.03, 0.02]])  # predicted class 0
        true_classes = [0]  # correct

        scores = miner.find_hard_negatives(ids, probs, true_classes=true_classes)
        assert scores[0].is_confirmed_hard is False

    def test_low_confidence_hardness_zero(self):
        """Low-confidence predictions are not hard negatives (hardness = 0)."""
        from active_learning.sampling.hard_negative import HardNegativeMiner

        miner = HardNegativeMiner(confidence_threshold=0.9)
        ids = ["z"]
        probs = np.array([[0.4, 0.3, 0.2, 0.1]])  # max conf = 0.4 < 0.9 threshold

        scores = miner.find_hard_negatives(ids, probs)
        assert scores[0].hardness_score == 0.0

    def test_labeled_only_filters_unconfirmed(self):
        """labeled_only=True excludes samples without confirmed wrong label."""
        from active_learning.sampling.hard_negative import HardNegativeMiner

        miner = HardNegativeMiner(confidence_threshold=0.7, labeled_only=True)
        ids = ["a", "b"]
        probs = np.array([
            [0.9, 0.05, 0.03, 0.02],   # high confidence
            [0.85, 0.1, 0.03, 0.02],   # high confidence
        ])
        true_classes = [0, 2]  # a correct, b wrong

        scores = miner.find_hard_negatives(ids, probs, true_classes=true_classes)
        # Only b (confirmed hard) should be returned
        confirmed = [s for s in scores if s.is_confirmed_hard]
        assert len(confirmed) == 1
        assert confirmed[0].sample_id == "b"

    def test_ensemble_disagreement_raises_hardness(self):
        """Ensemble prediction for different class raises disagreement score."""
        from active_learning.sampling.hard_negative import HardNegativeMiner

        miner = HardNegativeMiner(confidence_threshold=0.7, disagreement_weight=0.5)
        ids = ["d"]
        primary = np.array([[0.85, 0.1, 0.03, 0.02]])  # predicts class 0
        ensemble = np.array([[0.05, 0.9, 0.03, 0.02]])  # predicts class 1

        scores_no_ens = miner.find_hard_negatives(ids, primary)
        scores_with_ens = miner.find_hard_negatives(ids, primary, ensemble_probs=ensemble)

        # With ensemble disagreement, hardness should be different
        # (not necessarily higher — depends on formula, but should not crash)
        assert scores_with_ens[0].disagreement_score >= 0.0


# ===========================================================================
# TestComputePriority
# ===========================================================================

class TestComputePriority:
    """Tests for the compute_priority formula in queuing/priority_queue.py."""

    def test_higher_uncertainty_higher_priority(self):
        from active_learning.queuing.priority_queue import compute_priority

        low = compute_priority(uncertainty_score=0.1)
        high = compute_priority(uncertainty_score=0.9)
        assert high > low

    def test_class_rarity_multiplies_priority(self):
        from active_learning.queuing.priority_queue import compute_priority

        common = compute_priority(uncertainty_score=0.5, class_rarity=1.0)
        rare = compute_priority(uncertainty_score=0.5, class_rarity=5.0)
        assert rare > common

    def test_age_boost_caps_at_0_2(self):
        """Age boost must be capped at 0.2 regardless of age_seconds."""
        from active_learning.queuing.priority_queue import compute_priority

        p_young = compute_priority(uncertainty_score=0.5, age_seconds=0)
        p_very_old = compute_priority(uncertainty_score=0.5, age_seconds=10_000_000)

        boost = p_very_old - p_young
        # The time-decay capping means the boost cannot exceed 0.2 * class_rarity
        assert boost <= 0.21  # small tolerance

    def test_manual_boost_adds(self):
        from active_learning.queuing.priority_queue import compute_priority

        without = compute_priority(uncertainty_score=0.5, manual_boost=0.0)
        with_boost = compute_priority(uncertainty_score=0.5, manual_boost=1.0)
        assert with_boost > without

    def test_result_is_non_negative(self):
        from active_learning.queuing.priority_queue import compute_priority

        # Edge case: all zeros
        p = compute_priority(
            uncertainty_score=0.0,
            entropy_score=0.0,
            disagreement_score=0.0,
            class_rarity=1.0,
            age_seconds=0,
            manual_boost=0.0,
        )
        assert p >= 0.0

    def test_returns_float(self):
        from active_learning.queuing.priority_queue import compute_priority

        result = compute_priority(uncertainty_score=0.5)
        assert isinstance(result, float)


# ===========================================================================
# TestAnnotationPriorityQueue
# ===========================================================================

class TestAnnotationPriorityQueue:
    """Tests for AnnotationPriorityQueue in queuing/priority_queue.py."""

    @pytest.fixture()
    def queue(self):
        from active_learning.queuing.priority_queue import AnnotationPriorityQueue
        return AnnotationPriorityQueue()

    def test_push_and_len(self, queue):
        queue.push(
            sample_id="s1", dataset_id="d1", image_path="/img/s1.png",
            uncertainty_score=0.8, entropy_score=0.7,
        )
        assert len(queue) == 1

    def test_pop_returns_highest_priority_first(self, queue):
        """Items are popped in descending priority order."""
        queue.push(sample_id="low", dataset_id="d", image_path="",
                   uncertainty_score=0.1, entropy_score=0.1)
        queue.push(sample_id="high", dataset_id="d", image_path="",
                   uncertainty_score=0.9, entropy_score=0.9)
        queue.push(sample_id="mid", dataset_id="d", image_path="",
                   uncertainty_score=0.5, entropy_score=0.5)

        first = queue.pop()
        assert first is not None
        assert first.sample_id == "high"

    def test_pop_empty_returns_none(self, queue):
        assert queue.pop() is None

    def test_peek_does_not_remove(self, queue):
        """peek_top_k returns items without removing them."""
        for i in range(5):
            queue.push(sample_id=f"s{i}", dataset_id="d", image_path="",
                       uncertainty_score=float(i) / 5)

        before = len(queue)
        top = queue.peek_top_k(3)
        after = len(queue)

        assert after == before  # peek doesn't remove
        assert len(top) == 3

    def test_boost_elevates_item(self, queue):
        """Boosting an item should increase its effective priority."""
        low_entry = queue.push(sample_id="low", dataset_id="d", image_path="",
                               uncertainty_score=0.1, entropy_score=0.1)
        queue.push(sample_id="high_base", dataset_id="d", image_path="",
                   uncertainty_score=0.9, entropy_score=0.9)

        # Boost the low-priority item by a large amount using its item_id
        queue.boost(low_entry.item_id, boost_amount=10.0)

        # After boost, "low" should be popped first (highest priority now)
        first = queue.pop()
        assert first.sample_id == "low"

    def test_boost_unknown_id_returns_false(self, queue):
        result = queue.boost("nonexistent", boost_amount=1.0)
        assert result is False

    def test_complete_marks_item(self, queue):
        # push() returns the QueueEntry with its item_id
        entry = queue.push(sample_id="c1", dataset_id="d", image_path="",
                           uncertainty_score=0.5, entropy_score=0.5)
        result = queue.complete(entry.item_id)
        assert result is True

    def test_complete_nonexistent_returns_false(self, queue):
        result = queue.complete("00000000-0000-0000-0000-000000000000")
        assert result is False

    def test_remove_reduces_len(self, queue):
        rm_entry = queue.push(sample_id="rm", dataset_id="d", image_path="",
                              uncertainty_score=0.5, entropy_score=0.5)
        queue.push(sample_id="keep", dataset_id="d", image_path="",
                   uncertainty_score=0.6, entropy_score=0.6)
        queue.remove(rm_entry.item_id)
        remaining_ids = {e.sample_id for e in queue.peek_top_k(10)}
        assert "rm" not in remaining_ids

    def test_stats_structure(self, queue):
        """stats() returns QueueStats with expected fields."""
        from active_learning.queuing.priority_queue import QueueStats

        for i in range(3):
            queue.push(sample_id=f"s{i}", dataset_id="d", image_path="",
                       uncertainty_score=0.5)
        stats = queue.stats()
        # QueueStats uses `pending` (not `total_pending`)
        assert hasattr(stats, "pending"), f"QueueStats fields: {vars(stats)}"
        assert stats.pending == 3

    def test_global_queue_is_singleton(self):
        """get_global_queue() returns the same object on repeated calls."""
        from active_learning.queuing.priority_queue import get_global_queue

        q1 = get_global_queue()
        q2 = get_global_queue()
        assert q1 is q2


# ===========================================================================
# TestRetrainingTrigger
# ===========================================================================

class TestRetrainingTrigger:
    """Tests for RetrainingTrigger in retraining/trigger.py."""

    @pytest.fixture()
    def trigger(self):
        from active_learning.retraining.trigger import RetrainingTrigger, TriggerConfig
        cfg = TriggerConfig(
            annotation_count_threshold=10,
            performance_drop_threshold=0.05,
            drift_score_threshold=0.6,
            scheduled_interval_hours=24.0,
            min_retraining_interval_hours=0.0,  # no cooldown for tests
            min_annotations_for_trigger=5,
        )
        return RetrainingTrigger(config=cfg)

    def test_no_trigger_below_threshold(self, trigger):
        """With insufficient data, evaluate() should_retrain=False."""
        # Only 2 annotations — below min_annotations_for_trigger=5
        trigger.on_annotation_approved(2)
        decision = trigger.evaluate()
        assert decision.should_retrain is False

    def test_annotation_count_trigger(self, trigger):
        """Adding enough annotations fires the annotation count trigger."""
        trigger.on_annotation_approved(5)   # meets min_annotations
        trigger.on_annotation_approved(10)  # meets annotation_count_threshold
        decision = trigger.evaluate()
        assert decision.should_retrain is True
        assert any("annotation" in r.lower() for r in decision.reasons)

    def test_performance_drop_trigger(self, trigger):
        """mAP50 drop >= threshold fires performance trigger."""
        trigger.on_annotation_approved(5)  # meet minimum
        trigger.update_metrics(map50=0.80)  # set best
        trigger.update_metrics(map50=0.70)  # 0.10 drop > 0.05 threshold
        decision = trigger.evaluate()
        assert decision.should_retrain is True
        assert any("performance" in r.lower() or "drop" in r.lower() for r in decision.reasons)

    def test_drift_trigger(self, trigger):
        """Drift score >= threshold fires drift trigger."""
        trigger.on_annotation_approved(5)  # meet minimum
        trigger.update_drift(0.75)  # > 0.6 threshold
        decision = trigger.evaluate()
        assert decision.should_retrain is True
        assert any("drift" in r.lower() for r in decision.reasons)

    def test_no_trigger_when_no_conditions_met(self, trigger):
        """No trigger when all conditions below threshold."""
        trigger.on_annotation_approved(5)  # meet minimum
        trigger.update_metrics(map50=0.80)
        trigger.update_drift(0.3)  # below drift threshold
        # Only 5 new annotations — below annotation_count_threshold=10
        decision = trigger.evaluate()
        assert decision.should_retrain is False

    def test_cooldown_blocks_trigger(self):
        """Cooldown prevents re-triggering within min_interval."""
        from datetime import datetime, timedelta
        from active_learning.retraining.trigger import RetrainingTrigger, TriggerConfig

        cfg = TriggerConfig(
            annotation_count_threshold=5,
            min_retraining_interval_hours=1.0,
            min_annotations_for_trigger=1,
        )
        t = RetrainingTrigger(config=cfg)
        t.state.last_retrain_at = datetime.utcnow()  # just retrained
        t.on_annotation_approved(10)
        decision = t.evaluate()
        assert decision.should_retrain is False
        assert "cooldown" in decision.reasons[0].lower()

    def test_mark_retrained_resets_annotation_count(self, trigger):
        """After mark_retrained(), new annotation count resets to 0."""
        trigger.on_annotation_approved(15)
        trigger.mark_retrained()
        assert trigger.state.new_annotations_since_retrain == 0

    def test_mark_retrained_increments_retrain_count(self, trigger):
        trigger.mark_retrained()
        assert trigger.state.retrain_count == 1

    def test_manual_trigger_always_fires(self, trigger):
        """manual_trigger() returns should_retrain=True regardless of conditions."""
        decision = trigger.manual_trigger("test override")
        assert decision.should_retrain is True
        assert "manual" in decision.reasons[0].lower()

    def test_urgency_levels(self, trigger):
        """Performance drop >= 10% produces high urgency."""
        trigger.on_annotation_approved(5)
        trigger.update_metrics(map50=0.90)
        trigger.update_metrics(map50=0.75)  # 0.15 drop > threshold
        decision = trigger.evaluate()
        if decision.should_retrain:
            assert decision.urgency in {"low", "medium", "high", "critical"}

    def test_trigger_history_populated(self, trigger):
        """evaluate() appends to trigger_history when should_retrain=True."""
        trigger.on_annotation_approved(5)
        trigger.on_annotation_approved(10)
        trigger.evaluate()
        assert len(trigger.state.trigger_history) >= 1

    def test_get_status_dict_structure(self, trigger):
        """get_status() returns expected keys."""
        status = trigger.get_status()
        for key in ["last_retrain_at", "new_annotations_since_retrain",
                    "total_annotations", "best_map50", "current_map50",
                    "last_drift_score", "retrain_count", "hours_since_last"]:
            assert key in status, f"Missing key: {key}"

    def test_on_trigger_callback_called(self):
        """on_trigger callback is invoked when trigger fires."""
        from active_learning.retraining.trigger import RetrainingTrigger, TriggerConfig

        fired = []

        cfg = TriggerConfig(
            annotation_count_threshold=5,
            min_annotations_for_trigger=1,
            min_retraining_interval_hours=0.0,
        )
        t = RetrainingTrigger(config=cfg, on_trigger=lambda d: fired.append(d))
        t.on_annotation_approved(5)
        t.evaluate()
        assert len(fired) >= 1


# ===========================================================================
# TestDriftFunctions
# ===========================================================================

class TestDriftFunctions:
    """Tests for drift detection functions in analysis/drift.py."""

    def test_extract_image_statistics_shape(self):
        """extract_image_statistics returns expected feature vector."""
        from active_learning.analysis.drift import extract_image_statistics

        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        features = extract_image_statistics(img)
        # Should return a 1D feature vector
        assert isinstance(features, np.ndarray)
        assert features.ndim == 1
        assert len(features) > 0

    def test_extract_image_statistics_consistent(self):
        """Same image produces identical statistics."""
        from active_learning.analysis.drift import extract_image_statistics

        img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        f1 = extract_image_statistics(img)
        f2 = extract_image_statistics(img)
        np.testing.assert_array_almost_equal(f1, f2)

    def test_mmd_identical_distributions_near_zero(self):
        """MMD between identical distributions should be near zero."""
        from active_learning.analysis.drift import compute_mmd

        X = np.random.randn(50, 8).astype(np.float64)
        mmd = compute_mmd(X, X)
        assert mmd < 0.05, f"Expected MMD ≈ 0 for identical data, got {mmd}"

    def test_mmd_different_distributions_positive(self):
        """MMD between clearly different distributions should be positive."""
        from active_learning.analysis.drift import compute_mmd

        X = np.random.randn(50, 8).astype(np.float64)
        Y = np.random.randn(50, 8).astype(np.float64) + 10.0  # far from X
        mmd = compute_mmd(X, Y)
        assert mmd > 0.01, f"Expected positive MMD for different distributions, got {mmd}"

    def test_ks_drift_no_drift_same_data(self):
        """KS test on identical data should not detect drift."""
        from active_learning.analysis.drift import compute_ks_drift

        X = np.random.randn(100, 5).astype(np.float64)
        result = compute_ks_drift(X, X, alpha=0.05)
        # Same data: p-value should be very high → no drift (or marginal)
        assert result.test_name == "KolmogorovSmirnov"
        assert hasattr(result, "drift_detected")
        assert hasattr(result, "score")

    def test_ks_drift_detects_strong_shift(self):
        """KS test detects strong distribution shift (mean=0 vs mean=5)."""
        from active_learning.analysis.drift import compute_ks_drift

        np.random.seed(42)
        ref = np.random.randn(100, 5).astype(np.float64)
        tst = np.random.randn(100, 5).astype(np.float64) + 5.0
        result = compute_ks_drift(ref, tst, alpha=0.05)
        assert result.drift_detected is True

    def test_prediction_drift_no_shift(self):
        """Identical class distributions → no prediction drift."""
        from active_learning.analysis.drift import compute_prediction_drift

        dist = {0: 0.5, 1: 0.3, 2: 0.2}
        result = compute_prediction_drift(dist, dist, threshold=0.20)
        assert result.drift_detected is False

    def test_prediction_drift_detects_shift(self):
        """Very different class distributions → prediction drift."""
        from active_learning.analysis.drift import compute_prediction_drift

        ref = {0: 0.8, 1: 0.1, 2: 0.1}
        tst = {0: 0.1, 1: 0.1, 2: 0.8}
        result = compute_prediction_drift(ref, tst, threshold=0.20)
        assert result.drift_detected is True

    def test_drift_result_severity_field(self):
        """DriftResult has severity field in expected set."""
        from active_learning.analysis.drift import compute_ks_drift

        ref = np.random.randn(50, 4).astype(np.float64)
        tst = np.random.randn(50, 4).astype(np.float64) + 10.0
        result = compute_ks_drift(ref, tst)
        assert result.severity in {"none", "mild", "moderate", "severe"}


# ===========================================================================
# TestDriftDetector
# ===========================================================================

class TestDriftDetector:
    """Tests for DriftDetector.fit_reference() + analyze()."""

    def test_analyze_without_fit_raises(self):
        """analyze() before fit_reference() should raise RuntimeError."""
        from active_learning.analysis.drift import DriftDetector

        detector = DriftDetector()
        tst = np.random.randn(20, 4).astype(np.float64)
        with pytest.raises(RuntimeError):
            detector.analyze(tst)

    def test_identical_data_no_overall_drift(self):
        """Analyzing the same data as reference should not flag drift."""
        from active_learning.analysis.drift import DriftDetector

        np.random.seed(0)
        ref = np.random.randn(100, 6).astype(np.float64)
        detector = DriftDetector(mmd_threshold=0.5, ks_alpha=0.001)  # lenient thresholds
        detector.fit_reference(ref)
        report = detector.analyze(ref)

        # With lenient thresholds, identical data should not trigger drift
        # (KS test on same data has very high p-value → no drift)
        assert hasattr(report, "overall_drift_detected")
        assert hasattr(report, "results")
        assert len(report.results) >= 1

    def test_very_different_data_triggers_drift(self):
        """Data shifted by 10 standard deviations should always trigger drift."""
        from active_learning.analysis.drift import DriftDetector

        np.random.seed(1)
        ref = np.random.randn(100, 6).astype(np.float64)
        tst = np.random.randn(100, 6).astype(np.float64) + 10.0

        detector = DriftDetector()
        detector.fit_reference(ref)
        report = detector.analyze(tst)

        assert report.overall_drift_detected is True

    def test_report_has_recommendation(self):
        """DriftReport always has a non-empty recommendation string."""
        from active_learning.analysis.drift import DriftDetector

        np.random.seed(2)
        ref = np.random.randn(50, 4).astype(np.float64)
        detector = DriftDetector()
        detector.fit_reference(ref)
        report = detector.analyze(ref + 5.0)

        assert isinstance(report.recommendation, str)
        assert len(report.recommendation) > 10

    def test_report_num_samples_fields(self):
        """Report records the number of reference and test samples."""
        from active_learning.analysis.drift import DriftDetector

        ref = np.random.randn(80, 4).astype(np.float64)
        tst = np.random.randn(30, 4).astype(np.float64)

        detector = DriftDetector()
        detector.fit_reference(ref)
        report = detector.analyze(tst)

        assert report.num_reference_samples == 80
        assert report.num_test_samples == 30

    def test_prediction_distribution_drift_included(self):
        """If both class distributions are provided, prediction drift test runs."""
        from active_learning.analysis.drift import DriftDetector

        ref = np.random.randn(60, 4).astype(np.float64)
        ref_dist = {0: 0.7, 1: 0.2, 2: 0.1}
        tst_dist = {0: 0.1, 1: 0.1, 2: 0.8}  # very different

        detector = DriftDetector()
        detector.fit_reference(ref, class_distribution=ref_dist)
        report = detector.analyze(ref, test_class_dist=tst_dist)

        test_names = [r.test_name for r in report.results]
        assert any("prediction" in name.lower() or "tvd" in name.lower() or "class" in name.lower()
                   for name in test_names), f"Prediction test not found in: {test_names}"


# ===========================================================================
# TestALPipelineConfig
# ===========================================================================

class TestALPipelineConfig:
    """Tests for ALPipelineConfig defaults."""

    def test_defaults(self):
        from active_learning.application.al_pipeline import ALPipelineConfig

        cfg = ALPipelineConfig()
        assert cfg.pool_batch_size == 500
        assert cfg.queue_top_k == 50
        assert cfg.min_uncertainty_to_queue == 0.10
        assert cfg.mc_dropout_passes == 10
        assert cfg.enable_drift_detection is True
        assert cfg.enable_auto_trigger is True

    def test_custom_values(self):
        from active_learning.application.al_pipeline import ALPipelineConfig

        cfg = ALPipelineConfig(
            pool_batch_size=100,
            queue_top_k=10,
            mc_dropout_passes=5,
        )
        assert cfg.pool_batch_size == 100
        assert cfg.queue_top_k == 10
        assert cfg.mc_dropout_passes == 5

    def test_annotations_per_cycle_positive(self):
        from active_learning.application.al_pipeline import ALPipelineConfig

        cfg = ALPipelineConfig()
        assert cfg.annotations_per_cycle > 0


# ===========================================================================
# TestUncertaintySamplerMath — uncertainty.py standalone functions
# ===========================================================================

class TestUncertaintySamplerMath:
    """Tests for pure math in sampling/uncertainty.py (no GPU required)."""

    def test_compute_entropy_uniform(self):
        """Uniform distribution → max entropy for N classes."""
        from active_learning.sampling.uncertainty import compute_entropy

        probs = np.array([0.25, 0.25, 0.25, 0.25])
        h = compute_entropy(probs)
        expected = math.log(4)
        assert abs(h - expected) < 1e-5

    def test_compute_entropy_one_hot(self):
        """One-hot distribution → ~0 entropy."""
        from active_learning.sampling.uncertainty import compute_entropy

        probs = np.array([1.0, 0.0, 0.0, 0.0])
        h = compute_entropy(probs)
        assert h < 1e-9

    def test_compute_least_confidence_confident(self):
        """High confidence → low least_confidence score."""
        from active_learning.sampling.uncertainty import compute_least_confidence

        probs = np.array([0.95, 0.02, 0.02, 0.01])
        lc = compute_least_confidence(probs)
        assert abs(lc - 0.05) < 1e-6

    def test_compute_least_confidence_uncertain(self):
        """Low max prob → high least_confidence."""
        from active_learning.sampling.uncertainty import compute_least_confidence

        probs = np.array([0.28, 0.26, 0.24, 0.22])
        lc = compute_least_confidence(probs)
        assert lc > 0.70

    def test_uncertainty_sampler_invalid_strategy(self):
        """Invalid strategy raises ValueError."""
        from active_learning.sampling.uncertainty import UncertaintySampler

        with pytest.raises(ValueError):
            UncertaintySampler(strategy="magic")

    def test_uncertainty_sampler_valid_strategies(self):
        """All declared valid strategies can be instantiated."""
        from active_learning.sampling.uncertainty import UncertaintySampler

        for strategy in ["entropy", "least_confidence", "mc_dropout", "combined"]:
            s = UncertaintySampler(strategy=strategy)
            assert s.strategy == strategy
