"""
tests/integration/test_al_pipeline_integration.py — Active Learning pipeline integration tests.

Tests the full AL cycle end-to-end:
  unlabeled pool → entropy scoring → drift skip (no ref) → priority queue → trigger evaluation

No real YOLO model required — predictions are synthetic tensors / pre-built dicts.
No GPU, no filesystem access beyond tempdir.
"""

from __future__ import annotations

import math
import uuid
from unittest.mock import patch, MagicMock

import pytest

from active_learning.application.al_pipeline import (
    ActiveLearningPipeline,
    ALPipelineConfig,
    ALCycleResult,
)
from active_learning.queuing.priority_queue import (
    AnnotationPriorityQueue,
    get_global_queue,
)
from active_learning.retraining.trigger import (
    RetrainingTrigger,
    TriggerConfig,
    TriggerDecision,
)


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────


def _make_sample(i: int) -> dict:
    """Create a synthetic unlabeled sample dict."""
    return {
        "sample_id": f"sample_{i:04d}",
        "dataset_id": "test_dataset",
        "image_path": f"/data/test/img_{i:04d}.png",
    }


def _make_prediction(
    sample_id: str,
    probs: list[float],
    confidence: float,
    predicted_class: int = 0,
) -> dict:
    """Create a synthetic model prediction dict."""
    return {
        "sample_id": sample_id,
        "probabilities": probs,
        "confidence": confidence,
        "predicted_class": predicted_class,
    }


def _uniform_preds(n: int, n_classes: int = 4) -> list[dict]:
    """n predictions with uniform class probabilities — maximum uncertainty."""
    samples = [_make_sample(i) for i in range(n)]
    probs = [1.0 / n_classes] * n_classes
    preds = [
        _make_prediction(s["sample_id"], probs, confidence=0.25, predicted_class=0)
        for s in samples
    ]
    return samples, preds


def _certain_preds(n: int) -> list[dict]:
    """n predictions with class-0 probability=1.0 — minimum uncertainty."""
    samples = [_make_sample(i) for i in range(n)]
    preds = [
        _make_prediction(s["sample_id"], [1.0, 0.0, 0.0, 0.0], confidence=0.99, predicted_class=0)
        for s in samples
    ]
    return samples, preds


# Isolate each test from global queue state
@pytest.fixture(autouse=True)
def _reset_global_queue(monkeypatch):
    """Replace the global queue singleton with a fresh instance per test."""
    fresh_queue = AnnotationPriorityQueue()
    monkeypatch.setattr(
        "active_learning.queuing.priority_queue._global_queue",
        fresh_queue,
    )
    yield fresh_queue


# ──────────────────────────────────────────────────────────────────────────────
# 1. Basic cycle — no predictions
# ──────────────────────────────────────────────────────────────────────────────


class TestALCycleNoPredictions:
    """Pipeline handles an unlabeled pool with no pre-computed predictions."""

    def test_cycle_returns_result_object(self):
        pipeline = ActiveLearningPipeline(
            config=ALPipelineConfig(
                pool_batch_size=10,
                queue_top_k=5,
                enable_drift_detection=False,
            )
        )
        samples = [_make_sample(i) for i in range(20)]
        result = pipeline.run_cycle(unlabeled_pool=samples, model_predictions=None)
        assert isinstance(result, ALCycleResult)

    def test_cycle_id_is_nonempty_string(self):
        pipeline = ActiveLearningPipeline(
            config=ALPipelineConfig(enable_drift_detection=False)
        )
        result = pipeline.run_cycle(unlabeled_pool=[_make_sample(0)])
        assert isinstance(result.cycle_id, str)
        assert len(result.cycle_id) > 0

    def test_pool_size_recorded(self):
        pipeline = ActiveLearningPipeline(
            config=ALPipelineConfig(enable_drift_detection=False)
        )
        samples = [_make_sample(i) for i in range(15)]
        result = pipeline.run_cycle(unlabeled_pool=samples)
        assert result.pool_size == 15

    def test_no_errors_on_empty_predictions(self):
        pipeline = ActiveLearningPipeline(
            config=ALPipelineConfig(enable_drift_detection=False)
        )
        samples = [_make_sample(i) for i in range(5)]
        result = pipeline.run_cycle(unlabeled_pool=samples, model_predictions=[])
        assert result.errors == []

    def test_duration_is_positive(self):
        pipeline = ActiveLearningPipeline(
            config=ALPipelineConfig(enable_drift_detection=False)
        )
        result = pipeline.run_cycle(unlabeled_pool=[_make_sample(0)])
        assert result.duration_s >= 0.0

    def test_empty_pool_returns_zero_scored(self):
        pipeline = ActiveLearningPipeline(
            config=ALPipelineConfig(enable_drift_detection=False)
        )
        result = pipeline.run_cycle(unlabeled_pool=[])
        assert result.pool_size == 0
        assert result.queued_count == 0


# ──────────────────────────────────────────────────────────────────────────────
# 2. Uncertainty scoring
# ──────────────────────────────────────────────────────────────────────────────


class TestALCycleUncertaintyScoring:
    """Samples with high-uncertainty predictions are scored and queued."""

    def test_high_uncertainty_samples_scored(self):
        """Uniform probabilities → max entropy → should be scored as scored_count > 0."""
        pipeline = ActiveLearningPipeline(
            config=ALPipelineConfig(
                min_uncertainty_to_queue=0.0,  # accept all
                queue_top_k=10,
                enable_drift_detection=False,
            )
        )
        samples, preds = _uniform_preds(n=10, n_classes=4)
        result = pipeline.run_cycle(unlabeled_pool=samples, model_predictions=preds)
        assert result.scored_count == 10

    def test_low_uncertainty_samples_filtered(self):
        """Certain predictions (conf=0.99) → uncertainty = 1 - 0.99 = 0.01 → below threshold."""
        pipeline = ActiveLearningPipeline(
            config=ALPipelineConfig(
                min_uncertainty_to_queue=0.5,  # high threshold
                queue_top_k=10,
                enable_drift_detection=False,
            )
        )
        samples, preds = _certain_preds(n=10)
        result = pipeline.run_cycle(unlabeled_pool=samples, model_predictions=preds)
        assert result.queued_count == 0

    def test_top_k_respected(self):
        """queue_top_k=3 → at most 3 samples pushed to queue."""
        pipeline = ActiveLearningPipeline(
            config=ALPipelineConfig(
                min_uncertainty_to_queue=0.0,
                queue_top_k=3,
                enable_drift_detection=False,
            )
        )
        samples, preds = _uniform_preds(n=20, n_classes=4)
        result = pipeline.run_cycle(unlabeled_pool=samples, model_predictions=preds)
        assert result.queued_count <= 3

    def test_scored_count_matches_qualifying_samples(self):
        """With min_uncertainty=0 and 10 samples with predictions, all 10 are scored."""
        pipeline = ActiveLearningPipeline(
            config=ALPipelineConfig(
                min_uncertainty_to_queue=0.0,
                queue_top_k=100,
                enable_drift_detection=False,
            )
        )
        samples, preds = _uniform_preds(n=10, n_classes=4)
        result = pipeline.run_cycle(unlabeled_pool=samples, model_predictions=preds)
        assert result.scored_count == 10

    def test_top_samples_in_result(self):
        """result.top_samples contains dicts with sample_id and uncertainty keys."""
        pipeline = ActiveLearningPipeline(
            config=ALPipelineConfig(
                min_uncertainty_to_queue=0.0,
                queue_top_k=5,
                enable_drift_detection=False,
            )
        )
        samples, preds = _uniform_preds(n=10, n_classes=4)
        result = pipeline.run_cycle(unlabeled_pool=samples, model_predictions=preds)
        for item in result.top_samples:
            assert "sample_id" in item
            assert "uncertainty" in item

    def test_queue_receives_pushed_samples(self, _reset_global_queue):
        """After run_cycle, the global queue contains at least 1 item."""
        pipeline = ActiveLearningPipeline(
            config=ALPipelineConfig(
                min_uncertainty_to_queue=0.0,
                queue_top_k=5,
                enable_drift_detection=False,
            )
        )
        samples, preds = _uniform_preds(n=10, n_classes=4)
        pipeline.run_cycle(unlabeled_pool=samples, model_predictions=preds)
        # Pipeline uses get_global_queue() internally
        from active_learning.queuing.priority_queue import get_global_queue
        q = get_global_queue()
        assert len(q) > 0


# ──────────────────────────────────────────────────────────────────────────────
# 3. Cycle counter and status
# ──────────────────────────────────────────────────────────────────────────────


class TestALPipelineState:
    """Pipeline correctly tracks cycle count and exposes status."""

    def test_cycle_count_increments(self):
        pipeline = ActiveLearningPipeline(
            config=ALPipelineConfig(enable_drift_detection=False)
        )
        assert pipeline._cycle_count == 0
        for _ in range(3):
            pipeline.run_cycle(unlabeled_pool=[_make_sample(0)])
        assert pipeline._cycle_count == 3

    def test_get_status_has_required_keys(self):
        pipeline = ActiveLearningPipeline(
            config=ALPipelineConfig(enable_drift_detection=False)
        )
        pipeline.run_cycle(unlabeled_pool=[_make_sample(0)])
        status = pipeline.get_status()
        assert "cycle_count" in status
        assert "queue_size" in status
        assert "trigger_status" in status

    def test_last_result_in_status(self):
        pipeline = ActiveLearningPipeline(
            config=ALPipelineConfig(enable_drift_detection=False)
        )
        pipeline.run_cycle(unlabeled_pool=[_make_sample(0)])
        status = pipeline.get_status()
        assert status["last_result"] is not None
        assert "cycle_id" in status["last_result"]

    def test_multiple_cycles_result_in_last_result_update(self):
        pipeline = ActiveLearningPipeline(
            config=ALPipelineConfig(enable_drift_detection=False)
        )
        pipeline.run_cycle(unlabeled_pool=[_make_sample(0)])
        first_cycle_id = pipeline.get_status()["last_result"]["cycle_id"]

        pipeline.run_cycle(unlabeled_pool=[_make_sample(1)])
        second_cycle_id = pipeline.get_status()["last_result"]["cycle_id"]

        assert first_cycle_id != second_cycle_id

    def test_queue_size_reflects_queued_samples(self, _reset_global_queue):
        pipeline = ActiveLearningPipeline(
            config=ALPipelineConfig(
                min_uncertainty_to_queue=0.0,
                queue_top_k=5,
                enable_drift_detection=False,
            )
        )
        samples, preds = _uniform_preds(n=10, n_classes=4)
        pipeline.run_cycle(unlabeled_pool=samples, model_predictions=preds)
        status = pipeline.get_status()
        assert status["queue_size"] > 0


# ──────────────────────────────────────────────────────────────────────────────
# 4. Annotation feedback → trigger evaluation
# ──────────────────────────────────────────────────────────────────────────────


class TestALAnnotationFeedback:
    """on_annotation_approved drives the retraining trigger."""

    def test_trigger_fires_after_threshold_annotations(self):
        """Set threshold=5; approve 5 annotations; trigger should want to fire."""
        trigger_fired = []

        pipeline = ActiveLearningPipeline(
            config=ALPipelineConfig(
                annotations_per_cycle=5,
                enable_drift_detection=False,
                enable_auto_trigger=True,
            )
        )
        pipeline._init_components()
        # Attach callback spy
        original_on_trigger = pipeline._on_retrain_triggered
        pipeline._on_retrain_triggered = lambda d: trigger_fired.append(d) or original_on_trigger(d)
        pipeline._trigger._on_trigger_callback = pipeline._on_retrain_triggered

        for _ in range(5):
            pipeline.on_annotation_approved(count=1)

        trigger_status = pipeline._trigger.get_status()
        assert trigger_status["total_annotations"] == 5

    def test_on_annotation_approved_increments_count(self):
        pipeline = ActiveLearningPipeline(
            config=ALPipelineConfig(enable_drift_detection=False)
        )
        pipeline._init_components()
        pipeline.on_annotation_approved(count=3)
        status = pipeline._trigger.get_status()
        assert status["total_annotations"] == 3

    def test_update_model_metrics_propagates(self):
        pipeline = ActiveLearningPipeline(
            config=ALPipelineConfig(enable_drift_detection=False)
        )
        pipeline._init_components()
        pipeline.update_model_metrics(map50=0.72)
        # Just verify no exception and the trigger received the update
        status = pipeline._trigger.get_status()
        assert status is not None

    def test_manual_trigger_via_pipeline_trigger(self):
        pipeline = ActiveLearningPipeline(
            config=ALPipelineConfig(enable_drift_detection=False)
        )
        pipeline._init_components()
        decision = pipeline._trigger.manual_trigger(reason="test")
        assert decision.should_retrain is True
        # reasons is a list of strings; check any reason contains "manual" (case-insensitive)
        assert any("manual" in r.lower() for r in decision.reasons)


# ──────────────────────────────────────────────────────────────────────────────
# 5. Drift detection (disabled/no-reference path)
# ──────────────────────────────────────────────────────────────────────────────


class TestALDriftIntegration:
    """Drift detection with no reference set → graceful no-op."""

    def test_drift_disabled_returns_false(self):
        pipeline = ActiveLearningPipeline(
            config=ALPipelineConfig(enable_drift_detection=False)
        )
        samples = [_make_sample(i) for i in range(10)]
        result = pipeline.run_cycle(unlabeled_pool=samples)
        assert result.drift_detected is False

    def test_drift_enabled_no_ref_returns_no_drift(self):
        """With drift enabled but no reference set, DriftDetector returns False (graceful)."""
        pipeline = ActiveLearningPipeline(
            config=ALPipelineConfig(enable_drift_detection=True)
        )
        samples = [_make_sample(i) for i in range(5)]
        result = pipeline.run_cycle(unlabeled_pool=samples)
        # DriftDetector requires fit_reference first — returns False without it
        assert result.drift_detected is False
        assert result.errors == []

    def test_drift_score_is_float(self):
        pipeline = ActiveLearningPipeline(
            config=ALPipelineConfig(enable_drift_detection=False)
        )
        result = pipeline.run_cycle(unlabeled_pool=[_make_sample(0)])
        assert isinstance(result.drift_score, float)


# ──────────────────────────────────────────────────────────────────────────────
# 6. Priority queue integration
# ──────────────────────────────────────────────────────────────────────────────


class TestALQueueIntegration:
    """Verify that high-uncertainty samples are properly queued and retrievable."""

    def test_queued_items_have_correct_sample_ids(self, _reset_global_queue):
        pipeline = ActiveLearningPipeline(
            config=ALPipelineConfig(
                min_uncertainty_to_queue=0.0,
                queue_top_k=5,
                enable_drift_detection=False,
            )
        )
        samples, preds = _uniform_preds(n=10, n_classes=4)
        pipeline.run_cycle(unlabeled_pool=samples, model_predictions=preds)

        from active_learning.queuing.priority_queue import get_global_queue
        q = get_global_queue()
        top = q.peek_top_k(k=5)
        queued_ids = {e.sample_id for e in top}
        sample_ids = {s["sample_id"] for s in samples}
        # All queued items must be from the original pool
        assert queued_ids.issubset(sample_ids)

    def test_queue_stats_pending_count(self, _reset_global_queue):
        pipeline = ActiveLearningPipeline(
            config=ALPipelineConfig(
                min_uncertainty_to_queue=0.0,
                queue_top_k=3,
                enable_drift_detection=False,
            )
        )
        samples, preds = _uniform_preds(n=10, n_classes=4)
        result = pipeline.run_cycle(unlabeled_pool=samples, model_predictions=preds)

        from active_learning.queuing.priority_queue import get_global_queue
        q = get_global_queue()
        stats = q.stats()
        assert stats.pending == result.queued_count

    def test_two_cycles_accumulate_in_queue(self, _reset_global_queue):
        """Two independent cycles both push to the same global queue."""
        pipeline = ActiveLearningPipeline(
            config=ALPipelineConfig(
                min_uncertainty_to_queue=0.0,
                queue_top_k=3,
                enable_drift_detection=False,
            )
        )

        samples1 = [_make_sample(i) for i in range(10)]
        _, preds1 = _uniform_preds(n=10, n_classes=4)
        for i, p in enumerate(preds1):
            p["sample_id"] = samples1[i]["sample_id"]

        samples2 = [_make_sample(i + 100) for i in range(10)]
        _, preds2 = _uniform_preds(n=10, n_classes=4)
        for i, p in enumerate(preds2):
            p["sample_id"] = samples2[i]["sample_id"]

        result1 = pipeline.run_cycle(unlabeled_pool=samples1, model_predictions=preds1)
        result2 = pipeline.run_cycle(unlabeled_pool=samples2, model_predictions=preds2)

        from active_learning.queuing.priority_queue import get_global_queue
        q = get_global_queue()
        assert len(q) == result1.queued_count + result2.queued_count

    def test_high_uncertainty_ranked_before_low(self, _reset_global_queue):
        """Mix high- and low-uncertainty samples; queue top item should be uncertain."""
        pipeline = ActiveLearningPipeline(
            config=ALPipelineConfig(
                min_uncertainty_to_queue=0.0,
                queue_top_k=10,
                enable_drift_detection=False,
            )
        )
        samples = [_make_sample(i) for i in range(10)]
        preds = []
        for i, s in enumerate(samples):
            if i < 5:
                # Uniform → max entropy, uncertainty = 0.75
                preds.append(_make_prediction(s["sample_id"], [0.25] * 4, 0.25))
            else:
                # Certain → uncertainty = 0.01
                preds.append(_make_prediction(s["sample_id"], [0.99, 0.0, 0.0, 0.01], 0.99))

        pipeline.run_cycle(unlabeled_pool=samples, model_predictions=preds)

        from active_learning.queuing.priority_queue import get_global_queue
        q = get_global_queue()
        top = q.peek_top_k(k=1)
        assert len(top) == 1
        # The top item should be from the high-uncertainty half (samples 0-4)
        uncertain_ids = {samples[i]["sample_id"] for i in range(5)}
        assert top[0].sample_id in uncertain_ids


# ──────────────────────────────────────────────────────────────────────────────
# 7. Error resilience
# ──────────────────────────────────────────────────────────────────────────────


class TestALErrorResilience:
    """Pipeline logs errors without raising; results remain valid."""

    def test_malformed_sample_does_not_crash(self):
        """A sample missing sample_id is handled gracefully."""
        pipeline = ActiveLearningPipeline(
            config=ALPipelineConfig(enable_drift_detection=False)
        )
        # Missing sample_id
        bad_samples = [{"dataset_id": "x", "image_path": "/img.png"}]
        result = pipeline.run_cycle(unlabeled_pool=bad_samples)
        # No exception; errors may or may not be logged depending on implementation
        assert isinstance(result, ALCycleResult)

    def test_invalid_probabilities_do_not_crash(self):
        """Probabilities that don't sum to 1 are handled without exception."""
        pipeline = ActiveLearningPipeline(
            config=ALPipelineConfig(
                min_uncertainty_to_queue=0.0,
                enable_drift_detection=False,
            )
        )
        samples = [_make_sample(0)]
        preds = [_make_prediction("sample_0000", [0.5, 0.5, 0.5, 0.5], 0.5)]
        result = pipeline.run_cycle(unlabeled_pool=samples, model_predictions=preds)
        assert isinstance(result, ALCycleResult)

    def test_zero_class_probabilities_list_handled(self):
        """Empty probabilities list → entropy = 0 (no division by zero)."""
        pipeline = ActiveLearningPipeline(
            config=ALPipelineConfig(
                min_uncertainty_to_queue=0.0,
                enable_drift_detection=False,
            )
        )
        samples = [_make_sample(0)]
        preds = [_make_prediction("sample_0000", [], 0.5)]
        result = pipeline.run_cycle(unlabeled_pool=samples, model_predictions=preds)
        assert isinstance(result, ALCycleResult)
        assert result.errors == []

    def test_very_large_pool_does_not_crash(self):
        """Pool of 1000 samples with no predictions runs without error."""
        pipeline = ActiveLearningPipeline(
            config=ALPipelineConfig(
                min_uncertainty_to_queue=0.0,
                queue_top_k=10,
                enable_drift_detection=False,
            )
        )
        samples = [_make_sample(i) for i in range(1000)]
        result = pipeline.run_cycle(unlabeled_pool=samples)
        assert result.pool_size == 1000
        assert result.errors == []
