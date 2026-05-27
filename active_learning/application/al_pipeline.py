"""
active_learning/application/al_pipeline.py — Full active learning pipeline.

Orchestrates:
  1. Run inference on unlabeled pool with uncertainty estimation
  2. Score samples by uncertainty (entropy) + ensemble disagreement
  3. Detect dataset drift in new samples
  4. Enqueue high-value samples to annotation priority queue
  5. Monitor annotation progress and trigger retraining when thresholds met

This is the central coordinator — it calls into the individual sampling,
drift, queue, and trigger modules.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("trichome.active_learning")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ALPipelineConfig:
    """Active learning pipeline configuration."""

    # How many unlabeled samples to score per cycle
    pool_batch_size: int = 500

    # How many top samples to push to annotation queue per cycle
    queue_top_k: int = 50

    # Minimum uncertainty score to queue (filter trivially easy samples)
    min_uncertainty_to_queue: float = 0.10

    # Use ensemble disagreement in addition to entropy
    use_disagreement: bool = False  # requires multiple model checkpoints

    # Number of MC Dropout forward passes for uncertainty estimation
    mc_dropout_passes: int = 10

    # Drift detection: compare new batch against training reference
    enable_drift_detection: bool = True

    # Auto-trigger: check trigger conditions after each annotation batch
    enable_auto_trigger: bool = True

    # Annotation count to trigger one AL cycle
    annotations_per_cycle: int = 25


# ---------------------------------------------------------------------------
# Cycle result
# ---------------------------------------------------------------------------


@dataclass
class ALCycleResult:
    """Result of one active learning cycle."""

    cycle_id: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    pool_size: int = 0
    scored_count: int = 0
    queued_count: int = 0
    drift_detected: bool = False
    drift_score: float = 0.0
    trigger_decision: dict = field(default_factory=dict)
    top_samples: list[dict] = field(default_factory=list)
    duration_s: float = 0.0
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class ActiveLearningPipeline:
    """
    Full active learning pipeline for trichome detection.

    Instantiate once per application lifecycle. Call run_cycle() periodically
    or when triggered by new annotations.
    """

    def __init__(
        self,
        config: ALPipelineConfig | None = None,
        model_path: str | None = None,
    ) -> None:
        self.config = config or ALPipelineConfig()
        self.model_path = model_path
        self._cycle_count = 0
        self._results: list[ALCycleResult] = []

        # Lazy-initialized components
        self._entropy_sampler: Any = None
        self._drift_detector: Any = None
        self._trigger: Any = None
        self._priority_queue: Any = None

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _init_components(self) -> None:
        """Lazily initialize AL components on first use."""
        if self._entropy_sampler is None:
            from active_learning.sampling.entropy import EntropySampler
            from active_learning.queuing.priority_queue import get_global_queue
            from active_learning.retraining.trigger import RetrainingTrigger, TriggerConfig

            self._entropy_sampler = EntropySampler()
            self._priority_queue = get_global_queue()

            trigger_config = TriggerConfig(
                annotation_count_threshold=self.config.annotations_per_cycle,
            )
            self._trigger = RetrainingTrigger(
                config=trigger_config,
                on_trigger=self._on_retrain_triggered,
            )

        if self.config.enable_drift_detection and self._drift_detector is None:
            try:
                from active_learning.analysis.drift import DriftDetector
                self._drift_detector = DriftDetector()
            except ImportError:
                logger.warning("Drift detector not available, skipping drift detection")

    def _on_retrain_triggered(self, decision: Any) -> None:
        """Callback when retraining trigger fires."""
        logger.info(
            "Retraining trigger fired! Urgency=%s Reasons=%s",
            decision.urgency,
            decision.reasons,
        )
        # In production: submit training job via task_router
        # For now: log the event

    # ------------------------------------------------------------------
    # Core cycle
    # ------------------------------------------------------------------

    def run_cycle(
        self,
        unlabeled_pool: list[dict],
        model_predictions: list[dict] | None = None,
    ) -> ALCycleResult:
        """
        Run one active learning cycle.

        Args:
            unlabeled_pool: List of sample dicts with keys:
                sample_id, dataset_id, image_path, [image_features]
            model_predictions: Pre-computed predictions. If None, will attempt
                to run inference (requires model_path to be set).

        Returns:
            ALCycleResult with queued samples and trigger decision.
        """
        import time
        import uuid

        self._init_components()

        cycle_id = str(uuid.uuid4())[:8]
        start = time.time()

        result = ALCycleResult(
            cycle_id=cycle_id,
            pool_size=len(unlabeled_pool),
        )

        logger.info(
            "AL cycle %s started — pool size: %d", cycle_id, len(unlabeled_pool)
        )

        try:
            # 1. Score samples by uncertainty
            scored_samples = self._score_uncertainty(
                unlabeled_pool,
                model_predictions or [],
            )
            result.scored_count = len(scored_samples)

            # 2. Drift detection
            if self.config.enable_drift_detection and self._drift_detector is not None:
                drift_result = self._check_drift(unlabeled_pool)
                result.drift_detected = drift_result.get("drifted", False)
                result.drift_score = drift_result.get("score", 0.0)
                if self._trigger:
                    self._trigger.update_drift(result.drift_score)

            # 3. Select top-k and push to queue
            queued = self._push_to_queue(scored_samples)
            result.queued_count = len(queued)
            result.top_samples = [
                {
                    "sample_id": q.get("sample_id"),
                    "uncertainty": q.get("uncertainty_score"),
                    "entropy": q.get("entropy_score"),
                }
                for q in queued[:10]
            ]

            # 4. Evaluate trigger conditions
            if self.config.enable_auto_trigger and self._trigger:
                decision = self._trigger.evaluate()
                result.trigger_decision = decision.to_dict()

        except Exception as exc:  # noqa: BLE001
            logger.error("AL cycle %s error: %s", cycle_id, exc, exc_info=True)
            result.errors.append(str(exc))

        result.duration_s = round(time.time() - start, 2)
        self._cycle_count += 1
        self._results.append(result)

        logger.info(
            "AL cycle %s done: scored=%d queued=%d drift=%s duration=%.2fs",
            cycle_id,
            result.scored_count,
            result.queued_count,
            result.drift_detected,
            result.duration_s,
        )

        return result

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _score_uncertainty(
        self,
        pool: list[dict],
        predictions: list[dict],
    ) -> list[dict]:
        """
        Score pool samples by uncertainty.

        Uses entropy if predictions available, otherwise returns pool as-is
        with zero scores.
        """
        scored: list[dict] = []
        pred_map = {p.get("sample_id", ""): p for p in predictions}

        for sample in pool:
            sample_id = sample.get("sample_id", "")
            pred = pred_map.get(sample_id)

            if pred and "probabilities" in pred:
                from active_learning.sampling.entropy import compute_entropy, compute_normalized_entropy
                probs = pred["probabilities"]
                entropy = compute_entropy(probs)
                num_classes = len(probs)
                import math
                norm_entropy = entropy / math.log(num_classes) if num_classes > 1 else 0.0
                uncertainty = float(1.0 - pred.get("confidence", 0.5))
            else:
                entropy = 0.0
                norm_entropy = 0.0
                uncertainty = 0.5  # Unknown = medium priority

            scored.append(
                {
                    **sample,
                    "entropy_score": round(entropy, 6),
                    "normalized_entropy": round(norm_entropy, 6),
                    "uncertainty_score": round(uncertainty, 6),
                    "predicted_class": pred.get("predicted_class", -1) if pred else -1,
                    "predicted_confidence": pred.get("confidence", 0.0) if pred else 0.0,
                }
            )

        # Filter by minimum uncertainty
        scored = [s for s in scored if s["uncertainty_score"] >= self.config.min_uncertainty_to_queue]

        # Sort by composite score
        scored.sort(
            key=lambda s: 0.6 * s["uncertainty_score"] + 0.4 * s["normalized_entropy"],
            reverse=True,
        )
        return scored

    def _check_drift(self, pool: list[dict]) -> dict:
        """Check for dataset drift in new pool samples."""
        try:
            if self._drift_detector is None:
                return {"drifted": False, "score": 0.0}

            # Extract image paths
            image_paths = [s.get("image_path", "") for s in pool if s.get("image_path")]
            if len(image_paths) < 10:
                return {"drifted": False, "score": 0.0, "reason": "too_few_samples"}

            report = self._drift_detector.analyze(image_paths[:100])  # cap for speed
            return {
                "drifted": report.overall_drifted if report else False,
                "score": report.max_drift_score if report else 0.0,
            }
        except Exception as exc:  # noqa: BLE001
            logger.debug("Drift check failed: %s", exc)
            return {"drifted": False, "score": 0.0, "error": str(exc)}

    def _push_to_queue(self, scored_samples: list[dict]) -> list[dict]:
        """Push top-k scored samples to the annotation priority queue."""
        top_k = scored_samples[: self.config.queue_top_k]

        for sample in top_k:
            try:
                self._priority_queue.push(
                    sample_id=sample["sample_id"],
                    dataset_id=sample.get("dataset_id", ""),
                    image_path=sample.get("image_path", ""),
                    uncertainty_score=sample["uncertainty_score"],
                    entropy_score=sample["entropy_score"],
                    predicted_class=sample.get("predicted_class", -1),
                    predicted_confidence=sample.get("predicted_confidence", 0.0),
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("Failed to queue sample %s: %s", sample.get("sample_id"), exc)

        return top_k

    # ------------------------------------------------------------------
    # Annotation feedback
    # ------------------------------------------------------------------

    def on_annotation_approved(self, count: int = 1) -> None:
        """Notify the pipeline that annotations were approved."""
        self._init_components()
        if self._trigger:
            self._trigger.on_annotation_approved(count)

    def update_model_metrics(self, map50: float) -> None:
        """Update current model performance for trigger evaluation."""
        self._init_components()
        if self._trigger:
            self._trigger.update_metrics(map50)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> dict:
        """Return pipeline status summary."""
        self._init_components()
        return {
            "cycle_count": self._cycle_count,
            "queue_size": len(self._priority_queue) if self._priority_queue else 0,
            "trigger_status": (
                self._trigger.get_status() if self._trigger else {}
            ),
            "last_result": (
                {
                    "cycle_id": self._results[-1].cycle_id,
                    "scored": self._results[-1].scored_count,
                    "queued": self._results[-1].queued_count,
                    "drift": self._results[-1].drift_detected,
                }
                if self._results
                else None
            ),
        }
