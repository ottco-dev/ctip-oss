"""
active_learning/retraining/trigger.py — Auto-trigger retraining thresholds.

Decides when to kick off a retraining run based on:
  1. New annotation count threshold (N new annotations since last run)
  2. Performance degradation detection (val mAP drops below threshold)
  3. Dataset drift alarm (drift detector flags significant shift)
  4. Scheduled interval (retraining at least every X hours regardless)
  5. Manual override via trigger()

The trigger is stateful: it tracks the last retraining time, the count of
new annotations, and current metric snapshots.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class TriggerConfig:
    """Configuration for automatic retraining triggers."""

    # Annotation count threshold: retrain after N new approved annotations
    annotation_count_threshold: int = 100

    # Performance threshold: retrain if mAP50 drops more than this from best
    performance_drop_threshold: float = 0.05  # 5% absolute drop

    # Drift trigger: retrain if drift score exceeds this (0-1 scale)
    drift_score_threshold: float = 0.6

    # Scheduled interval: retrain at least every N hours (0 = disabled)
    scheduled_interval_hours: float = 24.0

    # Minimum interval between retraining runs (cooldown)
    min_retraining_interval_hours: float = 2.0

    # Minimum annotations required before any auto-trigger
    min_annotations_for_trigger: int = 50


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


@dataclass
class TriggerState:
    """Mutable state tracking trigger conditions."""

    last_retrain_at: Optional[datetime] = None
    new_annotations_since_retrain: int = 0
    best_map50: float = 0.0
    current_map50: float = 0.0
    last_drift_score: float = 0.0
    total_annotations: int = 0
    retrain_count: int = 0
    trigger_history: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Trigger reason
# ---------------------------------------------------------------------------


@dataclass
class TriggerDecision:
    """Result of a trigger evaluation."""

    should_retrain: bool
    reasons: list[str]
    urgency: str  # low | medium | high | critical
    annotation_count: int
    performance_drop: float
    drift_score: float
    hours_since_last: float

    def to_dict(self) -> dict:
        return {
            "should_retrain": self.should_retrain,
            "reasons": self.reasons,
            "urgency": self.urgency,
            "annotation_count": self.annotation_count,
            "performance_drop": round(self.performance_drop, 4),
            "drift_score": round(self.drift_score, 4),
            "hours_since_last": round(self.hours_since_last, 2),
        }


# ---------------------------------------------------------------------------
# Retraining trigger
# ---------------------------------------------------------------------------


class RetrainingTrigger:
    """
    Monitors conditions and decides when to trigger retraining.

    Usage:
        trigger = RetrainingTrigger(config)
        trigger.on_annotation_approved()  # call when annotation is approved
        trigger.update_metrics(map50=0.82)
        trigger.update_drift(drift_score=0.71)

        decision = trigger.evaluate()
        if decision.should_retrain:
            orchestrator.start_training(...)
    """

    def __init__(
        self,
        config: TriggerConfig | None = None,
        on_trigger: Optional[Callable[[TriggerDecision], None]] = None,
    ) -> None:
        self.config = config or TriggerConfig()
        self.state = TriggerState()
        self._on_trigger = on_trigger  # Callback when trigger fires

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def on_annotation_approved(self, count: int = 1) -> None:
        """Call when annotations are approved by human reviewer."""
        self.state.new_annotations_since_retrain += count
        self.state.total_annotations += count

    def on_annotation_rejected(self, count: int = 1) -> None:
        """Track rejections for quality monitoring (no trigger)."""
        pass  # Could log or adjust confidence thresholds

    def update_metrics(
        self,
        map50: float,
        map50_95: float | None = None,
    ) -> None:
        """Update current model performance metrics."""
        self.state.current_map50 = map50
        if map50 > self.state.best_map50:
            self.state.best_map50 = map50

    def update_drift(self, drift_score: float) -> None:
        """Update dataset drift score (0=no drift, 1=severe drift)."""
        self.state.last_drift_score = drift_score

    def mark_retrained(self) -> None:
        """Call after a retraining run completes."""
        self.state.last_retrain_at = datetime.utcnow()
        self.state.new_annotations_since_retrain = 0
        self.state.retrain_count += 1

    def manual_trigger(self, reason: str = "manual") -> TriggerDecision:
        """Force a trigger regardless of conditions."""
        decision = TriggerDecision(
            should_retrain=True,
            reasons=[f"Manual override: {reason}"],
            urgency="high",
            annotation_count=self.state.new_annotations_since_retrain,
            performance_drop=0.0,
            drift_score=self.state.last_drift_score,
            hours_since_last=self._hours_since_last(),
        )
        self._fire(decision)
        return decision

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------

    def evaluate(self) -> TriggerDecision:
        """
        Evaluate all trigger conditions and return a decision.

        Does NOT fire any retraining — caller decides what to do with the result.
        To fire automatically on trigger, set on_trigger callback in constructor.
        """
        reasons: list[str] = []
        urgency_levels: list[int] = []  # 0=low 1=med 2=high 3=crit

        hours_since = self._hours_since_last()
        annotation_count = self.state.new_annotations_since_retrain
        performance_drop = self.state.best_map50 - self.state.current_map50
        drift_score = self.state.last_drift_score

        # Cooldown check (hours_since < 0 means never retrained — skip cooldown)
        if hours_since >= 0 and hours_since < self.config.min_retraining_interval_hours:
            return TriggerDecision(
                should_retrain=False,
                reasons=[f"Cooldown: {hours_since:.1f}h since last retrain "
                         f"(min {self.config.min_retraining_interval_hours}h)"],
                urgency="low",
                annotation_count=annotation_count,
                performance_drop=performance_drop,
                drift_score=drift_score,
                hours_since_last=hours_since,
            )

        # Minimum annotations check
        if self.state.total_annotations < self.config.min_annotations_for_trigger:
            return TriggerDecision(
                should_retrain=False,
                reasons=[f"Insufficient data: {self.state.total_annotations} annotations "
                         f"(need {self.config.min_annotations_for_trigger})"],
                urgency="low",
                annotation_count=annotation_count,
                performance_drop=performance_drop,
                drift_score=drift_score,
                hours_since_last=hours_since,
            )

        # --- Trigger 1: Annotation count ---
        if annotation_count >= self.config.annotation_count_threshold:
            reasons.append(
                f"New annotations: {annotation_count} ≥ {self.config.annotation_count_threshold}"
            )
            urgency_levels.append(1)

        # --- Trigger 2: Performance degradation ---
        if (
            self.state.best_map50 > 0.0
            and performance_drop >= self.config.performance_drop_threshold
        ):
            reasons.append(
                f"Performance drop: {performance_drop:.3f} (best={self.state.best_map50:.3f}, "
                f"current={self.state.current_map50:.3f})"
            )
            urgency_levels.append(2 if performance_drop >= 0.10 else 1)

        # --- Trigger 3: Dataset drift ---
        if drift_score >= self.config.drift_score_threshold:
            reasons.append(f"Dataset drift detected: score={drift_score:.3f}")
            urgency_levels.append(2 if drift_score >= 0.80 else 1)

        # --- Trigger 4: Scheduled interval ---
        if (
            self.config.scheduled_interval_hours > 0
            and hours_since >= self.config.scheduled_interval_hours
            and annotation_count > 0
        ):
            reasons.append(
                f"Scheduled: {hours_since:.1f}h since last retrain "
                f"(interval={self.config.scheduled_interval_hours}h)"
            )
            urgency_levels.append(0)

        if not reasons:
            return TriggerDecision(
                should_retrain=False,
                reasons=["No trigger conditions met"],
                urgency="low",
                annotation_count=annotation_count,
                performance_drop=performance_drop,
                drift_score=drift_score,
                hours_since_last=hours_since,
            )

        max_urgency = max(urgency_levels)
        urgency_map = {0: "low", 1: "medium", 2: "high", 3: "critical"}
        urgency = urgency_map.get(max_urgency, "medium")

        decision = TriggerDecision(
            should_retrain=True,
            reasons=reasons,
            urgency=urgency,
            annotation_count=annotation_count,
            performance_drop=performance_drop,
            drift_score=drift_score,
            hours_since_last=hours_since,
        )

        self._fire(decision)
        return decision

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _hours_since_last(self) -> float:
        if self.state.last_retrain_at is None:
            return -1.0  # Sentinel: never retrained
        delta = datetime.utcnow() - self.state.last_retrain_at
        return delta.total_seconds() / 3600.0

    def _fire(self, decision: TriggerDecision) -> None:
        record = {
            **decision.to_dict(),
            "fired_at": datetime.utcnow().isoformat(),
        }
        self.state.trigger_history.append(record)
        if len(self.state.trigger_history) > 100:
            self.state.trigger_history = self.state.trigger_history[-100:]

        if self._on_trigger and decision.should_retrain:
            try:
                self._on_trigger(decision)
            except Exception:  # noqa: BLE001
                pass

    def get_status(self) -> dict:
        """Return current trigger status as a dict."""
        return {
            "last_retrain_at": (
                self.state.last_retrain_at.isoformat()
                if self.state.last_retrain_at
                else None
            ),
            "new_annotations_since_retrain": self.state.new_annotations_since_retrain,
            "total_annotations": self.state.total_annotations,
            "best_map50": self.state.best_map50,
            "current_map50": self.state.current_map50,
            "last_drift_score": self.state.last_drift_score,
            "retrain_count": self.state.retrain_count,
            "hours_since_last": self._hours_since_last(),
            "trigger_history_last_5": self.state.trigger_history[-5:],
            "config": {
                "annotation_count_threshold": self.config.annotation_count_threshold,
                "performance_drop_threshold": self.config.performance_drop_threshold,
                "drift_score_threshold": self.config.drift_score_threshold,
                "scheduled_interval_hours": self.config.scheduled_interval_hours,
            },
        }
