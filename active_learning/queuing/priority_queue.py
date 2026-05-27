"""
active_learning/queuing/priority_queue.py — Priority-based annotation queue.

Combines multiple uncertainty signals to compute a composite priority score
for each candidate sample, then maintains a sorted queue for annotators.

Priority factors:
  1. Uncertainty score (entropy / BALD / disagreement)
  2. Representativeness (distance from already-annotated samples)
  3. Class-rarity bonus (rare classes weighted higher)
  4. Time decay (older candidates get slight boost to prevent starvation)
  5. Human override: annotator can manually elevate priority

Thread-safe: uses a lock for concurrent backend + frontend access.
"""

from __future__ import annotations

import heapq
import time
import uuid
from dataclasses import dataclass, field
from threading import Lock
from typing import Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(order=True)
class QueueEntry:
    """
    Priority queue entry.

    Heap is min-heap, so we negate priority for max-heap behavior.
    """

    neg_priority: float = field(compare=True)
    inserted_at: float = field(compare=True)  # tiebreaker
    item_id: str = field(compare=False)
    sample_id: str = field(compare=False)
    dataset_id: str = field(compare=False)
    image_path: str = field(compare=False)

    # Uncertainty scores
    entropy_score: float = field(compare=False, default=0.0)
    disagreement_score: float = field(compare=False, default=0.0)
    uncertainty_score: float = field(compare=False, default=0.0)

    # Metadata
    predicted_class: int = field(compare=False, default=-1)
    predicted_confidence: float = field(compare=False, default=0.0)
    class_rarity: float = field(compare=False, default=1.0)  # 1.0 = common, >1 = rare
    manual_priority_boost: float = field(compare=False, default=0.0)

    status: str = field(compare=False, default="pending")  # pending | assigned | completed


@dataclass
class QueueStats:
    total_items: int
    pending: int
    assigned: int
    completed: int
    mean_priority: float
    min_priority: float
    max_priority: float
    class_distribution: dict[int, int]


# ---------------------------------------------------------------------------
# Priority computation
# ---------------------------------------------------------------------------


CLASS_RARITY_WEIGHTS: dict[int, float] = {
    0: 1.0,   # capitate-stalked (60%) — common
    1: 1.5,   # capitate-sessile (25%) — moderate
    2: 3.0,   # bulbous (10%) — rare
    3: 6.0,   # non-glandular (5%) — very rare
}


def compute_priority(
    uncertainty_score: float,
    entropy_score: float = 0.0,
    disagreement_score: float = 0.0,
    class_rarity: float = 1.0,
    age_seconds: float = 0.0,
    manual_boost: float = 0.0,
    w_uncertainty: float = 0.5,
    w_entropy: float = 0.2,
    w_disagreement: float = 0.2,
    w_rarity: float = 0.1,
) -> float:
    """
    Compute composite priority score in [0, ∞).

    Higher score = higher priority for annotation.

    Args:
        uncertainty_score: Base uncertainty (e.g. from MC Dropout), in [0,1].
        entropy_score: Prediction entropy, in [0,1] (normalized).
        disagreement_score: Ensemble disagreement, in [0,1].
        class_rarity: Multiplier for rare classes (1.0=common, >1=rare).
        age_seconds: Seconds since item was added (prevents starvation).
        manual_boost: Manual annotator priority override (0=none, 1=max boost).
        w_*: Mixing weights (should sum to ~1.0 before class_rarity scaling).

    Returns:
        Composite priority score.
    """
    base = (
        w_uncertainty * uncertainty_score
        + w_entropy * entropy_score
        + w_disagreement * disagreement_score
        + w_rarity * min(1.0, (class_rarity - 1.0) / 10.0)  # normalize rarity
    )

    # Time decay boost: +0.01 per hour, capped at +0.2
    age_boost = min(0.20, age_seconds / 3600.0 * 0.01)

    priority = (base + age_boost) * class_rarity + manual_boost
    return round(max(0.0, priority), 6)


# ---------------------------------------------------------------------------
# Priority queue
# ---------------------------------------------------------------------------


class AnnotationPriorityQueue:
    """
    Thread-safe priority queue for annotation candidates.

    Items are ordered by composite priority score (highest first).
    Supports: push, pop, peek, boost, complete, and stats.
    """

    def __init__(self) -> None:
        self._heap: list[QueueEntry] = []
        self._items: dict[str, QueueEntry] = {}  # item_id → entry
        self._lock = Lock()
        self._completed: list[QueueEntry] = []

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def push(
        self,
        sample_id: str,
        dataset_id: str,
        image_path: str,
        uncertainty_score: float = 0.0,
        entropy_score: float = 0.0,
        disagreement_score: float = 0.0,
        predicted_class: int = -1,
        predicted_confidence: float = 0.0,
        class_rarity: float | None = None,
        manual_priority_boost: float = 0.0,
    ) -> QueueEntry:
        """Add a new candidate to the queue. Returns the created entry."""
        if class_rarity is None:
            class_rarity = CLASS_RARITY_WEIGHTS.get(predicted_class, 1.0)

        now = time.time()
        priority = compute_priority(
            uncertainty_score=uncertainty_score,
            entropy_score=entropy_score,
            disagreement_score=disagreement_score,
            class_rarity=class_rarity,
            age_seconds=0.0,
            manual_boost=manual_priority_boost,
        )

        entry = QueueEntry(
            neg_priority=-priority,  # negate for min-heap
            inserted_at=now,
            item_id=str(uuid.uuid4()),
            sample_id=sample_id,
            dataset_id=dataset_id,
            image_path=image_path,
            entropy_score=entropy_score,
            disagreement_score=disagreement_score,
            uncertainty_score=uncertainty_score,
            predicted_class=predicted_class,
            predicted_confidence=predicted_confidence,
            class_rarity=class_rarity,
            manual_priority_boost=manual_priority_boost,
            status="pending",
        )

        with self._lock:
            heapq.heappush(self._heap, entry)
            self._items[entry.item_id] = entry

        return entry

    def pop(self) -> Optional[QueueEntry]:
        """
        Pop the highest-priority pending item.

        Skips completed/assigned items at heap top (lazy deletion).
        Returns None if queue is empty.
        """
        with self._lock:
            while self._heap:
                entry = heapq.heappop(self._heap)
                current = self._items.get(entry.item_id)
                if current is None or current.status != "pending":
                    continue  # Lazy deletion of stale entries
                current.status = "assigned"
                return current
        return None

    def peek_top_k(self, k: int = 10) -> list[QueueEntry]:
        """Return top-k pending items by priority without removing them."""
        with self._lock:
            pending = [e for e in self._items.values() if e.status == "pending"]
            # Re-score with current age to handle starvation correctly
            now = time.time()
            scored: list[tuple[float, QueueEntry]] = []
            for entry in pending:
                age = now - entry.inserted_at
                priority = compute_priority(
                    uncertainty_score=entry.uncertainty_score,
                    entropy_score=entry.entropy_score,
                    disagreement_score=entry.disagreement_score,
                    class_rarity=entry.class_rarity,
                    age_seconds=age,
                    manual_boost=entry.manual_priority_boost,
                )
                scored.append((-priority, entry))
            scored.sort()
            return [e for _, e in scored[:k]]

    def boost(self, item_id: str, boost_amount: float = 0.5) -> bool:
        """Manually increase priority of a queued item."""
        with self._lock:
            entry = self._items.get(item_id)
            if entry is None or entry.status != "pending":
                return False
            entry.manual_priority_boost += boost_amount
            # Re-insert with updated priority
            age = time.time() - entry.inserted_at
            new_priority = compute_priority(
                uncertainty_score=entry.uncertainty_score,
                entropy_score=entry.entropy_score,
                disagreement_score=entry.disagreement_score,
                class_rarity=entry.class_rarity,
                age_seconds=age,
                manual_boost=entry.manual_priority_boost,
            )
            entry.neg_priority = -new_priority
            heapq.heappush(self._heap, entry)
        return True

    def complete(self, item_id: str) -> bool:
        """Mark an item as completed (annotation done)."""
        with self._lock:
            entry = self._items.get(item_id)
            if entry is None:
                return False
            entry.status = "completed"
            self._completed.append(entry)
        return True

    def remove(self, item_id: str) -> bool:
        """Remove item from queue (e.g. sample was deleted)."""
        with self._lock:
            if item_id in self._items:
                del self._items[item_id]
                return True
        return False

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> QueueStats:
        """Return queue statistics."""
        with self._lock:
            items = list(self._items.values())

        pending = [e for e in items if e.status == "pending"]
        assigned = [e for e in items if e.status == "assigned"]

        priorities = [-e.neg_priority for e in pending]  # stored negated

        class_dist: dict[int, int] = {}
        for e in pending:
            c = e.predicted_class
            class_dist[c] = class_dist.get(c, 0) + 1

        return QueueStats(
            total_items=len(items),
            pending=len(pending),
            assigned=len(assigned),
            completed=len(self._completed),
            mean_priority=round(sum(priorities) / len(priorities), 4) if priorities else 0.0,
            min_priority=round(min(priorities), 4) if priorities else 0.0,
            max_priority=round(max(priorities), 4) if priorities else 0.0,
            class_distribution=class_dist,
        )

    def __len__(self) -> int:
        with self._lock:
            return sum(1 for e in self._items.values() if e.status == "pending")


# ---------------------------------------------------------------------------
# Singleton queue (shared across the process)
# ---------------------------------------------------------------------------

_global_queue: AnnotationPriorityQueue | None = None


def get_global_queue() -> AnnotationPriorityQueue:
    """Return the process-singleton annotation priority queue."""
    global _global_queue
    if _global_queue is None:
        _global_queue = AnnotationPriorityQueue()
    return _global_queue
