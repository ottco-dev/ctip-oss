"""
annotation.review.queue — Human review queue management.

DESIGN:
The review queue is the critical human-in-loop component that sits between
VLM auto-labeling and the training dataset.

QUEUE INVARIANT:
Items enter the queue as VLM_AUTO pseudo-labels.
Items leave the queue as HUMAN_REVIEWED or HUMAN_CORRECTED labels.
Training data loader only accepts items with reviewed=True.

PRIORITY SYSTEM:
Priority 3 (critical): Invalid JSON, unknown class predictions
Priority 2 (high):     Semantic inconsistencies, rule disagreements
Priority 1 (medium):   Low confidence predictions
Priority 0 (low):      Routine review (passed all filters)

THROUGHPUT:
A trained annotator can review ~200-400 trichome maturity images/hour
(compared to ~5-15 from scratch annotation).
VLM auto-labeling + human review is ~5-10× faster than manual annotation.
"""

from __future__ import annotations

import heapq
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterator


class ReviewAction(str, Enum):
    APPROVE = "approve"
    """Accept VLM label as-is."""

    CORRECT = "correct"
    """Accept with corrected label."""

    REJECT = "reject"
    """Discard — image not suitable or label too uncertain."""

    SKIP = "skip"
    """Return to queue for later review."""


@dataclass
class ReviewItem:
    """
    A single item in the review queue.

    Wraps a PseudoLabel and adds queue management metadata.
    """

    item_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    pseudo_label_id: str = ""

    # Display data (for review UI)
    image_path: str = ""
    image_id: str = ""
    maturity_stage: str | None = None
    vlm_confidence: float = 0.0
    hallucination_flags: list[str] = field(default_factory=list)
    raw_response: dict[str, Any] | None = None

    # Queue metadata
    priority: int = 0
    """0=low, 1=medium, 2=high, 3=critical"""

    queued_at: float = field(default_factory=time.time)
    assigned_to: str | None = None
    """Reviewer ID if item is being reviewed."""

    assigned_at: float | None = None

    # Review result
    reviewed: bool = False
    review_action: ReviewAction | None = None
    corrected_label: str | None = None
    reviewer_id: str | None = None
    review_comment: str | None = None
    reviewed_at: float | None = None

    def __lt__(self, other: "ReviewItem") -> bool:
        """For heapq: higher priority first, then older items first."""
        if self.priority != other.priority:
            return self.priority > other.priority  # Higher priority = smaller in heap
        return self.queued_at < other.queued_at  # Older = smaller in heap

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "pseudo_label_id": self.pseudo_label_id,
            "image_path": self.image_path,
            "image_id": self.image_id,
            "maturity_stage": self.maturity_stage,
            "vlm_confidence": self.vlm_confidence,
            "hallucination_flags": self.hallucination_flags,
            "priority": self.priority,
            "queued_at": self.queued_at,
            "assigned_to": self.assigned_to,
            "reviewed": self.reviewed,
            "review_action": self.review_action.value if self.review_action else None,
            "corrected_label": self.corrected_label,
            "reviewer_id": self.reviewer_id,
        }


@dataclass
class QueueStats:
    total_queued: int = 0
    pending: int = 0
    in_review: int = 0
    completed: int = 0
    approved: int = 0
    corrected: int = 0
    rejected: int = 0
    mean_review_time_s: float = 0.0
    throughput_per_hour: float = 0.0

    priority_counts: dict[int, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_queued": self.total_queued,
            "pending": self.pending,
            "in_review": self.in_review,
            "completed": self.completed,
            "approved": self.approved,
            "corrected": self.corrected,
            "rejected": self.rejected,
            "mean_review_time_s": self.mean_review_time_s,
            "throughput_per_hour": self.throughput_per_hour,
            "priority_counts": self.priority_counts,
        }


class ReviewQueue:
    """
    Priority-based human review queue.

    In-memory implementation suitable for single-user local workflow.
    Production: back with database (SQLite or PostgreSQL via SQLModel).

    Usage:
        queue = ReviewQueue()

        # Add items from auto-label pipeline
        for label in pseudo_labels:
            queue.add(ReviewItem(
                pseudo_label_id=label.label_id,
                image_path=label.image_path,
                maturity_stage=label.maturity_stage,
                vlm_confidence=label.vlm_confidence,
                hallucination_flags=label.hallucination_flags,
                priority=label.filter_result.review_priority if label.filter_result else 0,
            ))

        # Reviewer fetches next item
        item = queue.get_next(reviewer_id="reviewer_1")
        # ... display to reviewer ...
        queue.submit_review(
            item_id=item.item_id,
            action=ReviewAction.APPROVE,
            reviewer_id="reviewer_1",
        )
    """

    def __init__(self) -> None:
        self._items: dict[str, ReviewItem] = {}  # item_id → ReviewItem
        self._heap: list[tuple[tuple[int, float], str]] = []
        # Heap entries: ((-priority, queued_at), item_id)
        self._review_times: list[float] = []

    @property
    def size(self) -> int:
        """Number of pending (unreviewed) items in queue."""
        return sum(
            1 for item in self._items.values()
            if not item.reviewed and item.assigned_to is None
        )

    def add(self, item: ReviewItem) -> str:
        """
        Add a review item to the queue.

        Returns the item_id.
        """
        self._items[item.item_id] = item
        # Heap key: negative priority (so higher priority pops first), then timestamp
        heap_key = (-item.priority, item.queued_at)
        heapq.heappush(self._heap, (heap_key, item.item_id))
        return item.item_id

    def add_from_pseudo_label(self, label: Any) -> str:
        """
        Convenience method: create ReviewItem from PseudoLabel and add to queue.

        Args:
            label: PseudoLabel from auto_label_pipeline.
        """
        priority = 0
        if label.filter_result:
            priority = label.filter_result.review_priority

        item = ReviewItem(
            pseudo_label_id=label.label_id,
            image_path=label.image_path,
            image_id=label.image_id,
            maturity_stage=label.maturity_stage,
            vlm_confidence=label.vlm_confidence,
            hallucination_flags=label.hallucination_flags,
            raw_response=label.parsed_vlm_response,
            priority=priority,
        )
        return self.add(item)

    def get_next(self, reviewer_id: str | None = None) -> ReviewItem | None:
        """
        Get next highest-priority item from queue and assign it.

        Args:
            reviewer_id: Optional reviewer ID for assignment tracking.

        Returns:
            ReviewItem or None if queue is empty.
        """
        # Pop from heap, skipping already-reviewed or assigned items
        while self._heap:
            heap_key, item_id = heapq.heappop(self._heap)
            item = self._items.get(item_id)
            if item is None:
                continue
            if item.reviewed or item.assigned_to is not None:
                continue

            # Assign to reviewer
            item.assigned_to = reviewer_id
            item.assigned_at = time.time()
            return item

        return None

    def get_batch(
        self,
        n: int,
        reviewer_id: str | None = None,
        min_priority: int = 0,
    ) -> list[ReviewItem]:
        """
        Get up to N items from the queue for batch review.

        Args:
            n: Maximum items to return.
            reviewer_id: Reviewer ID for assignment.
            min_priority: Only return items with priority >= this.
        """
        items: list[ReviewItem] = []
        temp_heap = []

        while self._heap and len(items) < n:
            heap_key, item_id = heapq.heappop(self._heap)
            item = self._items.get(item_id)
            if item is None:
                continue
            if item.reviewed or item.assigned_to is not None:
                continue
            if item.priority < min_priority:
                temp_heap.append((heap_key, item_id))
                continue

            item.assigned_to = reviewer_id
            item.assigned_at = time.time()
            items.append(item)

        # Push back items we skipped due to priority filter
        for entry in temp_heap:
            heapq.heappush(self._heap, entry)

        return items

    def submit_review(
        self,
        item_id: str,
        action: ReviewAction,
        reviewer_id: str,
        corrected_label: str | None = None,
        comment: str | None = None,
    ) -> ReviewItem:
        """
        Submit a review decision for an item.

        Args:
            item_id: ID of the item being reviewed.
            action: APPROVE, CORRECT, REJECT, or SKIP.
            reviewer_id: ID of the reviewer.
            corrected_label: Required if action=CORRECT.
            comment: Optional free-text comment.

        Returns:
            Updated ReviewItem.

        Raises:
            KeyError: If item_id not found.
            ValueError: If action=CORRECT but corrected_label not provided.
        """
        if item_id not in self._items:
            raise KeyError(f"Review item '{item_id}' not found in queue")

        item = self._items[item_id]

        if action == ReviewAction.CORRECT and not corrected_label:
            raise ValueError("corrected_label required when action=CORRECT")

        if action == ReviewAction.SKIP:
            # Return to queue
            item.assigned_to = None
            item.assigned_at = None
            heap_key = (-item.priority, item.queued_at)
            heapq.heappush(self._heap, (heap_key, item_id))
            return item

        # Record review
        review_time = time.time()
        if item.assigned_at:
            self._review_times.append(review_time - item.assigned_at)

        item.reviewed = True
        item.review_action = action
        item.reviewer_id = reviewer_id
        item.review_comment = comment
        item.reviewed_at = review_time

        if action == ReviewAction.CORRECT:
            item.corrected_label = corrected_label

        return item

    def unassign(self, item_id: str) -> None:
        """Release an item back to the queue (e.g., reviewer disconnected)."""
        item = self._items.get(item_id)
        if item and not item.reviewed:
            item.assigned_to = None
            item.assigned_at = None
            heap_key = (-item.priority, item.queued_at)
            heapq.heappush(self._heap, (heap_key, item_id))

    def get_stats(self) -> QueueStats:
        """Get current queue statistics."""
        all_items = list(self._items.values())
        pending = [i for i in all_items if not i.reviewed and i.assigned_to is None]
        in_review = [i for i in all_items if not i.reviewed and i.assigned_to is not None]
        completed = [i for i in all_items if i.reviewed]

        priority_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        for item in pending:
            priority_counts[item.priority] = priority_counts.get(item.priority, 0) + 1

        mean_review = 0.0
        throughput = 0.0
        if self._review_times:
            mean_review = sum(self._review_times) / len(self._review_times)
            if mean_review > 0:
                throughput = 3600 / mean_review  # items per hour

        return QueueStats(
            total_queued=len(all_items),
            pending=len(pending),
            in_review=len(in_review),
            completed=len(completed),
            approved=sum(1 for i in completed if i.review_action == ReviewAction.APPROVE),
            corrected=sum(1 for i in completed if i.review_action == ReviewAction.CORRECT),
            rejected=sum(1 for i in completed if i.review_action == ReviewAction.REJECT),
            mean_review_time_s=mean_review,
            throughput_per_hour=throughput,
            priority_counts=priority_counts,
        )

    def get_approved_labels(self) -> list[ReviewItem]:
        """Get all approved (ready for training) items."""
        return [
            item for item in self._items.values()
            if item.reviewed and item.review_action in (
                ReviewAction.APPROVE, ReviewAction.CORRECT
            )
        ]

    def iter_pending(self) -> Iterator[ReviewItem]:
        """Iterate over pending items in priority order."""
        pending = [
            i for i in self._items.values()
            if not i.reviewed and i.assigned_to is None
        ]
        yield from sorted(
            pending,
            key=lambda i: (-i.priority, i.queued_at),
        )
