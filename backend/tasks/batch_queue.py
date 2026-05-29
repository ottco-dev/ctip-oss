"""
backend.tasks.batch_queue — Dynamic batching queue for GPU inference.

Problem solved
--------------
When multiple single-image detection requests arrive concurrently, the current
architecture acquires the GPU semaphore once *per request* and processes images
sequentially.  This leaves the GPU under-utilised: YOLO on RTX 4060 achieves
~45 ms/image sequentially but only ~12-15 ms/image in batches of 8 (3× throughput).

Solution
--------
A collection window (default 50 ms) accumulates incoming requests.  When the
window expires OR the batch reaches ``max_size``, a single YOLO detect_batch()
call processes all accumulated images in one GPU forward pass.

Each caller awaits an ``asyncio.Future`` that resolves with its individual
result when the batch completes.

GPU semaphore contract
----------------------
The batch flush task acquires the GPU semaphore *once* for the entire batch.
Callers at ``/inference/detect/queued`` do NOT acquire the semaphore themselves.
This is the key architectural difference vs. ``/inference/detect`` where each
request holds the slot independently.

Configuration (.env)
--------------------
  BATCH_QUEUE_WINDOW_MS=50    collect window in ms (lower = lower latency, smaller batches)
  BATCH_QUEUE_MAX_SIZE=8      max images per batch (8 is optimal for RTX 4060 at 1280 px)

Throughput (RTX 4060, YOLO11s, 1280 px, estimated)
----------------------------------------------------
  Sequential (/detect):         ~45 ms/image
  Batched (/detect/queued):     ~15 ms/image at batch_size=8  (3× throughput)
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from shared.logging.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Internal entry — one per submitted request
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _QueueEntry:
    image: NDArray[np.uint8]
    conf_threshold: float
    iou_threshold: float
    model_variant: str
    use_tiled: bool
    model_path: str | None
    # resolved by the batch flush task
    future: asyncio.Future = field(default_factory=asyncio.Future)
    queued_at: float = field(default_factory=time.monotonic)


# ─────────────────────────────────────────────────────────────────────────────
# Public result type
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BatchedDetectionResult:
    """Individual result returned to each caller via their asyncio.Future."""

    detections: list[Any]
    """list[DetectionBox] — same schema as /inference/detect."""

    inference_time_ms: float
    """GPU time divided by batch size — per-image share of the forward pass."""

    queue_wait_ms: float
    """Time from submit() to GPU slot acquisition."""

    batch_size: int
    """How many images were processed together in this flush."""


# ─────────────────────────────────────────────────────────────────────────────
# Batch queue
# ─────────────────────────────────────────────────────────────────────────────

class DetectionBatchQueue:
    """
    Dynamic batching queue for YOLO inference.

    Thread-safe within a single asyncio event loop.
    All public methods must be awaited from coroutines sharing the same loop.

    Lifecycle of a single request::

        caller → submit()
                 ├─ appended to _pending
                 ├─ window timer started (or batch flushed immediately if full)
                 └─ awaits entry.future

        window expires / batch full
                 └─ _run_batch(batch)
                         ├─ acquire GPU semaphore (once for entire batch)
                         ├─ _sync_batch() in thread executor
                         │       ├─ group by (model_variant, use_tiled, model_path)
                         │       ├─ detect_batch() per group  [true GPU batching]
                         │       └─ fallback: sequential _run_detection() on failure
                         └─ resolve each entry.future with its BatchedDetectionResult

        caller ← BatchedDetectionResult
    """

    def __init__(self, window_ms: float = 50.0, max_size: int = 8) -> None:
        if window_ms <= 0:
            raise ValueError(f"window_ms must be positive, got {window_ms}")
        if max_size < 1:
            raise ValueError(f"max_size must be >= 1, got {max_size}")

        self._window_ms = window_ms
        self._max_size = max_size
        self._pending: list[_QueueEntry] = []
        self._lock = asyncio.Lock()
        self._flush_task: asyncio.Task | None = None

        # EMA stats (α = 0.2, reset on process restart)
        self._total_batches: int = 0
        self._total_images: int = 0
        self._ema_batch_size: float = 0.0
        self._ema_latency_ms: float = 0.0

    # ── public API ────────────────────────────────────────────────────────────

    async def submit(
        self,
        image: NDArray[np.uint8],
        conf_threshold: float,
        iou_threshold: float,
        model_variant: str,
        use_tiled: bool,
        model_path: str | None = None,
    ) -> BatchedDetectionResult:
        """
        Submit an image for batched detection.

        Suspends until the batch containing this image has been processed.
        Returns a ``BatchedDetectionResult`` with detections and timing info.
        """
        entry = _QueueEntry(
            image=image,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            model_variant=model_variant,
            use_tiled=use_tiled,
            model_path=model_path,
        )

        async with self._lock:
            self._pending.append(entry)

            if len(self._pending) >= self._max_size:
                # Batch full — flush immediately, cancel any running window timer
                batch = self._drain()
                if self._flush_task and not self._flush_task.done():
                    self._flush_task.cancel()
                    self._flush_task = None
                asyncio.get_event_loop().create_task(self._run_batch(batch))

            elif self._flush_task is None or self._flush_task.done():
                # First entry in a new collection window — start the expiry timer
                self._flush_task = asyncio.get_event_loop().create_task(
                    self._window_flush()
                )

        return await entry.future

    @property
    def stats(self) -> dict:
        """Current queue statistics — suitable for monitoring endpoints."""
        return {
            "total_batches": self._total_batches,
            "total_images": self._total_images,
            "ema_batch_size": round(self._ema_batch_size, 2),
            "ema_latency_ms": round(self._ema_latency_ms, 2),
            "pending": len(self._pending),
            "window_ms": self._window_ms,
            "max_size": self._max_size,
        }

    # ── internal ──────────────────────────────────────────────────────────────

    def _drain(self) -> list[_QueueEntry]:
        batch = self._pending[:]
        self._pending.clear()
        return batch

    async def _window_flush(self) -> None:
        """Wait for the collection window, then flush whatever has accumulated."""
        await asyncio.sleep(self._window_ms / 1000.0)
        async with self._lock:
            if self._pending:
                batch = self._drain()
                asyncio.get_event_loop().create_task(self._run_batch(batch))

    async def _run_batch(self, batch: list[_QueueEntry]) -> None:
        """Acquire the GPU semaphore once and dispatch a single batched inference."""
        if not batch:
            return

        t_gpu_start = time.monotonic()

        try:
            from backend.dependencies.gpu import acquire_gpu_slot

            async with acquire_gpu_slot():
                loop = asyncio.get_event_loop()
                results = await loop.run_in_executor(None, self._sync_batch, batch)

            elapsed_ms = (time.monotonic() - t_gpu_start) * 1000
            self._update_stats(len(batch), elapsed_ms)

            for entry, result in zip(batch, results):
                if not entry.future.done():
                    entry.future.set_result(result)

        except Exception as exc:
            logger.error(
                "Batch inference failed",
                batch_size=len(batch),
                error=str(exc),
            )
            for entry in batch:
                if not entry.future.done():
                    entry.future.set_exception(exc)

    def _sync_batch(self, batch: list[_QueueEntry]) -> list[BatchedDetectionResult]:
        """
        Synchronous batch inference — runs in a thread pool executor.

        Groups entries by ``(model_variant, use_tiled, model_path)`` and fires
        a single ``detect_batch()`` call per group, then reassembles in the
        original order.

        Tiled mode does not support true batching; those entries fall back to
        sequential single-image inference automatically.
        """
        from backend.api.v1.inference import _run_detection, _run_detection_batch

        t_start = time.monotonic()

        # Group entries by inference config key
        groups: dict[tuple, list[tuple[int, _QueueEntry]]] = {}
        for i, entry in enumerate(batch):
            key = (entry.model_variant, entry.use_tiled, entry.model_path)
            groups.setdefault(key, []).append((i, entry))

        result_map: dict[int, tuple[list, float]] = {}

        for (variant, use_tiled, model_path), group in groups.items():
            indices = [i for i, _ in group]
            entries = [e for _, e in group]

            if use_tiled or len(entries) == 1:
                # Tiled mode or singleton — sequential fallback (no true batch gain)
                for idx, entry in zip(indices, entries):
                    boxes, ms = _run_detection(
                        entry.image,
                        entry.conf_threshold,
                        entry.iou_threshold,
                        variant,
                        use_tiled,
                        model_path,
                    )
                    result_map[idx] = (boxes, ms)
            else:
                # True GPU batch — single YOLO forward pass for all images in group
                batch_out = _run_detection_batch(
                    images=[e.image for e in entries],
                    conf_threshold=entries[0].conf_threshold,
                    iou_threshold=entries[0].iou_threshold,
                    model_variant=variant,
                    model_path_override=model_path,
                )
                for idx, (boxes, ms) in zip(indices, batch_out):
                    result_map[idx] = (boxes, ms)

        results: list[BatchedDetectionResult] = []
        for i, entry in enumerate(batch):
            boxes, inf_ms = result_map.get(i, ([], 0.0))
            queue_wait = max(
                (time.monotonic() - entry.queued_at) * 1000 - inf_ms, 0.0
            )
            results.append(
                BatchedDetectionResult(
                    detections=boxes,
                    inference_time_ms=round(inf_ms, 2),
                    queue_wait_ms=round(queue_wait, 2),
                    batch_size=len(batch),
                )
            )

        return results

    def _update_stats(self, batch_size: int, elapsed_ms: float) -> None:
        self._total_batches += 1
        self._total_images += batch_size
        alpha = 0.2
        self._ema_batch_size = (
            alpha * batch_size + (1 - alpha) * self._ema_batch_size
            if self._ema_batch_size > 0 else float(batch_size)
        )
        self._ema_latency_ms = (
            alpha * elapsed_ms + (1 - alpha) * self._ema_latency_ms
            if self._ema_latency_ms > 0 else elapsed_ms
        )


# ─────────────────────────────────────────────────────────────────────────────
# Process-wide singleton
# ─────────────────────────────────────────────────────────────────────────────

_queue_instance: DetectionBatchQueue | None = None


def get_batch_queue() -> DetectionBatchQueue:
    """Return the process-wide DetectionBatchQueue singleton."""
    global _queue_instance
    if _queue_instance is None:
        try:
            from backend.config import get_settings
            s = get_settings()
            window_ms = float(getattr(s, "batch_queue_window_ms", 50.0))
            max_size = int(getattr(s, "batch_queue_max_size", 8))
        except Exception:
            window_ms, max_size = 50.0, 8

        _queue_instance = DetectionBatchQueue(window_ms=window_ms, max_size=max_size)
        logger.info(
            "DetectionBatchQueue initialised",
            window_ms=window_ms,
            max_size=max_size,
        )
    return _queue_instance


def reset_batch_queue() -> None:
    """Reset the singleton — for use in tests only."""
    global _queue_instance
    _queue_instance = None
