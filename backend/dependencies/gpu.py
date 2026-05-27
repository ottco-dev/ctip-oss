"""
backend.dependencies.gpu — GPU semaphore FastAPI dependency.

Design
------
The RTX 4060 has 8 GB VRAM. Running multiple GPU inference or training
tasks concurrently causes OOM errors and non-deterministic latency spikes.

This module exposes a *single shared* asyncio.Semaphore(1) that all
GPU-capable endpoints must acquire before executing model inference.

The semaphore is the same object used by backend.tasks.task_router for
background training jobs, ensuring:

  1. Only ONE GPU task runs at any moment across the entire API.
  2. Training and synchronous inference cannot overlap.
  3. Direct REST inference calls and background jobs share the budget.

Usage in router endpoints
--------------------------
Method 1 — FastAPI dependency (recommended for stateless endpoints):

    from fastapi import Depends
    from backend.dependencies.gpu import gpu_slot

    @router.post("/analyze/crop")
    async def analyze_crop(
        ...,
        _slot: None = Depends(gpu_slot),
    ) -> AnalysisResponse:
        # semaphore held for duration of this handler
        result = model(image)
        return result

Method 2 — Manual context manager (when semaphore scope must be explicit):

    from backend.dependencies.gpu import acquire_gpu_slot

    async def my_handler():
        async with acquire_gpu_slot():
            result = model(image)
        return result

Bypass / CPU-only paths
------------------------
Endpoints that are provably CPU-only (e.g. scale-bar calibration math,
density map computation) should NOT use this dependency — it adds ~0 overhead
when not contested but it wrongly serialises CPU work.

Add `_slot: None = Depends(gpu_slot)` ONLY to endpoints that:
  - load a PyTorch / ONNX model on GPU at init-time, OR
  - call .predict() / .forward() / model(tensor), OR
  - run SAM2, YOLO, VLM, or similar GPU-resident model inference.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import HTTPException, status
from shared.logging.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Shared GPU semaphore
# ---------------------------------------------------------------------------
# Initialized once at module import time. All routers that import this module
# share the same semaphore instance — import-time singleton, no FastAPI state.
#
# max_concurrent=1: serialize all GPU tasks (RTX 4060, 8 GB VRAM budget).
# Increase to 2 only if you switch to a 16+ GB card and profile carefully.
# ---------------------------------------------------------------------------
_GPU_SEMAPHORE: asyncio.Semaphore | None = None
_MAX_CONCURRENT_GPU = 1

# ---------------------------------------------------------------------------
# Waiting-request counter (for rate-limiting)
# ---------------------------------------------------------------------------
# Tracks how many coroutines are currently *waiting* for the semaphore
# (not yet holding it). Used by gpu_slot_or_429() to reject excess requests.
_waiting_count: int = 0

# Maximum number of requests allowed to queue before returning 429.
# Default=0: if the slot is busy, ALL new requests get 429 (no queuing).
# Set to N>0 to allow N requests to wait in line before 429 fires.
_MAX_GPU_QUEUE_DEPTH: int = 0


def _get_semaphore() -> asyncio.Semaphore:
    """
    Lazily create the GPU semaphore the first time it is needed.

    Creating asyncio.Semaphore at module level (outside a running loop) can
    raise DeprecationWarning in Python ≥3.10 if no event loop is running yet.
    Lazy init avoids this while still giving a singleton within a process.
    """
    global _GPU_SEMAPHORE
    if _GPU_SEMAPHORE is None:
        _GPU_SEMAPHORE = asyncio.Semaphore(_MAX_CONCURRENT_GPU)
        logger.debug(
            "GPU semaphore created",
            max_concurrent=_MAX_CONCURRENT_GPU,
        )
    return _GPU_SEMAPHORE


# ---------------------------------------------------------------------------
# Sync the semaphore with task_router on first use
# ---------------------------------------------------------------------------
# The task_router creates its own Semaphore. We wire ours together so that
# both the REST endpoints and the background task queue contend on the SAME
# semaphore object, guaranteeing true mutual exclusion.
#
# Called once from backend/main.py lifespan after task_router is initialised.
# ---------------------------------------------------------------------------

def wire_task_router_semaphore() -> None:
    """
    Replace task_router's internal semaphore with the shared one from this
    module (or vice-versa — whichever is already created wins).

    Call once from the FastAPI lifespan / startup handler.
    """
    global _GPU_SEMAPHORE
    try:
        from backend.tasks.task_router import task_router  # noqa: PLC0415

        if _GPU_SEMAPHORE is None:
            # Use the task_router's semaphore as the canonical one
            _GPU_SEMAPHORE = task_router._gpu_semaphore
            logger.info("GPU semaphore wired from task_router")
        else:
            # Our semaphore was created first; hand it to task_router
            task_router._gpu_semaphore = _GPU_SEMAPHORE
            logger.info("GPU semaphore handed to task_router")

    except Exception as exc:  # pragma: no cover
        logger.warning("Could not wire GPU semaphore to task_router", error=str(exc))


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------

@asynccontextmanager
async def acquire_gpu_slot(
    timeout: float | None = None,
) -> AsyncGenerator[None, None]:
    """
    Async context manager that acquires the GPU semaphore.

    Args:
        timeout: Maximum seconds to wait. None = wait forever.
                 Raises asyncio.TimeoutError if the slot is not acquired.

    Usage::

        async with acquire_gpu_slot(timeout=30):
            result = model.predict(image)
    """
    sem = _get_semaphore()
    acquire_coro = sem.acquire()

    if timeout is not None:
        try:
            await asyncio.wait_for(acquire_coro, timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning("GPU slot acquire timeout", timeout_s=timeout)
            raise
    else:
        await acquire_coro

    logger.debug("GPU slot acquired")
    try:
        yield
    finally:
        sem.release()
        logger.debug("GPU slot released")


# ---------------------------------------------------------------------------
# FastAPI dependency
# ---------------------------------------------------------------------------

async def gpu_slot(timeout: float | None = None) -> None:
    """
    FastAPI dependency: acquire GPU semaphore for the duration of the request.

    Inject via ``_slot: None = Depends(gpu_slot)`` in route signatures.
    The semaphore is released automatically when the handler returns,
    including on exception.

    Example::

        @router.post("/analyze/crop")
        async def analyze_crop(
            file: UploadFile,
            _slot: None = Depends(gpu_slot),
        ) -> AnalysisResponse: ...
    """
    async with acquire_gpu_slot(timeout=timeout):
        yield


# ---------------------------------------------------------------------------
# Rate-limited GPU dependency  (429 if queue is full)
# ---------------------------------------------------------------------------

async def gpu_slot_or_429(
    max_queue_depth: int | None = None,
    timeout: float | None = None,
) -> None:
    """
    FastAPI dependency: acquire GPU semaphore OR return HTTP 429 if the slot
    is already contested beyond ``max_queue_depth``.

    Behaviour:
      - If the GPU slot is free → acquire immediately, yield, release on exit.
      - If the slot is held AND current waiters < max_queue_depth → wait.
      - If the slot is held AND current waiters >= max_queue_depth → HTTP 429.

    Use this dependency on latency-sensitive inference endpoints where it is
    better to fast-fail the client than to pile up indefinite server queues.

    Args:
        max_queue_depth: Maximum number of waiting coroutines before 429.
            Defaults to ``_MAX_GPU_QUEUE_DEPTH`` (module-level config, default 0).
            Set to 0 to reject immediately when the slot is busy.
        timeout: Maximum wait time in seconds (applies when queuing is allowed).

    Example::

        @router.post("/inference/detect")
        async def detect(
            file: UploadFile,
            _slot: None = Depends(gpu_slot_or_429),
        ) -> DetectionResponse: ...

    To allow up to 2 requests to queue:

        from functools import partial
        gpu_slot_2q = partial(gpu_slot_or_429, max_queue_depth=2)

        @router.post("/analyze")
        async def analyze(_slot: None = Depends(gpu_slot_2q)): ...
    """
    global _waiting_count

    depth = max_queue_depth if max_queue_depth is not None else _MAX_GPU_QUEUE_DEPTH
    sem = _get_semaphore()
    available = sem._value  # type: ignore[attr-defined]

    # Slot busy → check queue depth
    if available == 0 and _waiting_count >= depth:
        logger.warning(
            "GPU rate limit: slot busy, queue full",
            waiting=_waiting_count,
            max_queue=depth,
        )
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "error": "GPU slot busy",
                "message": (
                    "All GPU inference slots are occupied. "
                    f"Currently {_waiting_count} request(s) queued. "
                    "Retry after the current job completes."
                ),
                "retry_after_s": 5,
            },
            headers={"Retry-After": "5"},
        )

    # Slot available or queue has room → wait for slot
    _waiting_count += 1
    try:
        async with acquire_gpu_slot(timeout=timeout):
            _waiting_count -= 1
            logger.debug("GPU slot acquired (rate-limited path)", waiting=_waiting_count)
            yield
    except asyncio.TimeoutError:
        _waiting_count -= 1
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "GPU slot timeout",
                "message": "GPU slot not available within the timeout period.",
            },
        )
    except Exception:
        _waiting_count -= 1
        raise


def configure_gpu_rate_limit(max_queue_depth: int) -> None:
    """
    Set the module-level GPU queue depth for ``gpu_slot_or_429``.

    Call once from application startup (e.g., lifespan handler or settings).
    ``max_queue_depth=0`` (default) → no queueing: 429 whenever slot is busy.
    ``max_queue_depth=2`` → up to 2 requests may wait before 429.

    Args:
        max_queue_depth: Non-negative integer.
    """
    global _MAX_GPU_QUEUE_DEPTH
    if max_queue_depth < 0:
        raise ValueError(f"max_queue_depth must be >= 0, got {max_queue_depth}")
    _MAX_GPU_QUEUE_DEPTH = max_queue_depth
    logger.info("GPU rate limit configured", max_queue_depth=max_queue_depth)


# ---------------------------------------------------------------------------
# Status helper (for /system/health reporting)
# ---------------------------------------------------------------------------

def gpu_semaphore_status() -> dict:
    """Return a dict describing the current GPU semaphore state."""
    sem = _get_semaphore()
    # asyncio.Semaphore._value is the number of permits currently available.
    available = sem._value  # type: ignore[attr-defined]
    return {
        "max_concurrent": _MAX_CONCURRENT_GPU,
        "available_slots": available,
        "busy": available == 0,
        "waiting_requests": _waiting_count,
        "max_queue_depth": _MAX_GPU_QUEUE_DEPTH,
    }
