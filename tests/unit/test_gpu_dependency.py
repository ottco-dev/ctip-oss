"""
tests.unit.test_gpu_dependency — Unit tests for backend.dependencies.gpu.

Tests:
  - Semaphore created lazily and shared across calls
  - acquire_gpu_slot context manager acquires and releases correctly
  - Concurrent calls are serialised (only one holder at a time)
  - acquire_gpu_slot raises TimeoutError if slot is not available within timeout
  - gpu_slot FastAPI dependency yields once and releases on exit
  - gpu_semaphore_status returns correct available/busy counts
  - wire_task_router_semaphore unifies semaphores (smoke test)
  - gpu_slot_or_429: 429 when slot busy and queue full (depth=0)
  - gpu_slot_or_429: waits when queue has room (depth=1)
  - gpu_slot_or_429: waiting count increments/decrements correctly
  - configure_gpu_rate_limit: sets module-level depth
  - gpu_semaphore_status: includes waiting_requests + max_queue_depth
"""

from __future__ import annotations

import asyncio

import pytest


# ---------------------------------------------------------------------------
# Reset module-level semaphore before each test to avoid cross-test pollution
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_gpu_semaphore():
    """Reset the module-level semaphore and counters so each test starts clean."""
    import backend.dependencies.gpu as gpu_mod
    gpu_mod._GPU_SEMAPHORE = None
    gpu_mod._waiting_count = 0
    gpu_mod._MAX_GPU_QUEUE_DEPTH = 0
    yield
    gpu_mod._GPU_SEMAPHORE = None
    gpu_mod._waiting_count = 0
    gpu_mod._MAX_GPU_QUEUE_DEPTH = 0


# ---------------------------------------------------------------------------
# _get_semaphore — lazy creation
# ---------------------------------------------------------------------------

class TestGetSemaphore:

    def test_semaphore_created_on_first_call(self):
        from backend.dependencies.gpu import _get_semaphore
        sem = _get_semaphore()
        assert sem is not None

    def test_semaphore_is_singleton(self):
        from backend.dependencies.gpu import _get_semaphore
        sem1 = _get_semaphore()
        sem2 = _get_semaphore()
        assert sem1 is sem2

    def test_semaphore_starts_with_one_slot(self):
        from backend.dependencies.gpu import _get_semaphore
        sem = _get_semaphore()
        # asyncio.Semaphore._value = remaining slots
        assert sem._value == 1


# ---------------------------------------------------------------------------
# acquire_gpu_slot context manager
# ---------------------------------------------------------------------------

class TestAcquireGpuSlot:

    @pytest.mark.asyncio
    async def test_slot_acquired_and_released(self):
        from backend.dependencies.gpu import acquire_gpu_slot, _get_semaphore
        sem = _get_semaphore()

        assert sem._value == 1
        async with acquire_gpu_slot():
            assert sem._value == 0     # held
        assert sem._value == 1         # released

    @pytest.mark.asyncio
    async def test_slot_released_on_exception(self):
        from backend.dependencies.gpu import acquire_gpu_slot, _get_semaphore
        sem = _get_semaphore()

        with pytest.raises(ValueError):
            async with acquire_gpu_slot():
                raise ValueError("intentional error")

        assert sem._value == 1         # must be released even on exception

    @pytest.mark.asyncio
    async def test_concurrent_calls_serialised(self):
        from backend.dependencies.gpu import acquire_gpu_slot

        order: list[str] = []

        async def task_a():
            async with acquire_gpu_slot():
                order.append("a_enter")
                await asyncio.sleep(0.05)
                order.append("a_exit")

        async def task_b():
            # Give A time to acquire first
            await asyncio.sleep(0.01)
            async with acquire_gpu_slot():
                order.append("b_enter")
                await asyncio.sleep(0.01)
                order.append("b_exit")

        await asyncio.gather(task_a(), task_b())

        # B must not enter until A has exited
        a_exit_idx = order.index("a_exit")
        b_enter_idx = order.index("b_enter")
        assert b_enter_idx > a_exit_idx, f"B entered before A exited: {order}"

    @pytest.mark.asyncio
    async def test_timeout_raises_on_held_slot(self):
        from backend.dependencies.gpu import acquire_gpu_slot

        # Hold the slot in the background
        held = asyncio.Event()
        released = asyncio.Event()

        async def holder():
            async with acquire_gpu_slot():
                held.set()
                await released.wait()

        holder_task = asyncio.create_task(holder())
        await held.wait()   # slot is now held

        with pytest.raises(asyncio.TimeoutError):
            async with acquire_gpu_slot(timeout=0.05):
                pass

        released.set()
        await holder_task


# ---------------------------------------------------------------------------
# gpu_slot FastAPI dependency
# ---------------------------------------------------------------------------

class TestGpuSlotDependency:

    @pytest.mark.asyncio
    async def test_dependency_yields_and_releases(self):
        from backend.dependencies.gpu import gpu_slot, _get_semaphore
        sem = _get_semaphore()

        # Simulate FastAPI calling the dependency as an async generator
        gen = gpu_slot()
        await gen.__anext__()          # acquire
        assert sem._value == 0        # slot held

        with pytest.raises(StopAsyncIteration):
            await gen.__anext__()      # release

        assert sem._value == 1        # slot freed


# ---------------------------------------------------------------------------
# gpu_semaphore_status
# ---------------------------------------------------------------------------

class TestGpuSemaphoreStatus:

    @pytest.mark.asyncio
    async def test_idle_status(self):
        from backend.dependencies.gpu import gpu_semaphore_status
        status = gpu_semaphore_status()
        assert status["available_slots"] == 1
        assert status["busy"] is False
        assert status["max_concurrent"] == 1

    @pytest.mark.asyncio
    async def test_busy_status_while_held(self):
        from backend.dependencies.gpu import acquire_gpu_slot, gpu_semaphore_status

        async with acquire_gpu_slot():
            status = gpu_semaphore_status()
            assert status["available_slots"] == 0
            assert status["busy"] is True

        status = gpu_semaphore_status()
        assert status["busy"] is False


# ---------------------------------------------------------------------------
# wire_task_router_semaphore (smoke test — no actual task_router in unit env)
# ---------------------------------------------------------------------------

class TestWireTaskRouterSemaphore:

    def test_wire_does_not_raise_without_task_router(self, monkeypatch):
        """wire_task_router_semaphore must not raise even if task_router is absent."""
        import sys
        import backend.dependencies.gpu as gpu_mod

        # Temporarily hide task_router from sys.modules
        original = sys.modules.pop("backend.tasks.task_router", None)
        try:
            # Should silently log warning, not raise
            gpu_mod.wire_task_router_semaphore()
        finally:
            if original is not None:
                sys.modules["backend.tasks.task_router"] = original

    def test_wire_unifies_semaphore_with_task_router(self):
        """After wiring, task_router and dependency share the same semaphore."""
        import backend.dependencies.gpu as gpu_mod

        try:
            from backend.tasks.task_router import task_router

            gpu_mod.wire_task_router_semaphore()

            # After wiring, both must reference the same semaphore
            assert (
                gpu_mod._GPU_SEMAPHORE is task_router._gpu_semaphore
            ), "Semaphores not unified after wire"

        except ImportError:
            pytest.skip("task_router not available in this test environment")


# ---------------------------------------------------------------------------
# gpu_slot_or_429 — rate limiting behaviour
# ---------------------------------------------------------------------------

class TestGpuSlotOr429:

    @pytest.mark.asyncio
    async def test_429_when_slot_busy_and_depth_zero(self):
        """With depth=0, any busy slot triggers 429."""
        from fastapi import HTTPException
        from backend.dependencies.gpu import acquire_gpu_slot, gpu_slot_or_429

        held = asyncio.Event()
        released = asyncio.Event()

        async def holder():
            async with acquire_gpu_slot():
                held.set()
                await released.wait()

        task = asyncio.create_task(holder())
        await held.wait()  # slot now held

        with pytest.raises(HTTPException) as exc_info:
            gen = gpu_slot_or_429(max_queue_depth=0)
            await gen.__anext__()

        assert exc_info.value.status_code == 429
        assert "Retry-After" in exc_info.value.headers

        released.set()
        await task

    @pytest.mark.asyncio
    async def test_no_429_when_slot_free(self):
        """With depth=0, a free slot is acquired normally."""
        from backend.dependencies.gpu import gpu_slot_or_429, _get_semaphore

        sem = _get_semaphore()
        gen = gpu_slot_or_429(max_queue_depth=0)
        await gen.__anext__()       # should acquire
        assert sem._value == 0      # slot held

        with pytest.raises(StopAsyncIteration):
            await gen.__anext__()   # releases

        assert sem._value == 1      # slot freed

    @pytest.mark.asyncio
    async def test_queues_when_depth_allows(self):
        """With depth=1, one request may queue rather than get 429."""
        from backend.dependencies.gpu import acquire_gpu_slot, gpu_slot_or_429
        import backend.dependencies.gpu as gpu_mod

        held = asyncio.Event()
        released = asyncio.Event()
        queued_done = asyncio.Event()

        async def holder():
            async with acquire_gpu_slot():
                held.set()
                await released.wait()

        async def waiter():
            gen = gpu_slot_or_429(max_queue_depth=1)
            await gen.__anext__()
            queued_done.set()
            with pytest.raises(StopAsyncIteration):
                await gen.__anext__()

        holder_task = asyncio.create_task(holder())
        await held.wait()

        # At this point slot is busy; waiter should queue (not 429)
        waiter_task = asyncio.create_task(waiter())
        await asyncio.sleep(0.02)

        # waiting_count should be 1
        assert gpu_mod._waiting_count == 1

        released.set()
        await asyncio.wait_for(queued_done.wait(), timeout=1.0)
        await holder_task
        await waiter_task

        # After both complete, waiting_count returns to 0
        assert gpu_mod._waiting_count == 0

    @pytest.mark.asyncio
    async def test_429_when_depth_exceeded(self):
        """With depth=1, a second waiter gets 429."""
        from fastapi import HTTPException
        from backend.dependencies.gpu import acquire_gpu_slot, gpu_slot_or_429
        import backend.dependencies.gpu as gpu_mod

        held = asyncio.Event()
        released = asyncio.Event()

        async def holder():
            async with acquire_gpu_slot():
                held.set()
                await released.wait()

        holder_task = asyncio.create_task(holder())
        await held.wait()

        # Manually bump waiting_count to simulate 1 waiter already queued
        gpu_mod._waiting_count = 1

        # Now a new request with depth=1 should be rejected (1 waiter already == depth)
        with pytest.raises(HTTPException) as exc_info:
            gen = gpu_slot_or_429(max_queue_depth=1)
            await gen.__anext__()

        assert exc_info.value.status_code == 429

        released.set()
        await holder_task

    @pytest.mark.asyncio
    async def test_waiting_count_decremented_on_error(self):
        """If acquire raises, _waiting_count must return to its original value."""
        from backend.dependencies.gpu import gpu_slot_or_429, _get_semaphore
        import backend.dependencies.gpu as gpu_mod

        # Fill the slot manually by acquiring the semaphore without a holder
        sem = _get_semaphore()
        await sem.acquire()  # slot now held (manual)

        initial_count = gpu_mod._waiting_count

        # Attempt to acquire with depth=1 and very short timeout → TimeoutError → 503
        from fastapi import HTTPException
        try:
            gen = gpu_slot_or_429(max_queue_depth=1, timeout=0.05)
            await gen.__anext__()
        except HTTPException:
            pass  # expected 503
        finally:
            sem.release()

        # waiting_count must be back to original
        assert gpu_mod._waiting_count == initial_count


# ---------------------------------------------------------------------------
# configure_gpu_rate_limit
# ---------------------------------------------------------------------------

class TestConfigureGpuRateLimit:

    def test_sets_max_queue_depth(self):
        from backend.dependencies.gpu import configure_gpu_rate_limit
        import backend.dependencies.gpu as gpu_mod

        configure_gpu_rate_limit(3)
        assert gpu_mod._MAX_GPU_QUEUE_DEPTH == 3

    def test_zero_allowed(self):
        from backend.dependencies.gpu import configure_gpu_rate_limit
        import backend.dependencies.gpu as gpu_mod

        configure_gpu_rate_limit(0)
        assert gpu_mod._MAX_GPU_QUEUE_DEPTH == 0

    def test_negative_raises(self):
        from backend.dependencies.gpu import configure_gpu_rate_limit
        with pytest.raises(ValueError, match="must be >= 0"):
            configure_gpu_rate_limit(-1)


# ---------------------------------------------------------------------------
# gpu_semaphore_status extended fields
# ---------------------------------------------------------------------------

class TestGpuSemaphoreStatusExtended:

    def test_includes_waiting_requests(self):
        from backend.dependencies.gpu import gpu_semaphore_status
        import backend.dependencies.gpu as gpu_mod

        gpu_mod._waiting_count = 2
        status = gpu_semaphore_status()
        assert status["waiting_requests"] == 2

    def test_includes_max_queue_depth(self):
        from backend.dependencies.gpu import gpu_semaphore_status, configure_gpu_rate_limit
        configure_gpu_rate_limit(5)
        status = gpu_semaphore_status()
        assert status["max_queue_depth"] == 5
