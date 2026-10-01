"""Background tasks are kept alive and errors are logged; coroutines reach the app loop from worker threads."""

from __future__ import annotations

import asyncio
import gc
import threading

from shared import async_utils
from shared.async_utils import pending_tasks, set_app_loop, spawn, submit_from_thread


async def test_spawned_task_survives_garbage_collection():
    done = asyncio.Event()

    async def work():
        await asyncio.sleep(0.01)
        done.set()

    spawn(work(), name="gc-test")
    gc.collect()
    await asyncio.wait_for(done.wait(), 1)
    await asyncio.sleep(0)
    assert pending_tasks() == 0


async def test_failing_task_is_logged_not_lost(monkeypatch):
    errors = []
    monkeypatch.setattr(async_utils.logger, "error", lambda msg, **kw: errors.append((msg, kw)))

    async def boom():
        raise ValueError("broken")

    task = spawn(boom(), name="boom")
    await asyncio.sleep(0.01)
    assert task.done()
    assert errors and "broken" in errors[0][1]["error"]


async def test_submit_from_worker_thread_runs_on_app_loop():
    loop = asyncio.get_running_loop()
    set_app_loop(loop)
    seen = []

    async def record():
        seen.append(threading.current_thread() is threading.main_thread())

    try:
        await asyncio.to_thread(submit_from_thread, record())
        for _ in range(50):
            if seen:
                break
            await asyncio.sleep(0.01)
    finally:
        set_app_loop(None)
    assert seen == [True]                   # ran on the loop's (main) thread, not in the worker


def test_submit_without_app_loop_runs_in_place():
    set_app_loop(None)
    seen = []

    async def record():
        seen.append(1)

    submit_from_thread(record())
    assert seen == [1]
