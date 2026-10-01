"""
Background coroutines that are neither lost nor garbage-collected.

- ``spawn(coro)``: fire-and-forget inside a running event loop. asyncio keeps only a weak reference to tasks, so a
  bare ``asyncio.create_task(...)`` can be collected before it finishes; ``spawn`` holds the task until it is done and
  logs an exception instead of losing it.
- ``submit_from_thread(coro)``: schedule a coroutine from synchronous code - e.g. training callbacks that run in a
  worker thread - onto the application's event loop registered with ``set_app_loop`` (backend lifespan). Without a
  registered loop the coroutine runs to completion in the calling thread.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Coroutine
from typing import Any

from shared.logging.logger import get_logger

logger = get_logger(__name__)

_tasks: set[asyncio.Task[Any]] = set()
_app_loop: asyncio.AbstractEventLoop | None = None
_lock = threading.Lock()


def _done(task: asyncio.Task[Any]) -> None:
    _tasks.discard(task)
    if not task.cancelled() and task.exception() is not None:
        logger.error("Background task failed", task=task.get_name(), error=repr(task.exception()))


def spawn(coro: Coroutine[Any, Any, Any], *, name: str | None = None) -> asyncio.Task[Any]:
    """Start ``coro`` as a task of the running loop and keep it alive until it finishes."""
    task = asyncio.get_running_loop().create_task(coro, name=name)
    _tasks.add(task)
    task.add_done_callback(_done)
    return task


def set_app_loop(loop: asyncio.AbstractEventLoop | None) -> None:
    """Register (or clear) the application's event loop for ``submit_from_thread``."""
    global _app_loop
    with _lock:
        _app_loop = loop


def submit_from_thread(coro: Coroutine[Any, Any, Any]) -> None:
    """Run ``coro`` on the app loop from any thread; never blocks on a running loop."""
    try:
        running = asyncio.get_running_loop()
    except RuntimeError:
        running = None
    if running is not None:                      # called from inside a coroutine/loop thread
        spawn(coro)
        return
    with _lock:
        loop = _app_loop
    if loop is not None and loop.is_running() and not loop.is_closed():
        asyncio.run_coroutine_threadsafe(coro, loop)
        return
    asyncio.run(coro)                            # no application loop (CLI, tests): run here


def pending_tasks() -> int:
    """Number of background tasks still running (diagnostics, tests)."""
    return len(_tasks)
