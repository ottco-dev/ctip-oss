"""
backend.websocket.router — WebSocket endpoints.

Endpoints:
    WS /ws/training  — Live training metrics stream
    WS /ws/jobs      — Background job status stream
    WS /ws/system    — GPU/RAM stats stream (2s interval)
    WS /ws/global    — General notifications
    WS /ws/logs      — Live log stream for process monitoring tray

Usage from frontend:
    const ws = new WebSocket('ws://localhost:8000/ws/training?client_id=browser-1');
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === 'training_metrics') {
            updateLossChart(data.epoch, data.metrics);
        }
    };
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query

from backend.websocket.manager import ws_manager
from backend.api.v1.system import _get_gpu_stats, _get_cpu_ram_stats
from shared.logging.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.websocket("/ws/training")
async def ws_training(
    websocket: WebSocket,
    client_id: str = Query(default=""),
) -> None:
    """
    WebSocket for live training metrics.

    Receives: epoch metrics every 2s during active training
    Format: {type: "training_metrics", epoch: N, metrics: {...}}
    """
    cid = client_id or str(uuid.uuid4())
    await ws_manager.connect(websocket, cid, topic="training")

    try:
        while True:
            # Keep connection alive — handle client messages
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                # Echo back any client messages
                await ws_manager.send_personal(cid, {
                    "type": "echo",
                    "data": data,
                })
            except asyncio.TimeoutError:
                # Send heartbeat
                await ws_manager.heartbeat(cid)

    except WebSocketDisconnect:
        logger.info("Training WS disconnected", client_id=cid)
    finally:
        await ws_manager.disconnect(cid, websocket)


@router.websocket("/ws/jobs")
async def ws_jobs(
    websocket: WebSocket,
    client_id: str = Query(default=""),
) -> None:
    """WebSocket for background job status updates."""
    cid = client_id or str(uuid.uuid4())
    await ws_manager.connect(websocket, cid, topic="jobs")

    try:
        while True:
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
            except asyncio.TimeoutError:
                await ws_manager.heartbeat(cid)
    except WebSocketDisconnect:
        pass
    finally:
        await ws_manager.disconnect(cid, websocket)


@router.websocket("/ws/system")
async def ws_system(
    websocket: WebSocket,
    client_id: str = Query(default=""),
    interval: float = Query(default=2.0, ge=1.0, le=60.0),
) -> None:
    """
    WebSocket for real-time GPU/RAM stats.

    Pushes stats every `interval` seconds (default: 2s).
    Frontend uses this for the GPU monitor widget.
    """
    cid = client_id or str(uuid.uuid4())
    await ws_manager.connect(websocket, cid, topic="system")

    async def push_stats_loop() -> None:
        while True:
            try:
                gpu = _get_gpu_stats()
                cpu_ram = _get_cpu_ram_stats()

                await ws_manager.send_personal(cid, {
                    "type": "gpu_stats",
                    "timestamp": time.time(),
                    "gpu": gpu,
                    "cpu_ram": cpu_ram,
                })
            except Exception as e:
                logger.debug("Stats push error", error=str(e))

            await asyncio.sleep(interval)

    # Start background stats pusher
    stats_task = asyncio.create_task(push_stats_loop())

    try:
        while True:
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
            except asyncio.TimeoutError:
                pass  # Stats are pushed proactively
    except WebSocketDisconnect:
        pass
    finally:
        stats_task.cancel()
        await ws_manager.disconnect(cid, websocket)


@router.websocket("/ws/global")
async def ws_global(
    websocket: WebSocket,
    client_id: str = Query(default=""),
) -> None:
    """General notification channel for alerts and system messages."""
    cid = client_id or str(uuid.uuid4())
    await ws_manager.connect(websocket, cid, topic="global")

    try:
        while True:
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
            except asyncio.TimeoutError:
                await ws_manager.heartbeat(cid)
    except WebSocketDisconnect:
        pass
    finally:
        await ws_manager.disconnect(cid, websocket)


# ── Log streaming for process monitoring tray ────────────────────

import logging as _logging
import collections
import time as _time

_log_buffer: collections.deque = collections.deque(maxlen=1000)

# ── Loguru sink (primary — captures all app logs) ─────────────────

def _loguru_ws_sink(message) -> None:
    """Loguru sink that feeds the WebSocket log ring buffer. Non-blocking."""
    try:
        record = message.record
        level_name = record["level"].name  # "INFO", "DEBUG", etc.
        _log_buffer.append({
            "ts": record["time"].timestamp(),
            "level": level_name,
            "logger": record["name"] or "root",
            "msg": record["message"],
        })
    except Exception:
        pass


# ── Stdlib handler (fallback — captures uvicorn/httpx/sqlalchemy logs) ────

class _WebSocketLogHandler(_logging.Handler):
    """Capture stdlib log records into the shared ring buffer."""

    def emit(self, record: _logging.LogRecord) -> None:
        try:
            _log_buffer.append({
                "ts": record.created,
                "level": record.levelname,
                "logger": record.name,
                "msg": self.format(record),
            })
        except Exception:
            pass


# ── Install handlers once (idempotent guard) ──────────────────────

_handlers_installed = False

def _install_log_handlers() -> None:
    global _handlers_installed
    if _handlers_installed:
        return
    _handlers_installed = True

    # Stdlib handler — catches uvicorn, httpx, sqlalchemy
    _ws_log_handler = _WebSocketLogHandler()
    _ws_log_handler.setFormatter(_logging.Formatter("%(name)s — %(message)s"))
    _ws_log_handler.setLevel(_logging.DEBUG)
    root = _logging.getLogger()
    if not any(isinstance(h, _WebSocketLogHandler) for h in root.handlers):
        root.addHandler(_ws_log_handler)
    root.setLevel(_logging.DEBUG)

    # Loguru sink — catches all app-level logs
    try:
        from loguru import logger as _loguru
        _loguru.add(_loguru_ws_sink, level="DEBUG", format="{message}", enqueue=False)
    except ImportError:
        pass

    # Seed the buffer with a startup entry
    _log_buffer.append({
        "ts": _time.time(),
        "level": "INFO",
        "logger": "trichome.ws.logs",
        "msg": "Log streaming active — WebSocket /ws/logs ready",
    })


_install_log_handlers()


@router.websocket("/ws/logs")
async def ws_logs(
    websocket: WebSocket,
    client_id: str = Query(default=""),
    level: str = Query(default="INFO"),
) -> None:
    """
    WebSocket for live log streaming.

    Sends buffered logs on connect, then streams new logs in real time.
    Filter: level=DEBUG|INFO|WARNING|ERROR
    """
    cid = client_id or str(uuid.uuid4())
    await ws_manager.connect(websocket, cid, topic="logs")

    # Send buffered log history immediately
    level_num = getattr(_logging, level.upper(), _logging.INFO)
    history = [
        entry for entry in list(_log_buffer)
        if _logging.getLevelName(entry["level"]) >= level_num
    ]
    await ws_manager.send_personal(cid, {
        "type": "log_history",
        "entries": history[-200:],
    })

    # Track by timestamp — robust against deque wrap-around
    last_ts = history[-1]["ts"] if history else 0.0

    async def push_new_logs() -> None:
        nonlocal last_ts
        while True:
            await asyncio.sleep(1.0)
            try:
                new_entries = [
                    e for e in list(_log_buffer)
                    if e["ts"] > last_ts
                    and _logging.getLevelName(e["level"]) >= level_num
                ]
                if new_entries:
                    last_ts = new_entries[-1]["ts"]
                    await ws_manager.send_personal(cid, {
                        "type": "log_batch",
                        "entries": new_entries,
                    })
            except Exception:
                pass

    log_push_task = asyncio.create_task(push_new_logs())

    try:
        while True:
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
            except asyncio.TimeoutError:
                await ws_manager.heartbeat(cid)
    except WebSocketDisconnect:
        pass
    finally:
        log_push_task.cancel()
        await ws_manager.disconnect(cid, websocket)
