"""
backend/middleware/gpu_guard.py — VRAM-aware request throttling middleware.

Rejects GPU-bound requests when available VRAM is below the configured threshold.
Passes WebSocket connections through unconditionally (ASGI-safe).
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable

logger = logging.getLogger("trichome.gpu_guard")

GPU_ROUTES: frozenset[str] = frozenset({
    "/api/v1/inference/detect",
    "/api/v1/inference/maturity",
    "/api/v1/training/start",
    "/api/v1/annotation/auto-label",
    "/api/v1/video/analyze",
})

MIN_FREE_VRAM_BYTES = 512 * 1024 * 1024  # 512 MB
CACHE_TTL = 5.0

_vram_cache: dict[str, Any] = {"free": None, "ts": 0.0}
_cache_lock: asyncio.Lock | None = None


def _get_cache_lock() -> asyncio.Lock:
    global _cache_lock
    if _cache_lock is None:
        _cache_lock = asyncio.Lock()
    return _cache_lock


async def _get_free_vram() -> int | None:
    now = time.monotonic()
    lock = _get_cache_lock()
    async with lock:
        if _vram_cache["free"] is not None and (now - _vram_cache["ts"]) < CACHE_TTL:
            return _vram_cache["free"]
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        free, _ = torch.cuda.mem_get_info(0)
        async with lock:
            _vram_cache["free"] = free
            _vram_cache["ts"] = now
        return free
    except Exception:
        return None


class GpuGuardMiddleware:
    """Pure ASGI middleware — WebSocket-safe."""

    def __init__(self, app: Any, min_free_vram_bytes: int = MIN_FREE_VRAM_BYTES) -> None:
        self.app = app
        self.min_free_vram_bytes = min_free_vram_bytes

    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        # Always pass WebSocket, lifespan, and non-GPU routes through
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        query = scope.get("query_string", b"").decode()
        is_gpu_route = path in GPU_ROUTES or "gpu=1" in query

        if not is_gpu_route:
            await self.app(scope, receive, send)
            return

        free_vram = await _get_free_vram()
        if free_vram is not None and free_vram < self.min_free_vram_bytes:
            free_mb = free_vram / (1024 * 1024)
            req_mb = self.min_free_vram_bytes / (1024 * 1024)
            logger.warning("GPU guard triggered: %.0f MB free < %.0f MB required", free_mb, req_mb)
            import json
            body = json.dumps({
                "detail": f"Insufficient GPU memory. Free: {free_mb:.0f} MB, Required: {req_mb:.0f} MB.",
                "free_vram_mb": round(free_mb, 1),
                "required_vram_mb": round(req_mb, 1),
            }).encode()
            await send({"type": "http.response.start", "status": 503,
                        "headers": [(b"content-type", b"application/json"),
                                    (b"retry-after", b"30")]})
            await send({"type": "http.response.body", "body": body})
            return

        await self.app(scope, receive, send)
