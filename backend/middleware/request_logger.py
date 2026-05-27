"""
backend/middleware/request_logger.py — Structured request/response logging.
Pure ASGI middleware — WebSocket-safe.
"""

from __future__ import annotations

import time
from typing import Any, Callable

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger("trichome.requests")  # type: ignore[assignment]

SKIP_PATHS: frozenset[str] = frozenset({
    "/api/v1/system/health",
    "/api/v1/system/gpu",
    "/favicon.ico",
})

SKIP_PREFIXES: tuple[str, ...] = ("/ws/", "/static/", "/_next/")


class RequestLoggerMiddleware:
    """Pure ASGI request logger — passes WebSocket connections through untouched."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        # Pass non-HTTP scopes (websocket, lifespan) straight through
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        if path in SKIP_PATHS or any(path.startswith(p) for p in SKIP_PREFIXES):
            await self.app(scope, receive, send)
            return

        method = scope.get("method", "?")
        client = (scope.get("client") or ("?", 0))[0]
        start = time.perf_counter()
        status_code = 500

        async def send_wrapper(message: dict) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message.get("status", 500)
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
            latency_ms = (time.perf_counter() - start) * 1000
            level = "error" if status_code >= 500 else "warning" if status_code >= 400 else "info"
            getattr(logger, level)(
                f"{method} {path} → {status_code} [{latency_ms:.1f}ms] {client}"
            )
        except Exception as exc:
            latency_ms = (time.perf_counter() - start) * 1000
            logger.error(f"{method} {path} → EXCEPTION [{latency_ms:.1f}ms] {client}: {exc}")
            raise
