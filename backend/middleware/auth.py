"""
backend.middleware.auth — Single-user API token authentication.

Design:
  - Single-user system (local lab deployment, not multi-tenant SaaS)
  - Token stored in settings.api_token (env var: API_TOKEN)
  - If API_TOKEN is empty/not set → auth is DISABLED (development mode)
  - If API_TOKEN is set → every request must include:
      Authorization: Bearer <token>
    OR
      X-API-Key: <token>
    OR
      ?api_key=<token>  (query param, for WebSocket upgrade compatibility)

Token validation:
  - Constant-time HMAC comparison (hmac.compare_digest) to prevent timing attacks
  - 401 on missing token, 403 on invalid token

Excluded paths (no auth required):
  - /health
  - /api/v1/system/health
  - /docs, /redoc, /openapi.json (API documentation)
  - /ws/* (WebSocket upgrades handled by the WS router itself)

Usage in main.py:
    from backend.middleware.auth import APITokenMiddleware
    app.add_middleware(APITokenMiddleware)

Environment:
    API_TOKEN=your-secret-token-here   # Set to enable auth
    API_TOKEN=                          # Empty → auth disabled (dev mode)
"""

from __future__ import annotations

import hmac
import logging
from typing import Sequence

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)

# Paths that never require authentication
_AUTH_EXCLUDED_PREFIXES: tuple[str, ...] = (
    "/health",
    "/api/v1/system/health",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/ws/",
    "/favicon.ico",
)


def _constant_time_equal(a: str, b: str) -> bool:
    """Compare two strings in constant time to prevent timing attacks."""
    return hmac.compare_digest(a.encode(), b.encode())


def _extract_token(request: Request) -> str | None:
    """
    Extract the API token from the request.

    Checks in priority order:
    1. Authorization: Bearer <token>
    2. X-API-Key: <token>
    3. ?api_key=<token> query parameter
    """
    # 1. Authorization header (Bearer scheme)
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:].strip()
        return token if token else None  # "Bearer " with nothing after → None

    # 2. X-API-Key header
    x_api_key = request.headers.get("X-API-Key", "").strip()
    if x_api_key:
        return x_api_key

    # 3. Query parameter (for WebSocket and simple browser access)
    api_key_param = request.query_params.get("api_key", "").strip()
    if api_key_param:
        return api_key_param

    return None


class APITokenMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for single-user API token authentication.

    Transparently disabled if no token is configured.
    """

    def __init__(self, app: ASGIApp, api_token: str = "") -> None:
        super().__init__(app)
        self._token = api_token.strip()
        self._enabled = bool(self._token)

        if self._enabled:
            logger.info(
                "API token authentication ENABLED "
                "(token length=%d, first 4 chars: %s…)",
                len(self._token),
                self._token[:4] if len(self._token) >= 4 else "****",
            )
        else:
            logger.warning(
                "API token authentication DISABLED — "
                "set API_TOKEN env var to enable (production recommended)."
            )

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    async def dispatch(self, request: Request, call_next) -> Response:
        # Auth disabled → pass through
        if not self._enabled:
            return await call_next(request)

        # Excluded paths → pass through
        path = request.url.path
        for prefix in _AUTH_EXCLUDED_PREFIXES:
            if path == prefix or path.startswith(prefix):
                return await call_next(request)

        # OPTIONS preflight → pass through (CORS handled by CORSMiddleware)
        if request.method == "OPTIONS":
            return await call_next(request)

        # Extract and validate token
        token = _extract_token(request)

        if token is None:
            logger.debug("Auth: missing token for %s %s", request.method, path)
            return JSONResponse(
                status_code=401,
                content={
                    "detail": "Authentication required. "
                              "Provide 'Authorization: Bearer <token>', "
                              "'X-API-Key: <token>', or '?api_key=<token>'."
                },
                headers={"WWW-Authenticate": "Bearer"},
            )

        if not _constant_time_equal(token, self._token):
            logger.warning(
                "Auth: invalid token for %s %s (from %s)",
                request.method,
                path,
                request.client.host if request.client else "unknown",
            )
            return JSONResponse(
                status_code=403,
                content={"detail": "Invalid API token."},
            )

        return await call_next(request)
