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

from fastapi import Request, Response, WebSocket
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)



def _constant_time_equal(a: str, b: str) -> bool:
    """Compare two strings in constant time to prevent timing attacks."""
    return hmac.compare_digest(a.encode(), b.encode())


AUTH_COOKIE = "ctip_token"


def _extract_token(request: Request | WebSocket) -> str | None:
    """
    Extract the API token from an HTTP request or a WebSocket handshake.

    Checks in priority order:
    1. Authorization: Bearer <token>
    2. X-API-Key: <token>
    3. ?api_key=<token> query parameter
    4. the ``ctip_token`` cookie set by the web UI (also sent with <img> requests and WebSocket upgrades)
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

    # 4. Cookie from the web UI
    cookie = request.cookies.get(AUTH_COOKIE, "").strip()
    if cookie:
        return cookie

    return None


SESSION_COOKIE = "ctip_session"

# reachable without being logged in (each checks what it needs itself)
PUBLIC_PREFIXES: tuple[str, ...] = (
    "/health", "/api/v1/system/health", "/docs", "/redoc", "/openapi.json", "/favicon.ico",
    "/api/v1/auth/status", "/api/v1/auth/session", "/api/v1/auth/login", "/api/v1/auth/logout",
    "/api/v1/auth/register", "/api/v1/compute/agent/", "/api/v1/compute/install/", "/ws/",
)

# admin only: (prefix, methods) - "*" = every method
ADMIN_RULES: tuple[tuple[str, str], ...] = (
    ("/api/v1/auth/users", "*"), ("/api/v1/auth/invites", "*"), ("/api/v1/containers", "*"),
    ("/api/v1/setup/status", ""),                         # "" = nobody special: members may read the setup state
    ("/api/v1/setup", "*"), ("/api/v1/settings", "POST PUT PATCH DELETE"),
    ("/api/v1/compute/enrollments", "*"), ("/api/v1/compute/workers", "POST PUT PATCH DELETE"),
    ("/api/v1/compute/jobs", "POST PUT PATCH DELETE"), ("/api/v1/compute/artifacts", "POST PUT PATCH DELETE"),
)


def admin_only(path: str, method: str) -> bool:
    for prefix, methods in ADMIN_RULES:
        if path == prefix or path.startswith(prefix + "/") or path.startswith(prefix + "?"):
            return methods == "*" or method in methods.split()
    return False


def resolve_mode(mode: str, api_token: str) -> str:
    mode = (mode or "auto").lower()
    if mode == "auto":
        return "token" if api_token.strip() else "off"
    if mode not in ("off", "token", "accounts"):
        raise ValueError(f"AUTH_MODE must be auto, off, token or accounts, not {mode!r}")
    return mode


def principal_from(conn: Request | WebSocket, api_token: str, mode: str):
    """The caller: API token (scripts; legacy UI cookie in token mode) or a login session (accounts mode)."""
    from backend.accounts.service import API_TOKEN_PRINCIPAL

    token = _extract_token(conn)
    expected = api_token.strip()
    if token and expected and _constant_time_equal(token, expected):
        if mode == "accounts" and conn.cookies.get(AUTH_COOKIE) == token and not conn.headers.get("Authorization"):
            pass                                          # accounts mode: the old token cookie is no login
        else:
            return API_TOKEN_PRINCIPAL
    if mode == "accounts":
        session = conn.cookies.get(SESSION_COOKIE, "").strip()
        if session:
            return _session_principal(session)
    return None


_cache: dict[str, tuple[float, object]] = {}


def _session_principal(token: str):
    """Session lookup with a 10 s cache (logout/revocation takes effect within 10 s)."""
    import time as _t

    from sqlmodel import Session as _Session

    from backend.accounts.service import principal_for_session
    from backend.database import engine

    hit = _cache.get(token)
    if hit and _t.time() - hit[0] < 10:
        return hit[1]
    with _Session(engine) as db:
        p = principal_for_session(db, token)
    if len(_cache) > 5000:
        _cache.clear()
    _cache[token] = (_t.time(), p)
    return p


def forget_session(token: str) -> None:
    _cache.pop(token, None)


def websocket_authorized(websocket: WebSocket, api_token: str, mode: str = "auto") -> bool:
    """True if auth is off or the handshake carries a valid token / login session (HTTP middleware skips WebSockets)."""
    mode = resolve_mode(mode, api_token)
    if mode == "off":
        return True
    return principal_from(websocket, api_token, mode) is not None


class APITokenMiddleware(BaseHTTPMiddleware):
    """
    Authentication for the whole API.

    off       nothing required (local use)
    token     the instance API token (header, query or the UI's token cookie)
    accounts  a login session (users with roles) - the API token still works for scripts and acts as admin

    Sets request.state.principal; admin-only routes (ADMIN_RULES) reject members with 403.
    """

    def __init__(self, app: ASGIApp, api_token: str = "", mode: str = "auto") -> None:
        super().__init__(app)
        self._token = api_token.strip()
        self._mode = resolve_mode(mode, self._token)
        self._enabled = self._mode != "off"
        logger.info("Authentication mode: %s", self._mode)

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    async def dispatch(self, request: Request, call_next) -> Response:
        import asyncio

        request.state.principal = None
        if not self._enabled or request.method == "OPTIONS":
            return await call_next(request)
        path = request.url.path
        public = any(path == p or path.startswith(p) for p in PUBLIC_PREFIXES)
        if self._mode == "token" and public:
            return await call_next(request)
        principal = await asyncio.to_thread(principal_from, request, self._token, self._mode)
        request.state.principal = principal
        if public:
            return await call_next(request)
        if principal is None:
            if self._mode == "token" and _extract_token(request) is not None:
                logger.warning("Auth: invalid token for %s %s (from %s)", request.method, path,
                               request.client.host if request.client else "unknown")
                return JSONResponse(status_code=403, content={"detail": "Invalid API token."})
            detail = ("Login required." if self._mode == "accounts" else
                      "Authentication required. Provide 'Authorization: Bearer <token>', 'X-API-Key: <token>', or '?api_key=<token>'.")
            return JSONResponse(status_code=401, content={"detail": detail}, headers={"WWW-Authenticate": "Bearer"})
        if admin_only(path, request.method) and not principal.is_admin:
            return JSONResponse(status_code=403, content={"detail": "Admins only."})
        return await call_next(request)
