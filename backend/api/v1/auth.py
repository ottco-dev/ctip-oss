"""
Browser session for the single-user API token.

The web UI cannot add an Authorization header to <img> requests or WebSocket upgrades. It posts the token once to
``/auth/session``; the backend checks it and sets an HttpOnly, SameSite=Strict cookie that the auth middleware and
the WebSocket manager accept. Without ``API_TOKEN`` the endpoints report that auth is off.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, Field

from backend.config import get_settings
from backend.middleware.auth import AUTH_COOKIE, _constant_time_equal, _extract_token

router = APIRouter(prefix="/auth", tags=["auth"])

_COOKIE_MAX_AGE = 30 * 24 * 3600


class SessionRequest(BaseModel):
    token: str = Field(min_length=1, max_length=512)


class AuthStatus(BaseModel):
    enabled: bool
    authenticated: bool


@router.get("/status", response_model=AuthStatus)
async def auth_status(request: Request) -> AuthStatus:
    """Is token auth on, and does this browser/client already carry a valid token?"""
    expected = get_settings().api_token.strip()
    if not expected:
        return AuthStatus(enabled=False, authenticated=True)
    token = _extract_token(request)
    return AuthStatus(enabled=True, authenticated=bool(token) and _constant_time_equal(token, expected))


@router.post("/session", response_model=AuthStatus)
async def create_session(body: SessionRequest, request: Request, response: Response) -> AuthStatus:
    """Check the token and store it in an HttpOnly cookie for this browser."""
    expected = get_settings().api_token.strip()
    if not expected:
        return AuthStatus(enabled=False, authenticated=True)
    if not _constant_time_equal(body.token.strip(), expected):
        raise HTTPException(status_code=403, detail="Invalid API token.")
    response.set_cookie(
        AUTH_COOKIE, expected, max_age=_COOKIE_MAX_AGE, httponly=True, samesite="strict",
        secure=request.url.scheme == "https", path="/",
    )
    return AuthStatus(enabled=True, authenticated=True)


@router.delete("/session", response_model=AuthStatus)
async def end_session(response: Response) -> AuthStatus:
    response.delete_cookie(AUTH_COOKIE, path="/")
    return AuthStatus(enabled=bool(get_settings().api_token.strip()), authenticated=False)
