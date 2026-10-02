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
from backend.middleware.auth import AUTH_COOKIE, _constant_time_equal

router = APIRouter(prefix="/auth", tags=["auth"])

_COOKIE_MAX_AGE = 30 * 24 * 3600


class SessionRequest(BaseModel):
    token: str = Field(min_length=1, max_length=512)


class AuthUser(BaseModel):
    username: str
    role: str
    display_name: str = ""
    must_change_password: bool = False


class AuthStatus(BaseModel):
    enabled: bool
    authenticated: bool
    mode: str = "off"                 # off | token | accounts
    user: AuthUser | None = None


@router.get("/status", response_model=AuthStatus)
async def auth_status(request: Request) -> AuthStatus:
    """Which login this instance wants, and who the caller is."""
    from backend.middleware.auth import principal_from, resolve_mode

    st = get_settings()
    mode = resolve_mode(st.auth_mode, st.api_token)
    if mode == "off":
        return AuthStatus(enabled=False, authenticated=True, mode=mode)
    p = getattr(request.state, "principal", None) or principal_from(request, st.api_token, mode)
    if p is None:
        return AuthStatus(enabled=True, authenticated=False, mode=mode)
    user = None
    if p.kind == "user":
        from sqlmodel import Session

        from backend.accounts.models import User
        from backend.database import engine

        with Session(engine) as db:
            u = db.get(User, p.user_id)
            if u is not None:
                user = AuthUser(username=u.username, role=u.role, display_name=u.display_name,
                                must_change_password=u.must_change_password)
    else:
        user = AuthUser(username=p.name, role=p.role)
    return AuthStatus(enabled=True, authenticated=True, mode=mode, user=user)


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
