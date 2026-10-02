"""Accounts API (AUTH_MODE=accounts): login/logout, own password, users and invitations (admins)."""

from __future__ import annotations

import secrets

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field
from sqlmodel import Session, col, select

from backend.accounts import service
from backend.accounts.models import Invite, User
from backend.accounts.service import AccountError, Principal
from backend.config import get_settings
from backend.database import get_session
from backend.middleware.auth import SESSION_COOKIE, forget_session, resolve_mode

router = APIRouter(prefix="/auth", tags=["accounts"])


def _accounts_mode() -> None:
    st = get_settings()
    if resolve_mode(st.auth_mode, st.api_token) != "accounts":
        raise HTTPException(400, "user accounts are off on this instance (AUTH_MODE=accounts)")


def _principal(request: Request) -> Principal:
    p = getattr(request.state, "principal", None)
    if p is None:
        raise HTTPException(401, "Login required.")
    return p


def _admin(request: Request) -> Principal:
    p = _principal(request)
    if not p.is_admin:
        raise HTTPException(403, "Admins only.")
    return p


def _raise(e: AccountError):
    raise HTTPException(e.status, e.detail) from e


def _set_cookie(request: Request, response: Response, token: str) -> None:
    secure = request.headers.get("x-forwarded-proto", request.url.scheme).split(",")[0].strip() == "https"
    response.set_cookie(SESSION_COOKIE, token, max_age=service.SESSION_DAYS * 86400, httponly=True, samesite="strict",
                        secure=secure, path="/")


def _ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for", "")
    return (fwd.split(",")[0].strip() if fwd else (request.client.host if request.client else ""))[:64]


def _temp_password() -> str:
    return secrets.token_urlsafe(12)


class LoginIn(BaseModel):
    username: str = Field(min_length=1, max_length=40)
    password: str = Field(min_length=1, max_length=200)


@router.post("/login", dependencies=[Depends(_accounts_mode)])
def login(body: LoginIn, request: Request, response: Response, db: Session = Depends(get_session)) -> dict:
    try:
        u, token = service.login(db, body.username, body.password, _ip(request), request.headers.get("user-agent", ""))
    except AccountError as e:
        _raise(e)
    _set_cookie(request, response, token)
    return service.user_view(u)


@router.post("/logout")
def logout(request: Request, response: Response, db: Session = Depends(get_session)) -> dict:
    token = request.cookies.get(SESSION_COOKIE, "")
    if token:
        service.logout(db, token)
        forget_session(token)
    response.delete_cookie(SESSION_COOKIE, path="/")
    return {"ok": True}


@router.get("/me", dependencies=[Depends(_accounts_mode)])
def me(request: Request, db: Session = Depends(get_session)) -> dict:
    p = _principal(request)
    u = db.get(User, p.user_id) if p.user_id else None
    return service.user_view(u) if u else {"username": p.name, "role": p.role, "kind": p.kind}


class PasswordIn(BaseModel):
    current: str = Field(max_length=200)
    new: str = Field(min_length=1, max_length=200)


@router.post("/password", dependencies=[Depends(_accounts_mode)])
def change_password(body: PasswordIn, request: Request, db: Session = Depends(get_session)) -> dict:
    try:
        service.change_own_password(db, _principal(request), body.current, body.new)
    except AccountError as e:
        _raise(e)
    return {"ok": True}


# ── admins ───────────────────────────────────────────────────────────────────

@router.get("/users", dependencies=[Depends(_accounts_mode)])
def users(request: Request, db: Session = Depends(get_session)) -> list[dict]:
    _admin(request)
    return [service.user_view(u) for u in db.exec(select(User).order_by(col(User.created_at))).all()]


class UserIn(BaseModel):
    username: str = Field(min_length=3, max_length=40)
    role: str = "member"
    display_name: str = Field("", max_length=80)
    email: str = Field("", max_length=200)
    password: str | None = Field(None, max_length=200, description="empty: a temporary password is generated")


@router.post("/users", dependencies=[Depends(_accounts_mode)])
def create_user(body: UserIn, request: Request, db: Session = Depends(get_session)) -> dict:
    _admin(request)
    temp = body.password is None or body.password == ""
    password = _temp_password() if temp else body.password
    try:
        u = service.create_user(db, body.username, password, body.role, body.display_name, body.email, must_change_password=temp)
    except AccountError as e:
        _raise(e)
    return service.user_view(u) | ({"temporary_password": password} if temp else {})


class UserPatch(BaseModel):
    role: str | None = None
    active: bool | None = None
    display_name: str | None = Field(None, max_length=80)
    email: str | None = Field(None, max_length=200)
    reset_password: bool = False


@router.patch("/users/{user_id}", dependencies=[Depends(_accounts_mode)])
def update_user(user_id: str, body: UserPatch, request: Request, db: Session = Depends(get_session)) -> dict:
    actor = _admin(request)
    temp = _temp_password() if body.reset_password else None
    try:
        u = service.update_user(db, actor, user_id, role=body.role, active=body.active, display_name=body.display_name,
                                email=body.email, new_password=temp)
    except AccountError as e:
        _raise(e)
    return service.user_view(u) | ({"temporary_password": temp} if temp else {})


class InviteIn(BaseModel):
    role: str = "member"
    note: str = Field("", max_length=200)
    days: float = Field(7, gt=0, le=30)


@router.post("/invites", dependencies=[Depends(_accounts_mode)])
def create_invite(body: InviteIn, request: Request, db: Session = Depends(get_session)) -> dict:
    actor = _admin(request)
    try:
        token, inv = service.create_invite(db, actor, body.role, body.note, body.days)
    except AccountError as e:
        _raise(e)
    return {"token": token, "role": inv.role, "expires_at": inv.expires_at, "note": inv.note}


@router.get("/invites", dependencies=[Depends(_accounts_mode)])
def invites(request: Request, db: Session = Depends(get_session)) -> list[dict]:
    _admin(request)
    rows = db.exec(select(Invite).order_by(col(Invite.created_at).desc()).limit(100)).all()
    return [{"id": i.id, "role": i.role, "note": i.note, "created_by": i.created_by, "created_at": i.created_at,
             "expires_at": i.expires_at, "used_at": i.used_at} for i in rows]


class RegisterIn(BaseModel):
    invite: str = Field(min_length=10, max_length=200)
    username: str = Field(min_length=3, max_length=40)
    password: str = Field(min_length=1, max_length=200)
    display_name: str = Field("", max_length=80)


@router.post("/register", dependencies=[Depends(_accounts_mode)])
def register(body: RegisterIn, request: Request, response: Response, db: Session = Depends(get_session)) -> dict:
    try:
        service.accept_invite(db, body.invite, body.username, body.password, body.display_name)
        u, token = service.login(db, body.username, body.password, _ip(request), request.headers.get("user-agent", ""))
    except AccountError as e:
        _raise(e)
    _set_cookie(request, response, token)
    return service.user_view(u)
