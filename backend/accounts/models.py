"""Tables for accounts: users, sessions (server-side, revocable), invitations."""

from __future__ import annotations

import time
import uuid

from sqlmodel import Field, SQLModel

ROLES = ("admin", "member")


def _id() -> str:
    return uuid.uuid4().hex


class User(SQLModel, table=True):
    __tablename__ = "users"

    id: str = Field(default_factory=_id, primary_key=True)
    username: str = Field(index=True, unique=True)
    display_name: str = ""
    email: str = ""
    password_hash: str
    role: str = "member"
    active: bool = True
    must_change_password: bool = False
    created_at: float = Field(default_factory=time.time)
    last_login: float | None = None


class UserSession(SQLModel, table=True):
    __tablename__ = "user_sessions"

    id: str = Field(default_factory=_id, primary_key=True)
    token_hash: str = Field(index=True, unique=True)
    user_id: str = Field(index=True)
    created_at: float = Field(default_factory=time.time)
    expires_at: float
    last_seen: float = Field(default_factory=time.time)
    user_agent: str = ""
    ip: str = ""


class Invite(SQLModel, table=True):
    __tablename__ = "user_invites"

    id: str = Field(default_factory=_id, primary_key=True)
    token_hash: str = Field(index=True, unique=True)
    role: str = "member"
    note: str = ""
    created_by: str = ""
    created_at: float = Field(default_factory=time.time)
    expires_at: float
    used_at: float | None = None
    user_id: str | None = None
