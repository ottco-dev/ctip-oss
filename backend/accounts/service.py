"""Accounts logic: users, login with lockout, server-side sessions, invitations, first admin."""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass

from sqlmodel import Session, col, select

from backend.accounts.models import ROLES, Invite, User, UserSession
from backend.accounts.passwords import hash_password, password_problem, verify_password
from shared.compute import tokens

SESSION_DAYS = 14
INVITE_DAYS = 7
MAX_FAILURES = 5
LOCKOUT_S = 15 * 60
USERNAME = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{2,39}$")
SESSION_PREFIX, INVITE_PREFIX = "ctips_", "ctipi_"
_DUMMY = hash_password("timing-equaliser-not-a-password")   # makes unknown users cost as much as wrong passwords


class AccountError(Exception):
    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status, self.detail = status, detail


@dataclass(frozen=True)
class Principal:
    """Who is calling: a logged-in user, or a script with the instance API token."""
    kind: str                # "user" | "token"
    name: str
    role: str
    user_id: str | None = None
    session_id: str | None = None

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"


API_TOKEN_PRINCIPAL = Principal("token", "api-token", "admin")


# ── lockout (per username and per IP, in memory) ─────────────────────────────

class _Lockout:
    def __init__(self) -> None:
        self._fails: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def blocked(self, *keys: str) -> float:
        now = time.time()
        with self._lock:
            worst = 0.0
            for k in keys:
                recent = [t for t in self._fails.get(k, []) if now - t < LOCKOUT_S]
                self._fails[k] = recent
                if len(recent) >= MAX_FAILURES:
                    worst = max(worst, recent[0] + LOCKOUT_S - now)
            return worst

    def fail(self, *keys: str) -> None:
        with self._lock:
            for k in keys:
                self._fails.setdefault(k, []).append(time.time())

    def clear(self, *keys: str) -> None:
        with self._lock:
            for k in keys:
                self._fails.pop(k, None)


lockout = _Lockout()


# ── users ────────────────────────────────────────────────────────────────────

def _check_new(username: str, password: str, role: str) -> None:
    if not USERNAME.match(username):
        raise AccountError(422, "username: 3-40 characters, letters, digits, . _ -")
    if role not in ROLES:
        raise AccountError(422, f"role must be one of {', '.join(ROLES)}")
    if problem := password_problem(password, username):
        raise AccountError(422, f"password: {problem}")


def create_user(db: Session, username: str, password: str, role: str = "member", display_name: str = "",
                email: str = "", must_change_password: bool = False) -> User:
    username = username.strip()
    _check_new(username, password, role)
    if db.exec(select(User).where(col(User.username).ilike(username))).first():
        raise AccountError(409, f"user {username} exists")
    u = User(username=username, password_hash=hash_password(password), role=role, display_name=display_name[:80],
             email=email[:200], must_change_password=must_change_password)
    db.add(u)
    db.commit()
    db.refresh(u)
    return u


def admins(db: Session) -> list[User]:
    return list(db.exec(select(User).where(User.role == "admin", User.active == True)).all())  # noqa: E712


def ensure_admin(db: Session, username: str, password: str) -> User | None:
    """First start: create the configured admin when no active admin exists. Never overwrites an existing user."""
    if not (username and password) or admins(db):
        return None
    existing = db.exec(select(User).where(User.username == username)).first()
    if existing:
        existing.role, existing.active = "admin", True
        db.add(existing)
        db.commit()
        return existing
    return create_user(db, username, password, "admin", must_change_password=False)


def update_user(db: Session, actor: Principal, user_id: str, *, role: str | None = None, active: bool | None = None,
                display_name: str | None = None, email: str | None = None, new_password: str | None = None) -> User:
    u = db.get(User, user_id)
    if u is None:
        raise AccountError(404, "no such user")
    demote = (role is not None and role != "admin") or active is False
    if u.role == "admin" and demote and len(admins(db)) <= 1:
        raise AccountError(409, "the last admin cannot be demoted or deactivated")
    if role is not None:
        if role not in ROLES:
            raise AccountError(422, "unknown role")
        u.role = role
    if active is not None:
        u.active = active
    if display_name is not None:
        u.display_name = display_name[:80]
    if email is not None:
        u.email = email[:200]
    if new_password is not None:
        if problem := password_problem(new_password, u.username):
            raise AccountError(422, f"password: {problem}")
        u.password_hash, u.must_change_password = hash_password(new_password), actor.user_id != u.id
    db.add(u)
    if active is False or new_password is not None:
        revoke_sessions(db, u.id, keep=actor.session_id if actor.user_id == u.id else None)
    db.commit()
    db.refresh(u)
    return u


def change_own_password(db: Session, actor: Principal, old: str, new: str) -> None:
    u = db.get(User, actor.user_id) if actor.user_id else None
    if u is None:
        raise AccountError(403, "only logged-in users have a password")
    if not verify_password(old, u.password_hash):
        raise AccountError(403, "current password is wrong")
    if problem := password_problem(new, u.username):
        raise AccountError(422, f"password: {problem}")
    u.password_hash, u.must_change_password = hash_password(new), False
    db.add(u)
    revoke_sessions(db, u.id, keep=actor.session_id)
    db.commit()


# ── login and sessions ───────────────────────────────────────────────────────

def login(db: Session, username: str, password: str, ip: str = "", user_agent: str = "") -> tuple[User, str]:
    key_u, key_ip = f"u:{username.strip().lower()}", f"ip:{ip}"
    if wait := lockout.blocked(key_u, key_ip):
        raise AccountError(429, f"too many failed logins - try again in {int(wait // 60) + 1} min")
    u = db.exec(select(User).where(col(User.username).ilike(username.strip()))).first()
    ok = verify_password(password, u.password_hash if u else _DUMMY)
    if not (u and ok and u.active):
        lockout.fail(key_u, key_ip)
        raise AccountError(401, "wrong username or password")       # same answer for every reason
    lockout.clear(key_u)
    token = tokens.new_token(SESSION_PREFIX)
    s = UserSession(token_hash=tokens.token_hash(token), user_id=u.id, expires_at=time.time() + SESSION_DAYS * 86400,
                    user_agent=user_agent[:200], ip=ip[:64])
    u.last_login = time.time()
    db.add(s)
    db.add(u)
    db.commit()
    return u, token


def principal_for_session(db: Session, token: str) -> Principal | None:
    if not token.startswith(SESSION_PREFIX):
        return None
    s = db.exec(select(UserSession).where(UserSession.token_hash == tokens.token_hash(token))).first()
    now = time.time()
    if s is None or s.expires_at < now:
        return None
    u = db.get(User, s.user_id)
    if u is None or not u.active:
        return None
    if now - s.last_seen > 300:                                      # sliding expiry, written at most every 5 min
        s.last_seen, s.expires_at = now, now + SESSION_DAYS * 86400
        db.add(s)
        db.commit()
    return Principal("user", u.username, u.role, u.id, s.id)


def logout(db: Session, token: str) -> None:
    s = db.exec(select(UserSession).where(UserSession.token_hash == tokens.token_hash(token))).first()
    if s:
        db.delete(s)
        db.commit()


def revoke_sessions(db: Session, user_id: str, keep: str | None = None) -> int:
    n = 0
    for s in db.exec(select(UserSession).where(UserSession.user_id == user_id)).all():
        if s.id != keep:
            db.delete(s)
            n += 1
    return n


# ── invitations ──────────────────────────────────────────────────────────────

def create_invite(db: Session, actor: Principal, role: str = "member", note: str = "", days: float = INVITE_DAYS) -> tuple[str, Invite]:
    if role not in ROLES:
        raise AccountError(422, "unknown role")
    token = tokens.new_token(INVITE_PREFIX)
    inv = Invite(token_hash=tokens.token_hash(token), role=role, note=note[:200], created_by=actor.name,
                 expires_at=time.time() + days * 86400)
    db.add(inv)
    db.commit()
    db.refresh(inv)
    return token, inv


def accept_invite(db: Session, token: str, username: str, password: str, display_name: str = "") -> User:
    inv = db.exec(select(Invite).where(Invite.token_hash == tokens.token_hash(token))).first()
    if inv is None or inv.used_at is not None or inv.expires_at < time.time():
        raise AccountError(403, "invitation unknown, used or expired")
    u = create_user(db, username, password, inv.role, display_name)
    inv.used_at, inv.user_id = time.time(), u.id
    db.add(inv)
    db.commit()
    return u


def user_view(u: User) -> dict:
    return {"id": u.id, "username": u.username, "display_name": u.display_name, "email": u.email, "role": u.role,
            "active": u.active, "must_change_password": u.must_change_password, "created_at": u.created_at,
            "last_login": u.last_login}
