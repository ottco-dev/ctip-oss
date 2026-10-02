"""Accounts mode: login, lockout, roles, invites, deactivation, last admin, API token for scripts, WebSockets."""

from __future__ import annotations

import pytest
from fastapi import FastAPI, WebSocket
from fastapi.testclient import TestClient
from sqlmodel import Session, SQLModel, create_engine
from sqlmodel.pool import StaticPool
from starlette.websockets import WebSocketDisconnect

from backend.accounts import service
from backend.accounts.passwords import hash_password, password_problem, verify_password

API = "script-token"


@pytest.fixture()
def env(monkeypatch, tmp_path):
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    from backend.accounts import models  # noqa: F401
    from backend.compute import models as cm  # noqa: F401

    SQLModel.metadata.create_all(engine)
    import backend.database as database
    import backend.middleware.auth as auth_mod

    monkeypatch.setattr(database, "engine", engine)
    auth_mod._cache.clear()
    service.lockout._fails.clear()
    from backend.config import get_settings

    st = get_settings()
    monkeypatch.setattr(st, "auth_mode", "accounts")
    monkeypatch.setattr(st, "api_token", API)

    from backend.accounts.api import router as accounts_router
    from backend.api.v1 import auth as auth_api
    from backend.compute import api as compute_api
    from backend.database import get_session
    from backend.middleware.auth import APITokenMiddleware, websocket_authorized

    monkeypatch.setattr(compute_api, "artifact_root", lambda: tmp_path)
    app = FastAPI()
    app.add_middleware(APITokenMiddleware, api_token=API, mode="accounts")
    for r in (auth_api.router, accounts_router, compute_api.router):
        app.include_router(r, prefix="/api/v1")

    @app.get("/api/v1/datasets")
    def datasets() -> list:
        return []

    @app.websocket("/ws/test")
    async def ws(websocket: WebSocket) -> None:
        if not websocket_authorized(websocket, API, "accounts"):
            await websocket.close(code=1008)
            return
        await websocket.accept()
        await websocket.send_json({"ok": True})
        await websocket.close()

    def _db():
        with Session(engine) as s:
            yield s

    app.dependency_overrides[get_session] = _db
    with Session(engine) as db:
        service.ensure_admin(db, "ottco", "admin-password-123")
    return app, engine


def login(app, user, pw):
    c = TestClient(app)
    r = c.post("/api/v1/auth/login", json={"username": user, "password": pw})
    return c, r


def test_passwords():
    h = hash_password("long enough pass")
    assert verify_password("long enough pass", h) and not verify_password("other", h)
    assert password_problem("short") and password_problem("ottco-is-me-123", "ottco") and password_problem("aaaaaaaaaaaa")
    assert password_problem("a Good passphrase 9") is None


def test_login_status_logout(env):
    app, _ = env
    anon = TestClient(app)
    assert anon.get("/api/v1/datasets").status_code == 401
    assert anon.get("/api/v1/auth/status").json() == {"enabled": True, "authenticated": False, "mode": "accounts", "user": None}
    c, r = login(app, "ottco", "admin-password-123")
    assert r.status_code == 200 and "ctip_session" in r.headers["set-cookie"] and "HttpOnly" in r.headers["set-cookie"]
    st = c.get("/api/v1/auth/status").json()
    assert st["authenticated"] and st["user"]["username"] == "ottco" and st["user"]["role"] == "admin"
    assert c.get("/api/v1/datasets").status_code == 200
    c.post("/api/v1/auth/logout")
    c.cookies.clear()
    assert c.get("/api/v1/datasets").status_code == 401


def test_wrong_password_and_lockout(env):
    app, _ = env
    unknown = login(app, "nobody", "wrong")[1].json()["detail"]
    for _ in range(4):
        r = login(app, "ottco", "wrong")[1]
        assert r.status_code == 401
    assert r.json()["detail"] == unknown == "wrong username or password"      # no hint which part was wrong
    r = login(app, "ottco", "admin-password-123")[1]
    assert r.status_code == 429 and "try again" in r.json()["detail"]         # 5 failures from this address


def test_roles_and_temporary_password(env):
    app, _ = env
    admin, _ = login(app, "ottco", "admin-password-123")
    r = admin.post("/api/v1/auth/users", json={"username": "lisa", "role": "member"}).json()
    temp = r["temporary_password"]
    assert r["must_change_password"] is True
    lisa, ok = login(app, "lisa", temp)
    assert ok.status_code == 200
    assert lisa.get("/api/v1/datasets").status_code == 200
    assert lisa.get("/api/v1/auth/users").status_code == 403                  # admin area
    assert lisa.post("/api/v1/compute/enrollments", json={}).status_code == 403
    assert lisa.get("/api/v1/compute/workers").status_code == 200             # members may look
    assert lisa.post("/api/v1/auth/password", json={"current": temp, "new": "a fresh passphrase 7"}).status_code == 200
    assert lisa.get("/api/v1/auth/me").json()["must_change_password"] is False


def test_invite_register_and_deactivate(env):
    app, _ = env
    admin, _ = login(app, "ottco", "admin-password-123")
    token = admin.post("/api/v1/auth/invites", json={"role": "member", "note": "tester"}).json()["token"]
    anon = TestClient(app)
    r = anon.post("/api/v1/auth/register", json={"invite": token, "username": "max_m", "password": "maxs passphrase 1"})
    assert r.status_code == 200 and anon.get("/api/v1/datasets").status_code == 200   # logged in right away
    again = TestClient(app).post("/api/v1/auth/register", json={"invite": token, "username": "other", "password": "another phrase 22"})
    assert again.status_code == 403                                                    # invite used
    uid = r.json()["id"]
    assert admin.patch(f"/api/v1/auth/users/{uid}", json={"active": False}).status_code == 200
    import backend.middleware.auth as auth_mod

    auth_mod._cache.clear()                                                            # skip the 10 s cache
    assert anon.get("/api/v1/datasets").status_code == 401                             # sessions revoked


def test_last_admin_and_api_token(env):
    app, _ = env
    admin, _ = login(app, "ottco", "admin-password-123")
    me = admin.get("/api/v1/auth/me").json()
    assert admin.patch(f"/api/v1/auth/users/{me['id']}", json={"role": "member"}).status_code == 409
    script = TestClient(app, headers={"Authorization": f"Bearer {API}"})
    assert script.post("/api/v1/compute/enrollments", json={}).status_code == 200      # scripts act as admin
    legacy = TestClient(app)
    legacy.cookies.set("ctip_token", API)
    assert legacy.get("/api/v1/datasets").status_code == 401                           # old UI token cookie is no login


def test_websocket_needs_login(env):
    app, _ = env
    anon = TestClient(app)
    with pytest.raises(WebSocketDisconnect), anon.websocket_connect("/ws/test") as ws:
        ws.receive_json()
    c, _ = login(app, "ottco", "admin-password-123")
    with c.websocket_connect("/ws/test") as ws:
        assert ws.receive_json() == {"ok": True}
