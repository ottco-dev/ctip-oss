"""Browser session cookie for the API token, and token checks on WebSocket handshakes."""

from __future__ import annotations

import pytest
from fastapi import FastAPI, WebSocket
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from backend.middleware.auth import AUTH_COOKIE, APITokenMiddleware, websocket_authorized

TOKEN = "s3cret-token-for-tests"


@pytest.fixture()
def app(monkeypatch):
    from backend.api.v1 import auth as auth_api
    from backend.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "api_token", TOKEN)
    application = FastAPI()
    application.add_middleware(APITokenMiddleware, api_token=TOKEN)
    application.include_router(auth_api.router, prefix="/api/v1")

    @application.get("/api/v1/protected")
    async def protected() -> dict:
        return {"ok": True}

    @application.websocket("/ws/test")
    async def ws(websocket: WebSocket) -> None:
        if not websocket_authorized(websocket, TOKEN):
            await websocket.close(code=1008)
            return
        await websocket.accept()
        await websocket.send_json({"ok": True})
        await websocket.close()

    return application


def test_status_and_login_flow(app):
    c = TestClient(app)
    assert c.get("/api/v1/protected").status_code == 401
    st = c.get("/api/v1/auth/status").json()
    assert st["enabled"] is True and st["authenticated"] is False and st["mode"] == "token"
    assert c.post("/api/v1/auth/session", json={"token": "wrong"}).status_code == 403
    r = c.post("/api/v1/auth/session", json={"token": TOKEN})
    assert r.status_code == 200
    cookie = r.headers["set-cookie"]
    assert AUTH_COOKIE in cookie and "HttpOnly" in cookie and "SameSite=strict" in cookie
    assert c.get("/api/v1/protected").status_code == 200          # cookie carried by the client
    assert c.get("/api/v1/auth/status").json()["authenticated"] is True
    c.delete("/api/v1/auth/session")
    c.cookies.clear()
    assert c.get("/api/v1/protected").status_code == 401


def test_websocket_needs_token(app):
    c = TestClient(app)
    with pytest.raises(WebSocketDisconnect) as exc, c.websocket_connect("/ws/test") as ws:
        ws.receive_json()
    assert exc.value.code == 1008
    with c.websocket_connect(f"/ws/test?api_key={TOKEN}") as ws:
        assert ws.receive_json() == {"ok": True}
    c.cookies.set(AUTH_COOKIE, TOKEN)
    with c.websocket_connect("/ws/test") as ws:
        assert ws.receive_json() == {"ok": True}


def test_auth_off_without_token():
    assert websocket_authorized(object(), "") is True               # type: ignore[arg-type]
