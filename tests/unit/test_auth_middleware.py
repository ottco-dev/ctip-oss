"""
tests.unit.test_auth_middleware — Unit tests for API token authentication middleware.

Tests:
  - Auth disabled when api_token is empty
  - Auth enabled with valid token
  - Bearer token accepted
  - X-API-Key header accepted
  - ?api_key query parameter accepted
  - 401 on missing token
  - 403 on invalid token
  - Excluded paths bypass auth
  - OPTIONS requests bypass auth (CORS preflight)
  - Constant-time comparison (no timing leak)
  - Token extraction priority order
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.middleware.auth import APITokenMiddleware, _constant_time_equal, _extract_token


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_app(api_token: str = "secret-test-token") -> tuple[FastAPI, TestClient]:
    """Create a minimal FastAPI app with auth middleware."""
    app = FastAPI()
    app.add_middleware(APITokenMiddleware, api_token=api_token)

    @app.get("/api/v1/test")
    async def protected():
        return {"ok": True}

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/api/v1/system/health")
    async def system_health():
        return {"status": "ok"}

    @app.get("/docs")
    async def docs():
        return {"docs": True}

    return app, TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Auth disabled (empty token)
# ---------------------------------------------------------------------------

class TestAuthDisabled:

    def setup_method(self):
        _, self.client = _make_app(api_token="")

    def test_protected_route_accessible_without_token(self):
        resp = self.client.get("/api/v1/test")
        assert resp.status_code == 200

    def test_protected_route_accessible_with_any_token(self):
        resp = self.client.get("/api/v1/test", headers={"Authorization": "Bearer anything"})
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Auth enabled — valid token scenarios
# ---------------------------------------------------------------------------

class TestAuthEnabled:
    TOKEN = "my-super-secret-token-123"

    def setup_method(self):
        _, self.client = _make_app(api_token=self.TOKEN)

    def test_bearer_token_accepted(self):
        resp = self.client.get("/api/v1/test", headers={"Authorization": f"Bearer {self.TOKEN}"})
        assert resp.status_code == 200

    def test_x_api_key_header_accepted(self):
        resp = self.client.get("/api/v1/test", headers={"X-API-Key": self.TOKEN})
        assert resp.status_code == 200

    def test_api_key_query_param_accepted(self):
        resp = self.client.get(f"/api/v1/test?api_key={self.TOKEN}")
        assert resp.status_code == 200

    def test_missing_token_returns_401(self):
        resp = self.client.get("/api/v1/test")
        assert resp.status_code == 401

    def test_wrong_token_returns_403(self):
        resp = self.client.get("/api/v1/test", headers={"Authorization": "Bearer wrong-token"})
        assert resp.status_code == 403

    def test_empty_bearer_returns_401(self):
        resp = self.client.get("/api/v1/test", headers={"Authorization": "Bearer "})
        assert resp.status_code == 401

    def test_bearer_prefix_required(self):
        # "Token <secret>" (not Bearer) should fail
        resp = self.client.get("/api/v1/test", headers={"Authorization": f"Token {self.TOKEN}"})
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Excluded paths — always bypass auth
# ---------------------------------------------------------------------------

class TestExcludedPaths:
    TOKEN = "secret"

    def setup_method(self):
        _, self.client = _make_app(api_token=self.TOKEN)

    def test_health_bypass(self):
        resp = self.client.get("/health")
        assert resp.status_code == 200

    def test_system_health_bypass(self):
        resp = self.client.get("/api/v1/system/health")
        assert resp.status_code == 200

    def test_docs_bypass(self):
        resp = self.client.get("/docs")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# OPTIONS (CORS preflight) bypass
# ---------------------------------------------------------------------------

class TestOptionsBypass:
    TOKEN = "secret"

    def setup_method(self):
        _, self.client = _make_app(api_token=self.TOKEN)

    def test_options_bypasses_auth(self):
        resp = self.client.options("/api/v1/test")
        # OPTIONS is allowed through — response may be 200 or 405 depending on routing,
        # but must NOT be 401/403
        assert resp.status_code not in (401, 403)


# ---------------------------------------------------------------------------
# Constant-time comparison
# ---------------------------------------------------------------------------

class TestConstantTimeCompare:

    def test_equal_strings_match(self):
        assert _constant_time_equal("abc123", "abc123") is True

    def test_different_strings_no_match(self):
        assert _constant_time_equal("abc123", "xyz999") is False

    def test_empty_strings_match(self):
        assert _constant_time_equal("", "") is True

    def test_empty_vs_nonempty_no_match(self):
        assert _constant_time_equal("", "abc") is False

    def test_case_sensitive(self):
        assert _constant_time_equal("Secret", "secret") is False


# ---------------------------------------------------------------------------
# Token extraction priority
# ---------------------------------------------------------------------------

class TestTokenExtraction:

    def _make_request(self, headers: dict, params: dict | None = None) -> object:
        """Build a minimal mock request for _extract_token."""
        from unittest.mock import MagicMock
        req = MagicMock()
        req.headers = headers
        req.query_params = params or {}
        return req

    def test_bearer_header_extracted(self):
        req = self._make_request({"Authorization": "Bearer my-token"})
        assert _extract_token(req) == "my-token"

    def test_x_api_key_extracted(self):
        req = self._make_request({"X-API-Key": "my-token"})
        assert _extract_token(req) == "my-token"

    def test_query_param_extracted(self):
        req = self._make_request({}, {"api_key": "my-token"})
        assert _extract_token(req) == "my-token"

    def test_no_token_returns_none(self):
        req = self._make_request({})
        assert _extract_token(req) is None

    def test_bearer_takes_priority_over_x_api_key(self):
        req = self._make_request({
            "Authorization": "Bearer bearer-token",
            "X-API-Key": "xapikey-token",
        })
        assert _extract_token(req) == "bearer-token"

    def test_x_api_key_takes_priority_over_query(self):
        req = self._make_request(
            {"X-API-Key": "xapikey-token"},
            {"api_key": "queryparam-token"},
        )
        assert _extract_token(req) == "xapikey-token"

    def test_whitespace_stripped_from_bearer(self):
        req = self._make_request({"Authorization": "Bearer  spaced-token  "})
        assert _extract_token(req) == "spaced-token"


# ---------------------------------------------------------------------------
# Middleware is_enabled property
# ---------------------------------------------------------------------------

class TestMiddlewareState:

    def test_enabled_with_token(self):
        app = FastAPI()
        mw = APITokenMiddleware(app, api_token="my-token")
        assert mw.is_enabled is True

    def test_disabled_without_token(self):
        app = FastAPI()
        mw = APITokenMiddleware(app, api_token="")
        assert mw.is_enabled is False

    def test_disabled_with_whitespace_only(self):
        app = FastAPI()
        mw = APITokenMiddleware(app, api_token="   ")
        # Whitespace-only token is stripped → empty → disabled
        assert mw.is_enabled is False
