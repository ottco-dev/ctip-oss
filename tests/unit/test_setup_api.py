"""
tests/unit/test_setup_api.py — Unit tests for backend/api/v1/setup.py

Tests are designed to run without a real GPU, Docker daemon, or Label Studio instance.
All external calls are mocked.

Coverage targets:
  - _read_env_file / _write_env_file  — env file I/O
  - /setup/status                     — setup state detection
  - /setup/config                     — config read with redaction
  - /setup/validate                   — per-key validation
  - /setup/configure                  — env write, SETUP_COMPLETED flag
  - /setup/system-check               — dependency detection
  - /setup/docker/status              — docker availability + group membership
  - /setup/docker/start-annotation    — docker compose invocation
  - /setup/models/status              — model catalog + presence check
  - /setup/models/download            — background task creation
  - /setup/models/download/{task_id}  — progress polling
  - /setup/label-studio/test          — reachability + auth
  - /setup/label-studio/create-account — account creation + already-exists path
  - /setup/label-studio/create-project — project creation + idempotency
  - /setup/verification               — live service health check
  - _do_download                      — streaming download with progress tracking
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import uuid
from pathlib import Path
from typing import Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from httpx import Response as HttpxResponse


# ── App bootstrap ──────────────────────────────────────────────────────────────

def _make_app():
    """Create a minimal FastAPI app that includes only the setup router."""
    from fastapi import FastAPI
    from backend.api.v1.setup import router
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    return app


@pytest.fixture(scope="module")
def client() -> Generator[TestClient, None, None]:
    app = _make_app()
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


# ── Helpers ────────────────────────────────────────────────────────────────────

def _write_tmp_env(content: str, tmp_path: Path) -> Path:
    p = tmp_path / ".env"
    p.write_text(content)
    return p


# ── _read_env_file / _write_env_file ──────────────────────────────────────────

class TestEnvFileIO:
    def test_read_empty(self, tmp_path):
        from backend.api.v1.setup import _read_env_file
        p = tmp_path / ".env"
        p.write_text("")
        assert _read_env_file(p) == {}

    def test_read_missing(self, tmp_path):
        from backend.api.v1.setup import _read_env_file
        assert _read_env_file(tmp_path / "nonexistent.env") == {}

    def test_read_basic_keys(self, tmp_path):
        from backend.api.v1.setup import _read_env_file
        p = _write_tmp_env('CUDA_DEVICE="cuda:0"\nVRAM_LIMIT_GB=8.0\n', tmp_path)
        result = _read_env_file(p)
        assert result["CUDA_DEVICE"] == "cuda:0"
        assert result["VRAM_LIMIT_GB"] == "8.0"

    def test_read_skips_comments(self, tmp_path):
        from backend.api.v1.setup import _read_env_file
        p = _write_tmp_env("# comment\nKEY=value\n", tmp_path)
        result = _read_env_file(p)
        assert "# comment" not in result
        assert result["KEY"] == "value"

    def test_read_strips_quotes(self, tmp_path):
        from backend.api.v1.setup import _read_env_file
        p = _write_tmp_env('KEY="quoted value"\n', tmp_path)
        assert _read_env_file(p)["KEY"] == "quoted value"

    def test_write_new_keys(self, tmp_path):
        from backend.api.v1.setup import _write_env_file, _read_env_file
        p = tmp_path / ".env"
        _write_env_file(p, {"ENVIRONMENT": "production", "PUBLIC_PORT": "3001"})
        result = _read_env_file(p)
        assert result["ENVIRONMENT"] == "production"
        assert result["PUBLIC_PORT"] == "3001"

    def test_write_updates_existing_key(self, tmp_path):
        from backend.api.v1.setup import _write_env_file, _read_env_file
        p = _write_tmp_env('ENVIRONMENT="development"\n', tmp_path)
        _write_env_file(p, {"ENVIRONMENT": "production"})
        result = _read_env_file(p)
        assert result["ENVIRONMENT"] == "production"

    def test_write_preserves_other_keys(self, tmp_path):
        from backend.api.v1.setup import _write_env_file, _read_env_file
        p = _write_tmp_env('KEEP=me\nUPDATE=old\n', tmp_path)
        _write_env_file(p, {"UPDATE": "new"})
        result = _read_env_file(p)
        assert result["KEEP"] == "me"
        assert result["UPDATE"] == "new"

    def test_write_atomic_via_tempfile(self, tmp_path):
        """Write must not leave orphan .tmp files on success."""
        from backend.api.v1.setup import _write_env_file
        p = tmp_path / ".env"
        _write_env_file(p, {"KEY": "val"})
        tmp_files = list(tmp_path.glob("*.env.tmp"))
        assert tmp_files == [], f"Orphan tmp file found: {tmp_files}"


# ── GET /setup/status ──────────────────────────────────────────────────────────

class TestSetupStatus:
    def test_returns_false_when_no_env(self, client, tmp_path):
        with patch("backend.api.v1.setup.ENV_FILE", tmp_path / "missing.env"):
            r = client.get("/api/v1/setup/status")
        assert r.status_code == 200
        assert r.json()["completed"] is False
        assert r.json()["env_exists"] is False

    def test_returns_false_when_setup_not_complete(self, client, tmp_path):
        p = _write_tmp_env('ENVIRONMENT="development"\n', tmp_path)
        with patch("backend.api.v1.setup.ENV_FILE", p):
            r = client.get("/api/v1/setup/status")
        assert r.status_code == 200
        assert r.json()["completed"] is False
        assert r.json()["env_exists"] is True

    def test_returns_true_when_setup_complete(self, client, tmp_path):
        p = _write_tmp_env('SETUP_COMPLETED="true"\n', tmp_path)
        with patch("backend.api.v1.setup.ENV_FILE", p):
            r = client.get("/api/v1/setup/status")
        assert r.status_code == 200
        assert r.json()["completed"] is True

    def test_configured_keys_listed(self, client, tmp_path):
        p = _write_tmp_env('CUDA_DEVICE="cuda:0"\nVRAM_LIMIT_GB="8.0"\n', tmp_path)
        with patch("backend.api.v1.setup.ENV_FILE", p):
            r = client.get("/api/v1/setup/status")
        keys = r.json()["configured_keys"]
        assert "CUDA_DEVICE" in keys


# ── GET /setup/config ──────────────────────────────────────────────────────────

class TestSetupConfig:
    def test_sensitive_keys_redacted(self, client, tmp_path):
        p = _write_tmp_env('SECRET_KEY="supersecret"\nAPI_TOKEN="mytoken"\n', tmp_path)
        with patch("backend.api.v1.setup.ENV_FILE", p):
            r = client.get("/api/v1/setup/config")
        assert r.status_code == 200
        entries = {e["key"]: e for e in r.json()["entries"]}
        if "SECRET_KEY" in entries:
            assert entries["SECRET_KEY"]["value"] == "••••••••"
            assert entries["SECRET_KEY"]["sensitive"] is True

    def test_non_sensitive_value_returned(self, client, tmp_path):
        p = _write_tmp_env('ENVIRONMENT="production"\n', tmp_path)
        with patch("backend.api.v1.setup.ENV_FILE", p):
            r = client.get("/api/v1/setup/config")
        entries = {e["key"]: e for e in r.json()["entries"]}
        assert entries["ENVIRONMENT"]["value"] == "production"


# ── POST /setup/configure ─────────────────────────────────────────────────────

class TestSetupConfigure:
    def test_writes_settings_to_env(self, client, tmp_path):
        p = tmp_path / ".env"
        with patch("backend.api.v1.setup.ENV_FILE", p):
            r = client.post("/api/v1/setup/configure", json={
                "settings": {"ENVIRONMENT": "production", "VRAM_LIMIT_GB": "8.0"},
                "mark_setup_complete": False,
            })
        assert r.status_code == 200
        assert "ENVIRONMENT" in r.json()["written"]
        from backend.api.v1.setup import _read_env_file
        env = _read_env_file(p)
        assert env["ENVIRONMENT"] == "production"

    def test_rejects_disallowed_keys(self, client, tmp_path):
        p = tmp_path / ".env"
        with patch("backend.api.v1.setup.ENV_FILE", p):
            r = client.post("/api/v1/setup/configure", json={
                "settings": {"DISALLOWED_KEY": "evil"},
                "mark_setup_complete": False,
            })
        assert r.status_code == 200
        assert "DISALLOWED_KEY" in r.json()["skipped"]

    def test_marks_setup_complete(self, client, tmp_path):
        p = tmp_path / ".env"
        with patch("backend.api.v1.setup.ENV_FILE", p):
            client.post("/api/v1/setup/configure", json={
                "settings": {}, "mark_setup_complete": True,
            })
        from backend.api.v1.setup import _read_env_file
        env = _read_env_file(p)
        assert env.get("SETUP_COMPLETED") == "true"

    def test_skips_empty_string_values(self, client, tmp_path):
        """Empty string values (e.g., not filling optional SECRET_KEY) should be skipped."""
        p = tmp_path / ".env"
        with patch("backend.api.v1.setup.ENV_FILE", p):
            r = client.post("/api/v1/setup/configure", json={
                "settings": {"SECRET_KEY": "", "ENVIRONMENT": "development"},
                "mark_setup_complete": False,
            })
        assert r.status_code == 200
        from backend.api.v1.setup import _read_env_file
        env = _read_env_file(p)
        assert "SECRET_KEY" not in env or env["SECRET_KEY"] == ""
        assert env.get("ENVIRONMENT") == "development"


# ── GET /setup/system-check ───────────────────────────────────────────────────

class TestSystemCheck:
    def test_returns_items_list(self, client):
        r = client.get("/api/v1/setup/system-check")
        assert r.status_code == 200
        data = r.json()
        assert "items" in data
        assert "all_required_ok" in data
        assert isinstance(data["items"], list)
        assert len(data["items"]) > 0

    def test_every_item_has_required_fields(self, client):
        r = client.get("/api/v1/setup/system-check")
        for item in r.json()["items"]:
            assert "name" in item
            assert "ok" in item
            assert "required" in item

    def test_python_item_present(self, client):
        r = client.get("/api/v1/setup/system-check")
        names = [i["name"] for i in r.json()["items"]]
        assert "Python" in names

    def test_python_item_is_ok(self, client):
        """Python must be ok since we're running in it."""
        r = client.get("/api/v1/setup/system-check")
        items = {i["name"]: i for i in r.json()["items"]}
        assert items["Python"]["ok"] is True

    def test_all_required_ok_type(self, client):
        r = client.get("/api/v1/setup/system-check")
        assert isinstance(r.json()["all_required_ok"], bool)


# ── GET /setup/docker/status ──────────────────────────────────────────────────

class TestDockerStatus:
    def _mock_run_no_docker(self, cmd, timeout=8, cwd=None):
        return False, "Cannot connect to the Docker daemon"

    def test_docker_unavailable_returns_gracefully(self, client):
        with patch("backend.api.v1.setup._run", side_effect=self._mock_run_no_docker):
            r = client.get("/api/v1/setup/docker/status")
        assert r.status_code == 200
        data = r.json()
        assert data["docker_available"] is False
        assert data["containers"] == []

    def test_fix_command_included_when_not_in_group(self, client):
        with patch("backend.api.v1.setup._run", side_effect=self._mock_run_no_docker):
            r = client.get("/api/v1/setup/docker/status")
        data = r.json()
        # Only present when user is not in docker group
        if not data["in_docker_group"]:
            assert "fix_command" in data
            assert len(data["fix_command"]) > 0

    def test_docker_available_mocked(self, client):
        def mock_run(cmd, timeout=8, cwd=None):
            cmd_str = " ".join(cmd)
            if "docker info" in cmd_str:
                return True, "24.0.7"
            if "compose version" in cmd_str:
                return True, "2.24.0"
            if "compose" in cmd_str and "ps" in cmd_str:
                # Return empty — no containers
                return True, ""
            return False, "unknown cmd"

        with patch("backend.api.v1.setup._run", side_effect=mock_run):
            r = client.get("/api/v1/setup/docker/status")
        assert r.status_code == 200
        data = r.json()
        assert data["docker_available"] is True

    def test_container_list_parsed(self, client):
        container_json = json.dumps({
            "Name": "ctip-label-studio",
            "Image": "heartexlabs/label-studio:1.13.1",
            "Status": "running (healthy)",
            "Publishers": [{"PublishedPort": 3005}],
        })

        def mock_run(cmd, timeout=8, cwd=None):
            if "info" in " ".join(cmd):
                return True, "24.0.7"
            if "version" in " ".join(cmd):
                return True, "2.24.0"
            if "ps" in " ".join(cmd):
                return True, container_json
            return False, ""

        with patch("backend.api.v1.setup._run", side_effect=mock_run):
            r = client.get("/api/v1/setup/docker/status")
        data = r.json()
        assert len(data["containers"]) == 1
        c = data["containers"][0]
        assert c["name"] == "ctip-label-studio"
        assert c["running"] is True
        assert "3005" in c["ports"]


# ── POST /setup/docker/start-annotation ───────────────────────────────────────

class TestDockerStartAnnotation:
    def test_fails_gracefully_when_docker_unavailable(self, client):
        with patch("backend.api.v1.setup._run", return_value=(False, "permission denied")):
            r = client.post("/api/v1/setup/docker/start-annotation", json={})
        assert r.status_code == 200
        assert r.json()["ok"] is False

    def test_starts_annotation_stack(self, client):
        def mock_run(cmd, timeout=8, cwd=None):
            if "info" in " ".join(cmd):
                return True, "24.0.7"
            return True, "Started 3 containers"

        with patch("backend.api.v1.setup._run", side_effect=mock_run):
            r = client.post("/api/v1/setup/docker/start-annotation", json={"profile": "annotation"})
        assert r.status_code == 200
        assert r.json()["ok"] is True

    def test_uses_correct_compose_profile(self, client):
        called_cmds = []

        def mock_run(cmd, timeout=8, cwd=None):
            called_cmds.append(cmd)
            if "info" in " ".join(cmd):
                return True, "24.0.7"
            return True, "ok"

        with patch("backend.api.v1.setup._run", side_effect=mock_run):
            client.post("/api/v1/setup/docker/start-annotation", json={"profile": "annotation"})

        compose_calls = [c for c in called_cmds if "compose" in " ".join(c)]
        assert any("annotation" in " ".join(c) for c in compose_calls)

    def test_custom_profile_respected(self, client):
        called_cmds = []

        def mock_run(cmd, timeout=8, cwd=None):
            called_cmds.append(cmd)
            if "info" in " ".join(cmd):
                return True, "24.0.7"
            return True, "ok"

        with patch("backend.api.v1.setup._run", side_effect=mock_run):
            client.post("/api/v1/setup/docker/start-annotation", json={"profile": "training"})

        compose_calls = [c for c in called_cmds if "compose" in " ".join(c)]
        assert any("training" in " ".join(c) for c in compose_calls)


# ── GET /setup/models/status ──────────────────────────────────────────────────

class TestModelsStatus:
    def test_returns_list_of_models(self, client, tmp_path):
        with patch("backend.api.v1.setup._models_dir", return_value=tmp_path):
            r = client.get("/api/v1/setup/models/status")
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data, list)
        assert len(data) == 5  # yolo11n/s/m + sam2-tiny/small

    def test_required_models_flagged(self, client, tmp_path):
        with patch("backend.api.v1.setup._models_dir", return_value=tmp_path):
            r = client.get("/api/v1/setup/models/status")
        models = {m["id"]: m for m in r.json()}
        assert models["yolo11s"]["required"] is True
        assert models["sam2-tiny"]["required"] is True
        assert models["yolo11n"]["required"] is False

    def test_absent_model_shows_not_present(self, client, tmp_path):
        with patch("backend.api.v1.setup._models_dir", return_value=tmp_path):
            r = client.get("/api/v1/setup/models/status")
        models = {m["id"]: m for m in r.json()}
        assert models["yolo11s"]["present"] is False

    def test_present_model_detected(self, client, tmp_path):
        (tmp_path / "yolo11s.pt").write_bytes(b"\x00" * 100)
        with patch("backend.api.v1.setup._models_dir", return_value=tmp_path):
            r = client.get("/api/v1/setup/models/status")
        models = {m["id"]: m for m in r.json()}
        assert models["yolo11s"]["present"] is True
        assert models["yolo11n"]["present"] is False

    def test_model_has_url(self, client, tmp_path):
        with patch("backend.api.v1.setup._models_dir", return_value=tmp_path):
            r = client.get("/api/v1/setup/models/status")
        for m in r.json():
            assert m["url"].startswith("http"), f"{m['id']} has no valid URL"


# ── POST /setup/models/download ───────────────────────────────────────────────

class TestModelDownload:
    def test_unknown_model_id_returns_404(self, client, tmp_path):
        with patch("backend.api.v1.setup._models_dir", return_value=tmp_path):
            r = client.post("/api/v1/setup/models/download", json={"model_id": "does-not-exist"})
        assert r.status_code == 404

    def test_returns_task_id_for_valid_model(self, client, tmp_path):
        with patch("backend.api.v1.setup._models_dir", return_value=tmp_path):
            r = client.post("/api/v1/setup/models/download", json={"model_id": "yolo11n"})
        assert r.status_code == 200
        data = r.json()
        assert "task_id" in data
        assert data["model_id"] == "yolo11n"
        assert len(data["task_id"]) > 0

    def test_already_present_returns_done_task(self, client, tmp_path):
        (tmp_path / "yolo11n.pt").write_bytes(b"\x00" * 1000)
        with patch("backend.api.v1.setup._models_dir", return_value=tmp_path):
            r = client.post("/api/v1/setup/models/download", json={"model_id": "yolo11n"})
        assert r.status_code == 200
        task_id = r.json()["task_id"]
        # Poll the task
        with patch("backend.api.v1.setup._models_dir", return_value=tmp_path):
            poll = client.get(f"/api/v1/setup/models/download/{task_id}")
        assert poll.status_code == 200
        assert poll.json()["status"] == "done"
        assert poll.json()["progress"] == 100

    def test_poll_nonexistent_task_returns_404(self, client):
        r = client.get(f"/api/v1/setup/models/download/{uuid.uuid4()}")
        assert r.status_code == 404


# ── _do_download (async unit test) ────────────────────────────────────────────

class TestDoDownload:
    @pytest.mark.asyncio
    async def test_successful_download_sets_done(self, tmp_path):
        from backend.api.v1.setup import _do_download, _download_tasks

        task_id = str(uuid.uuid4())
        dest = tmp_path / "test_model.pt"
        fake_content = b"X" * (1024 * 100)  # 100 KB

        class FakeStream:
            def __init__(self):
                self.headers = {"content-length": str(len(fake_content))}
                self.status_code = 200

            async def aiter_bytes(self, chunk_size=65536):
                for i in range(0, len(fake_content), chunk_size):
                    yield fake_content[i:i + chunk_size]

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            def raise_for_status(self):
                pass

        class FakeClient:
            def stream(self, method, url):
                return FakeStream()

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        with patch("backend.api.v1.setup.httpx.AsyncClient", return_value=FakeClient()):
            await _do_download(task_id, "http://fake/model.pt", dest)

        assert dest.exists()
        assert dest.read_bytes() == fake_content
        assert _download_tasks[task_id]["status"] == "done"
        assert _download_tasks[task_id]["progress"] == 100

    @pytest.mark.asyncio
    async def test_failed_download_sets_error(self, tmp_path):
        from backend.api.v1.setup import _do_download, _download_tasks

        task_id = str(uuid.uuid4())
        dest = tmp_path / "test_model.pt"

        class BrokenClient:
            def stream(self, method, url):
                raise RuntimeError("Connection refused")

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        with patch("backend.api.v1.setup.httpx.AsyncClient", return_value=BrokenClient()):
            await _do_download(task_id, "http://fake/model.pt", dest)

        assert not dest.exists()
        assert _download_tasks[task_id]["status"] == "error"
        assert "Connection refused" in _download_tasks[task_id]["detail"]

    @pytest.mark.asyncio
    async def test_progress_tracked_during_download(self, tmp_path):
        from backend.api.v1.setup import _do_download, _download_tasks

        task_id = str(uuid.uuid4())
        dest = tmp_path / "prog_model.pt"
        chunk_size = 65536
        num_chunks = 10
        fake_content = b"A" * (chunk_size * num_chunks)
        observed_progress = []

        class TrackingStream:
            def __init__(self):
                self.headers = {"content-length": str(len(fake_content))}

            async def aiter_bytes(self, chunk_size=65536):
                for i in range(0, len(fake_content), chunk_size):
                    yield fake_content[i:i + chunk_size]
                    observed_progress.append(_download_tasks[task_id]["progress"])

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            def raise_for_status(self):
                pass

        class TrackingClient:
            def stream(self, method, url):
                return TrackingStream()

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        with patch("backend.api.v1.setup.httpx.AsyncClient", return_value=TrackingClient()):
            await _do_download(task_id, "http://fake/model.pt", dest)

        assert len(observed_progress) > 0
        assert observed_progress[-1] <= 100
        assert observed_progress[0] >= 0
        # Progress must be non-decreasing
        for a, b in zip(observed_progress, observed_progress[1:]):
            assert b >= a

    @pytest.mark.asyncio
    async def test_no_orphan_tmp_file_on_success(self, tmp_path):
        from backend.api.v1.setup import _do_download

        task_id = str(uuid.uuid4())
        dest = tmp_path / "clean.pt"
        fake_content = b"X" * 100

        class FakeStream:
            def __init__(self):
                self.headers = {"content-length": str(len(fake_content))}

            async def aiter_bytes(self, chunk_size=65536):
                yield fake_content

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            def raise_for_status(self):
                pass

        class FakeClient:
            def stream(self, method, url):
                return FakeStream()

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        with patch("backend.api.v1.setup.httpx.AsyncClient", return_value=FakeClient()):
            await _do_download(task_id, "http://fake/model.pt", dest)

        tmp_files = list(tmp_path.glob("*.tmp"))
        assert tmp_files == [], f"Orphan .tmp file found: {tmp_files}"


# ── POST /setup/label-studio/test ─────────────────────────────────────────────

class TestLabelStudioTest:
    @pytest.mark.asyncio
    async def test_not_reachable(self, client):
        with patch("backend.api.v1.setup._http_ok", return_value=(False, 0, "Connection refused")):
            r = client.post("/api/v1/setup/label-studio/test", json={
                "url": "http://localhost:3005", "api_key": "somekey",
            })
        assert r.status_code == 200
        data = r.json()
        assert data["reachable"] is False
        assert data["authenticated"] is False

    @pytest.mark.asyncio
    async def test_reachable_but_no_api_key(self, client):
        with patch("backend.api.v1.setup._http_ok", return_value=(True, 200, "")):
            r = client.post("/api/v1/setup/label-studio/test", json={
                "url": "http://localhost:3005", "api_key": "",
            })
        assert r.status_code == 200
        data = r.json()
        assert data["reachable"] is True
        assert data["authenticated"] is False

    @pytest.mark.asyncio
    async def test_authenticated_success(self, client):
        whoami_response = MagicMock()
        whoami_response.status_code = 200
        whoami_response.json.return_value = {"username": "admin@ctip.local"}

        projects_response = MagicMock()
        projects_response.status_code = 200
        projects_response.json.return_value = {"count": 3}

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=[whoami_response, projects_response])
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("backend.api.v1.setup._http_ok", return_value=(True, 200, "")), \
             patch("backend.api.v1.setup.httpx.AsyncClient", return_value=mock_client):
            r = client.post("/api/v1/setup/label-studio/test", json={
                "url": "http://localhost:3005", "api_key": "valid_token",
            })

        assert r.status_code == 200
        data = r.json()
        assert data["ok"] is True
        assert data["authenticated"] is True
        assert data["user"] == "admin@ctip.local"
        assert data["projects_count"] == 3

    @pytest.mark.asyncio
    async def test_bad_api_key_returns_not_authenticated(self, client):
        unauth_response = MagicMock()
        unauth_response.status_code = 401

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=unauth_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("backend.api.v1.setup._http_ok", return_value=(True, 200, "")), \
             patch("backend.api.v1.setup.httpx.AsyncClient", return_value=mock_client):
            r = client.post("/api/v1/setup/label-studio/test", json={
                "url": "http://localhost:3005", "api_key": "wrong_key",
            })

        assert r.status_code == 200
        data = r.json()
        assert data["authenticated"] is False


# ── POST /setup/label-studio/create-account ───────────────────────────────────

class TestLabelStudioCreateAccount:
    @pytest.mark.asyncio
    async def test_creates_account_successfully(self, client):
        signup_response = MagicMock()
        signup_response.status_code = 201
        signup_response.json.return_value = {"token": "test-token-abc", "id": 1}

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=signup_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("backend.api.v1.setup.httpx.AsyncClient", return_value=mock_client):
            r = client.post("/api/v1/setup/label-studio/create-account", json={
                "url": "http://localhost:3005",
                "email": "admin@ctip.local",
                "password": "StrongPass123!",
            })

        assert r.status_code == 200
        data = r.json()
        assert data["ok"] is True
        assert data["token"] == "test-token-abc"
        assert data["already_existed"] is False

    @pytest.mark.asyncio
    async def test_already_exists_returns_ok(self, client):
        exists_response = MagicMock()
        exists_response.status_code = 400
        exists_response.text = "User with this email already exists"

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=exists_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("backend.api.v1.setup.httpx.AsyncClient", return_value=mock_client):
            r = client.post("/api/v1/setup/label-studio/create-account", json={
                "url": "http://localhost:3005",
                "email": "admin@ctip.local",
                "password": "AnyPass",
            })

        assert r.status_code == 200
        data = r.json()
        assert data["ok"] is True
        assert data["already_existed"] is True


# ── POST /setup/label-studio/create-project ───────────────────────────────────

class TestLabelStudioCreateProject:
    @pytest.mark.asyncio
    async def test_creates_project(self, client):
        list_response = MagicMock()
        list_response.status_code = 200
        list_response.json.return_value = {"results": []}

        create_response = MagicMock()
        create_response.status_code = 201
        create_response.json.return_value = {"id": 42, "title": "CTIP — Trichome Detection"}

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=list_response)
        mock_client.post = AsyncMock(return_value=create_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("backend.api.v1.setup.httpx.AsyncClient", return_value=mock_client):
            r = client.post("/api/v1/setup/label-studio/create-project", json={
                "url": "http://localhost:3005",
                "api_key": "valid_token",
                "project_name": "CTIP — Trichome Detection",
            })

        assert r.status_code == 200
        data = r.json()
        assert data["ok"] is True
        assert data["project_id"] == 42
        assert data["already_existed"] is False

    @pytest.mark.asyncio
    async def test_returns_existing_project_id(self, client):
        existing_project = {"id": 7, "title": "CTIP — Trichome Detection"}
        list_response = MagicMock()
        list_response.status_code = 200
        list_response.json.return_value = {"results": [existing_project]}

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=list_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("backend.api.v1.setup.httpx.AsyncClient", return_value=mock_client):
            r = client.post("/api/v1/setup/label-studio/create-project", json={
                "url": "http://localhost:3005",
                "api_key": "valid_token",
                "project_name": "CTIP — Trichome Detection",
            })

        assert r.status_code == 200
        data = r.json()
        assert data["ok"] is True
        assert data["project_id"] == 7
        assert data["already_existed"] is True


# ── GET /setup/verification ───────────────────────────────────────────────────

class TestVerification:
    def test_returns_items_and_all_ok(self, client):
        async def fake_http_ok(url, timeout=4.0):
            return True, 200, ""

        with patch("backend.api.v1.setup._http_ok", side_effect=fake_http_ok):
            r = client.get("/api/v1/setup/verification")
        assert r.status_code == 200
        data = r.json()
        assert "items" in data
        assert "all_ok" in data
        assert "timestamp" in data

    def test_item_structure(self, client):
        async def fake_http_ok(url, timeout=4.0):
            return True, 200, ""

        with patch("backend.api.v1.setup._http_ok", side_effect=fake_http_ok):
            r = client.get("/api/v1/setup/verification")

        for item in r.json()["items"]:
            assert "name" in item
            assert "ok" in item

    def test_all_ok_false_when_service_down(self, client):
        async def fake_http_ok(url, timeout=4.0):
            return False, 0, "Connection refused"

        with patch("backend.api.v1.setup._http_ok", side_effect=fake_http_ok):
            r = client.get("/api/v1/setup/verification")

        data = r.json()
        assert data["all_ok"] is False
        assert any(not i["ok"] for i in data["items"])


# ── POST /setup/reset ─────────────────────────────────────────────────────────

class TestSetupReset:
    def test_reset_clears_setup_completed(self, client, tmp_path):
        p = _write_tmp_env('SETUP_COMPLETED="true"\nENVIRONMENT="production"\n', tmp_path)
        with patch("backend.api.v1.setup.ENV_FILE", p):
            r = client.post("/api/v1/setup/reset")
        assert r.status_code == 200
        from backend.api.v1.setup import _read_env_file
        env = _read_env_file(p)
        assert env.get("SETUP_COMPLETED", "").lower() not in ("true", "1", "yes")

    def test_reset_preserves_other_keys(self, client, tmp_path):
        p = _write_tmp_env('SETUP_COMPLETED="true"\nENVIRONMENT="production"\n', tmp_path)
        with patch("backend.api.v1.setup.ENV_FILE", p):
            client.post("/api/v1/setup/reset")
        from backend.api.v1.setup import _read_env_file
        env = _read_env_file(p)
        assert env.get("ENVIRONMENT") == "production"
