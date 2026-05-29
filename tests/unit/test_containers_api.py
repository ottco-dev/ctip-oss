"""
tests/unit/test_containers_api.py — Unit tests for backend/api/v1/containers.py

All Docker and compose calls are mocked — no daemon required.

Coverage:
  _detect_port_conflict    — regex extraction of conflicting port from docker output
  read_env_file            — parse .env into dict
  write_env_key            — write / update key in .env without clobbering others
  write_env_keys           — multi-key atomic write
  GET  /containers         — list containers (mocked docker ps output)
  POST /containers/{name}/start|stop|restart  — container lifecycle
  DELETE /containers/{name}                   — stop + rm
  GET  /containers/{name}/logs                — tail logs
  GET  /compose/config                        — compose ps + env file read
  POST /compose/up                            — synchronous compose up
  POST /compose/down                          — compose down
  POST /compose/up/background                 — fire-and-forget, returns task_id
  GET  /compose/task/{task_id}                — poll task status
  GET  /compose/tasks                         — list all tasks
  POST /compose/reinstall/background          — pull + up (pull skips buildable)
  GET  /compose/ports                         — list PORT_* entries from registry
  PATCH /compose/ports                        — write port to .env + derived vars
  POST /{name}/pull                           — per-container pull + restart
"""

from __future__ import annotations

import json
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


# ──────────────────────────────────────────────────────────────────────────────
# App fixture
# ──────────────────────────────────────────────────────────────────────────────

def _make_app() -> FastAPI:
    from backend.api.v1.containers import router
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    return app


@pytest.fixture()
def client() -> Generator[TestClient, None, None]:
    app = _make_app()
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


# ──────────────────────────────────────────────────────────────────────────────
# env_file utility tests (pure, no HTTP)
# ──────────────────────────────────────────────────────────────────────────────

class TestReadEnvFile:
    def test_empty_file(self, tmp_path):
        from backend.utils.env_file import read_env_file
        f = tmp_path / ".env"
        f.write_text("")
        assert read_env_file(f) == {}

    def test_parses_key_value(self, tmp_path):
        from backend.utils.env_file import read_env_file
        f = tmp_path / ".env"
        f.write_text('FOO="bar"\nBAZ="qux"\n')
        r = read_env_file(f)
        assert r["FOO"] == "bar"
        assert r["BAZ"] == "qux"

    def test_strips_quotes(self, tmp_path):
        from backend.utils.env_file import read_env_file
        f = tmp_path / ".env"
        f.write_text("A='single'\nB=\"double\"\nC=noquotes\n")
        r = read_env_file(f)
        assert r["A"] == "single"
        assert r["B"] == "double"
        assert r["C"] == "noquotes"

    def test_ignores_comments(self, tmp_path):
        from backend.utils.env_file import read_env_file
        f = tmp_path / ".env"
        f.write_text("# comment\nFOO=bar\n# another comment\n")
        r = read_env_file(f)
        assert "# comment" not in r
        assert r["FOO"] == "bar"

    def test_missing_file_returns_empty(self, tmp_path):
        from backend.utils.env_file import read_env_file
        assert read_env_file(tmp_path / "nonexistent.env") == {}


class TestWriteEnvKey:
    def test_updates_existing_key(self, tmp_path):
        from backend.utils.env_file import write_env_key, read_env_file
        f = tmp_path / ".env"
        f.write_text('FOO="old"\nBAR="keep"\n')
        write_env_key("FOO", "new", path=f)
        r = read_env_file(f)
        assert r["FOO"] == "new"
        assert r["BAR"] == "keep"

    def test_appends_new_key(self, tmp_path):
        from backend.utils.env_file import write_env_key, read_env_file
        f = tmp_path / ".env"
        f.write_text('FOO="bar"\n')
        write_env_key("NEW_KEY", "value", path=f)
        r = read_env_file(f)
        assert r["NEW_KEY"] == "value"
        assert r["FOO"] == "bar"

    def test_preserves_comments(self, tmp_path):
        from backend.utils.env_file import write_env_key
        f = tmp_path / ".env"
        f.write_text("# My comment\nFOO=\"bar\"\n")
        write_env_key("FOO", "new", path=f)
        content = f.read_text()
        assert "# My comment" in content

    def test_creates_file_if_missing(self, tmp_path):
        from backend.utils.env_file import write_env_key, read_env_file
        f = tmp_path / "new.env"
        write_env_key("KEY", "val", path=f)
        assert read_env_file(f)["KEY"] == "val"

    def test_write_env_keys_multiple_atomic(self, tmp_path):
        from backend.utils.env_file import write_env_keys, read_env_file
        f = tmp_path / ".env"
        f.write_text('A="1"\nB="2"\nC="3"\n')
        write_env_keys({"A": "10", "C": "30"}, path=f)
        r = read_env_file(f)
        assert r["A"] == "10"
        assert r["B"] == "2"   # untouched
        assert r["C"] == "30"


# ──────────────────────────────────────────────────────────────────────────────
# Port conflict detection
# ──────────────────────────────────────────────────────────────────────────────

class TestDetectPortConflict:
    def _detect(self, lines, env_overrides=None):
        """Call _detect_port_conflict with controlled env."""
        from backend.api.v1.containers import _detect_port_conflict, _PORT_REGISTRY
        with patch.dict(os.environ, env_overrides or {}):
            return _detect_port_conflict(lines)

    def test_detects_real_docker_error(self):
        lines = [
            "Error response from daemon: failed to set up container networking: "
            "driver failed programming external connectivity on endpoint trichome-mlflow: "
            "failed to bind host port 0.0.0.0:3004/tcp: address already in use",
        ]
        r = self._detect(lines, {"PORT_MLFLOW": "3004"})
        assert r is not None
        assert r.port == 3004
        assert r.env_var == "PORT_MLFLOW"
        assert "MLflow" in r.service

    def test_detects_short_format(self):
        lines = ["Error: 3005/tcp: address already in use"]
        r = self._detect(lines, {"PORT_LABEL_STUDIO": "3005"})
        assert r is not None
        assert r.port == 3005
        assert r.env_var == "PORT_LABEL_STUDIO"

    def test_returns_none_if_no_conflict(self):
        lines = ["Everything started successfully", "All containers healthy"]
        assert self._detect(lines) is None

    def test_unknown_port_returns_generic(self):
        lines = ["Error: 9999/tcp: address already in use"]
        r = self._detect(lines, {})
        assert r is not None
        assert r.port == 9999
        assert r.env_var == ""   # unknown → no env var

    def test_detects_nginx_port(self):
        lines = ["failed to bind host port 0.0.0.0:3001/tcp: address already in use"]
        r = self._detect(lines, {"PORT_NGINX": "3001"})
        assert r is not None
        assert r.port == 3001


# ──────────────────────────────────────────────────────────────────────────────
# Container list endpoint
# ──────────────────────────────────────────────────────────────────────────────

_FAKE_CONTAINER = json.dumps({
    "id": "abc123def456",
    "name": "trichome-backend",
    "image": "trichome-backend:dev",
    "state": "running",
    "status": "Up 5 minutes",
    "ports": "0.0.0.0:3002->8000/tcp",
    "labels": "com.docker.compose.project=trichome,com.docker.compose.service=backend",
})


class TestListContainers:
    def test_returns_empty_on_docker_failure(self, client):
        with patch("backend.api.v1.containers._docker", return_value=(False, "docker not found")):
            r = client.get("/api/v1/containers")
        assert r.status_code == 200
        assert r.json() == []

    def test_parses_running_container(self, client):
        with patch("backend.api.v1.containers._docker", return_value=(True, _FAKE_CONTAINER)):
            r = client.get("/api/v1/containers")
        assert r.status_code == 200
        data = r.json()
        assert len(data) == 1
        assert data[0]["name"] == "trichome-backend"
        assert data[0]["running"] is True
        assert data[0]["compose_service"] == "backend"

    def test_all_flag_passed(self, client):
        calls = []
        def mock_docker(*args, **kwargs):
            calls.append(args)
            return True, _FAKE_CONTAINER
        with patch("backend.api.v1.containers._docker", side_effect=mock_docker):
            client.get("/api/v1/containers?all=true")
        assert any("--all" in c for c in calls)

    def test_skips_malformed_json_lines(self, client):
        bad_output = "not json\n" + _FAKE_CONTAINER + "\nalso not json"
        with patch("backend.api.v1.containers._docker", return_value=(True, bad_output)):
            r = client.get("/api/v1/containers")
        assert r.status_code == 200
        assert len(r.json()) == 1   # only the valid line


# ──────────────────────────────────────────────────────────────────────────────
# Container lifecycle (start / stop / restart / delete)
# ──────────────────────────────────────────────────────────────────────────────

class TestContainerLifecycle:
    def _mock_ok(self, *_args, **_kwargs):
        return True, "done"

    def _mock_fail(self, *_args, **_kwargs):
        return False, "error: no such container"

    def test_start_ok(self, client):
        with patch("backend.api.v1.containers._docker", side_effect=self._mock_ok):
            r = client.post("/api/v1/containers/trichome-backend/start")
        assert r.status_code == 200
        assert r.json()["ok"] is True

    def test_stop_ok(self, client):
        with patch("backend.api.v1.containers._docker", side_effect=self._mock_ok):
            r = client.post("/api/v1/containers/trichome-backend/stop")
        assert r.status_code == 200
        assert r.json()["ok"] is True

    def test_restart_ok(self, client):
        with patch("backend.api.v1.containers._docker", side_effect=self._mock_ok):
            r = client.post("/api/v1/containers/trichome-backend/restart")
        assert r.status_code == 200
        assert r.json()["ok"] is True

    def test_start_failure_returns_ok_false(self, client):
        with patch("backend.api.v1.containers._docker", side_effect=self._mock_fail):
            r = client.post("/api/v1/containers/nonexistent/start")
        assert r.status_code == 200
        assert r.json()["ok"] is False

    def test_delete_calls_stop_then_rm(self, client):
        calls = []
        def track(*args, **kw):
            calls.append(args)
            return True, "ok"
        with patch("backend.api.v1.containers._docker", side_effect=track):
            r = client.delete("/api/v1/containers/trichome-backend")
        assert r.status_code == 200
        verbs = [c[0] for c in calls]
        assert "stop" in verbs
        assert "rm" in verbs


# ──────────────────────────────────────────────────────────────────────────────
# Container logs
# ──────────────────────────────────────────────────────────────────────────────

class TestContainerLogs:
    def test_returns_logs_string(self, client):
        with patch("backend.api.v1.containers._docker", return_value=(True, "line1\nline2\n")):
            r = client.get("/api/v1/containers/trichome-backend/logs?tail=10")
        assert r.status_code == 200
        assert "line1" in r.json()["logs"]

    def test_error_wrapped_in_response(self, client):
        with patch("backend.api.v1.containers._docker", return_value=(False, "no such container")):
            r = client.get("/api/v1/containers/nonexistent/logs")
        assert r.status_code == 200
        assert "[error]" in r.json()["logs"]


# ──────────────────────────────────────────────────────────────────────────────
# compose config
# ──────────────────────────────────────────────────────────────────────────────

class TestComposeConfig:
    def test_returns_compose_config(self, client, tmp_path):
        svc_json = json.dumps({"Service": "backend", "Image": "trichome:dev", "Status": "running"})
        env_file = tmp_path / ".env"
        env_file.write_text('FOO="bar"\n')

        with patch("backend.api.v1.containers._compose", return_value=(True, svc_json)), \
             patch("backend.api.v1.containers.ENV_FILE", env_file):
            r = client.get("/api/v1/containers/compose/config")

        assert r.status_code == 200
        data = r.json()
        assert "services" in data
        assert len(data["services"]) == 1
        assert data["services"][0]["service"] == "backend"

    def test_env_file_path_returned(self, client, tmp_path):
        """compose/config returns env_file as path string and raw_env as dict."""
        env_file = tmp_path / ".env"
        env_file.write_text('FOO="bar"\nBAZ="qux"\n')

        with patch("backend.api.v1.containers._compose", return_value=(True, "")), \
             patch.object(
                 __import__("backend.api.v1.containers", fromlist=["REPO_ROOT"]),
                 "REPO_ROOT", tmp_path
             ):
            r = client.get("/api/v1/containers/compose/config")

        assert r.status_code == 200
        data = r.json()
        assert "env_file" in data          # path string
        assert isinstance(data["env_file"], str)
        assert "raw_env" in data           # plain dict
        assert isinstance(data["raw_env"], dict)


# ──────────────────────────────────────────────────────────────────────────────
# compose up / down (synchronous)
# ──────────────────────────────────────────────────────────────────────────────

class TestComposeUpDown:
    def test_compose_up_ok(self, client):
        with patch("backend.api.v1.containers._compose", return_value=(True, "done")):
            r = client.post("/api/v1/containers/compose/up?profile=annotation")
        assert r.status_code == 200
        assert r.json()["ok"] is True

    def test_compose_up_failure(self, client):
        with patch("backend.api.v1.containers._compose", return_value=(False, "port in use")):
            r = client.post("/api/v1/containers/compose/up")
        assert r.status_code == 200
        assert r.json()["ok"] is False

    def test_compose_down_ok(self, client):
        with patch("backend.api.v1.containers._compose", return_value=(True, "Networks removed")):
            r = client.post("/api/v1/containers/compose/down")
        assert r.status_code == 200
        assert r.json()["ok"] is True


# ──────────────────────────────────────────────────────────────────────────────
# Background task system
# ──────────────────────────────────────────────────────────────────────────────

class TestBackgroundTasks:
    def test_up_background_returns_task_id(self, client):
        with patch("asyncio.create_task"):
            r = client.post("/api/v1/containers/compose/up/background?profile=annotation")
        assert r.status_code == 200
        task_id = r.json()["task_id"]
        assert isinstance(task_id, str) and len(task_id) > 0

    def test_poll_task_found(self, client, tmp_path):
        # Inject a fake completed task directly into the TaskStore cache
        from backend.tasks.task_store import TaskRecord, get_task_store
        from backend.api.v1.containers import _TASK_DB
        store = get_task_store(_TASK_DB)
        tid = str(uuid.uuid4())
        rec = TaskRecord(
            id=tid, profile="annotation", status="done",
            started_at=time.time() - 10, finished_at=time.time(),
            ok=True, log=["step 1", "step 2"],
        )
        store._cache[tid] = rec
        try:
            r = client.get(f"/api/v1/containers/compose/task/{tid}")
            assert r.status_code == 200
            data = r.json()
            assert data["status"] == "done"
            assert data["ok"] is True
            assert len(data["log"]) == 2
            assert data["elapsed_seconds"] is not None
        finally:
            store._cache.pop(tid, None)

    def test_poll_task_not_found(self, client):
        r = client.get("/api/v1/containers/compose/task/nonexistent-uuid")
        assert r.status_code == 404

    def test_list_tasks_returns_list(self, client):
        from backend.tasks.task_store import TaskRecord, get_task_store
        from backend.api.v1.containers import _TASK_DB
        store = get_task_store(_TASK_DB)
        tid = str(uuid.uuid4())
        rec = TaskRecord(
            id=tid, profile="annotation", status="error",
            started_at=time.time() - 5, finished_at=time.time(), ok=False,
        )
        store._cache[tid] = rec
        try:
            r = client.get("/api/v1/containers/compose/tasks")
            assert r.status_code == 200
            ids = [t["id"] for t in r.json()]
            assert tid in ids
        finally:
            store._cache.pop(tid, None)

    def test_port_conflict_in_task(self, client):
        """Task with status=port_conflict exposes PortConflictInfo."""
        from backend.tasks.task_store import TaskRecord, PortConflictData, get_task_store
        from backend.api.v1.containers import _TASK_DB
        store = get_task_store(_TASK_DB)
        tid = str(uuid.uuid4())
        rec = TaskRecord(
            id=tid, profile="annotation", status="port_conflict",
            started_at=time.time() - 3, finished_at=time.time(), ok=False,
            port_conflict=PortConflictData(port=3004, service="MLflow Tracking", env_var="PORT_MLFLOW"),
        )
        store._cache[tid] = rec
        try:
            r = client.get(f"/api/v1/containers/compose/task/{tid}")
            assert r.status_code == 200
            data = r.json()
            assert data["status"] == "port_conflict"
            assert data["port_conflict"]["port"] == 3004
            assert data["port_conflict"]["env_var"] == "PORT_MLFLOW"
        finally:
            store._cache.pop(tid, None)

    def test_reinstall_background_returns_task_id(self, client):
        with patch("asyncio.create_task"):
            r = client.post("/api/v1/containers/compose/reinstall/background?profile=annotation")
        assert r.status_code == 200
        assert "task_id" in r.json()


# ──────────────────────────────────────────────────────────────────────────────
# Port management endpoints
# ──────────────────────────────────────────────────────────────────────────────

class TestPortEndpoints:
    def test_get_ports_returns_all_services(self, client):
        r = client.get("/api/v1/containers/compose/ports")
        assert r.status_code == 200
        entries = r.json()
        assert isinstance(entries, list)
        env_vars = {e["env_var"] for e in entries}
        assert "PORT_NGINX" in env_vars
        assert "PORT_MLFLOW" in env_vars
        assert "PORT_BACKEND" in env_vars
        assert "PORT_LABEL_STUDIO" in env_vars

    def test_get_ports_has_required_fields(self, client):
        r = client.get("/api/v1/containers/compose/ports")
        for entry in r.json():
            assert "env_var" in entry
            assert "current_port" in entry
            assert "default_port" in entry
            assert "label" in entry
            assert isinstance(entry["current_port"], int)

    def test_patch_port_valid(self, client, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text('PORT_MLFLOW="3004"\nMLFLOW_TRACKING_URI="http://localhost:3004"\n')
        with patch("backend.api.v1.containers.ENV_FILE", env_file), \
             patch("backend.utils.env_file._DEFAULT_ENV_FILE", env_file):
            r = client.patch("/api/v1/containers/compose/ports", json={
                "env_var": "PORT_MLFLOW",
                "port": 3099,
            })
        assert r.status_code == 200
        assert r.json()["ok"] is True
        content = env_file.read_text()
        assert 'PORT_MLFLOW="3099"' in content

    def test_patch_port_updates_mlflow_uri(self, client, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text('PORT_MLFLOW="3004"\nMLFLOW_TRACKING_URI="http://localhost:3004"\n')
        with patch("backend.api.v1.containers.ENV_FILE", env_file), \
             patch("backend.utils.env_file._DEFAULT_ENV_FILE", env_file):
            client.patch("/api/v1/containers/compose/ports", json={
                "env_var": "PORT_MLFLOW",
                "port": 4000,
            })
        content = env_file.read_text()
        assert 'MLFLOW_TRACKING_URI="http://localhost:4000"' in content

    def test_patch_port_rejects_privileged(self, client):
        r = client.patch("/api/v1/containers/compose/ports", json={
            "env_var": "PORT_NGINX",
            "port": 80,
        })
        assert r.status_code == 400

    def test_patch_port_rejects_out_of_range(self, client):
        r = client.patch("/api/v1/containers/compose/ports", json={
            "env_var": "PORT_NGINX",
            "port": 99999,
        })
        assert r.status_code == 400

    def test_patch_port_rejects_unknown_env_var(self, client):
        r = client.patch("/api/v1/containers/compose/ports", json={
            "env_var": "SOME_UNKNOWN_VAR",
            "port": 5000,
        })
        assert r.status_code == 400


# ──────────────────────────────────────────────────────────────────────────────
# Per-container pull endpoint
# ──────────────────────────────────────────────────────────────────────────────

class TestPerContainerPull:
    def test_pull_unknown_container_returns_404(self, client):
        with patch("backend.api.v1.containers._docker", return_value=(False, "no such container")):
            r = client.post("/api/v1/containers/nonexistent/pull")
        assert r.status_code == 404

    def test_pull_success(self, client):
        call_count = [0]
        def mock_docker(*args, **kw):
            call_count[0] += 1
            if "inspect" in args:
                return True, "trichome-backend:dev"
            return True, "ok"
        with patch("backend.api.v1.containers._docker", side_effect=mock_docker):
            r = client.post("/api/v1/containers/trichome-backend/pull")
        assert r.status_code == 200
        assert r.json()["ok"] is True
        # inspect + pull + restart = 3 calls
        assert call_count[0] == 3

    def test_pull_failure_returns_ok_false(self, client):
        def mock_docker(*args, **kw):
            if "inspect" in args:
                return True, "some-image:latest"
            if "pull" in args:
                return False, "unauthorized"
            return True, "ok"
        with patch("backend.api.v1.containers._docker", side_effect=mock_docker):
            r = client.post("/api/v1/containers/trichome-backend/pull")
        assert r.status_code == 200
        assert r.json()["ok"] is False
        assert "pull failed" in r.json()["detail"]
