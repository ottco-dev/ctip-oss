"""
Container management API — list, start, stop, restart, logs, compose config.
Background task system for long-running docker compose operations.
All docker operations run in a thread to avoid blocking the event loop.

Port conflict handling:
  When `docker compose up` fails with "address already in use", the task transitions
  to status "port_conflict" and exposes the conflicting port + service.  The frontend
  shows a dialog so the user can pick a new host port.  The new port is written to
  .env (PORT_<SERVICE>=NNNN) and derived env vars (MLFLOW_TRACKING_URI etc.) are
  updated automatically.  The reinstall is then retried with the updated config.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

router = APIRouter(prefix="/containers", tags=["containers"])

# ── Background task store ─────────────────────────────────────────────────────
# SQLite-backed via backend.tasks.task_store — survives backend restarts.
# In-memory cache keeps the hot path fast; DB writes happen at status transitions.

TaskStatus = Literal["queued", "running", "done", "error", "port_conflict"]


class PortConflictInfo(BaseModel):
    port: int
    service: str          # human label, e.g. "MLflow"
    env_var: str          # e.g. "PORT_MLFLOW"


class BgTask(BaseModel):
    id: str
    status: TaskStatus
    started_at: float
    finished_at: float | None = None
    ok: bool | None = None
    log: list[str] = []
    profile: str = "annotation"
    port_conflict: PortConflictInfo | None = None


from backend.tasks.task_store import (
    TaskRecord,
    PortConflictData,
    get_task_store,
    MAX_LOG_LINES,
)

# Initialised at first request (lazy) — avoids import-time side-effects.
# Explicitly initialised by the startup event in backend/main.py (if present).
_TASK_DB: Path | None = None  # resolved once REPO_ROOT is known


def _store():
    """Return the initialized TaskStore singleton."""
    from backend.tasks.task_store import get_task_store
    return get_task_store(_TASK_DB)


def _task_to_bg(t: TaskRecord) -> BgTask:
    """Convert a TaskRecord to a BgTask Pydantic model for API responses."""
    pc = None
    if t.port_conflict:
        pc = PortConflictInfo(
            port=t.port_conflict.port,
            service=t.port_conflict.service,
            env_var=t.port_conflict.env_var,
        )
    return BgTask(
        id=t.id,
        status=t.status,  # type: ignore[arg-type]
        started_at=t.started_at,
        finished_at=t.finished_at,
        ok=t.ok,
        log=t.log,
        profile=t.profile,
        port_conflict=pc,
    )


# ── Port config registry ───────────────────────────────────────────────────────
# Maps service key → (env_var, default_port, human label, derived env vars to update)
_PORT_REGISTRY: dict[str, tuple[str, int, str, list[str]]] = {
    "nginx":         ("PORT_NGINX",         3001, "Nginx (Public Entry Point)", ["PUBLIC_PORT"]),
    "backend":       ("PORT_BACKEND",       3002, "FastAPI Backend",            []),
    "frontend":      ("PORT_FRONTEND",      3003, "Next.js Frontend",           []),
    "mlflow":        ("PORT_MLFLOW",        3004, "MLflow Tracking",            ["MLFLOW_TRACKING_URI"]),
    "label-studio":  ("PORT_LABEL_STUDIO",  3005, "Label Studio",               ["LABEL_STUDIO_URL"]),
    "cvat":          ("PORT_CVAT",          3006, "CVAT",                       []),
}

# Pattern: "failed to bind host port 0.0.0.0:3004/tcp: address already in use"
_PORT_CONFLICT_RE = re.compile(
    r"(?:bind host port|already in use|address already in use)[^\d]*(\d{3,5})"
    r"|"
    r"(\d{3,5})/tcp.*address already in use"
)

from backend.utils.env_file import get_env_path, read_env_file, write_env_key as _write_env_key

ENV_FILE = get_env_path()


# ── Repo root ─────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[3]
COMPOSE_FILE = REPO_ROOT / "docker" / "docker-compose.yml"
COMPOSE_DIR = REPO_ROOT / "docker"

# Resolve task DB path now that REPO_ROOT is set
_TASK_DB = REPO_ROOT / "data" / "tasks.db"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _docker(*args: str, timeout: int = 30) -> tuple[bool, str]:
    try:
        r = subprocess.run(
            ["docker", *args],
            capture_output=True, text=True, timeout=timeout,
        )
        out = (r.stdout + r.stderr).strip()
        return r.returncode == 0, out
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except FileNotFoundError:
        return False, "docker not found"


def _compose(*args: str, timeout: int = 60) -> tuple[bool, str]:
    cmd = [
        "docker", "compose",
        "--project-directory", str(COMPOSE_DIR),
        "-f", str(COMPOSE_FILE),
        *args,
    ]
    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            cwd=str(COMPOSE_DIR),
        )
        out = (r.stdout + r.stderr).strip()
        return r.returncode == 0, out
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except FileNotFoundError:
        return False, "docker not found"


# ── Schemas ───────────────────────────────────────────────────────────────────

class ContainerSummary(BaseModel):
    id: str
    name: str
    image: str
    status: str          # "running" | "exited" | "created" | ...
    state: str           # raw State field
    ports: str
    running: bool
    compose_project: str | None = None
    compose_service: str | None = None


class ContainerAction(BaseModel):
    pass  # empty — action identified by route


class ContainerActionResponse(BaseModel):
    ok: bool
    detail: str


class ComposeService(BaseModel):
    service: str
    image: str
    status: str
    ports: list[str]


class ComposeConfig(BaseModel):
    project: str
    services: list[ComposeService]
    env_file: str
    raw_env: dict[str, str]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("", response_model=list[ContainerSummary])
async def list_containers(all: bool = True) -> list[ContainerSummary]:
    """List all Docker containers (default: all, including stopped)."""
    fmt = (
        '{"id":"{{.ID}}","name":"{{.Names}}","image":"{{.Image}}",'
        '"state":"{{.State}}","status":"{{.Status}}","ports":"{{.Ports}}",'
        '"labels":"{{.Labels}}"}'
    )
    flag = "--all" if all else ""
    args = ["ps", "--format", fmt]
    if flag:
        args.insert(1, flag)

    ok, out = await asyncio.to_thread(_docker, *args)
    if not ok:
        return []

    result: list[ContainerSummary] = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            raw: dict[str, Any] = json.loads(line)
        except json.JSONDecodeError:
            continue

        labels: dict[str, str] = {}
        for part in raw.get("labels", "").split(","):
            if "=" in part:
                k, v = part.split("=", 1)
                labels[k.strip()] = v.strip()

        result.append(ContainerSummary(
            id=raw.get("id", "")[:12],
            name=raw.get("name", "").lstrip("/"),
            image=raw.get("image", ""),
            state=raw.get("state", ""),
            status=raw.get("status", ""),
            ports=raw.get("ports", ""),
            running=raw.get("state", "") == "running",
            compose_project=labels.get("com.docker.compose.project"),
            compose_service=labels.get("com.docker.compose.service"),
        ))
    return result


@router.post("/{name}/start", response_model=ContainerActionResponse)
async def start_container(name: str) -> ContainerActionResponse:
    ok, out = await asyncio.to_thread(_docker, "start", name)
    return ContainerActionResponse(ok=ok, detail=out[:500])


@router.post("/{name}/stop", response_model=ContainerActionResponse)
async def stop_container(name: str) -> ContainerActionResponse:
    ok, out = await asyncio.to_thread(_docker, "stop", name, timeout=30)
    return ContainerActionResponse(ok=ok, detail=out[:500])


@router.post("/{name}/restart", response_model=ContainerActionResponse)
async def restart_container(name: str) -> ContainerActionResponse:
    ok, out = await asyncio.to_thread(_docker, "restart", name, timeout=30)
    return ContainerActionResponse(ok=ok, detail=out[:500])


@router.delete("/{name}", response_model=ContainerActionResponse)
async def remove_container(name: str) -> ContainerActionResponse:
    """Stop and remove a container (does NOT remove image or volume)."""
    await asyncio.to_thread(_docker, "stop", name, timeout=15)
    ok, out = await asyncio.to_thread(_docker, "rm", "-f", name)
    return ContainerActionResponse(ok=ok, detail=out[:500])


@router.get("/{name}/logs")
async def container_logs(name: str, tail: int = 200) -> dict[str, str]:
    ok, out = await asyncio.to_thread(_docker, "logs", "--tail", str(tail), name, timeout=10)
    return {"logs": out if ok else f"[error] {out}"}


@router.get("/{name}/logs/stream")
async def container_logs_stream(name: str, tail: int = 50):
    """SSE stream of live container logs (docker logs -f)."""
    async def _gen():
        import asyncio.subprocess as _asp
        proc = await _asp.create_subprocess_exec(
            "docker", "logs", "-f", "--tail", str(tail), name,
            stdout=_asp.PIPE, stderr=_asp.STDOUT,
        )
        assert proc.stdout is not None
        try:
            async for raw in proc.stdout:
                line = raw.decode(errors="replace").rstrip()
                if line:
                    yield f"data: {line}\n\n"
        finally:
            try:
                proc.kill()
            except Exception:
                pass
            await proc.wait()
            yield "data: [STREAM_END]\n\n"

    return StreamingResponse(
        _gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/compose/config", response_model=ComposeConfig)
async def compose_config() -> ComposeConfig:
    """Read docker compose project state + env file."""
    ok, out = await asyncio.to_thread(
        _compose, "--profile", "annotation", "ps", "--format", "json",
        timeout=15,
    )

    services: list[ComposeService] = []
    if ok and out:
        for line in out.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                svc = json.loads(line)
                services.append(ComposeService(
                    service=svc.get("Service", svc.get("Name", "")),
                    image=svc.get("Image", ""),
                    status=svc.get("Status", ""),
                    ports=[p.get("URL", "") for p in svc.get("Publishers", []) if p.get("URL")],
                ))
            except (json.JSONDecodeError, AttributeError):
                continue

    # Read .env file
    env_path = REPO_ROOT / ".env"
    raw_env: dict[str, str] = {}
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                raw_env[k.strip()] = v.strip()

    return ComposeConfig(
        project="ctip-oss",
        services=services,
        env_file=str(env_path),
        raw_env=raw_env,
    )


@router.post("/compose/up", response_model=ContainerActionResponse)
async def compose_up(profile: str = "annotation") -> ContainerActionResponse:
    ok, out = await asyncio.to_thread(_compose, "--profile", profile, "up", "-d", "--remove-orphans", timeout=600)
    return ContainerActionResponse(ok=ok, detail=out[:2000])


@router.post("/compose/down", response_model=ContainerActionResponse)
async def compose_down(profile: str = "annotation") -> ContainerActionResponse:
    ok, out = await asyncio.to_thread(_compose, "--profile", profile, "down", timeout=60)
    return ContainerActionResponse(ok=ok, detail=out[:500])


@router.get("/compose/up/stream")
async def compose_up_stream(profile: str = "annotation"):
    """SSE stream of docker compose up output (requires open connection)."""
    async def _gen():
        import asyncio.subprocess as _asp
        cmd = [
            "docker", "compose",
            "--project-directory", str(COMPOSE_DIR),
            "-f", str(COMPOSE_FILE),
            "--profile", profile,
            "up", "-d", "--remove-orphans",
        ]
        proc = await _asp.create_subprocess_exec(
            *cmd,
            stdout=_asp.PIPE, stderr=_asp.STDOUT,
            cwd=str(COMPOSE_DIR),
        )
        assert proc.stdout is not None
        try:
            async for raw in proc.stdout:
                line = raw.decode(errors="replace").rstrip()
                if line:
                    yield f"data: {line}\n\n"
        finally:
            await proc.wait()
            ok = proc.returncode == 0
            yield f"data: [DONE:{'OK' if ok else f'ERROR({proc.returncode})'}]\n\n"

    return StreamingResponse(
        _gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Background task endpoints ─────────────────────────────────────────────────

async def _run_compose_bg(task_id: str, profile: str) -> None:
    """Async background worker — runs docker compose up and updates persistent task store."""
    import asyncio.subprocess as _asp

    store = _store()
    task = store.get(task_id)
    if task is None:
        return

    task.status = "running"
    await store.flush(task)

    cmd = [
        "docker", "compose",
        "--project-directory", str(COMPOSE_DIR),
        "-f", str(COMPOSE_FILE),
        "--profile", profile,
        "up", "-d", "--remove-orphans",
    ]
    try:
        proc = await _asp.create_subprocess_exec(
            *cmd,
            stdout=_asp.PIPE, stderr=_asp.STDOUT,
            cwd=str(COMPOSE_DIR),
        )
        assert proc.stdout is not None
        async for raw in proc.stdout:
            line = raw.decode(errors="replace").rstrip()
            if line:
                task.log.append(line)
                if len(task.log) > MAX_LOG_LINES:
                    task.log = task.log[-MAX_LOG_LINES:]
                # Periodic log flush to survive crashes mid-run
                if store.should_flush_log(task):
                    await store.flush(task)
        await proc.wait()
        task.ok = proc.returncode == 0
        task.status = "done" if task.ok else "error"
    except Exception as exc:
        task.log.append(f"[fatal] {exc}")
        task.ok = False
        task.status = "error"
    finally:
        task.finished_at = time.time()
        await store.flush(task)


class BgTaskResponse(BaseModel):
    task_id: str


class BgTaskStatus(BaseModel):
    id: str
    status: TaskStatus
    started_at: float
    finished_at: float | None
    ok: bool | None
    log: list[str]
    profile: str
    elapsed_seconds: float | None
    port_conflict: PortConflictInfo | None = None


@router.post("/compose/up/background", response_model=BgTaskResponse)
async def compose_up_background(profile: str = "annotation") -> BgTaskResponse:
    """Start docker compose up as a fire-and-forget background task.
    Returns task_id immediately — poll /compose/task/{task_id} for status.
    Task is persisted to SQLite so it survives backend restarts.
    """
    store = _store()
    task = store.create(profile)
    asyncio.create_task(_run_compose_bg(task.id, profile))
    return BgTaskResponse(task_id=task.id)


@router.get("/compose/task/{task_id}", response_model=BgTaskStatus)
async def get_compose_task(task_id: str) -> BgTaskStatus:
    """Poll a background compose task for its current status and log.
    Tasks survive backend restarts — history kept for 24 hours.
    """
    task = _store().get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    bg = _task_to_bg(task)
    elapsed = task.elapsed_seconds
    return BgTaskStatus(
        id=bg.id,
        status=bg.status,
        started_at=bg.started_at,
        finished_at=bg.finished_at,
        ok=bg.ok,
        log=bg.log,
        profile=bg.profile,
        elapsed_seconds=elapsed,
        port_conflict=bg.port_conflict,
    )


@router.get("/compose/tasks", response_model=list[BgTaskStatus])
async def list_compose_tasks() -> list[BgTaskStatus]:
    """List background compose tasks — most recent first, last 20, up to 24h history."""
    tasks = _store().list_recent(20)
    result = []
    for task in tasks:
        bg = _task_to_bg(task)
        result.append(BgTaskStatus(
            id=bg.id,
            status=bg.status,
            started_at=bg.started_at,
            finished_at=bg.finished_at,
            ok=bg.ok,
            log=bg.log,
            profile=bg.profile,
            elapsed_seconds=task.elapsed_seconds,
            port_conflict=bg.port_conflict,
        ))
    return result


def _detect_port_conflict(log_lines: list[str]) -> PortConflictData | None:
    """
    Scan docker compose output for port-conflict errors.

    Returns PortConflictData if found, None otherwise.
    Pattern examples:
      "failed to bind host port 0.0.0.0:3004/tcp: address already in use"
      "Error response from daemon: … 3004/tcp: address already in use"
    """
    combined = " ".join(log_lines)
    m = _PORT_CONFLICT_RE.search(combined)
    if not m:
        return None
    port_str = m.group(1) or m.group(2)
    if not port_str:
        return None
    conflict_port = int(port_str)
    # Reverse-map port → registry entry
    for svc, (env_var, default_port, label, _) in _PORT_REGISTRY.items():
        current = int(os.getenv(env_var, str(default_port)))
        if current == conflict_port:
            return PortConflictData(port=conflict_port, service=label, env_var=env_var)
    # Port found but not mapped — return generic entry
    return PortConflictData(
        port=conflict_port,
        service=f"Port {conflict_port}",
        env_var="",
    )


async def _run_reinstall_bg(task_id: str, profile: str) -> None:
    """
    Pull fresh images + recreate all compose containers.

    Pull skips services with a build: section (--ignore-buildable) so
    trichome-backend:dev (locally built) is never pulled from a registry.

    If docker compose up fails due to a port conflict, the task transitions to
    status="port_conflict" instead of "error" so the frontend can show a dialog.

    Task is persisted to SQLite at each status transition and every LOG_FLUSH_EVERY
    log lines so state survives backend restarts.
    """
    import asyncio.subprocess as _asp

    store = _store()
    task = store.get(task_id)
    if task is None:
        return

    task.status = "running"
    await store.flush(task)

    base = [
        "docker", "compose",
        "--project-directory", str(COMPOSE_DIR),
        "-f", str(COMPOSE_FILE),
        "--profile", profile,
    ]

    pull_cmd = [*base, "pull", "--ignore-buildable"]
    up_cmd   = [*base, "up", "-d", "--force-recreate", "--remove-orphans"]

    async def _stream(cmd: list[str]) -> bool:
        try:
            proc = await _asp.create_subprocess_exec(
                *cmd,
                stdout=_asp.PIPE, stderr=_asp.STDOUT,
                cwd=str(COMPOSE_DIR),
            )
            assert proc.stdout is not None
            async for raw in proc.stdout:
                line = raw.decode(errors="replace").rstrip()
                if line:
                    task.log.append(line)
                    if len(task.log) > MAX_LOG_LINES:
                        task.log = task.log[-MAX_LOG_LINES:]
                    if store.should_flush_log(task):
                        await store.flush(task)
            await proc.wait()
            return proc.returncode == 0
        except Exception as exc:
            task.log.append(f"[fatal] {exc}")
            return False

    try:
        task.log.append("=== Step 1: docker compose pull (skipping locally-built images) ===")
        ok_pull = await _stream(pull_cmd)
        task.log.append(f"=== Pull {'OK' if ok_pull else 'FAILED'} ===")
        task.log.append("=== Step 2: docker compose up --force-recreate ===")
        ok_up = await _stream(up_cmd)
        task.log.append(f"=== Recreate {'OK' if ok_up else 'FAILED'} ===")

        if not ok_up:
            conflict = _detect_port_conflict(task.log)
            if conflict:
                task.port_conflict = conflict
                task.status = "port_conflict"
                task.ok = False
                task.finished_at = time.time()
                await store.flush(task)
                return

        task.ok = ok_up
        task.status = "done" if task.ok else "error"
    except Exception as exc:
        task.log.append(f"[fatal] {exc}")
        task.ok = False
        task.status = "error"
    finally:
        task.finished_at = time.time()
        await store.flush(task)


@router.post("/compose/reinstall/background", response_model=BgTaskResponse)
async def compose_reinstall_background(profile: str = "annotation") -> BgTaskResponse:
    """Pull fresh images + force-recreate all containers for the given profile.
    Runs as a background task — returns task_id immediately.
    Task persisted to SQLite so it survives backend restarts.
    """
    store = _store()
    task = store.create(profile)
    asyncio.create_task(_run_reinstall_bg(task.id, profile))
    return BgTaskResponse(task_id=task.id)


# ── Port management ───────────────────────────────────────────────────────────

class PortEntry(BaseModel):
    env_var: str
    current_port: int
    default_port: int
    label: str


class PortUpdateRequest(BaseModel):
    env_var: str   # e.g. "PORT_MLFLOW"
    port: int      # new host port to use




@router.get("/compose/ports", response_model=list[PortEntry])
async def get_compose_ports() -> list[PortEntry]:
    """
    Return current host port configuration for all services.

    Reads PORT_* env vars from environment (populated from .env at startup).
    """
    result = []
    for _svc, (env_var, default_port, label, _derived) in _PORT_REGISTRY.items():
        current = int(os.getenv(env_var, str(default_port)))
        result.append(PortEntry(
            env_var=env_var,
            current_port=current,
            default_port=default_port,
            label=label,
        ))
    return result


@router.patch("/compose/ports", response_model=ContainerActionResponse)
async def update_compose_port(req: PortUpdateRequest) -> ContainerActionResponse:
    """
    Update a service host port in .env and all derived env vars.

    Writes the new port to .env so it survives restarts and is picked up by
    docker compose (which reads .env from the project directory).

    Also updates derived vars:
      PORT_MLFLOW → MLFLOW_TRACKING_URI=http://localhost:<port>
      PORT_LABEL_STUDIO → LABEL_STUDIO_URL=http://localhost:<port>
      PORT_NGINX → PUBLIC_PORT=<port>
    """
    if req.port < 1024 or req.port > 65535:
        raise HTTPException(status_code=400, detail="Port must be between 1024 and 65535")

    # Validate env_var is known
    known_vars = {cfg[0] for cfg in _PORT_REGISTRY.values()}
    if req.env_var not in known_vars:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown env var '{req.env_var}'. Known: {sorted(known_vars)}",
        )

    # Write primary port var
    _write_env_key(req.env_var, str(req.port))
    os.environ[req.env_var] = str(req.port)

    # Update derived env vars
    for _svc, (env_var, _default, _label, derived) in _PORT_REGISTRY.items():
        if env_var != req.env_var:
            continue
        for derived_key in derived:
            if derived_key == "MLFLOW_TRACKING_URI":
                new_val = f"http://localhost:{req.port}"
                _write_env_key(derived_key, new_val)
                os.environ[derived_key] = new_val
            elif derived_key == "LABEL_STUDIO_URL":
                new_val = f"http://localhost:{req.port}"
                _write_env_key(derived_key, new_val)
                os.environ[derived_key] = new_val
            elif derived_key == "PUBLIC_PORT":
                _write_env_key(derived_key, str(req.port))
                os.environ[derived_key] = str(req.port)

    return ContainerActionResponse(
        ok=True,
        detail=f"{req.env_var} updated to {req.port} in .env — changes take effect on next compose up",
    )


@router.post("/{name}/pull", response_model=ContainerActionResponse)
async def pull_container_image(name: str) -> ContainerActionResponse:
    """Pull the latest image for a specific container (by container name).
    Stops → pulls → starts the container.
    """
    # Inspect to get the image name first
    ok_insp, out_insp = await asyncio.to_thread(
        _docker, "inspect", "--format", "{{.Config.Image}}", name
    )
    if not ok_insp:
        raise HTTPException(status_code=404, detail=f"Container {name!r} not found")
    image = out_insp.strip()
    if not image:
        raise HTTPException(status_code=400, detail="Could not determine image name")

    # Pull new image
    ok_pull, out_pull = await asyncio.to_thread(_docker, "pull", image, timeout=300)
    if not ok_pull:
        return ContainerActionResponse(ok=False, detail=f"pull failed: {out_pull[:400]}")

    # Restart to pick up new image
    ok_restart, out_restart = await asyncio.to_thread(_docker, "restart", name, timeout=30)
    detail = f"pull OK | restart {'OK' if ok_restart else 'FAILED'}: {out_restart[:200]}"
    return ContainerActionResponse(ok=ok_restart, detail=detail)
