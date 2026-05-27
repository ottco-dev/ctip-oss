"""
Container management API — list, start, stop, restart, logs, compose config.
Background task system for long-running docker compose operations.
All docker operations run in a thread to avoid blocking the event loop.
"""

from __future__ import annotations

import asyncio
import json
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
# In-memory; survives per uvicorn worker lifetime. Fine for single-worker dev mode.

TaskStatus = Literal["queued", "running", "done", "error"]

class BgTask(BaseModel):
    id: str
    status: TaskStatus
    started_at: float
    finished_at: float | None = None
    ok: bool | None = None
    log: list[str] = []
    profile: str = "annotation"

_bg_tasks: dict[str, BgTask] = {}


# ── Repo root ─────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[3]
COMPOSE_FILE = REPO_ROOT / "docker" / "docker-compose.yml"
COMPOSE_DIR = REPO_ROOT / "docker"


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
    """Async background worker — runs docker compose up and updates task store."""
    import asyncio.subprocess as _asp

    task = _bg_tasks[task_id]
    task.status = "running"

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
                # Keep last 500 log lines to avoid unbounded growth
                if len(task.log) > 500:
                    task.log = task.log[-500:]
        await proc.wait()
        task.ok = proc.returncode == 0
        task.status = "done" if task.ok else "error"
    except Exception as exc:
        task.log.append(f"[fatal] {exc}")
        task.ok = False
        task.status = "error"
    finally:
        task.finished_at = time.time()


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


@router.post("/compose/up/background", response_model=BgTaskResponse)
async def compose_up_background(profile: str = "annotation") -> BgTaskResponse:
    """Start docker compose up as a fire-and-forget background task.
    Returns task_id immediately — poll /compose/task/{task_id} for status.
    """
    task_id = str(uuid.uuid4())
    task = BgTask(
        id=task_id,
        status="queued",
        started_at=time.time(),
        profile=profile,
    )
    _bg_tasks[task_id] = task
    # Fire and forget — runs independently of the HTTP connection
    asyncio.create_task(_run_compose_bg(task_id, profile))
    return BgTaskResponse(task_id=task_id)


@router.get("/compose/task/{task_id}", response_model=BgTaskStatus)
async def get_compose_task(task_id: str) -> BgTaskStatus:
    """Poll a background compose task for its current status and log."""
    task = _bg_tasks.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    elapsed = None
    if task.finished_at:
        elapsed = round(task.finished_at - task.started_at, 1)
    elif task.status in ("running", "queued"):
        elapsed = round(time.time() - task.started_at, 1)
    return BgTaskStatus(
        id=task.id,
        status=task.status,
        started_at=task.started_at,
        finished_at=task.finished_at,
        ok=task.ok,
        log=task.log,
        profile=task.profile,
        elapsed_seconds=elapsed,
    )


@router.get("/compose/tasks", response_model=list[BgTaskStatus])
async def list_compose_tasks() -> list[BgTaskStatus]:
    """List all background compose tasks (most recent first, last 20)."""
    tasks = sorted(_bg_tasks.values(), key=lambda t: t.started_at, reverse=True)[:20]
    result = []
    for task in tasks:
        elapsed = None
        if task.finished_at:
            elapsed = round(task.finished_at - task.started_at, 1)
        elif task.status in ("running", "queued"):
            elapsed = round(time.time() - task.started_at, 1)
        result.append(BgTaskStatus(
            id=task.id,
            status=task.status,
            started_at=task.started_at,
            finished_at=task.finished_at,
            ok=task.ok,
            log=task.log,
            profile=task.profile,
            elapsed_seconds=elapsed,
        ))
    return result


async def _run_reinstall_bg(task_id: str, profile: str) -> None:
    """Pull fresh images and recreate all compose containers."""
    import asyncio.subprocess as _asp

    task = _bg_tasks[task_id]
    task.status = "running"

    # Step 1: pull
    pull_cmd = [
        "docker", "compose",
        "--project-directory", str(COMPOSE_DIR),
        "-f", str(COMPOSE_FILE),
        "--profile", profile,
        "pull",
    ]
    # Step 2: up --force-recreate
    up_cmd = [
        "docker", "compose",
        "--project-directory", str(COMPOSE_DIR),
        "-f", str(COMPOSE_FILE),
        "--profile", profile,
        "up", "-d", "--force-recreate", "--remove-orphans",
    ]

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
                    if len(task.log) > 500:
                        task.log = task.log[-500:]
            await proc.wait()
            return proc.returncode == 0
        except Exception as exc:
            task.log.append(f"[fatal] {exc}")
            return False

    try:
        task.log.append("=== Step 1: docker compose pull ===")
        ok_pull = await _stream(pull_cmd)
        task.log.append(f"=== Pull {'OK' if ok_pull else 'FAILED'} ===")
        task.log.append("=== Step 2: docker compose up --force-recreate ===")
        ok_up = await _stream(up_cmd)
        task.log.append(f"=== Recreate {'OK' if ok_up else 'FAILED'} ===")
        task.ok = ok_pull and ok_up
        task.status = "done" if task.ok else "error"
    except Exception as exc:
        task.log.append(f"[fatal] {exc}")
        task.ok = False
        task.status = "error"
    finally:
        task.finished_at = time.time()


@router.post("/compose/reinstall/background", response_model=BgTaskResponse)
async def compose_reinstall_background(profile: str = "annotation") -> BgTaskResponse:
    """Pull fresh images + force-recreate all containers for the given profile.
    Runs as a background task — returns task_id immediately.
    """
    task_id = str(uuid.uuid4())
    task = BgTask(
        id=task_id,
        status="queued",
        started_at=time.time(),
        profile=profile,
    )
    _bg_tasks[task_id] = task
    asyncio.create_task(_run_reinstall_bg(task_id, profile))
    return BgTaskResponse(task_id=task_id)


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
