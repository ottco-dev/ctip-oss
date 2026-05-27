"""
Container management API — list, start, stop, restart, logs, compose config.
All docker operations run in a thread to avoid blocking the event loop.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

router = APIRouter(prefix="/containers", tags=["containers"])

# ── Repo root ─────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[4]
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
    """SSE stream of docker compose up output."""
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
