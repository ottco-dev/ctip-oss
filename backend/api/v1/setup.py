"""
backend.api.v1.setup — Full Installer & First-Time Configuration API.

Endpoints:
  GET  /setup/status
  GET  /setup/config
  POST /setup/validate
  POST /setup/configure
  POST /setup/reset
  GET  /setup/system-check
  GET  /setup/docker/status
  POST /setup/docker/start-annotation
  GET  /setup/models/catalog
  GET  /setup/models/status
  POST /setup/models/download
  GET  /setup/models/download/{task_id}
  POST /setup/label-studio/test
  POST /setup/label-studio/create-account
  POST /setup/label-studio/create-project
  GET  /setup/verification
"""

from __future__ import annotations

import asyncio
import importlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Literal

import httpx
from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/setup", tags=["setup"])

# ── Constants ─────────────────────────────────────────────────────────────────

ENV_FILE    = Path(".env")
ENV_EXAMPLE = Path(".env.example")
REPO_ROOT   = Path(".").resolve()

ALLOWED_KEYS: set[str] = {
    "PUBLIC_DOMAIN", "PUBLIC_PORT",
    "CUDA_DEVICE", "CUDA_VISIBLE_DEVICES", "VRAM_LIMIT_GB", "VRAM_INFERENCE_BUDGET_GB",
    "DATA_ROOT", "TRICHOME_ROOT", "MODELS_DIR", "OUTPUTS_DIR", "UPLOADS_DIR",
    "LABEL_STUDIO_URL", "LABEL_STUDIO_API_KEY",
    "MLFLOW_TRACKING_URI", "MLFLOW_EXPERIMENT_NAME",
    "USE_WANDB", "WANDB_API_KEY", "WANDB_PROJECT",
    "SECRET_KEY", "API_TOKEN",
    "CVAT_URL", "CVAT_USERNAME",
    "ENVIRONMENT", "LOG_LEVEL",
}

SENSITIVE_KEYS: set[str] = {
    "LABEL_STUDIO_API_KEY", "WANDB_API_KEY", "SECRET_KEY", "API_TOKEN", "CVAT_PASSWORD",
}

TRICHOME_LABEL_CONFIG = """<View>
  <Header value="Trichome Detection — CTIP"/>
  <Image name="image" value="$image" zoom="true" zoomControl="true"/>
  <RectangleLabels name="label" toName="image" showInline="true">
    <Label value="stalked"        background="#22d3ee" hotkey="1"/>
    <Label value="sessile"        background="#34d399" hotkey="2"/>
    <Label value="bulbous"        background="#a78bfa" hotkey="3"/>
    <Label value="non-glandular"  background="#fb923c" hotkey="4"/>
  </RectangleLabels>
  <Choices name="quality" toName="image" showInline="true">
    <Choice value="good"/>
    <Choice value="blurry"/>
    <Choice value="poor"/>
  </Choices>
  <TextArea name="notes" toName="image" placeholder="Session notes" maxSubmissions="1"/>
</View>"""

# ── Model catalog ─────────────────────────────────────────────────────────────

MODEL_CATALOG: list[dict] = [
    {
        "id": "yolo11n",
        "name": "YOLO11n — Nano (fast)",
        "filename": "yolo11n.pt",
        "size_mb": 5.4,
        "purpose": "Detection — fastest, lower accuracy. Good for CPU-only or weak GPUs.",
        "required": False,
        "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt",
    },
    {
        "id": "yolo11s",
        "name": "YOLO11s — Small (recommended)",
        "filename": "yolo11s.pt",
        "size_mb": 18.4,
        "purpose": "Detection — best speed/accuracy balance for RTX 4060. Default CTIP model.",
        "required": True,
        "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt",
    },
    {
        "id": "yolo11m",
        "name": "YOLO11m — Medium",
        "filename": "yolo11m.pt",
        "size_mb": 43.0,
        "purpose": "Detection — higher accuracy, ~2× VRAM vs small. Use with 12 GB+ VRAM.",
        "required": False,
        "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt",
    },
    {
        "id": "sam2-tiny",
        "name": "SAM2-tiny (segmentation)",
        "filename": "sam2_hiera_tiny.pt",
        "size_mb": 38.9,
        "purpose": "Instance segmentation — prompted by YOLO boxes → pixel masks.",
        "required": True,
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt",
    },
    {
        "id": "sam2-small",
        "name": "SAM2-small (segmentation)",
        "filename": "sam2_hiera_small.pt",
        "size_mb": 46.1,
        "purpose": "Segmentation — slightly better masks, ~20% more VRAM.",
        "required": False,
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt",
    },
]

# In-memory download task registry  {task_id → progress dict}
_download_tasks: dict[str, dict] = {}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _read_env_file(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    if not path.exists():
        return result
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, value = line.partition("=")
            result[key.strip()] = value.strip().strip('"').strip("'")
    return result


def _write_env_file(path: Path, updates: dict[str, str]) -> None:
    existing_lines: list[str] = []
    if path.exists():
        existing_lines = path.read_text(encoding="utf-8").splitlines()
    updated_keys: set[str] = set()
    new_lines: list[str] = []
    for line in existing_lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key = stripped.split("=", 1)[0].strip()
            if key in updates:
                new_lines.append(f'{key}="{updates[key]}"')
                updated_keys.add(key)
                continue
        new_lines.append(line)
    remaining = {k: v for k, v in updates.items() if k not in updated_keys}
    if remaining:
        if new_lines and new_lines[-1].strip():
            new_lines.append("")
        new_lines.append("# --- CTIP Setup Wizard ---")
        for key, value in sorted(remaining.items()):
            new_lines.append(f'{key}="{value}"')
    content = "\n".join(new_lines)
    if not content.endswith("\n"):
        content += "\n"
    dir_path = path.parent
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=dir_path, delete=False, suffix=".env.tmp"
    ) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


def _redact(key: str, value: str) -> str:
    return "••••••••" if key in SENSITIVE_KEYS and value else value


def _run(cmd: list[str], timeout: int = 8, cwd: str | None = None) -> tuple[bool, str]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                           cwd=cwd or str(REPO_ROOT))
        return r.returncode == 0, (r.stdout + r.stderr).strip()
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        return False, str(e)


async def _http_ok(url: str, timeout: float = 4.0) -> tuple[bool, int, str]:
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as c:
            r = await c.get(url)
            return r.status_code < 500, r.status_code, ""
    except Exception as e:
        return False, 0, str(e)


def _models_dir() -> Path:
    env = _read_env_file(ENV_FILE)
    p = Path(env.get("MODELS_DIR", "data/models")).expanduser()
    if not p.is_absolute():
        p = REPO_ROOT / p
    return p


# ── Background download task ──────────────────────────────────────────────────

async def _do_download(task_id: str, url: str, dest: Path) -> None:
    _download_tasks[task_id] = {
        "status": "downloading", "progress": 0,
        "filename": dest.name, "size_mb": 0, "downloaded_mb": 0.0, "detail": "",
    }
    tmp: Path | None = None
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".tmp")
        async with httpx.AsyncClient(timeout=None, follow_redirects=True) as client:
            async with client.stream("GET", url) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                total_mb = round(total / 1024**2, 1)
                _download_tasks[task_id]["size_mb"] = total_mb
                received = 0
                with open(tmp, "wb") as f:
                    async for chunk in r.aiter_bytes(chunk_size=65536):
                        f.write(chunk)
                        received += len(chunk)
                        received_mb = round(received / 1024**2, 1)
                        _download_tasks[task_id]["downloaded_mb"] = received_mb
                        if total:
                            pct = int(received / total * 100)
                            _download_tasks[task_id]["progress"] = pct
                            _download_tasks[task_id]["detail"] = (
                                f"Downloading… {received_mb} / {total_mb} MB"
                            )
        tmp.replace(dest)
        _download_tasks[task_id].update({
            "status": "done", "progress": 100,
            "detail": f"Saved to {dest.name}",
        })
    except Exception as e:
        if tmp is not None and tmp.exists():
            tmp.unlink(missing_ok=True)
        _download_tasks[task_id].update({"status": "error", "detail": str(e)})


# ── Schemas ───────────────────────────────────────────────────────────────────

class SetupStatus(BaseModel):
    completed: bool; env_exists: bool; configured_keys: list[str]

class ConfigEntry(BaseModel):
    key: str; value: str; sensitive: bool = False

class ConfigReadResponse(BaseModel):
    entries: list[ConfigEntry]; warnings: list[str] = []

class ConfigWriteRequest(BaseModel):
    settings: dict[str, str]; mark_setup_complete: bool = True

class ConfigWriteResponse(BaseModel):
    written: list[str]; skipped: list[str]; env_path: str

class ValidationResult(BaseModel):
    key: str; value: str; valid: bool; message: str = ""

class ValidateRequest(BaseModel):
    settings: dict[str, str]

class CheckItem(BaseModel):
    name: str; ok: bool; value: str = ""; detail: str = ""; required: bool = True

class SystemCheckResponse(BaseModel):
    items: list[CheckItem]; all_required_ok: bool; checked_at: str

class ContainerInfo(BaseModel):
    name: str; image: str = ""; status: str; running: bool; ports: str = ""

class DockerStatusResponse(BaseModel):
    docker_available: bool
    compose_available: bool
    docker_version: str = ""
    in_docker_group: bool
    fix_command: str = ""
    containers: list[ContainerInfo] = []
    detail: str = ""

class DockerStartRequest(BaseModel):
    profile: str = "annotation"  # annotation | training | inference

class DockerStartResponse(BaseModel):
    ok: bool; detail: str = ""; output: str = ""

class ModelInfo(BaseModel):
    id: str; name: str; filename: str; size_mb: float
    purpose: str; required: bool; present: bool
    path: str = ""; url: str = ""

class DownloadRequest(BaseModel):
    model_id: str

class DownloadStartResponse(BaseModel):
    task_id: str; model_id: str; filename: str

class DownloadProgressResponse(BaseModel):
    task_id: str; status: str  # downloading | done | error
    progress: int; filename: str; size_mb: float
    downloaded_mb: float = 0.0; detail: str = ""

class LabelStudioTestRequest(BaseModel):
    url: str; api_key: str = ""

class LabelStudioTestResponse(BaseModel):
    ok: bool; reachable: bool; authenticated: bool
    user: str = ""; projects_count: int = 0; detail: str = ""

class LabelStudioAccountRequest(BaseModel):
    url: str; email: str; password: str
    first_name: str = "Admin"; last_name: str = "CTIP"

class LabelStudioAccountResponse(BaseModel):
    ok: bool; token: str = ""; user_id: int = 0
    already_existed: bool = False; detail: str = ""

class LabelStudioProjectRequest(BaseModel):
    url: str; api_key: str; project_name: str = "CTIP — Trichome Detection"

class LabelStudioProjectResponse(BaseModel):
    ok: bool; project_id: int = 0; project_url: str = ""
    already_existed: bool = False; detail: str = ""

class VerificationItem(BaseModel):
    name: str; url: str; ok: bool
    status_code: int = 0; latency_ms: float = 0; detail: str = ""

class VerificationResponse(BaseModel):
    items: list[VerificationItem]; all_ok: bool; timestamp: str


# ── Config endpoints ──────────────────────────────────────────────────────────

@router.get("/status", response_model=SetupStatus)
async def get_setup_status() -> SetupStatus:
    env = _read_env_file(ENV_FILE)
    completed = env.get("SETUP_COMPLETED", "").lower() in ("true", "1", "yes")
    configured = [k for k in ALLOWED_KEYS if k in env and env[k]]
    return SetupStatus(completed=completed, env_exists=ENV_FILE.exists(),
                       configured_keys=sorted(configured))


@router.get("/config", response_model=ConfigReadResponse)
async def get_config() -> ConfigReadResponse:
    env = _read_env_file(ENV_FILE)
    example = _read_env_file(ENV_EXAMPLE)
    warnings: list[str] = []
    if not ENV_FILE.exists():
        warnings.append(".env not found — showing defaults from .env.example.")
    entries = [
        ConfigEntry(key=k, value=_redact(k, env.get(k, example.get(k, ""))),
                    sensitive=k in SENSITIVE_KEYS)
        for k in sorted(ALLOWED_KEYS)
    ]
    return ConfigReadResponse(entries=entries, warnings=warnings)


@router.post("/validate", response_model=list[ValidationResult])
async def validate_config(body: ValidateRequest) -> list[ValidationResult]:
    results: list[ValidationResult] = []
    for key, value in body.settings.items():
        if key not in ALLOWED_KEYS:
            results.append(ValidationResult(key=key, value=value, valid=False,
                                            message=f"'{key}' not allowed.")); continue
        valid, message = True, "OK"
        if key == "PUBLIC_DOMAIN" and value:
            if not re.match(r"^[a-zA-Z0-9]([a-zA-Z0-9\-.]{0,253}[a-zA-Z0-9])?$", value):
                valid, message = False, "Invalid domain format."
        elif key == "PUBLIC_PORT":
            try:
                if not (1 <= int(value) <= 65535): raise ValueError
            except ValueError:
                valid, message = False, "Port must be 1–65535."
        elif key == "VRAM_LIMIT_GB":
            try:
                if not (1 <= float(value) <= 80): raise ValueError
            except ValueError:
                valid, message = False, "Must be 1–80 GB."
        elif key in ("LABEL_STUDIO_URL", "MLFLOW_TRACKING_URI", "CVAT_URL"):
            if value and not re.match(r"^https?://", value):
                valid, message = False, "Must be HTTP/HTTPS URL."
        elif key == "DATA_ROOT":
            if not Path(value).expanduser().is_absolute():
                valid, message = False, "Must be an absolute path."
        results.append(ValidationResult(key=key, value=value, valid=valid, message=message))
    return results


@router.post("/configure", response_model=ConfigWriteResponse)
async def write_config(body: ConfigWriteRequest) -> ConfigWriteResponse:
    written, skipped, updates = [], [], {}
    for key, value in body.settings.items():
        if key not in ALLOWED_KEYS:
            skipped.append(key); continue
        updates[key] = value; written.append(key)
    if body.mark_setup_complete:
        updates["SETUP_COMPLETED"] = "true"
    _write_env_file(ENV_FILE, updates)
    try:
        from backend.config import get_settings
        get_settings.cache_clear()
    except Exception:
        pass
    return ConfigWriteResponse(written=sorted(written), skipped=sorted(skipped),
                               env_path=str(ENV_FILE.resolve()))


@router.post("/reset")
async def reset_setup_status() -> dict[str, str]:
    _write_env_file(ENV_FILE, {"SETUP_COMPLETED": "false"})
    return {"status": "reset"}


# ── System check ──────────────────────────────────────────────────────────────

@router.get("/system-check", response_model=SystemCheckResponse)
async def system_check() -> SystemCheckResponse:
    from datetime import datetime, timezone
    items: list[CheckItem] = []

    def add(name: str, ok: bool, value: str = "", detail: str = "", required: bool = True) -> None:
        items.append(CheckItem(name=name, ok=ok, value=value, detail=detail, required=required))

    ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    add("Python", sys.version_info >= (3, 11), ver)

    ok_node, node_out = await asyncio.to_thread(_run, ["node", "--version"])
    add("Node.js", ok_node, node_out.strip().lstrip("v") if ok_node else "", node_out if not ok_node else "")

    ok_npm, npm_out = await asyncio.to_thread(_run, ["npm", "--version"])
    add("npm", ok_npm, npm_out.strip() if ok_npm else "")

    ok_nx, nx_out = await asyncio.to_thread(_run, ["nginx", "-v"])
    add("nginx", ok_nx or bool(shutil.which("nginx")),
        nx_out.replace("nginx version: ", "") if ok_nx else "")

    ok_git, git_out = await asyncio.to_thread(_run, ["git", "--version"])
    add("git", ok_git, git_out.replace("git version ", "") if ok_git else "", required=False)

    ok_dc, dc_out = await asyncio.to_thread(_run, ["docker", "compose", "version", "--short"])
    add("Docker Compose", ok_dc, dc_out.strip() if ok_dc else "", required=False)

    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        if cuda_ok:
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            add("GPU / CUDA", True, f"{gpu_name} ({vram:.1f} GB)")
        else:
            add("GPU / CUDA", False, "CPU only", "CUDA not available", required=False)
    except Exception as e:
        add("GPU / CUDA", False, "", str(e), required=False)

    ok_smi, smi_out = await asyncio.to_thread(
        _run, ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"])
    add("nvidia-smi", ok_smi, smi_out.split("\n")[0] if ok_smi else "", required=False)

    pkg_checks = [
        ("torch", True), ("ultralytics", True), ("fastapi", True), ("pydantic", True),
        ("sqlmodel", True), ("numpy", True), ("cv2", True), ("PIL", True),
        ("mlflow", True), ("httpx", True), ("loguru", True),
        ("sam2", False), ("label_studio_sdk", False), ("tensorrt", False),
    ]
    for pkg, required in pkg_checks:
        try:
            m = importlib.import_module(pkg)
            add(f"pkg:{pkg}", True, getattr(m, "__version__", "installed"), required=required)
        except ImportError:
            add(f"pkg:{pkg}", False, "", "not installed", required=required)

    svc_checks = [
        ("Backend API",  "http://localhost:8000/api/v1/setup/status", True),
        ("Frontend",     "http://localhost:3000",                     True),
        ("nginx proxy",  "http://localhost:3001/health",              False),
        ("MLflow",       "http://localhost:3004",                     False),
        ("Label Studio", "http://localhost:3005",                     False),
    ]
    for name, url, required in svc_checks:
        ok_svc, code, detail = await _http_ok(url)
        add(name, ok_svc, f"HTTP {code}" if ok_svc else "", detail if not ok_svc else "", required=required)

    return SystemCheckResponse(
        items=items,
        all_required_ok=all(i.ok for i in items if i.required),
        checked_at=datetime.now(timezone.utc).isoformat(),
    )


# ── Docker endpoints ──────────────────────────────────────────────────────────

@router.get("/docker/status", response_model=DockerStatusResponse)
async def docker_status() -> DockerStatusResponse:
    """Check Docker availability, group membership, running containers."""
    import grp, os

    # Docker group check
    try:
        docker_gid = grp.getgrnam("docker").gr_gid
        in_group = docker_gid in os.getgroups()
    except KeyError:
        in_group = False

    ok_dk, dk_out = await asyncio.to_thread(_run, ["docker", "info", "--format", "{{.ServerVersion}}"])
    ok_dc, dc_out = await asyncio.to_thread(_run, ["docker", "compose", "version", "--short"])

    containers: list[ContainerInfo] = []
    if ok_dk:
        compose_file = str(REPO_ROOT / "docker" / "docker-compose.yml")
        ok_ps, ps_out = await asyncio.to_thread(
            _run,
            [
                "docker", "compose",
                "--project-directory", str(REPO_ROOT / "docker"),
                "-f", compose_file,
                "--profile", "annotation",
                "ps", "--format", "json",
            ],
        )
        if ok_ps and ps_out.strip():
            for line in ps_out.strip().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    c = json.loads(line)
                    pub = c.get("Publishers") or []
                    port_str = ", ".join(
                        str(p.get("PublishedPort", "")) for p in pub
                        if p.get("PublishedPort")
                    )
                    containers.append(ContainerInfo(
                        name=c.get("Name", ""),
                        image=c.get("Image", ""),
                        status=c.get("Status", ""),
                        running="running" in c.get("Status", "").lower(),
                        ports=port_str,
                    ))
                except Exception:
                    pass

    fix_cmd = ""
    if not in_group:
        import getpass
        user = getpass.getuser()
        fix_cmd = f"sudo usermod -aG docker {user} && newgrp docker"

    return DockerStatusResponse(
        docker_available=ok_dk,
        compose_available=ok_dc,
        docker_version=dk_out.strip() if ok_dk else "",
        in_docker_group=in_group,
        fix_command=fix_cmd,
        containers=containers,
        detail="" if ok_dk else dk_out,
    )


@router.post("/docker/start-annotation", response_model=DockerStartResponse)
async def docker_start_annotation(body: DockerStartRequest) -> DockerStartResponse:
    """Start Docker Compose annotation profile (Label Studio + CVAT + PostgreSQL)."""
    ok_dk, _ = await asyncio.to_thread(_run, ["docker", "info"])
    if not ok_dk:
        return DockerStartResponse(
            ok=False,
            detail="Docker not accessible. Add your user to the docker group first.",
        )
    profile = body.profile
    compose_file = str(REPO_ROOT / "docker" / "docker-compose.yml")
    cmd = [
        "docker", "compose",
        "--project-directory", str(REPO_ROOT / "docker"),
        "-f", compose_file,
        "--profile", profile,
        "up", "-d", "--remove-orphans",
    ]
    ok, out = await asyncio.to_thread(_run, cmd, timeout=600)
    return DockerStartResponse(ok=ok, output=out[:3000], detail=out[:500] if not ok else "")


@router.get("/docker/start-annotation/stream")
async def docker_start_annotation_stream(profile: str = "annotation"):
    """SSE stream of docker compose up output. Connect via EventSource."""
    from fastapi.responses import StreamingResponse as _SR
    import asyncio as _aio

    ok_dk, _ = await _aio.to_thread(_run, ["docker", "info"])
    if not ok_dk:
        async def _err():
            yield "data: [ERROR] Docker not accessible\n\n"
            yield "data: [DONE]\n\n"
        return _SR(_err(), media_type="text/event-stream")

    compose_file = str(REPO_ROOT / "docker" / "docker-compose.yml")
    cmd = [
        "docker", "compose",
        "--project-directory", str(REPO_ROOT / "docker"),
        "-f", compose_file,
        "--profile", profile,
        "up", "-d", "--remove-orphans",
    ]

    async def _stream():
        import asyncio.subprocess as _asp
        proc = await _asp.create_subprocess_exec(
            *cmd,
            stdout=_asp.PIPE,
            stderr=_asp.STDOUT,
            cwd=str(REPO_ROOT / "docker"),
        )
        assert proc.stdout is not None
        try:
            async for raw in proc.stdout:
                line = raw.decode(errors="replace").rstrip()
                if line:
                    yield f"data: {line}\n\n"
        finally:
            await proc.wait()
            status = "OK" if proc.returncode == 0 else f"ERROR (exit {proc.returncode})"
            yield f"data: [DONE:{status}]\n\n"

    return _SR(
        _stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Model endpoints ───────────────────────────────────────────────────────────

@router.get("/models/catalog", response_model=list[ModelInfo])
async def models_catalog() -> list[ModelInfo]:
    """Return all models with present/missing status."""
    mdir = _models_dir()
    result: list[ModelInfo] = []
    for m in MODEL_CATALOG:
        path = mdir / m["filename"]
        result.append(ModelInfo(
            id=m["id"], name=m["name"], filename=m["filename"],
            size_mb=m["size_mb"], purpose=m["purpose"], required=m["required"],
            present=path.exists(), path=str(path) if path.exists() else "",
            url=m["url"],
        ))
    return result


@router.get("/models/status", response_model=list[ModelInfo])
async def models_status() -> list[ModelInfo]:
    return await models_catalog()


@router.post("/models/download", response_model=DownloadStartResponse)
async def start_model_download(body: DownloadRequest,
                               background_tasks: BackgroundTasks) -> DownloadStartResponse:
    """Start background download for a model. Poll /models/download/{task_id} for progress."""
    model = next((m for m in MODEL_CATALOG if m["id"] == body.model_id), None)
    if not model:
        raise HTTPException(status_code=404, detail=f"Model '{body.model_id}' not in catalog.")

    mdir = _models_dir()
    dest = mdir / model["filename"]

    if dest.exists():
        # Already present — return a fake completed task
        task_id = str(uuid.uuid4())
        _download_tasks[task_id] = {
            "status": "done", "progress": 100,
            "filename": model["filename"],
            "size_mb": round(dest.stat().st_size / 1024**2, 1),
            "detail": "Already downloaded",
        }
        return DownloadStartResponse(task_id=task_id, model_id=body.model_id,
                                     filename=model["filename"])

    task_id = str(uuid.uuid4())
    background_tasks.add_task(_do_download, task_id, model["url"], dest)
    return DownloadStartResponse(task_id=task_id, model_id=body.model_id,
                                 filename=model["filename"])


@router.get("/models/download/{task_id}", response_model=DownloadProgressResponse)
async def get_download_progress(task_id: str) -> DownloadProgressResponse:
    task = _download_tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found.")
    return DownloadProgressResponse(task_id=task_id, **task)


# ── Label Studio endpoints ────────────────────────────────────────────────────

@router.post("/label-studio/test", response_model=LabelStudioTestResponse)
async def test_label_studio(body: LabelStudioTestRequest) -> LabelStudioTestResponse:
    url = body.url.rstrip("/")
    ok_reach, code, err = await _http_ok(f"{url}/health", timeout=5.0)
    if not ok_reach:
        ok_reach, code, err = await _http_ok(f"{url}/", timeout=5.0)
    if not ok_reach:
        return LabelStudioTestResponse(ok=False, reachable=False, authenticated=False,
                                       detail=f"Not reachable — {err or f'HTTP {code}'}")
    if not body.api_key:
        return LabelStudioTestResponse(ok=True, reachable=True, authenticated=False,
                                       detail="Reachable but no API key provided.")
    headers = {"Authorization": f"Token {body.api_key}"}
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            me = await client.get(f"{url}/api/current-user/whoami", headers=headers)
            if me.status_code != 200:
                return LabelStudioTestResponse(ok=False, reachable=True, authenticated=False,
                                               detail=f"Auth failed — HTTP {me.status_code}")
            data = me.json()
            user = data.get("username") or data.get("email", "unknown")
            proj = await client.get(f"{url}/api/projects/", headers=headers)
            count = proj.json().get("count", 0) if proj.status_code == 200 else 0
    except Exception as e:
        return LabelStudioTestResponse(ok=False, reachable=True, authenticated=False, detail=str(e))
    return LabelStudioTestResponse(ok=True, reachable=True, authenticated=True,
                                   user=user, projects_count=count)


@router.post("/label-studio/create-account", response_model=LabelStudioAccountResponse)
async def create_label_studio_account(body: LabelStudioAccountRequest) -> LabelStudioAccountResponse:
    """
    Create the first admin account in a fresh Label Studio instance.
    Uses the /user/signup REST endpoint (no auth required for first user).
    Also tries /api/user/signup for newer LS versions.
    """
    url = body.url.rstrip("/")

    # Check if a user already exists (any 200/401 from whoami without token)
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Try to sign up
            payload = {
                "email": body.email,
                "password": body.password,
                "first_name": body.first_name,
                "last_name": body.last_name,
            }

            # Try v1 API endpoint first
            for endpoint in ["/api/user/signup", "/user/signup"]:
                r = await client.post(f"{url}{endpoint}", json=payload,
                                      headers={"Content-Type": "application/json"})
                if r.status_code in (200, 201):
                    data = r.json()
                    token = data.get("token", "") or data.get("auth_token", "")
                    user_id = data.get("id", 0)
                    return LabelStudioAccountResponse(ok=True, token=token, user_id=user_id)
                if r.status_code == 400:
                    text = r.text.lower()
                    if "already" in text or "exist" in text or "unique" in text:
                        return LabelStudioAccountResponse(
                            ok=True, already_existed=True,
                            detail="Account already exists — use your existing credentials.")

            # Try form-based signup (older LS)
            form_data = {
                "email": body.email,
                "password": body.password,
                "firstName": body.first_name,
                "lastName": body.last_name,
            }
            r2 = await client.post(f"{url}/user/signup", data=form_data,
                                   headers={"Content-Type": "application/x-www-form-urlencoded"})
            if r2.status_code in (200, 201, 302):
                # Try to get token via login
                login_r = await client.post(
                    f"{url}/api/token-auth/",
                    json={"username": body.email, "password": body.password},
                )
                if login_r.status_code == 200:
                    token = login_r.json().get("token", "")
                    return LabelStudioAccountResponse(ok=True, token=token)
                return LabelStudioAccountResponse(
                    ok=True,
                    detail="Account created. Log in via the web UI to get your API token.")

            return LabelStudioAccountResponse(
                ok=False,
                detail=f"Signup failed (HTTP {r2.status_code}). "
                       "Try creating the account manually in the Label Studio web UI.")
    except Exception as e:
        return LabelStudioAccountResponse(ok=False, detail=str(e))


@router.post("/label-studio/create-project", response_model=LabelStudioProjectResponse)
async def create_label_studio_project(body: LabelStudioProjectRequest) -> LabelStudioProjectResponse:
    url = body.url.rstrip("/")
    headers = {"Authorization": f"Token {body.api_key}", "Content-Type": "application/json"}
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            r_list = await client.get(f"{url}/api/projects/", headers=headers)
            if r_list.status_code != 200:
                return LabelStudioProjectResponse(
                    ok=False, detail=f"Cannot list projects — HTTP {r_list.status_code}")
            data = r_list.json()
            projects = data.get("results", data) if isinstance(data, dict) else data
            for proj in projects:
                if proj.get("title") == body.project_name:
                    return LabelStudioProjectResponse(
                        ok=True, project_id=proj["id"],
                        project_url=f"{url}/projects/{proj['id']}/",
                        already_existed=True, detail="Project already exists — reusing.")
            payload = {
                "title": body.project_name,
                "description": "CTIP auto-created project for trichome detection annotation.",
                "label_config": TRICHOME_LABEL_CONFIG,
                "color": "#238636",
            }
            r_create = await client.post(f"{url}/api/projects/", headers=headers, json=payload)
            if r_create.status_code not in (200, 201):
                return LabelStudioProjectResponse(
                    ok=False,
                    detail=f"Creation failed — HTTP {r_create.status_code}: {r_create.text[:300]}")
            created = r_create.json()
            return LabelStudioProjectResponse(
                ok=True, project_id=created["id"],
                project_url=f"{url}/projects/{created['id']}/")
    except Exception as e:
        return LabelStudioProjectResponse(ok=False, detail=str(e))


# ── Verification ──────────────────────────────────────────────────────────────

@router.get("/verification", response_model=VerificationResponse)
async def run_verification() -> VerificationResponse:
    from datetime import datetime, timezone
    env = _read_env_file(ENV_FILE)
    endpoints = [
        ("Backend API",   "http://localhost:8000/api/v1/setup/status"),
        ("Frontend",      "http://localhost:3000/"),
        ("nginx proxy",   "http://localhost:3001/health"),
        ("nginx → API",   "http://localhost:3001/api/v1/setup/status"),
        ("MLflow",        env.get("MLFLOW_TRACKING_URI", "http://localhost:3004") + "/"),
        ("Label Studio",  env.get("LABEL_STUDIO_URL", "http://localhost:3005") + "/health"),
    ]
    items: list[VerificationItem] = []
    for name, url in endpoints:
        t0 = time.monotonic()
        ok, code, detail = await _http_ok(url, timeout=5.0)
        lat = (time.monotonic() - t0) * 1000
        items.append(VerificationItem(name=name, url=url, ok=ok, status_code=code,
                                      latency_ms=round(lat, 1), detail=detail))
    return VerificationResponse(items=items, all_ok=all(i.ok for i in items),
                                timestamp=datetime.now(timezone.utc).isoformat())
