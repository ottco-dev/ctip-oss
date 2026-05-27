"""
backend.api.v1.setup — First-time configuration & full installer API.

Endpoints:
  GET  /setup/status              — whether first-run setup is complete
  GET  /setup/config              — read current .env (sensitive values redacted)
  POST /setup/validate            — validate config values before writing
  POST /setup/configure           — atomically write to .env
  POST /setup/reset               — clear SETUP_COMPLETED flag
  GET  /setup/system-check        — full system/dependency check
  POST /setup/label-studio/test   — test Label Studio connection
  POST /setup/label-studio/create-project — create trichome detection project
  GET  /setup/verification        — post-setup health check of all subsystems

Security:
  - Sensitive values are redacted in GET responses.
  - .env is written atomically (temp file → rename).
  - Only ALLOWED_KEYS can be written.
"""

from __future__ import annotations

import asyncio
import importlib
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import httpx
from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter(prefix="/setup", tags=["setup"])

# ── Constants ─────────────────────────────────────────────────────────────────

ENV_FILE = Path(".env")
ENV_EXAMPLE = Path(".env.example")

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

# Label config for trichome detection in Label Studio
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
    <Choice value="good"   alias="good"/>
    <Choice value="blurry" alias="blurry"/>
    <Choice value="poor"   alias="poor"/>
  </Choices>
  <TextArea name="notes" toName="image" placeholder="Optional session notes" maxSubmissions="1"/>
</View>"""


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


def _run(cmd: list[str], timeout: int = 5) -> tuple[bool, str]:
    """Run a subprocess, return (ok, output)."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode == 0, (r.stdout + r.stderr).strip()
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        return False, str(e)


async def _http_ok(url: str, timeout: float = 4.0) -> tuple[bool, int, str]:
    """GET url, return (ok, status_code, detail)."""
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.get(url)
            return r.status_code < 500, r.status_code, ""
    except Exception as e:
        return False, 0, str(e)


# ── Schemas ───────────────────────────────────────────────────────────────────

class SetupStatus(BaseModel):
    completed: bool
    env_exists: bool
    configured_keys: list[str]


class ConfigEntry(BaseModel):
    key: str
    value: str
    sensitive: bool = False


class ConfigReadResponse(BaseModel):
    entries: list[ConfigEntry]
    warnings: list[str] = []


class ConfigWriteRequest(BaseModel):
    settings: dict[str, str]
    mark_setup_complete: bool = True


class ConfigWriteResponse(BaseModel):
    written: list[str]
    skipped: list[str]
    env_path: str


class ValidationResult(BaseModel):
    key: str
    value: str
    valid: bool
    message: str = ""


class ValidateRequest(BaseModel):
    settings: dict[str, str]


class CheckItem(BaseModel):
    name: str
    ok: bool
    value: str = ""          # version string, path, etc.
    detail: str = ""         # error or extra info
    required: bool = True    # False = optional/nice-to-have


class SystemCheckResponse(BaseModel):
    items: list[CheckItem]
    all_required_ok: bool
    checked_at: str


class LabelStudioTestRequest(BaseModel):
    url: str
    api_key: str = ""


class LabelStudioTestResponse(BaseModel):
    ok: bool
    reachable: bool
    authenticated: bool
    user: str = ""
    projects_count: int = 0
    detail: str = ""


class LabelStudioProjectRequest(BaseModel):
    url: str
    api_key: str
    project_name: str = "CTIP — Trichome Detection"


class LabelStudioProjectResponse(BaseModel):
    ok: bool
    project_id: int = 0
    project_url: str = ""
    already_existed: bool = False
    detail: str = ""


class VerificationItem(BaseModel):
    name: str
    url: str
    ok: bool
    status_code: int = 0
    latency_ms: float = 0
    detail: str = ""


class VerificationResponse(BaseModel):
    items: list[VerificationItem]
    all_ok: bool
    timestamp: str


# ── Endpoints — config ────────────────────────────────────────────────────────

@router.get("/status", response_model=SetupStatus)
async def get_setup_status() -> SetupStatus:
    env = _read_env_file(ENV_FILE)
    completed = env.get("SETUP_COMPLETED", "").lower() in ("true", "1", "yes")
    configured = [k for k in ALLOWED_KEYS if k in env and env[k]]
    return SetupStatus(
        completed=completed,
        env_exists=ENV_FILE.exists(),
        configured_keys=sorted(configured),
    )


@router.get("/config", response_model=ConfigReadResponse)
async def get_config() -> ConfigReadResponse:
    env = _read_env_file(ENV_FILE)
    example = _read_env_file(ENV_EXAMPLE)
    warnings: list[str] = []
    if not ENV_FILE.exists():
        warnings.append(".env not found — showing defaults from .env.example.")
    entries = [
        ConfigEntry(key=k, value=_redact(k, env.get(k, example.get(k, ""))), sensitive=k in SENSITIVE_KEYS)
        for k in sorted(ALLOWED_KEYS)
    ]
    return ConfigReadResponse(entries=entries, warnings=warnings)


@router.post("/validate", response_model=list[ValidationResult])
async def validate_config(body: ValidateRequest) -> list[ValidationResult]:
    results: list[ValidationResult] = []
    for key, value in body.settings.items():
        if key not in ALLOWED_KEYS:
            results.append(ValidationResult(key=key, value=value, valid=False,
                                            message=f"'{key}' not in allowed list."))
            continue
        valid, message = True, "OK"
        if key == "PUBLIC_DOMAIN" and value:
            if not re.match(r"^[a-zA-Z0-9]([a-zA-Z0-9\-.]{0,253}[a-zA-Z0-9])?$", value):
                valid, message = False, "Invalid domain format."
        elif key == "PUBLIC_PORT":
            try:
                p = int(value)
                if not (1 <= p <= 65535):
                    raise ValueError
            except ValueError:
                valid, message = False, "Port must be 1–65535."
        elif key == "VRAM_LIMIT_GB":
            try:
                if not (1 <= float(value) <= 80):
                    raise ValueError
            except ValueError:
                valid, message = False, "Must be 1–80 GB."
        elif key in ("LABEL_STUDIO_URL", "MLFLOW_TRACKING_URI", "CVAT_URL"):
            if value and not re.match(r"^https?://", value):
                valid, message = False, "Must be a valid HTTP/HTTPS URL."
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
            skipped.append(key)
            continue
        updates[key] = value
        written.append(key)
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


# ── Endpoint — system check ───────────────────────────────────────────────────

@router.get("/system-check", response_model=SystemCheckResponse)
async def system_check() -> SystemCheckResponse:
    """
    Full system/dependency check — runs synchronously via asyncio.to_thread
    for subprocess calls, async for HTTP checks.
    """
    from datetime import datetime, timezone
    items: list[CheckItem] = []

    def add(name: str, ok: bool, value: str = "", detail: str = "", required: bool = True) -> None:
        items.append(CheckItem(name=name, ok=ok, value=value, detail=detail, required=required))

    # ── Python ────────────────────────────────────────────────────────────────
    ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    ok_py = sys.version_info >= (3, 11)
    add("Python", ok_py, ver, "" if ok_py else "Python 3.11+ required")

    # ── Node.js ───────────────────────────────────────────────────────────────
    ok_node, node_out = await asyncio.to_thread(_run, ["node", "--version"])
    node_ver = node_out.strip().lstrip("v") if ok_node else ""
    add("Node.js", ok_node, node_ver if ok_node else "", node_out if not ok_node else "")

    # ── npm ───────────────────────────────────────────────────────────────────
    ok_npm, npm_out = await asyncio.to_thread(_run, ["npm", "--version"])
    add("npm", ok_npm, npm_out.strip() if ok_npm else "", "" if ok_npm else npm_out)

    # ── nginx ─────────────────────────────────────────────────────────────────
    ok_nx, nx_out = await asyncio.to_thread(_run, ["nginx", "-v"])
    nx_ver = nx_out.replace("nginx version: nginx/", "").split()[0] if ok_nx else nx_out
    add("nginx", ok_nx or bool(shutil.which("nginx")),
        nx_out.replace("nginx version: ", "") if ok_nx else "", "" if ok_nx else "not in PATH")

    # ── git ───────────────────────────────────────────────────────────────────
    ok_git, git_out = await asyncio.to_thread(_run, ["git", "--version"])
    add("git", ok_git, git_out.replace("git version ", "") if ok_git else "", required=False)

    # ── GPU / CUDA ────────────────────────────────────────────────────────────
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        if cuda_ok:
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            add("GPU / CUDA", True, f"{gpu_name} ({vram:.1f} GB)")
        else:
            add("GPU / CUDA", False, "CPU only",
                "CUDA not available — inference will be slow", required=False)
    except Exception as e:
        add("GPU / CUDA", False, "", str(e), required=False)

    # ── nvidia-smi ────────────────────────────────────────────────────────────
    ok_smi, smi_out = await asyncio.to_thread(_run, ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"])
    add("nvidia-smi", ok_smi, smi_out.split("\n")[0] if ok_smi else "", required=False)

    # ── Python packages ───────────────────────────────────────────────────────
    pkg_checks = [
        ("torch",          True),
        ("ultralytics",    True),
        ("fastapi",        True),
        ("pydantic",       True),
        ("sqlmodel",       True),
        ("numpy",          True),
        ("cv2",            True),
        ("PIL",            True),
        ("mlflow",         True),
        ("httpx",          True),
        ("loguru",         True),
        ("sam2",           False),
        ("label_studio_sdk", False),
        ("tensorrt",       False),
    ]
    for pkg, required in pkg_checks:
        try:
            m = importlib.import_module(pkg)
            ver_str = getattr(m, "__version__", "installed")
            add(f"pkg:{pkg}", True, ver_str, required=required)
        except ImportError:
            add(f"pkg:{pkg}", False, "",
                "not installed" + ("" if required else " (optional)"),
                required=required)

    # ── HTTP service checks ───────────────────────────────────────────────────
    svc_checks = [
        ("Backend API",  "http://localhost:8000/api/v1/setup/status", True),
        ("Frontend",     "http://localhost:3000",                     True),
        ("nginx proxy",  "http://localhost:3001/health",              False),
        ("MLflow",       "http://localhost:3004",                     False),
        ("Label Studio", "http://localhost:3005",                     False),
    ]
    for name, url, required in svc_checks:
        ok_svc, code, detail = await _http_ok(url)
        add(name, ok_svc,
            f"HTTP {code}" if ok_svc else "",
            detail if not ok_svc else "",
            required=required)

    # ── Docker ────────────────────────────────────────────────────────────────
    ok_dk, dk_out = await asyncio.to_thread(_run, ["docker", "info", "--format", "{{.ServerVersion}}"])
    add("Docker", ok_dk, dk_out.strip() if ok_dk else "",
        "Permission denied — add user to docker group" if not ok_dk else "",
        required=False)

    all_req_ok = all(i.ok for i in items if i.required)
    return SystemCheckResponse(
        items=items,
        all_required_ok=all_req_ok,
        checked_at=datetime.now(timezone.utc).isoformat(),
    )


# ── Endpoint — Label Studio ───────────────────────────────────────────────────

@router.post("/label-studio/test", response_model=LabelStudioTestResponse)
async def test_label_studio(body: LabelStudioTestRequest) -> LabelStudioTestResponse:
    """Test connectivity and authentication to a Label Studio instance."""
    url = body.url.rstrip("/")

    # 1. Basic reachability
    ok_reach, code, err = await _http_ok(f"{url}/health", timeout=5.0)
    if not ok_reach:
        # Some LS versions don't have /health — try root
        ok_reach, code, err = await _http_ok(f"{url}/", timeout=5.0)
    if not ok_reach:
        return LabelStudioTestResponse(
            ok=False, reachable=False, authenticated=False,
            detail=f"Not reachable at {url} — {err or f'HTTP {code}'}")

    # 2. Authentication check (requires API key)
    if not body.api_key:
        return LabelStudioTestResponse(ok=True, reachable=True, authenticated=False,
                                       detail="Reachable but no API key provided.")

    headers = {"Authorization": f"Token {body.api_key}"}
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            me = await client.get(f"{url}/api/current-user/whoami", headers=headers)
            if me.status_code == 200:
                data = me.json()
                user = data.get("username") or data.get("email", "unknown")
            else:
                return LabelStudioTestResponse(
                    ok=False, reachable=True, authenticated=False,
                    detail=f"Auth failed — HTTP {me.status_code}. Check API key.")

            proj = await client.get(f"{url}/api/projects/", headers=headers)
            proj_count = proj.json().get("count", 0) if proj.status_code == 200 else 0
    except Exception as e:
        return LabelStudioTestResponse(ok=False, reachable=True, authenticated=False,
                                       detail=str(e))

    return LabelStudioTestResponse(
        ok=True, reachable=True, authenticated=True,
        user=user, projects_count=proj_count)


@router.post("/label-studio/create-project", response_model=LabelStudioProjectResponse)
async def create_label_studio_project(body: LabelStudioProjectRequest) -> LabelStudioProjectResponse:
    """
    Create (or find existing) trichome detection project in Label Studio.

    Uses the Label Studio REST API directly — no SDK dependency required.
    """
    url = body.url.rstrip("/")
    headers = {
        "Authorization": f"Token {body.api_key}",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            # Check if project with same name already exists
            r_list = await client.get(f"{url}/api/projects/", headers=headers)
            if r_list.status_code != 200:
                return LabelStudioProjectResponse(
                    ok=False, detail=f"Cannot list projects — HTTP {r_list.status_code}")

            data = r_list.json()
            projects = data.get("results", data) if isinstance(data, dict) else data
            for proj in projects:
                if proj.get("title") == body.project_name:
                    proj_url = f"{url}/projects/{proj['id']}/"
                    return LabelStudioProjectResponse(
                        ok=True, project_id=proj["id"],
                        project_url=proj_url, already_existed=True,
                        detail="Project already exists — reusing it.")

            # Create new project
            payload = {
                "title": body.project_name,
                "description": (
                    "CTIP auto-created project for trichome detection annotation. "
                    "Label microscopy images with bounding boxes for 4 trichome classes."
                ),
                "label_config": TRICHOME_LABEL_CONFIG,
                "color": "#238636",
            }
            r_create = await client.post(f"{url}/api/projects/", headers=headers, json=payload)
            if r_create.status_code not in (200, 201):
                return LabelStudioProjectResponse(
                    ok=False,
                    detail=f"Project creation failed — HTTP {r_create.status_code}: {r_create.text[:300]}")

            created = r_create.json()
            proj_id = created["id"]
            proj_url = f"{url}/projects/{proj_id}/"
            return LabelStudioProjectResponse(
                ok=True, project_id=proj_id, project_url=proj_url)

    except Exception as e:
        return LabelStudioProjectResponse(ok=False, detail=str(e))


# ── Endpoint — verification ───────────────────────────────────────────────────

@router.get("/verification", response_model=VerificationResponse)
async def run_verification() -> VerificationResponse:
    """
    Post-setup verification: hit every configured service endpoint
    and return latency + status. Run after wizard completes.
    """
    from datetime import datetime, timezone
    env = _read_env_file(ENV_FILE)

    endpoints = [
        ("Backend API",    "http://localhost:8000/api/v1/setup/status"),
        ("Frontend",       "http://localhost:3000/"),
        ("nginx proxy",    "http://localhost:3001/health"),
        ("nginx → API",    "http://localhost:3001/api/v1/setup/status"),
        ("MLflow",         env.get("MLFLOW_TRACKING_URI", "http://localhost:3004") + "/"),
        ("Label Studio",   env.get("LABEL_STUDIO_URL", "http://localhost:3005") + "/health"),
    ]

    items: list[VerificationItem] = []
    for name, url in endpoints:
        t0 = time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(url)
                ok = r.status_code < 500
                code = r.status_code
                detail = ""
        except Exception as e:
            ok, code, detail = False, 0, str(e)[:120]
        latency = (time.monotonic() - t0) * 1000
        items.append(VerificationItem(
            name=name, url=url, ok=ok,
            status_code=code, latency_ms=round(latency, 1), detail=detail))

    return VerificationResponse(
        items=items,
        all_ok=all(i.ok for i in items),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
