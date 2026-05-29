"""
backend.api.v1.system — System health, GPU stats, and queue status endpoints.

Endpoints:
    GET /system/health           — Application health check
    GET /system/gpu              — GPU/VRAM/utilization stats
    GET /system/queue            — Background job queue status
    GET /system/info             — Full system info (GPU + queue + settings)
    GET /system/services         — Live health check for all CTIP services
    GET /system/logs             — Tail recent backend log lines
    GET /system/token/status     — API token auth status + masked preview
    POST /system/token/generate  — Generate a new random API token
    POST /system/token/clear     — Remove API token (disable auth)
"""

from __future__ import annotations

import os
import platform
import secrets
import socket
import time
from collections import deque
from typing import Any

from fastapi import APIRouter, HTTPException, Query

router = APIRouter(prefix="/system", tags=["system"])

# In-memory ring buffer for log lines (max 500); populated by request logger
_log_ring: deque[dict] = deque(maxlen=500)


def _get_gpu_stats() -> dict[str, Any]:
    """Query NVIDIA GPU stats via pynvml or torch."""
    try:
        import torch

        if not torch.cuda.is_available():
            return {
                "available": False,
                "reason": "CUDA not available",
            }

        device = torch.device("cuda:0")
        props = torch.cuda.get_device_properties(device)

        vram_total = props.total_memory / 1e9
        vram_used = torch.cuda.memory_allocated(0) / 1e9
        vram_reserved = torch.cuda.memory_reserved(0) / 1e9
        vram_free = vram_total - vram_reserved

        gpu_stats = {
            "available": True,
            "device_name": props.name,
            "device_index": 0,
            "vram_total_gb": round(vram_total, 2),
            "vram_used_gb": round(vram_used, 2),
            "vram_reserved_gb": round(vram_reserved, 2),
            "vram_free_gb": round(vram_free, 2),
            "vram_used_pct": round(vram_used / vram_total * 100, 1),
            "compute_capability": f"{props.major}.{props.minor}",
            "multi_processor_count": props.multi_processor_count,
        }

        # Try pynvml for utilization %
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000  # mW → W
            power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000

            gpu_stats.update({
                "gpu_utilization_pct": util.gpu,
                "memory_utilization_pct": util.memory,
                "temperature_c": temp,
                "power_draw_w": round(power, 1),
                "power_limit_w": round(power_limit, 1),
            })
        except Exception:
            gpu_stats["gpu_utilization_pct"] = None

        return gpu_stats

    except ImportError:
        return {"available": False, "reason": "PyTorch not installed"}
    except Exception as e:
        return {"available": False, "reason": str(e)}


def _get_cpu_ram_stats() -> dict[str, Any]:
    """Get CPU and RAM stats."""
    try:
        import psutil

        cpu_pct = psutil.cpu_percent(interval=0.1)
        ram = psutil.virtual_memory()
        disk = psutil.disk_usage("/")

        return {
            "cpu_count": psutil.cpu_count(),
            "cpu_utilization_pct": cpu_pct,
            "ram_total_gb": round(ram.total / 1e9, 1),
            "ram_used_gb": round(ram.used / 1e9, 1),
            "ram_free_gb": round(ram.available / 1e9, 1),
            "ram_used_pct": ram.percent,
            "disk_total_gb": round(disk.total / 1e9, 1),
            "disk_free_gb": round(disk.free / 1e9, 1),
            "disk_used_pct": round(disk.percent, 1),
        }
    except ImportError:
        return {"error": "psutil not installed"}


@router.get("/health")
async def health_check() -> dict[str, Any]:
    """Basic health check — used by Docker healthcheck and load balancers."""
    return {
        "status": "ok",
        "timestamp": time.time(),
        "service": "trichome-analysis-backend",
        "version": "0.1.0",
    }


@router.get("/gpu")
async def gpu_stats() -> dict[str, Any]:
    """
    Real-time GPU stats.

    Polled every 2s by frontend GPU monitor widget.
    Requires pynvml for utilization%; torch-only for VRAM.
    """
    return {
        "timestamp": time.time(),
        **_get_gpu_stats(),
    }


@router.get("/queue")
async def queue_status() -> dict[str, Any]:
    """Background job queue status, including GPU semaphore state."""
    from backend.tasks.task_router import task_router

    all_jobs = task_router.get_all_jobs()
    pending = [j for j in all_jobs if j["status"] == "pending"]
    running = [j for j in all_jobs if j["status"] == "running"]

    # GPU semaphore status (shared between REST inference + background tasks)
    gpu_sem: dict = {}
    try:
        from backend.dependencies.gpu import gpu_semaphore_status
        gpu_sem = gpu_semaphore_status()
    except Exception:
        pass

    return {
        "timestamp": time.time(),
        "gpu_task_running": task_router.gpu_task_running,
        "gpu_queue_depth": len(pending),
        "total_active_jobs": len(running) + len(pending),
        "gpu_semaphore": gpu_sem,
        "jobs": {
            "pending": len(pending),
            "running": len(running),
            "completed": sum(1 for j in all_jobs if j["status"] == "completed"),
            "failed": sum(1 for j in all_jobs if j["status"] == "failed"),
        },
    }


@router.get("/info")
async def system_info() -> dict[str, Any]:
    """
    Full system information dashboard.

    Called on initial page load and periodically refreshed.
    """
    from backend.config import get_settings

    settings = get_settings()

    return {
        "timestamp": time.time(),
        "platform": {
            "os": platform.system(),
            "os_version": platform.version(),
            "python_version": platform.python_version(),
            "hostname": platform.node(),
        },
        "gpu": _get_gpu_stats(),
        "cpu_ram": _get_cpu_ram_stats(),
        "config": {
            "api_host": settings.api_host,
            "api_port": settings.api_port,
            "database_url": settings.database_url.split("///")[0] + "///***",  # Hide path
            "mlflow_uri": settings.mlflow_tracking_uri,
            "default_vlm": settings.default_vlm_backend,
            "cuda_device": settings.cuda_device,
            "vram_limit_gb": settings.vram_limit_gb,
        },
    }


# ---------------------------------------------------------------------------
# Service health
# ---------------------------------------------------------------------------

_SERVICES = [
    {"name": "FastAPI Backend",   "port": 8000, "path": "/api/v1/system/health", "profile": None},
    {"name": "Next.js Frontend",  "port": 3000, "path": "/",                     "profile": None},
    {"name": "nginx Proxy",       "port": 3001, "path": "/",                     "profile": None},
    {"name": "MLflow",            "port": 3004, "path": "/",                     "profile": None},
    {"name": "Label Studio",      "port": 3005, "path": "/",                     "profile": "annotation"},
    {"name": "CVAT",              "port": 3006, "path": "/",                     "profile": "annotation"},
]


def _tcp_reachable(host: str, port: int, timeout: float = 0.5) -> bool:
    """Return True if a TCP connection to host:port succeeds within timeout."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


@router.get("/services")
async def service_health() -> dict[str, Any]:
    """
    Live TCP health check for all CTIP services.

    Uses a short timeout so the endpoint stays fast (~3 s worst case).
    """
    results = []
    for svc in _SERVICES:
        up = _tcp_reachable("127.0.0.1", svc["port"])
        results.append({
            "name": svc["name"],
            "port": svc["port"],
            "status": "running" if up else "stopped",
            "profile": svc["profile"],
            "url": f"http://localhost:{svc['port']}{svc['path']}",
        })
    return {"timestamp": time.time(), "services": results}


# ---------------------------------------------------------------------------
# Log ring — populated externally by request_logger middleware
# ---------------------------------------------------------------------------

def push_log_entry(entry: dict) -> None:
    """Called by the request logger middleware to append a log line."""
    _log_ring.append(entry)


@router.get("/logs")
async def tail_logs(limit: int = Query(default=50, le=200)) -> dict[str, Any]:
    """
    Return the most recent `limit` backend log entries from the in-memory ring.

    Supplements with recent lines from logs/backend.log on disk when available.
    """
    entries: list[dict] = list(_log_ring)[-limit:]

    # Supplement from on-disk log if ring is sparse
    if len(entries) < limit:
        log_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "..", "logs", "backend.log"
        )
        log_path = os.path.normpath(log_path)
        if os.path.exists(log_path):
            try:
                with open(log_path, "r", errors="replace") as f:
                    lines = f.readlines()
                disk_entries = [
                    {"level": "INFO", "msg": l.rstrip(), "ts": None}
                    for l in lines[-(limit - len(entries)):]
                ]
                entries = disk_entries + entries
            except OSError:
                pass

    return {"timestamp": time.time(), "count": len(entries), "entries": entries}


# ---------------------------------------------------------------------------
# API Token management
# ---------------------------------------------------------------------------

def _mask_token(token: str) -> str:
    """Return a masked token showing only first 4 and last 4 characters."""
    if len(token) <= 8:
        return "*" * len(token)
    return token[:4] + "*" * (len(token) - 8) + token[-4:]


@router.get("/token/status")
async def token_status() -> dict[str, Any]:
    """
    Return current API token authentication status.

    Returns whether token auth is enabled and a masked preview of the token
    (first 4 + last 4 characters only — never exposes the full token).
    """
    from backend.config import get_settings

    settings = get_settings()
    token = settings.api_token
    enabled = bool(token)

    return {
        "enabled": enabled,
        "token_preview": _mask_token(token) if enabled else None,
        "created_hint": None,
    }


@router.post("/token/generate")
async def token_generate() -> dict[str, Any]:
    """
    Generate a new 64-char hex API token and persist it to .env.

    The full token is returned exactly once — it cannot be retrieved again
    via this API (only a masked preview is available via /token/status).

    After writing, the settings cache is cleared so the new token is active
    immediately without a server restart.
    """
    from backend.config import get_settings
    from backend.utils.env_file import write_env_key

    new_token = secrets.token_hex(32)  # 64-character hex string
    write_env_key("API_TOKEN", new_token)
    get_settings.cache_clear()

    return {
        "token": new_token,
        "warning": "Copy now — not shown again",
    }


@router.post("/token/clear")
async def token_clear() -> dict[str, Any]:
    """
    Remove the API token from .env, effectively disabling authentication.

    Clears the settings cache so the change is active immediately.
    """
    from backend.config import get_settings
    from backend.utils.env_file import write_env_key

    write_env_key("API_TOKEN", "")
    get_settings.cache_clear()

    return {
        "enabled": False,
        "message": "API token removed — authentication disabled",
    }
