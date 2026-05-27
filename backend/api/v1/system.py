"""
backend.api.v1.system — System health, GPU stats, and queue status endpoints.

Endpoints:
    GET /system/health      — Application health check
    GET /system/gpu         — GPU/VRAM/utilization stats
    GET /system/queue       — Background job queue status
    GET /system/info        — Full system info (GPU + queue + settings)
"""

from __future__ import annotations

import platform
import time
from typing import Any

from fastapi import APIRouter

router = APIRouter(prefix="/system", tags=["system"])


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
