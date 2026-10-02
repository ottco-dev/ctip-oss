"""What can this machine do, and what can it give right now? CUDA, ROCm, Apple MPS or CPU."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess

import psutil

from apps.worker import __version__
from apps.worker.config import Limits
from shared.compute.schemas import Backend, Capabilities, Free, JobKind


def _torch():
    try:
        import torch

        return torch
    except Exception:
        return None


def detect(limits: Limits | None = None) -> Capabilities:
    torch = _torch()
    ram = psutil.virtual_memory().total / 2**30
    cores = os.cpu_count() or 1
    kinds = [JobKind(k) for k in (limits.kinds if limits else [k.value for k in JobKind])]
    common = {"ram_gb": round(ram, 1), "cpu_cores": cores, "torch": getattr(torch, "__version__", ""),
              "ctip": __version__, "os": f"{platform.system()} {platform.release()}"[:100], "kinds": kinds}
    if torch is not None and torch.cuda.is_available():
        backend = Backend.ROCM if getattr(torch.version, "hip", None) else Backend.CUDA
        n = torch.cuda.device_count()
        vram = min(torch.cuda.get_device_properties(i).total_memory for i in range(n)) / 2**30
        return Capabilities(backend=backend, device=torch.cuda.get_device_name(0), gpus=n, vram_gb=round(vram, 1), **common)
    if torch is not None and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        # unified memory: the GPU shares system RAM
        return Capabilities(backend=Backend.MPS, device=f"Apple {platform.machine()}", gpus=1, vram_gb=round(ram * 0.6, 1), **common)
    return Capabilities(backend=Backend.CPU, device=platform.processor() or platform.machine(), gpus=0, vram_gb=0.0, **common)


def others_gpu_util(own_pids: set[int] | None = None) -> int | None:
    """NVIDIA only: summed SM utilisation (%) of processes that are not ours - games, renders, other jobs.
    None when it cannot be measured (no nvidia-smi, other vendors)."""
    smi = shutil.which("nvidia-smi")
    if not smi:
        return None
    own = own_pids or set()
    try:
        out = subprocess.run([smi, "pmon", "-c", "1", "-s", "u"], capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return None
    total = 0
    for line in out.splitlines():
        parts = line.split()
        if not parts or parts[0].startswith("#") or len(parts) < 4 or not parts[1].isdigit():
            continue
        pid, sm = int(parts[1]), parts[3]
        if pid not in own and sm.isdigit():
            total += int(sm)
    return total


def gpu_busy_by_others(threshold: int, own_pids: set[int] | None = None) -> bool:
    util = others_gpu_util(own_pids)
    return util is not None and util >= threshold


def free(caps: Capabilities, limits: Limits, own_pids: set[int] | None = None) -> Free:
    torch = _torch()
    vram = 0.0
    if caps.backend in (Backend.CUDA, Backend.ROCM) and torch is not None:
        try:
            f, total = torch.cuda.mem_get_info(0)
            vram = max(0.0, min(f / 2**30 - limits.reserve_vram_gb, total / 2**30 * limits.max_vram_fraction))
        except Exception:
            vram = caps.vram_gb * limits.max_vram_fraction
    elif caps.backend == Backend.MPS:
        vram = max(0.0, psutil.virtual_memory().available / 2**30 * 0.6 - limits.reserve_vram_gb)
    ram = max(0.0, psutil.virtual_memory().available / 2**30 - limits.min_free_ram_gb)   # keep a reserve for the owner
    if limits.max_ram_gb:
        ram = min(ram, limits.max_ram_gb)
    busy = caps.backend in (Backend.CUDA,) and gpu_busy_by_others(limits.busy_gpu_util, own_pids)
    return Free(vram_gb=round(vram, 2), ram_gb=round(ram, 2), busy_by_others=busy)
