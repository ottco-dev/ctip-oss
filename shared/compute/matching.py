"""Which queued job may a worker take? Pure functions, shared by coordinator and tests."""

from __future__ import annotations

from shared.compute.schemas import Capabilities, Free, JobKind, Requirements


def fits(req: Requirements, kind: str, caps: Capabilities, free: Free) -> bool:
    """True if the worker supports the job kind and can give the resources the job needs right now."""
    if free.busy_by_others:
        return False
    if JobKind(kind) not in caps.kinds or caps.backend not in req.backends:
        return False
    gpu_backend = caps.backend.value != "cpu"
    if req.gpus > 1 and (not gpu_backend or caps.gpus < req.gpus):
        return False
    if req.min_vram_gb > 0 and (not gpu_backend or free.vram_gb < req.min_vram_gb):
        return False
    return free.ram_gb >= req.min_ram_gb
