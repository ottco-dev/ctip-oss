"""
backend.utils.compute — Shared compute-device resolution.

Single source of truth for converting the configured COMPUTE_BACKEND setting
into a concrete torch device string.  All inference components should call
get_torch_device() instead of hard-coding "cuda:0".
"""

from __future__ import annotations


def get_torch_device() -> str:
    """
    Return the torch device string for the currently configured backend.

    Priority:
      1. COMPUTE_BACKEND env var (set by /settings/compute or manually)
      2. Settings.compute_backend (reads .env via pydantic-settings)
      3. Auto-detect: cuda → mps → cpu
    """
    import os

    backend = os.environ.get("COMPUTE_BACKEND", "").strip().lower()
    if not backend:
        try:
            from backend.config import get_settings
            backend = get_settings().compute_backend.strip().lower()
        except Exception:
            backend = "auto"

    return _resolve(backend)


def _resolve(backend: str) -> str:
    try:
        import torch
    except ImportError:
        return "cpu"

    if backend == "cpu":
        return "cpu"

    if backend == "cuda":
        return "cuda:0" if torch.cuda.is_available() else "cpu"

    if backend == "rocm":
        # ROCm builds expose cuda.is_available() == True with torch.version.hip set
        if torch.cuda.is_available():
            return "cuda:0"
        return "cpu"

    if backend == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    # auto
    if torch.cuda.is_available():
        return "cuda:0"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
