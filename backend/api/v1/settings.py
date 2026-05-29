"""
backend.api.v1.settings — Platform-wide settings management.

Endpoints:
    GET  /settings                  — Read all editable settings
    PATCH /settings                 — Update settings (persists to .env)
    GET  /settings/compute          — Compute backend info + hardware detection
    POST /settings/compute          — Change compute backend
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from shared.logging.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/settings", tags=["settings"])


# ── Hardware detection ─────────────────────────────────────────────────────────

def _detect_hardware() -> dict[str, Any]:
    """
    Probe available compute hardware and return a structured dict.
    Never raises — returns cpu fallback on any error.
    """
    result: dict[str, Any] = {
        "cuda_available": False,
        "rocm_available": False,
        "mps_available": False,
        "cuda_device_count": 0,
        "devices": [],
        "recommended_backend": "cpu",
    }
    try:
        import torch
        result["cuda_available"] = torch.cuda.is_available()
        result["rocm_available"] = (
            torch.cuda.is_available() and hasattr(torch.version, "hip") and bool(torch.version.hip)
        )
        result["mps_available"] = (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )
        result["torch_version"] = torch.__version__
        result["cuda_version"] = getattr(torch.version, "cuda", None)
        result["hip_version"] = getattr(torch.version, "hip", None)

        if torch.cuda.is_available():
            result["cuda_device_count"] = torch.cuda.device_count()
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                result["devices"].append({
                    "index": i,
                    "name": props.name,
                    "vram_gb": round(props.total_memory / 1024**3, 1),
                    "compute_capability": f"{props.major}.{props.minor}",
                    "backend": "rocm" if result["rocm_available"] else "cuda",
                })
        elif result["mps_available"]:
            result["devices"].append({
                "index": 0,
                "name": "Apple Silicon MPS",
                "vram_gb": None,
                "compute_capability": None,
                "backend": "mps",
            })
        else:
            import platform, os
            result["devices"].append({
                "index": 0,
                "name": f"CPU ({os.cpu_count()} cores, {platform.processor() or 'unknown'})",
                "vram_gb": None,
                "compute_capability": None,
                "backend": "cpu",
            })

        # Recommended backend priority: cuda > rocm > mps > cpu
        if result["cuda_available"] and not result["rocm_available"]:
            result["recommended_backend"] = "cuda"
        elif result["rocm_available"]:
            result["recommended_backend"] = "rocm"
        elif result["mps_available"]:
            result["recommended_backend"] = "mps"

    except ImportError:
        pass

    return result


def _resolve_torch_device(backend: str) -> str:
    """Convert a backend name to a torch device string."""
    try:
        import torch
        if backend == "cpu":
            return "cpu"
        if backend in ("cuda", "auto"):
            if torch.cuda.is_available() and not (
                hasattr(torch.version, "hip") and torch.version.hip
            ):
                return "cuda:0"
        if backend == "rocm":
            if torch.cuda.is_available():
                return "cuda:0"  # ROCm uses the CUDA API surface via HIP
        if backend == "mps":
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        if backend == "auto":
            if torch.cuda.is_available():
                return "cuda:0"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
    except ImportError:
        pass
    return "cpu"


# ── Schemas ────────────────────────────────────────────────────────────────────

class DeviceInfo(BaseModel):
    index: int
    name: str
    vram_gb: float | None
    compute_capability: str | None
    backend: str


class ComputeInfo(BaseModel):
    configured_backend: str
    resolved_device: str
    recommended_backend: str
    cuda_available: bool
    rocm_available: bool
    mps_available: bool
    cuda_device_count: int
    torch_version: str | None
    cuda_version: str | None
    hip_version: str | None
    devices: list[DeviceInfo]
    gpu_semaphore_active: bool


class SetComputeRequest(BaseModel):
    backend: str = Field(
        description="Compute backend: auto | cuda | rocm | mps | cpu"
    )


class PlatformSettings(BaseModel):
    # Compute
    compute_backend: str
    cuda_device: str
    vram_limit_gb: float
    max_concurrent_gpu_tasks: int
    gpu_inference_queue_depth: int
    # VLM
    default_vlm_backend: str
    vlm_min_confidence: float
    active_vlm_provider: str
    active_vlm_model: str
    # Storage
    data_root: str
    models_dir: str
    uploads_dir: str
    max_upload_size_mb: int
    # Logging
    log_level: str
    # Security
    api_token_enabled: bool
    # MLflow
    mlflow_tracking_uri: str
    mlflow_experiment_name: str


class PatchSettingsRequest(BaseModel):
    compute_backend: str | None = None
    vram_limit_gb: float | None = None
    max_concurrent_gpu_tasks: int | None = None
    gpu_inference_queue_depth: int | None = None
    default_vlm_backend: str | None = None
    vlm_min_confidence: float | None = None
    log_level: str | None = None
    mlflow_tracking_uri: str | None = None
    max_upload_size_mb: int | None = None


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.get("", response_model=PlatformSettings)
async def get_all_settings() -> PlatformSettings:
    """Return all editable platform settings."""
    from backend.config import get_settings
    s = get_settings()
    return PlatformSettings(
        compute_backend=s.compute_backend,
        cuda_device=s.cuda_device,
        vram_limit_gb=s.vram_limit_gb,
        max_concurrent_gpu_tasks=s.max_concurrent_gpu_tasks,
        gpu_inference_queue_depth=s.gpu_inference_queue_depth,
        default_vlm_backend=s.default_vlm_backend,
        vlm_min_confidence=s.vlm_min_confidence,
        active_vlm_provider=s.active_vlm_provider,
        active_vlm_model=s.active_vlm_model,
        data_root=s.data_root,
        models_dir=s.models_dir,
        uploads_dir=s.uploads_dir,
        max_upload_size_mb=s.max_upload_size_mb,
        log_level=s.log_level,
        api_token_enabled=bool(s.api_token),
        mlflow_tracking_uri=s.mlflow_tracking_uri,
        mlflow_experiment_name=s.mlflow_experiment_name,
    )


@router.patch("", response_model=PlatformSettings)
async def patch_settings(req: PatchSettingsRequest) -> PlatformSettings:
    """
    Update one or more platform settings.
    Changes are persisted to .env and take effect immediately.
    """
    from backend.utils.env_file import write_env_keys
    from backend.config import get_settings

    VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    VALID_BACKENDS = {"auto", "cuda", "rocm", "mps", "cpu"}
    VALID_VLM = {"moondream", "qwen2vl", "florence2", "openai", "anthropic",
                 "google", "together", "groq", "huggingface"}

    to_write: dict[str, str] = {}

    if req.compute_backend is not None:
        if req.compute_backend not in VALID_BACKENDS:
            raise HTTPException(400, f"Invalid compute_backend. Choose from: {sorted(VALID_BACKENDS)}")
        to_write["COMPUTE_BACKEND"] = req.compute_backend

    if req.vram_limit_gb is not None:
        if req.vram_limit_gb <= 0:
            raise HTTPException(400, "vram_limit_gb must be > 0")
        to_write["VRAM_LIMIT_GB"] = str(req.vram_limit_gb)

    if req.max_concurrent_gpu_tasks is not None:
        if req.max_concurrent_gpu_tasks < 1:
            raise HTTPException(400, "max_concurrent_gpu_tasks must be >= 1")
        to_write["MAX_CONCURRENT_GPU_TASKS"] = str(req.max_concurrent_gpu_tasks)

    if req.gpu_inference_queue_depth is not None:
        if req.gpu_inference_queue_depth < 0:
            raise HTTPException(400, "gpu_inference_queue_depth must be >= 0")
        to_write["GPU_INFERENCE_QUEUE_DEPTH"] = str(req.gpu_inference_queue_depth)

    if req.default_vlm_backend is not None:
        if req.default_vlm_backend not in VALID_VLM:
            raise HTTPException(400, f"Invalid vlm backend. Choose from: {sorted(VALID_VLM)}")
        to_write["DEFAULT_VLM_BACKEND"] = req.default_vlm_backend

    if req.vlm_min_confidence is not None:
        if not 0.0 <= req.vlm_min_confidence <= 1.0:
            raise HTTPException(400, "vlm_min_confidence must be in [0, 1]")
        to_write["VLM_MIN_CONFIDENCE"] = str(req.vlm_min_confidence)

    if req.log_level is not None:
        if req.log_level.upper() not in VALID_LOG_LEVELS:
            raise HTTPException(400, f"Invalid log_level. Choose from: {sorted(VALID_LOG_LEVELS)}")
        to_write["LOG_LEVEL"] = req.log_level.upper()

    if req.mlflow_tracking_uri is not None:
        to_write["MLFLOW_TRACKING_URI"] = req.mlflow_tracking_uri

    if req.max_upload_size_mb is not None:
        if req.max_upload_size_mb < 1:
            raise HTTPException(400, "max_upload_size_mb must be >= 1")
        to_write["MAX_UPLOAD_SIZE_MB"] = str(req.max_upload_size_mb)

    if not to_write:
        raise HTTPException(400, "No settings provided to update")

    # Also update os.environ for immediate effect
    import os
    for k, v in to_write.items():
        os.environ[k] = v

    write_env_keys(to_write)
    get_settings.cache_clear()

    logger.info("Platform settings updated", keys=list(to_write.keys()))
    return await get_all_settings()


@router.get("/compute", response_model=ComputeInfo)
async def get_compute_info() -> ComputeInfo:
    """
    Return detected hardware info and current compute backend configuration.
    Used by the Settings page to show available GPU/CPU options.
    """
    from backend.config import get_settings
    from backend.dependencies.gpu import gpu_semaphore_status

    s = get_settings()
    hw = _detect_hardware()
    status = gpu_semaphore_status()

    return ComputeInfo(
        configured_backend=s.compute_backend,
        resolved_device=_resolve_torch_device(s.compute_backend),
        recommended_backend=hw["recommended_backend"],
        cuda_available=hw["cuda_available"],
        rocm_available=hw["rocm_available"],
        mps_available=hw["mps_available"],
        cuda_device_count=hw["cuda_device_count"],
        torch_version=hw.get("torch_version"),
        cuda_version=hw.get("cuda_version"),
        hip_version=hw.get("hip_version"),
        devices=[DeviceInfo(**d) for d in hw["devices"]],
        gpu_semaphore_active=status.get("busy", False),
    )


@router.post("/compute", response_model=ComputeInfo)
async def set_compute_backend(req: SetComputeRequest) -> ComputeInfo:
    """
    Change the active compute backend.

    - cuda  → NVIDIA GPU via CUDA (requires PyTorch with CUDA support)
    - rocm  → AMD GPU via ROCm/HIP (requires PyTorch ROCm build)
    - mps   → Apple Silicon unified memory (macOS only)
    - cpu   → CPU-only, no GPU required (slower but always available)
    - auto  → detect best available at next startup

    Changes are persisted to .env and take effect for new inference requests.
    Running jobs are NOT interrupted.
    """
    import os
    from backend.utils.env_file import write_env_keys
    from backend.config import get_settings

    VALID = {"auto", "cuda", "rocm", "mps", "cpu"}
    if req.backend not in VALID:
        raise HTTPException(400, f"Invalid backend '{req.backend}'. Choose from: {sorted(VALID)}")

    hw = _detect_hardware()

    # Warn if requesting unavailable backend but do NOT block — the user may be
    # pre-configuring for a different machine or installing ROCm separately.
    warnings = []
    if req.backend == "cuda" and not hw["cuda_available"]:
        warnings.append("CUDA not detected. Setting saved — will fall back to CPU at runtime.")
    if req.backend == "rocm" and not hw["rocm_available"]:
        warnings.append("ROCm/HIP not detected. Setting saved — requires PyTorch ROCm build.")
    if req.backend == "mps" and not hw["mps_available"]:
        warnings.append("MPS not detected. Setting saved — requires Apple Silicon macOS.")

    os.environ["COMPUTE_BACKEND"] = req.backend
    write_env_keys({"COMPUTE_BACKEND": req.backend})
    get_settings.cache_clear()

    if warnings:
        logger.warning("Compute backend set with warnings", backend=req.backend, warnings=warnings)
    else:
        logger.info("Compute backend changed", backend=req.backend)

    return await get_compute_info()
