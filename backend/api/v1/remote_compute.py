"""
backend.api.v1.remote_compute — Remote GPU compute management endpoints.

Endpoints:
    GET  /remote-compute/backends          — List backends + availability
    GET  /remote-compute/backends/active   — Active backend info
    POST /remote-compute/backends/test     — Test backend connectivity
    POST /remote-compute/infer             — Run remote VLM inference
    POST /remote-compute/sam2              — Run remote SAM2 segmentation

All compute offloading is optional. The platform works entirely on local GPU
by default; these endpoints allow routing heavy tasks to cloud GPUs.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from shared.logging.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/remote-compute", tags=["remote-compute"])


class BackendStatus(BaseModel):
    backend_id: str
    name: str
    kind: str
    free_tier: bool
    free_tier_note: str
    signup_url: str
    gpu_tiers: list[str]
    cost_per_hour: dict[str, float]
    available: bool
    required_env_vars: list[str]


class RemoteInferRequest(BaseModel):
    backend_id: str = Field(default="replicate", description="Backend to use")
    model_id: str = Field(default="meta/meta-llama-3-2-11b-vision-instruct")
    prompt: str = Field(description="Prompt for the vision model")
    gpu_tier: str = Field(default="t4")


class RemoteInferResult(BaseModel):
    success: bool
    backend_id: str
    model_id: str
    output: Any | None
    error: str | None
    latency_s: float
    gpu_tier: str
    estimated_cost_usd: float | None


def _load_image(upload: UploadFile) -> np.ndarray:
    import cv2
    content = upload.file.read()
    arr = np.frombuffer(content, np.uint8)
    image_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise HTTPException(status_code=422, detail="Cannot decode image")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


@router.get("/backends", response_model=list[BackendStatus])
async def list_backends() -> list[BackendStatus]:
    """
    List all remote compute backends with configuration status.

    Shows which backends are configured (API keys set), their GPU options,
    cost per hour, and free tier availability.
    """
    from services.remote_compute.registry import list_backends
    return [BackendStatus(**b) for b in list_backends()]


@router.post("/backends/test")
async def test_backend(backend_id: str = Form(default="modal")) -> dict[str, Any]:
    """
    Test connectivity to a remote compute backend.

    Verifies API keys are valid and the backend is reachable.
    """
    from services.remote_compute.registry import get_compute_backend

    try:
        backend = get_compute_backend(backend_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not backend.is_available:
        env_vars = ", ".join(v for v in [
            "MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET", "REPLICATE_API_KEY"
        ] if backend_id in v.lower() or True)
        return {
            "backend_id": backend_id,
            "available": False,
            "error": (
                f"Backend '{backend_id}' not configured. "
                f"Set required env vars in .env. See GET /remote-compute/backends for details."
            ),
        }

    return {
        "backend_id": backend_id,
        "available": True,
        "status": "configured",
        "note": f"Backend '{backend_id}' is configured and ready to use.",
    }


@router.post("/infer", response_model=RemoteInferResult)
async def remote_infer(
    file: UploadFile = File(..., description="Image to analyze"),
    backend_id: str = Form(default="replicate"),
    model_id: str = Form(default="meta/meta-llama-3-2-11b-vision-instruct"),
    prompt: str = Form(description="Vision model prompt"),
    gpu_tier: str = Form(default="t4"),
) -> RemoteInferResult:
    """
    Run VLM inference on a remote GPU.

    Offloads image analysis to Modal (serverless) or Replicate (hosted).
    Use when the local RTX 4060 is busy with training or lacks VRAM.
    """
    from services.remote_compute.registry import get_compute_backend
    from services.remote_compute.base import GpuTier

    image = _load_image(file)

    try:
        backend = get_compute_backend(backend_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not backend.is_available:
        raise HTTPException(
            status_code=503,
            detail=f"Backend '{backend_id}' is not configured. See GET /remote-compute/backends.",
        )

    try:
        gpu = GpuTier(gpu_tier)
    except ValueError:
        gpu = GpuTier.T4

    result = await backend.run_vlm_inference(image, prompt, model_id, gpu)

    cost = None
    if result.latency_s > 0 and backend_id in backend.info.cost_per_hour:
        cost = backend.info.cost_per_hour.get(gpu_tier, 0) * result.latency_s / 3600

    return RemoteInferResult(
        success=result.success,
        backend_id=backend_id,
        model_id=model_id,
        output=result.output,
        error=result.error,
        latency_s=result.latency_s,
        gpu_tier=result.gpu_tier,
        estimated_cost_usd=round(cost, 6) if cost else None,
    )


@router.post("/sam2")
async def remote_sam2(
    file: UploadFile = File(..., description="Image for SAM2 segmentation"),
    backend_id: str = Form(default="replicate"),
) -> dict[str, Any]:
    """
    Run SAM2 segmentation on a remote GPU via Replicate.

    Use when local VRAM is occupied or for SAM2-large (not feasible on 8GB).
    Returns mask URLs from Replicate's storage.
    """
    from services.remote_compute.registry import get_compute_backend
    from services.remote_compute.replicate_backend import ReplicateBackend

    if backend_id != "replicate":
        raise HTTPException(
            status_code=400,
            detail="SAM2 remote inference only supported via Replicate backend.",
        )

    image = _load_image(file)
    backend = get_compute_backend("replicate")

    if not isinstance(backend, ReplicateBackend) or not backend.is_available:
        raise HTTPException(
            status_code=503,
            detail="Replicate not configured. Set REPLICATE_API_KEY in .env.",
        )

    result = await backend.run_sam2_inference(image)

    return {
        "success": result.success,
        "output": result.output,
        "error": result.error,
        "latency_s": result.latency_s,
        "backend": "replicate",
    }
