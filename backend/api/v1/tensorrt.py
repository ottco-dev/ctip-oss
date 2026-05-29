"""
backend.api.v1.tensorrt — TensorRT engine management and inference endpoints.

Endpoints:
    GET  /tensorrt/status                  — TRT availability + installed engines
    GET  /tensorrt/engines                 — list .engine files in models dir
    GET  /tensorrt/inspect/{engine_name}   — I/O tensor info for a built engine
    POST /tensorrt/build                   — build .engine from ONNX (background)
    POST /tensorrt/infer                   — single-image inference via TRT engine
    GET  /tensorrt/build/{job_id}          — poll build job status

All endpoints degrade gracefully when TensorRT is not installed:
  - GET /tensorrt/status → 200 with available=false + install instructions
  - Other endpoints → 503 with install instructions

Install TensorRT (NVIDIA GPU required):
  pip install tensorrt pycuda
  (or) pip install tensorrt==10.x.x --extra-index-url https://pypi.nvidia.com

Performance reference (RTX 4060, YOLO11s, 1280px FP16):
  - ONNX Runtime:   ~15-25 ms/image
  - TensorRT FP16:   ~4-6  ms/image  (~4× speedup)
"""

from __future__ import annotations

import io
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from shared.logging.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/tensorrt", tags=["tensorrt"])

# ── In-memory build job store ─────────────────────────────────────────────────
# For production, this would be backed by the DB BackgroundJob table.
# Kept simple here since engine builds are rare (one-shot ops).
_build_jobs: dict[str, dict[str, Any]] = {}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _engines_dir() -> Path:
    try:
        from backend.config import get_settings
        return Path(get_settings().models_dir) / "engines"
    except Exception:
        return Path("models/engines")


def _trt_available() -> bool:
    from inference.tensorrt_engine.runner import tensorrt_available
    return tensorrt_available()


def _require_trt() -> None:
    if not _trt_available():
        raise HTTPException(
            status_code=503,
            detail={
                "available": False,
                "message": "TensorRT is not installed in this environment.",
                "install": (
                    "pip install tensorrt pycuda\n"
                    "  OR\n"
                    "pip install tensorrt==10.6.0 --extra-index-url https://pypi.nvidia.com\n"
                    "Requires: NVIDIA GPU + CUDA toolkit + NVIDIA Container Toolkit (Docker)"
                ),
                "docs": "https://docs.nvidia.com/deeplearning/tensorrt/install-guide/",
            },
        )


def _list_engine_files() -> list[dict[str, Any]]:
    d = _engines_dir()
    if not d.exists():
        return []
    engines = []
    for f in sorted(d.glob("*.engine")):
        stat = f.stat()
        engines.append({
            "name": f.name,
            "path": str(f),
            "size_mb": round(stat.st_size / (1024**2), 1),
            "modified": stat.st_mtime,
        })
    return engines


def _image_bytes_to_bgr(data: bytes) -> np.ndarray:
    import cv2
    arr = np.frombuffer(data, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(422, "Cannot decode uploaded image")
    return bgr


# ── Schemas ───────────────────────────────────────────────────────────────────

class TRTStatusResponse(BaseModel):
    available: bool
    trt_version: str | None
    pycuda_version: str | None
    cuda_available: bool
    engines: list[dict[str, Any]]
    engines_dir: str
    install_hint: str | None


class BuildRequest(BaseModel):
    onnx_path: str = Field(description="Path to the ONNX model file")
    engine_name: str = Field(
        default="",
        description="Output engine filename (default: same stem as onnx_path + _fp16.engine)"
    )
    imgsz: int = Field(default=1280, ge=32, le=4096)
    fp16: bool = Field(default=True, description="Enable FP16 precision (recommended for RTX 4060)")
    workspace_gb: float = Field(default=4.0, ge=0.5, le=24.0, description="GPU memory workspace for builder")
    overwrite: bool = Field(default=False)


class BuildJobStatus(BaseModel):
    job_id: str
    status: str          # queued | running | completed | failed
    onnx_path: str
    engine_path: str
    started_at: float | None
    finished_at: float | None
    duration_s: float | None
    engine_size_mb: float | None
    error: str | None


class TRTInferRequest(BaseModel):
    engine_name: str = Field(description="Engine filename in models/engines/")
    conf_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    iou_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    imgsz: int = Field(default=1280)


class TRTDetection(BaseModel):
    x1: float; y1: float; x2: float; y2: float
    confidence: float
    class_id: int
    class_name: str


class TRTInferResponse(BaseModel):
    engine_name: str
    detections: list[TRTDetection]
    inference_ms: float
    preprocess_ms: float
    postprocess_ms: float
    total_ms: float
    image_hw: tuple[int, int]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/status", response_model=TRTStatusResponse)
async def get_trt_status() -> TRTStatusResponse:
    """
    Return TensorRT availability, version, and list of built engines.
    Safe to call even when TensorRT is not installed.

    'available' is True only when BOTH tensorrt AND pycuda are importable
    (checked via tensorrt_available() which mirrors the runner's guard).
    """
    # Use the same check the runner uses (tensorrt + pycuda both required)
    available = _trt_available()

    trt_version: str | None = None
    pycuda_version: str | None = None

    try:
        import tensorrt as trt
        trt_version = trt.__version__
    except ImportError:
        pass

    try:
        import pycuda
        pycuda_version = pycuda.VERSION_TEXT
    except ImportError:
        pass

    cuda_available = False
    try:
        import torch
        cuda_available = torch.cuda.is_available()
    except ImportError:
        pass

    install_hint = None
    if not available:
        install_hint = (
            "pip install tensorrt pycuda\n"
            "  OR (pip with NVIDIA index):\n"
            "pip install tensorrt==10.6.0 --extra-index-url https://pypi.nvidia.com\n"
            "Requires NVIDIA GPU + CUDA + NVIDIA Container Toolkit"
        )

    return TRTStatusResponse(
        available=available,
        trt_version=trt_version,
        pycuda_version=pycuda_version,
        cuda_available=cuda_available,
        engines=_list_engine_files(),
        engines_dir=str(_engines_dir()),
        install_hint=install_hint,
    )


@router.get("/engines")
async def list_engines() -> list[dict[str, Any]]:
    """List all .engine files in the models/engines directory."""
    return _list_engine_files()


@router.get("/inspect/{engine_name}")
async def inspect_engine(engine_name: str) -> dict[str, Any]:
    """
    Return I/O tensor names, shapes, and dtypes for a built engine.
    Requires TensorRT to be installed.
    """
    _require_trt()

    engine_path = _engines_dir() / engine_name
    if not engine_path.exists():
        raise HTTPException(404, f"Engine not found: {engine_name}")

    from inference.tensorrt_engine.builder import inspect_engine as _inspect
    try:
        return _inspect(engine_path)
    except Exception as exc:
        raise HTTPException(500, f"Failed to inspect engine: {exc}") from exc


@router.post("/build", response_model=BuildJobStatus)
async def build_engine(
    req: BuildRequest,
    background_tasks: BackgroundTasks,
) -> BuildJobStatus:
    """
    Build a TensorRT engine from an ONNX model (background task).

    The build process takes 1-10 minutes depending on model complexity.
    Poll /tensorrt/build/{job_id} to track progress.
    """
    _require_trt()

    onnx_path = Path(req.onnx_path)
    if not onnx_path.exists():
        raise HTTPException(404, f"ONNX model not found: {req.onnx_path}")

    engine_name = req.engine_name or (onnx_path.stem + "_fp16.engine")
    if not engine_name.endswith(".engine"):
        engine_name += ".engine"

    engine_path = _engines_dir() / engine_name

    if engine_path.exists() and not req.overwrite:
        raise HTTPException(
            409,
            f"Engine already exists: {engine_name}. Set overwrite=true to rebuild.",
        )

    job_id = str(uuid.uuid4())
    _build_jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "onnx_path": str(onnx_path),
        "engine_path": str(engine_path),
        "started_at": None,
        "finished_at": None,
        "duration_s": None,
        "engine_size_mb": None,
        "error": None,
    }

    background_tasks.add_task(_run_build, job_id, req, engine_path)

    logger.info("TRT build job queued", job_id=job_id, onnx=str(onnx_path), engine=str(engine_path))
    return BuildJobStatus(**_build_jobs[job_id])


@router.get("/build/{job_id}", response_model=BuildJobStatus)
async def get_build_status(job_id: str) -> BuildJobStatus:
    """Poll TensorRT build job status."""
    job = _build_jobs.get(job_id)
    if not job:
        raise HTTPException(404, f"Build job not found: {job_id}")
    return BuildJobStatus(**job)


@router.post("/infer", response_model=TRTInferResponse)
async def infer_with_engine(
    file: UploadFile = File(..., description="Image file (JPEG/PNG)"),
    engine_name: str = Form(..., description="Engine filename in models/engines/"),
    conf_threshold: float = Form(default=0.25),
    iou_threshold: float = Form(default=0.45),
    imgsz: int = Form(default=1280),
) -> TRTInferResponse:
    """
    Run detection inference using a built TensorRT engine.

    The runner is not cached between requests — for production throughput,
    load the runner once at application startup and reuse it.
    """
    _require_trt()

    engine_path = _engines_dir() / engine_name
    if not engine_path.exists():
        raise HTTPException(404, f"Engine not found: {engine_name}")

    from inference.tensorrt_engine.runner import TensorRTRunner, TRTRunnerConfig

    raw = await file.read()
    bgr = _image_bytes_to_bgr(raw)

    cfg = TRTRunnerConfig(
        engine_path=str(engine_path),
        imgsz=imgsz,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
    )

    t0 = time.perf_counter()
    try:
        runner = TensorRTRunner(cfg)
        runner.load()
        result = runner.infer(bgr)
        runner.unload()
    except Exception as exc:
        logger.error("TRT inference failed", engine=engine_name, error=str(exc))
        raise HTTPException(500, f"TRT inference error: {exc}") from exc

    total_ms = round((time.perf_counter() - t0) * 1000, 1)

    return TRTInferResponse(
        engine_name=engine_name,
        detections=[
            TRTDetection(
                x1=d.x1, y1=d.y1, x2=d.x2, y2=d.y2,
                confidence=d.confidence,
                class_id=d.class_id,
                class_name=d.class_name,
            )
            for d in result.detections
        ],
        inference_ms=result.inference_ms,
        preprocess_ms=result.preprocess_ms,
        postprocess_ms=result.postprocess_ms,
        total_ms=total_ms,
        image_hw=result.image_hw,
    )


# ── Background build task ─────────────────────────────────────────────────────

def _run_build(job_id: str, req: BuildRequest, engine_path: Path) -> None:
    """Blocking TRT engine build — runs in FastAPI's thread pool."""
    job = _build_jobs[job_id]
    job["status"] = "running"
    job["started_at"] = time.time()

    try:
        _engines_dir().mkdir(parents=True, exist_ok=True)

        from inference.tensorrt_engine.builder import TRTBuildConfig, build_engine_from_onnx

        cfg = TRTBuildConfig(
            onnx_path=req.onnx_path,
            engine_path=str(engine_path),
            imgsz=req.imgsz,
            fp16=req.fp16,
            workspace_gb=req.workspace_gb,
            verbosity="INFO",
        )
        out = build_engine_from_onnx(cfg, overwrite=req.overwrite)

        job["status"] = "completed"
        job["engine_size_mb"] = round(out.stat().st_size / (1024**2), 1)
        logger.info("TRT build completed", job_id=job_id, size_mb=job["engine_size_mb"])

    except Exception as exc:
        job["status"] = "failed"
        job["error"] = str(exc)
        logger.error("TRT build failed", job_id=job_id, error=str(exc))
    finally:
        job["finished_at"] = time.time()
        if job["started_at"]:
            job["duration_s"] = round(job["finished_at"] - job["started_at"], 1)
