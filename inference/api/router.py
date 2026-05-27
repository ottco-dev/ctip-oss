"""
inference/api/router.py — Inference API endpoints for the standalone inference server.

This router is mounted by inference/api/main.py (the lightweight inference FastAPI app).
It is separate from backend/api/v1/inference.py which is part of the main backend.

Routes:
  GET  /infer/status         — health + loaded model info
  POST /infer/detect         — single image detection (ONNX / TRT / PyTorch)
  POST /infer/detect/batch   — batch image detection
  GET  /infer/models         — available exported models
  POST /infer/benchmark      — run latency benchmark
"""

from __future__ import annotations

import io
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, UploadFile
from pydantic import BaseModel, Field

router = APIRouter(prefix="/infer", tags=["inference"])

# ---------------------------------------------------------------------------
# Runtime state (lazily loaded)
# ---------------------------------------------------------------------------

_runner: Optional[object] = None
_runner_type: str = "none"
_model_path: str = ""


def _get_runner(model_path: str = "", backend: str = "auto"):
    """Lazily load the best available inference backend."""
    global _runner, _runner_type, _model_path

    if _runner is not None and (not model_path or model_path == _model_path):
        return _runner

    # Resolve backend
    if backend == "auto":
        # TRT > ONNX > PyTorch
        if model_path.endswith(".engine"):
            backend = "tensorrt"
        elif model_path.endswith(".onnx"):
            backend = "onnx"
        else:
            backend = "pytorch"

    if backend == "tensorrt":
        from inference.tensorrt_engine.runner import TensorRTRunner, TRTRunnerConfig
        _runner = TensorRTRunner(TRTRunnerConfig(engine_path=model_path))
        _runner_type = "tensorrt"

    elif backend == "onnx":
        from inference.onnx_runtime.runner import ONNXRuntimeRunner, ONNXRunnerConfig
        _runner = ONNXRuntimeRunner(ONNXRunnerConfig(model_path=model_path))
        _runner_type = "onnx"

    else:
        from inference.local.runner import LocalPyTorchRunner, LocalRunnerConfig
        _runner = LocalPyTorchRunner(LocalRunnerConfig(model_path=model_path))
        _runner_type = "pytorch"

    _runner.load()
    _model_path = model_path
    return _runner


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class Detection(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str


class InferRequest(BaseModel):
    model_path: str = ""
    backend: str = "auto"  # auto | pytorch | onnx | tensorrt
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    imgsz: int = 1280


class InferResponse(BaseModel):
    detections: list[Detection]
    inference_ms: float
    backend: str
    model_path: str
    image_width: int
    image_height: int
    total_ms: float


class BenchmarkRequest(BaseModel):
    model_path: str = ""
    backend: str = "auto"
    n_runs: int = Field(default=50, ge=5, le=500)
    imgsz: int = 1280


class BenchmarkResult(BaseModel):
    n_runs: int
    mean_ms: float
    min_ms: float
    max_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    fps: float
    backend: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/status")
def get_status():
    """Return inference server status."""
    return {
        "status": "running",
        "loaded_model": _model_path or None,
        "backend": _runner_type,
        "available_backends": {
            "pytorch": True,
            "onnx": _check_onnx(),
            "tensorrt": _check_trt(),
        },
    }


@router.post("/detect", response_model=InferResponse)
async def detect(file: UploadFile, request: InferRequest = InferRequest()):
    """Run detection on an uploaded image."""
    t_total = time.perf_counter()

    contents = await file.read()
    arr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(400, "Could not decode image")

    h, w = image.shape[:2]

    runner = _get_runner(request.model_path, request.backend)

    result = runner.infer(image)

    # Normalize detections to common schema
    detections = _parse_detections(result)

    total_ms = (time.perf_counter() - t_total) * 1000

    return InferResponse(
        detections=detections,
        inference_ms=getattr(result, "inference_ms", 0.0),
        backend=_runner_type,
        model_path=_model_path,
        image_width=w,
        image_height=h,
        total_ms=round(total_ms, 2),
    )


@router.post("/benchmark", response_model=BenchmarkResult)
def benchmark(request: BenchmarkRequest):
    """Run latency benchmark with random noise images."""
    runner = _get_runner(request.model_path, request.backend)

    imgsz = request.imgsz
    dummy = (np.random.rand(imgsz, imgsz, 3) * 255).astype(np.uint8)

    latencies = []
    for _ in range(request.n_runs):
        t0 = time.perf_counter()
        runner.infer(dummy)
        latencies.append((time.perf_counter() - t0) * 1000)

    latencies.sort()
    n = len(latencies)
    mean_ms = sum(latencies) / n

    return BenchmarkResult(
        n_runs=n,
        mean_ms=round(mean_ms, 2),
        min_ms=round(latencies[0], 2),
        max_ms=round(latencies[-1], 2),
        p50_ms=round(latencies[n // 2], 2),
        p95_ms=round(latencies[int(n * 0.95)], 2),
        p99_ms=round(latencies[int(n * 0.99)], 2),
        fps=round(1000.0 / mean_ms, 1),
        backend=_runner_type,
    )


@router.get("/models")
def list_models():
    """Scan common model directories for exported models."""
    model_dirs = [
        Path("/models/onnx"),
        Path("/models/tensorrt"),
        Path("/app/runs"),
        Path("runs"),
        Path("checkpoints"),
    ]

    models = []
    for d in model_dirs:
        if not d.exists():
            continue
        for ext in ("*.onnx", "*.engine", "*.pt"):
            for p in d.rglob(ext):
                models.append(
                    {
                        "path": str(p),
                        "name": p.name,
                        "format": p.suffix.lstrip("."),
                        "size_mb": round(p.stat().st_size / 1e6, 1),
                    }
                )

    return {"models": models}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_detections(result) -> list[Detection]:
    """Normalize detections from any runner to the common Detection schema."""
    raw_dets = getattr(result, "detections", [])
    detections = []
    for d in raw_dets:
        detections.append(
            Detection(
                x1=getattr(d, "x1", 0.0),
                y1=getattr(d, "y1", 0.0),
                x2=getattr(d, "x2", 0.0),
                y2=getattr(d, "y2", 0.0),
                confidence=getattr(d, "confidence", 0.0),
                class_id=getattr(d, "class_id", 0),
                class_name=getattr(d, "class_name", "unknown"),
            )
        )
    return detections


def _check_onnx() -> bool:
    try:
        import onnxruntime  # noqa: F401
        return True
    except ImportError:
        return False


def _check_trt() -> bool:
    try:
        import tensorrt  # noqa: F401
        return True
    except ImportError:
        return False
