"""
inference/api/main.py — Standalone FastAPI inference server.

Lightweight server that exposes only inference endpoints.
Separate from the main backend (backend/main.py) for deployment flexibility:
  - Can run on GPU nodes without the full backend stack
  - Lower latency (no ORM, no admin endpoints)
  - Can be containerized separately (Dockerfile.inference)

Default port: 8001 (backend: 8000, inference: 8001)

Usage:
    uvicorn inference.api.main:app --host 0.0.0.0 --port 8001 --workers 1

Note: --workers 1 is required — GPU state is not shareable across processes.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from inference.api.router import router as infer_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup and shutdown."""
    print("[inference server] Starting — no model pre-loaded (lazy init on first request)")
    yield
    print("[inference server] Shutdown")


app = FastAPI(
    title="Trichome Analysis — Inference Server",
    description=(
        "Standalone inference API for trichome detection. "
        "Supports PyTorch, ONNX Runtime, and TensorRT backends."
    ),
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS: allow main backend + frontend to call inference server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8000", "*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Mount inference router
app.include_router(infer_router, prefix="/api/v1")


@app.get("/", include_in_schema=False)
def root():
    return {"service": "trichome-inference", "version": "0.1.0", "docs": "/docs"}


@app.get("/api/v1/inference/status")
def health():
    """Health endpoint for Docker HEALTHCHECK."""
    return JSONResponse({"status": "ok"})


# ---------------------------------------------------------------------------
# Development entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "inference.api.main:app",
        host="0.0.0.0",
        port=8001,
        workers=1,  # Must be 1 for GPU state sharing
        reload=False,
    )
