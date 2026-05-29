"""
backend.main — FastAPI application factory.

Start:
    uvicorn backend.main:app --reload --port 8000

Or via CLI:
    trichome serve --port 8000
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.config import get_settings
from shared.logging.logger import configure_logging, get_logger

settings = get_settings()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan: startup and shutdown."""
    # ── STARTUP ──────────────────────────────────────────────────
    configure_logging(
        log_level=settings.log_level,
        log_file=settings.log_file,
    )

    # Re-install WebSocket log sink AFTER configure_logging() (which calls loguru.remove())
    try:
        from backend.websocket.router import _loguru_ws_sink, _log_buffer, _time
        from loguru import logger as _loguru
        _loguru.add(_loguru_ws_sink, level="DEBUG", format="{message}", enqueue=False)
        _log_buffer.append({
            "ts": _time.time(),
            "level": "INFO",
            "logger": "trichome.startup",
            "msg": f"Trichome Analysis Platform v{settings.app_version} starting…",
        })
    except Exception:
        pass

    logger.info(
        "Starting Trichome Analysis Platform",
        version=settings.app_version,
        environment=settings.environment,
        port=settings.api_port,
    )

    # Ensure storage directories exist
    settings.ensure_dirs()

    # Initialize database
    from backend.database import create_all_tables
    create_all_tables()
    logger.info("Database tables initialized")

    # Restore GPU task history from DB (marks any in-flight jobs as failed)
    try:
        from backend.database import get_session
        from backend.tasks.task_router import task_router
        with next(get_session()) as db:
            restored = await task_router.restore_from_db(db)
            logger.info("TaskRouter: job history restored from DB", count=restored)
    except Exception as exc:
        logger.warning("TaskRouter DB restore failed (non-fatal)", error=str(exc))

    # Initialize persistent background task store (SQLite)
    try:
        from backend.api.v1.containers import _TASK_DB
        from backend.tasks.task_store import get_task_store
        task_store = get_task_store(_TASK_DB)
        await task_store.initialize()
        logger.info("Background task store initialized", db=str(_TASK_DB))

        # Periodic expiry loop — runs every 6h
        async def task_expiry_loop() -> None:
            import asyncio as _aio
            while True:
                await _aio.sleep(6 * 3600)
                try:
                    n = await task_store.expire()
                    if n:
                        logger.info("Task store: expired old tasks", count=n)
                except Exception as exc:
                    logger.warning("Task expiry failed", error=str(exc))

        expiry_task = asyncio.create_task(task_expiry_loop())
        app.state.expiry_task = expiry_task
    except Exception as exc:
        logger.warning("Task store initialization failed (non-fatal)", error=str(exc))

    # Start GPU stats broadcaster (background task)
    import asyncio
    from backend.websocket.manager import ws_manager
    from backend.api.v1.system import _get_gpu_stats, _get_cpu_ram_stats

    async def gpu_broadcast_loop() -> None:
        while True:
            try:
                if ws_manager.connection_count > 0:
                    gpu = _get_gpu_stats()
                    await ws_manager.send_gpu_stats({
                        "gpu": gpu,
                        "timestamp": time.time(),
                    })
            except Exception:
                pass
            await asyncio.sleep(settings.gpu_poll_interval_s)

    gpu_task = asyncio.create_task(gpu_broadcast_loop())
    app.state.gpu_task = gpu_task

    # Periodic system heartbeat log (keeps Processes tray alive)
    async def log_heartbeat_loop() -> None:
        import psutil
        while True:
            await asyncio.sleep(30)
            try:
                cpu_pct = psutil.cpu_percent(interval=None)
                ram = psutil.virtual_memory()
                logger.debug(
                    f"System heartbeat — CPU {cpu_pct:.1f}% | RAM {ram.percent:.1f}% | "
                    f"WS clients {ws_manager.connection_count}"
                )
            except Exception:
                pass

    heartbeat_task = asyncio.create_task(log_heartbeat_loop())
    app.state.heartbeat_task = heartbeat_task

    # Wire GPU semaphore: unify the dependency semaphore with task_router's
    # so REST inference endpoints and background training jobs share one slot.
    try:
        from backend.dependencies.gpu import (
            wire_task_router_semaphore,
            configure_gpu_rate_limit,
        )
        wire_task_router_semaphore()
        configure_gpu_rate_limit(settings.gpu_inference_queue_depth)
        logger.info(
            "GPU semaphore wired to task_router",
            queue_depth=settings.gpu_inference_queue_depth,
        )
    except Exception as exc:
        logger.warning("GPU semaphore wire failed (non-fatal)", error=str(exc))

    logger.info("Application ready")
    yield

    # ── SHUTDOWN ─────────────────────────────────────────────────
    if hasattr(app.state, "expiry_task"):
        app.state.expiry_task.cancel()
    if hasattr(app.state, "gpu_task"):
        app.state.gpu_task.cancel()
    if hasattr(app.state, "heartbeat_task"):
        app.state.heartbeat_task.cancel()
    logger.info("Application shutting down")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "Research-grade Cannabis trichome analysis platform. "
            "Detection, segmentation, maturity analysis, VLM labeling, and training."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ── CORS ─────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── MIDDLEWARE (pure ASGI — WebSocket-safe) ───────────────────

    # Auth middleware — registered first so it runs outermost
    try:
        from backend.middleware.auth import APITokenMiddleware
        app.add_middleware(APITokenMiddleware, api_token=settings.api_token)
    except Exception:
        pass

    try:
        from backend.middleware.gpu_guard import GpuGuardMiddleware
        app.add_middleware(GpuGuardMiddleware)
    except Exception:
        pass

    try:
        from backend.middleware.request_logger import RequestLoggerMiddleware
        app.add_middleware(RequestLoggerMiddleware)
    except Exception:
        pass

    # ── API ROUTES ───────────────────────────────────────────────
    from backend.api.v1.router import router as v1_router
    app.include_router(v1_router, prefix=settings.api_prefix)

    # Detection router (separate prefix from v1 — /api/v1/detect/*)
    # Note: VLM router is registered inside backend.api.v1.router — do NOT add it here too.
    from detection.api.router import router as detection_router
    app.include_router(detection_router, prefix=settings.api_prefix)

    # ── WEBSOCKET ROUTES ─────────────────────────────────────────
    from backend.websocket.router import router as ws_router
    app.include_router(ws_router)

    # ── ROOT ENDPOINT ────────────────────────────────────────────
    @app.get("/")
    async def root() -> dict:
        return {
            "service": settings.app_name,
            "version": settings.app_version,
            "docs": "/docs",
            "health": f"{settings.api_prefix}/system/health",
            "frontend": "http://localhost:3000",
        }

    # ── EXCEPTION HANDLERS ───────────────────────────────────────
    @app.exception_handler(ValueError)
    async def value_error_handler(request, exc):
        return JSONResponse(
            status_code=422,
            content={"detail": str(exc), "type": "ValueError"},
        )

    @app.exception_handler(FileNotFoundError)
    async def file_not_found_handler(request, exc):
        return JSONResponse(
            status_code=404,
            content={"detail": str(exc), "type": "FileNotFoundError"},
        )

    return app


# Application instance (used by uvicorn)
app = create_app()
