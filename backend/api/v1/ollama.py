"""
backend.api.v1.ollama — Ollama local LLM management and narrative generation.

Endpoints:
    GET  /ollama/status          — Probe availability + installed models
    GET  /ollama/models          — List installed models with metadata
    POST /ollama/models/pull     — Trigger async model download (streaming progress)
    POST /ollama/narrative       — Generate a report narrative from analysis JSON
    PUT  /ollama/config          — Update Ollama settings (persisted to .env)

All narrative requests are routed to the local Ollama instance at the
configured base URL (default: http://localhost:11434).

SCIENTIFIC CONSTRAINT:
    THC/cannabinoid fields are stripped from analysis_result before any
    narrative is generated (enforced in OllamaProvider.build_prompt).
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

from shared.logging.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/ollama", tags=["ollama"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class OllamaStatusResponse(BaseModel):
    available: bool
    base_url: str
    installed_models: list[str]
    current_model: str


class OllamaModelInfo(BaseModel):
    name: str
    size_bytes: int | None = None
    size_gb: float | None = None
    modified_at: str | None = None
    digest: str | None = None


class PullModelRequest(BaseModel):
    model: str = Field(description="Ollama model tag to pull, e.g. 'llama3.2:3b'")


class PullModelResponse(BaseModel):
    task_id: str
    model: str
    status: str


class NarrativeRequest(BaseModel):
    analysis_result: dict[str, Any] = Field(
        description="Structured analysis result from the CTIP detection pipeline."
    )
    style: str = Field(
        default="scientific",
        description="Narrative style: 'scientific' | 'summary' | 'technical'",
    )
    language: str = Field(
        default="en",
        description="Output language: 'en' | 'de' | 'es'",
    )
    model: str | None = Field(
        default=None,
        description="Override model. None = use default from settings.",
    )


class NarrativeResponse(BaseModel):
    narrative: str
    model: str
    style: str
    language: str
    generation_time_ms: float


class OllamaConfigRequest(BaseModel):
    model: str | None = Field(default=None, description="Default Ollama model")
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1, le=8192)
    base_url: str | None = Field(default=None, description="Ollama base URL")


class OllamaConfigResponse(BaseModel):
    base_url: str
    model: str
    temperature: float
    max_tokens: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_ollama_config():
    """Build OllamaConfig from current settings."""
    from backend.config import get_settings
    from vlm_labeling.providers.local.ollama_provider import OllamaConfig

    s = get_settings()
    return OllamaConfig(
        base_url=s.ollama_base_url,
        model=s.ollama_model,
        timeout_s=30.0,
        max_tokens=s.ollama_max_tokens,
        temperature=s.ollama_temperature,
    )


def _get_provider(model_override: str | None = None):
    """Instantiate OllamaProvider with current settings, optionally overriding model."""
    from vlm_labeling.providers.local.ollama_provider import OllamaConfig, OllamaProvider

    cfg = _get_ollama_config()
    if model_override:
        from dataclasses import replace
        cfg = replace(cfg, model=model_override)
    return OllamaProvider(config=cfg)


_VALID_STYLES = {"scientific", "summary", "technical"}
_VALID_LANGUAGES = {"en", "de", "es"}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/status", response_model=OllamaStatusResponse)
async def get_ollama_status() -> OllamaStatusResponse:
    """
    Probe Ollama availability and return installed models.

    Returns available=False if Ollama is not running or unreachable.
    Does NOT raise HTTP errors — callers should check the 'available' field.
    """
    from backend.config import get_settings

    s = get_settings()
    provider = _get_provider()

    available = await provider.is_available()
    installed: list[str] = []

    if available:
        installed = await provider.list_models()

    return OllamaStatusResponse(
        available=available,
        base_url=s.ollama_base_url,
        installed_models=installed,
        current_model=s.ollama_model,
    )


@router.get("/models", response_model=list[OllamaModelInfo])
async def list_ollama_models() -> list[OllamaModelInfo]:
    """
    Return installed Ollama models with size metadata.

    Returns an empty list if Ollama is unavailable.
    """
    import aiohttp

    from backend.config import get_settings

    s = get_settings()
    url = f"{s.ollama_base_url}/api/tags"
    timeout = aiohttp.ClientTimeout(total=10.0)

    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url) as response:
                if response.status != 200:
                    return []
                data = await response.json()
    except Exception as exc:
        logger.warning("Failed to query Ollama models", error=str(exc))
        return []

    result: list[OllamaModelInfo] = []
    for m in data.get("models", []):
        size_bytes = m.get("size")
        result.append(
            OllamaModelInfo(
                name=m.get("name", ""),
                size_bytes=size_bytes,
                size_gb=round(size_bytes / 1024**3, 2) if size_bytes else None,
                modified_at=m.get("modified_at"),
                digest=m.get("digest"),
            )
        )
    return result


# ---------------------------------------------------------------------------
# Background pull task registry (simple in-process store)
# ---------------------------------------------------------------------------

_pull_tasks: dict[str, dict[str, Any]] = {}


async def _run_pull(task_id: str, model_name: str, base_url: str) -> None:
    """Background coroutine that streams model pull progress and updates task state."""
    from vlm_labeling.providers.local.ollama_provider import OllamaConfig, OllamaProvider

    cfg = OllamaConfig(base_url=base_url)
    provider = OllamaProvider(config=cfg)

    _pull_tasks[task_id] = {
        "task_id": task_id,
        "model": model_name,
        "status": "in_progress",
        "progress": [],
        "error": None,
    }

    try:
        async for chunk in provider.pull_model(model_name):
            _pull_tasks[task_id]["progress"].append(chunk)
            if chunk.get("status") == "success":
                break
        _pull_tasks[task_id]["status"] = "completed"
        logger.info("Model pull task completed", task_id=task_id, model=model_name)
    except Exception as exc:
        _pull_tasks[task_id]["status"] = "failed"
        _pull_tasks[task_id]["error"] = str(exc)
        logger.error("Model pull task failed", task_id=task_id, model=model_name, error=str(exc))


@router.post("/models/pull", response_model=PullModelResponse)
async def pull_model(
    req: PullModelRequest,
    background_tasks: BackgroundTasks,
) -> PullModelResponse:
    """
    Pull an Ollama model in the background.

    Returns a task_id immediately.  Use GET /ollama/models/pull/{task_id}
    to poll progress (or watch the server logs).

    The pull is performed via Ollama's streaming /api/pull endpoint.
    """
    import uuid

    from backend.config import get_settings

    s = get_settings()
    task_id = str(uuid.uuid4())

    background_tasks.add_task(_run_pull, task_id, req.model, s.ollama_base_url)

    logger.info(
        "Ollama model pull queued",
        model=req.model,
        task_id=task_id,
    )

    return PullModelResponse(
        task_id=task_id,
        model=req.model,
        status="queued",
    )


@router.get("/models/pull/{task_id}")
async def get_pull_status(task_id: str) -> dict[str, Any]:
    """
    Return the current status of a model pull task.

    Status values: "queued" | "in_progress" | "completed" | "failed"
    """
    task = _pull_tasks.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
    return task


@router.post("/narrative", response_model=NarrativeResponse)
async def generate_narrative(req: NarrativeRequest) -> NarrativeResponse:
    """
    Generate a human-readable report narrative from a structured analysis result.

    The model receives the detection counts, trichome type distribution,
    optical maturity distribution, confidence statistics and harvest
    recommendation.  It returns a formatted narrative in the requested style
    and language.

    THC/cannabinoid fields are unconditionally stripped before the prompt is
    constructed (scientific constraint — optical maturity only).
    """
    import aiohttp

    # Validate style / language before touching Ollama
    if req.style not in _VALID_STYLES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid style '{req.style}'. Choose from: {sorted(_VALID_STYLES)}",
        )
    if req.language not in _VALID_LANGUAGES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid language '{req.language}'. Choose from: {sorted(_VALID_LANGUAGES)}",
        )

    provider = _get_provider(model_override=req.model)

    # Verify Ollama is reachable before attempting generation
    if not await provider.is_available():
        raise HTTPException(
            status_code=503,
            detail=(
                f"Ollama is not available at {provider.config.base_url}. "
                "Ensure the Ollama service is running."
            ),
        )

    t0 = time.perf_counter()
    try:
        narrative = await provider.generate_narrative(
            analysis_result=req.analysis_result,
            style=req.style,
            language=req.language,
        )
    except asyncio.TimeoutError as exc:
        raise HTTPException(
            status_code=504,
            detail=f"Ollama request timed out after {provider.config.timeout_s}s",
        ) from exc
    except aiohttp.ClientError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Ollama communication error: {exc}",
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Ollama generation failed: {exc}",
        ) from exc

    generation_time_ms = (time.perf_counter() - t0) * 1000.0

    return NarrativeResponse(
        narrative=narrative,
        model=provider.config.model,
        style=req.style,
        language=req.language,
        generation_time_ms=round(generation_time_ms, 2),
    )


@router.put("/config", response_model=OllamaConfigResponse)
async def update_ollama_config(req: OllamaConfigRequest) -> OllamaConfigResponse:
    """
    Update Ollama configuration settings.

    Changes are persisted to .env and take effect immediately for subsequent
    requests.  The settings LRU cache is cleared after each write.
    """
    import os

    from backend.config import get_settings
    from backend.utils.env_file import write_env_keys

    to_write: dict[str, str] = {}

    if req.base_url is not None:
        to_write["OLLAMA_BASE_URL"] = req.base_url
        os.environ["OLLAMA_BASE_URL"] = req.base_url

    if req.model is not None:
        to_write["OLLAMA_MODEL"] = req.model
        os.environ["OLLAMA_MODEL"] = req.model

    if req.temperature is not None:
        to_write["OLLAMA_TEMPERATURE"] = str(req.temperature)
        os.environ["OLLAMA_TEMPERATURE"] = str(req.temperature)

    if req.max_tokens is not None:
        to_write["OLLAMA_MAX_TOKENS"] = str(req.max_tokens)
        os.environ["OLLAMA_MAX_TOKENS"] = str(req.max_tokens)

    if not to_write:
        raise HTTPException(
            status_code=400,
            detail="No configuration fields provided to update.",
        )

    write_env_keys(to_write)
    get_settings.cache_clear()

    logger.info("Ollama configuration updated and persisted to .env", keys=list(to_write.keys()))

    # Return freshly loaded settings
    s = get_settings()
    return OllamaConfigResponse(
        base_url=s.ollama_base_url,
        model=s.ollama_model,
        temperature=s.ollama_temperature,
        max_tokens=s.ollama_max_tokens,
    )
