"""
backend.api.v1.vlm_providers — Remote VLM provider management endpoints.

Endpoints:
    GET  /vlm/providers                 — List all providers + availability status
    GET  /vlm/providers/active          — Active provider info
    POST /vlm/providers/active          — Switch active provider
    POST /vlm/providers/test            — Test a provider with a sample image
    GET  /vlm/providers/{id}/models     — Models available for a provider
    POST /vlm/providers/{id}/configure  — Set API key for a provider (runtime only)

NOTES:
    - API keys are stored in .env for persistence, or set here for session only.
    - Switching providers does NOT affect ongoing batch jobs.
    - All outputs remain VLM_AUTO regardless of provider — HITL invariant preserved.
"""

from __future__ import annotations

import io
import time
from typing import Any

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from shared.logging.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/vlm/providers", tags=["vlm-providers"])


# ── Schemas ───────────────────────────────────────────────────────────────────

class ProviderStatus(BaseModel):
    provider_id: str
    name: str
    kind: str          # "local" | "remote"
    tier: str          # "free" | "freemium" | "paid"
    available: bool
    has_api_key: bool
    env_var: str
    models: list[str]
    default_model: str
    vram_gb: float | None
    cost_per_1k_tokens: float | None
    rate_limit_rpm: int | None
    free_tier_note: str
    signup_url: str
    is_active: bool = False


class ActiveProviderRequest(BaseModel):
    provider_id: str = Field(description="Provider to activate")
    model: str | None = Field(default=None, description="Override model (optional)")


class ActiveProviderResponse(BaseModel):
    provider_id: str
    model: str | None
    kind: str
    tier: str
    name: str


class ProviderTestResult(BaseModel):
    provider_id: str
    model_id: str
    success: bool
    latency_s: float
    error: str | None
    sample_response: dict[str, Any] | None
    tokens_used: int
    estimated_cost_usd: float | None


class ProviderConfigRequest(BaseModel):
    api_key: str = Field(description="API key to configure for this provider")
    model: str | None = Field(default=None, description="Default model override")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_image(upload: UploadFile) -> np.ndarray:
    import cv2
    content = upload.file.read()
    arr = np.frombuffer(content, np.uint8)
    image_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise HTTPException(status_code=422, detail="Cannot decode uploaded image")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _get_active_provider_id() -> str:
    # os.environ is updated immediately on write; get_settings() is lru-cached and
    # may return stale data until cache_clear() is called after a .env write.
    import os
    pid = os.environ.get("ACTIVE_VLM_PROVIDER")
    if pid:
        return pid
    try:
        from backend.config import get_settings
        return get_settings().active_vlm_provider
    except Exception:
        return "moondream"


def _get_active_model() -> str | None:
    import os
    m = os.environ.get("ACTIVE_VLM_MODEL")
    if m:
        return m or None
    try:
        from backend.config import get_settings
        m = get_settings().active_vlm_model
        return m or None
    except Exception:
        return None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("", response_model=list[ProviderStatus])
async def list_providers() -> list[ProviderStatus]:
    """
    List all VLM providers with configuration and availability status.

    Shows which providers are ready to use (API key present or local GPU available),
    their cost/rate tiers, free tier details, and signup URLs.
    """
    from vlm_labeling.provider_registry import get_registry
    registry = get_registry()
    active_id = _get_active_provider_id()

    return [
        ProviderStatus(**p, is_active=(p["provider_id"] == active_id))
        for p in registry.configured_providers()
    ]


@router.get("/active", response_model=ActiveProviderResponse)
async def get_active_provider() -> ActiveProviderResponse:
    """Return the currently active VLM provider."""
    from vlm_labeling.provider_registry import get_registry
    registry = get_registry()
    pid = _get_active_provider_id()
    model = _get_active_model()
    info = registry.get_info(pid)
    if not info:
        raise HTTPException(status_code=404, detail=f"Provider '{pid}' not found")
    return ActiveProviderResponse(
        provider_id=pid,
        model=model,
        kind=info.kind.value,
        tier=info.tier.value,
        name=info.name,
    )


@router.post("/active", response_model=ActiveProviderResponse)
async def set_active_provider(req: ActiveProviderRequest) -> ActiveProviderResponse:
    """
    Switch the active VLM provider.

    Changes take effect immediately for new inference requests.
    API key must already be configured (via .env or /providers/{id}/configure).
    """
    from vlm_labeling.provider_registry import get_registry
    import os

    registry = get_registry()
    info = registry.get_info(req.provider_id)
    if not info:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown provider '{req.provider_id}'",
        )

    # Validate that the provider is actually usable
    configured = {p["provider_id"]: p for p in registry.configured_providers()}
    status = configured.get(req.provider_id, {})
    if not status.get("available", False):
        env_var = status.get("env_var", "")
        raise HTTPException(
            status_code=400,
            detail=(
                f"Provider '{req.provider_id}' is not available. "
                + (f"Set {env_var} in your .env file." if env_var else "GPU not available.")
            ),
        )

    # Update in-process env
    os.environ["ACTIVE_VLM_PROVIDER"] = req.provider_id
    if req.model:
        os.environ["ACTIVE_VLM_MODEL"] = req.model
    elif "ACTIVE_VLM_MODEL" in os.environ:
        del os.environ["ACTIVE_VLM_MODEL"]

    # Persist to .env so the setting survives server restarts, then bust the
    # lru_cache so the next get_settings() re-reads the updated .env.
    from backend.utils.env_file import write_env_keys
    from backend.config import get_settings
    to_write: dict[str, str] = {"ACTIVE_VLM_PROVIDER": req.provider_id}
    to_write["ACTIVE_VLM_MODEL"] = req.model or ""
    write_env_keys(to_write)
    get_settings.cache_clear()

    logger.info(
        "Active VLM provider switched and persisted to .env",
        provider=req.provider_id,
        model=req.model,
    )

    return ActiveProviderResponse(
        provider_id=req.provider_id,
        model=req.model,
        kind=info.kind.value,
        tier=info.tier.value,
        name=info.name,
    )


@router.post("/test", response_model=ProviderTestResult)
async def test_provider(
    file: UploadFile = File(..., description="Test image (trichome microscopy)"),
    provider_id: str = Form(..., description="Provider to test"),
    model: str | None = Form(default=None, description="Model override"),
) -> ProviderTestResult:
    """
    Test a VLM provider with a real image.

    Runs a maturity classification and returns the raw result,
    latency, and estimated cost. Use this to compare providers
    before switching or to verify a new API key works.
    """
    from vlm_labeling.provider_registry import get_registry

    image = _load_image(file)
    registry = get_registry()

    try:
        provider = registry.get(provider_id, model=model)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        result = provider.label_maturity(image)
    except Exception as e:
        return ProviderTestResult(
            provider_id=provider_id,
            model_id=model or provider.info.default_model,
            success=False,
            latency_s=0.0,
            error=str(e),
            sample_response=None,
            tokens_used=0,
            estimated_cost_usd=None,
        )

    total_tokens = result.input_tokens + result.output_tokens
    cost = provider.estimate_cost(result.input_tokens, result.output_tokens)

    return ProviderTestResult(
        provider_id=provider_id,
        model_id=result.model_id or provider.info.default_model,
        success=result.is_valid,
        latency_s=round(result.latency_s, 3),
        error=result.error,
        sample_response=result.parsed_response,
        tokens_used=total_tokens,
        estimated_cost_usd=round(cost, 6) if cost is not None else None,
    )


@router.get("/{provider_id}/models")
async def list_provider_models(provider_id: str) -> dict[str, Any]:
    """List available models for a specific provider."""
    from vlm_labeling.provider_registry import get_registry
    registry = get_registry()
    info = registry.get_info(provider_id)
    if not info:
        raise HTTPException(status_code=404, detail=f"Unknown provider: {provider_id}")
    return {
        "provider_id": provider_id,
        "models": info.models,
        "default": info.default_model,
    }


@router.get("/prompts", response_model=list[dict])
async def list_vlm_prompts() -> list[dict]:
    """
    Return the catalogue of available VLM prompt presets.

    These presets control what the VLM is asked to classify / count.
    The frontend uses this list to populate the Prompt Preset selector in the
    VLM Configuration panel on the Annotation page.
    """
    return [
        {
            "name": "maturity_classification",
            "label": "Maturity Classification (default)",
            "description": "Classify trichomes as clear, cloudy or amber.",
            "is_default": True,
        },
        {
            "name": "morphology_classification",
            "label": "Morphology Classification",
            "description": "Classify trichomes as bulbous, sessile or stalked.",
            "is_default": False,
        },
        {
            "name": "trichome_detection_count",
            "label": "Trichome Detection Count",
            "description": "Count total trichomes visible in the image.",
            "is_default": False,
        },
        {
            "name": "custom",
            "label": "Custom…",
            "description": "Provide your own system and user prompt.",
            "is_default": False,
        },
    ]


@router.post("/{provider_id}/configure")
async def configure_provider(
    provider_id: str,
    req: ProviderConfigRequest,
) -> dict[str, str]:
    """
    Set API key for a provider at runtime (session only — not persisted to .env).

    For permanent configuration, add the key to your .env file instead.
    The appropriate env var name is shown in GET /vlm/providers.
    """
    from vlm_labeling.provider_registry import get_registry, _ENV_KEY_MAP
    import os

    registry = get_registry()
    if not registry.get_info(provider_id):
        raise HTTPException(status_code=404, detail=f"Unknown provider: {provider_id}")

    env_var = _ENV_KEY_MAP.get(provider_id, "")
    if not env_var:
        raise HTTPException(
            status_code=400,
            detail=f"Provider '{provider_id}' is local — no API key required.",
        )

    # Set in process env
    os.environ[env_var] = req.api_key
    if req.model:
        os.environ["ACTIVE_VLM_MODEL"] = req.model

    # Persist to .env so the key survives server restarts, then bust the cache.
    from backend.utils.env_file import write_env_keys
    from backend.config import get_settings
    to_write: dict[str, str] = {env_var: req.api_key}
    if req.model:
        to_write["ACTIVE_VLM_MODEL"] = req.model
    write_env_keys(to_write)
    get_settings.cache_clear()

    logger.info("Provider API key configured and persisted to .env", provider=provider_id)

    return {
        "status": "configured",
        "provider_id": provider_id,
        "env_var": env_var,
        "note": (
            f"{env_var} saved to .env — key persists across server restarts."
        ),
    }
