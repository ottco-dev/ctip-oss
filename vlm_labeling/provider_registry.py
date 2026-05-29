"""
vlm_labeling.provider_registry — Central VLM provider factory and registry.

USAGE:
    from vlm_labeling.provider_registry import ProviderRegistry, get_active_provider

    # List all available providers with their metadata
    registry = ProviderRegistry()
    all_info = registry.list_providers()

    # Get a configured provider (reads API keys from environment/config)
    provider = get_active_provider()  # uses settings.active_vlm_provider

    # Or explicitly:
    provider = registry.get("groq")
    result = provider.label_maturity(image)

PROVIDER IDs:
    Local (no API key, requires GPU):
        "moondream"     — 2.1 GB VRAM, fastest local
        "florence2"     — 8.0 GB VRAM, best detection
        "qwen2vl"       — 5.5 GB VRAM, best quality local

    Remote (API key required, no local GPU):
        "openai"        — GPT-4o-mini, paid, $0.15/1M tokens
        "anthropic"     — Claude 3.5 Haiku, paid, ~$0.80/1M tokens
        "google"        — Gemini 1.5 Flash, FREE tier (15 RPM, 1500 RPD)
        "together"      — Llama-3.2-Vision, freemium, cheapest paid
        "groq"          — Llama-3.2-Vision, FREE tier (fast, limited tokens)
        "huggingface"   — Various models, FREE tier (cold starts)

RECOMMENDED BY USE CASE:
    Development/testing  → groq (free, fast)
    Production free      → google (generous free tier, reliable)
    Production paid      → together (cheapest) or openai (best quality)
    Privacy/air-gap      → moondream or qwen2vl (local)
"""

from __future__ import annotations

import os
from typing import Any

from vlm_labeling.providers.base import VLMProvider, VLMProviderInfo


class ProviderRegistry:
    """
    Factory and metadata registry for all VLM providers.

    Reads API keys from environment variables (set in .env):
        OPENAI_API_KEY
        ANTHROPIC_API_KEY
        GOOGLE_API_KEY
        TOGETHER_API_KEY
        GROQ_API_KEY
        HUGGINGFACE_API_KEY
    """

    # ── Provider metadata (always available, no keys needed) ──────────
    _INFOS: dict[str, Any] = {}

    def __init__(self) -> None:
        self._load_infos()

    def _load_infos(self) -> None:
        """Eagerly load metadata without instantiating providers."""
        from vlm_labeling.providers.remote.openai_provider import _INFO as OAI
        from vlm_labeling.providers.remote.anthropic_provider import _INFO as ANT
        from vlm_labeling.providers.remote.google_provider import _INFO as GGL
        from vlm_labeling.providers.remote.together_provider import _INFO as TOG
        from vlm_labeling.providers.remote.groq_provider import _INFO as GRQ
        from vlm_labeling.providers.remote.hf_provider import _INFO as HF
        from vlm_labeling.providers.local.moondream_provider import _INFO as MD

        self._infos: dict[str, VLMProviderInfo] = {
            "openai":       OAI,
            "anthropic":    ANT,
            "google":       GGL,
            "together":     TOG,
            "groq":         GRQ,
            "huggingface":  HF,
            "moondream":    MD,
        }

        # Conditionally add local providers
        try:
            from vlm_labeling.providers.local.qwen_provider import _INFO as QW
            self._infos["qwen2vl"] = QW
        except ImportError:
            pass
        try:
            from vlm_labeling.providers.local.florence_provider import _INFO as FL
            self._infos["florence2"] = FL
        except ImportError:
            pass

    def list_providers(self) -> list[VLMProviderInfo]:
        """All known providers with static metadata."""
        return list(self._infos.values())

    def get_info(self, provider_id: str) -> VLMProviderInfo | None:
        return self._infos.get(provider_id)

    def configured_providers(self) -> list[dict[str, Any]]:
        """
        Return providers that are usable right now (API key set or local GPU available).

        Returns a list of dicts with info + availability status.
        """
        result = []
        for pid, info in self._infos.items():
            env_key = _ENV_KEY_MAP.get(pid, "")
            api_key = os.getenv(env_key, "") if env_key else ""
            is_local = info.kind.value == "local"
            available = is_local or bool(api_key)
            result.append({
                "provider_id": pid,
                "name": info.name,
                "kind": info.kind.value,
                "tier": info.tier.value,
                "available": available,
                "has_api_key": bool(api_key),
                "env_var": env_key,
                "models": info.models,
                "default_model": info.default_model,
                "vram_gb": info.vram_gb,
                "cost_per_1k_tokens": info.cost_per_1k_tokens,
                "rate_limit_rpm": info.rate_limit_rpm,
                "free_tier_note": info.free_tier_note,
                "signup_url": info.signup_url,
            })
        return result

    def get(
        self,
        provider_id: str,
        model: str | None = None,
        api_key: str | None = None,
    ) -> VLMProvider:
        """
        Instantiate a configured provider.

        Args:
            provider_id: Provider ID (e.g. 'groq', 'openai', 'moondream')
            model: Override the default model for this provider.
            api_key: Override API key (default: reads from environment).

        Returns:
            Configured VLMProvider instance.

        Raises:
            ValueError: If provider_id is unknown or required API key is missing.
        """
        if provider_id not in self._infos:
            raise ValueError(
                f"Unknown provider: '{provider_id}'. "
                f"Available: {list(self._infos.keys())}"
            )

        # Resolve API key
        env_var = _ENV_KEY_MAP.get(provider_id, "")
        resolved_key = api_key or (os.getenv(env_var, "") if env_var else "")

        return _FACTORY[provider_id](resolved_key, model)


# ── Environment variable map ──────────────────────────────────────────────────
_ENV_KEY_MAP: dict[str, str] = {
    "openai":       "OPENAI_API_KEY",
    "anthropic":    "ANTHROPIC_API_KEY",
    "google":       "GOOGLE_API_KEY",
    "together":     "TOGETHER_API_KEY",
    "groq":         "GROQ_API_KEY",
    "huggingface":  "HUGGINGFACE_API_KEY",
    "moondream":    "",   # local, no key
    "qwen2vl":      "",   # local, no key
    "florence2":    "",   # local, no key
}


# ── Provider factory functions ────────────────────────────────────────────────
def _make_openai(key: str, model: str | None) -> VLMProvider:
    from vlm_labeling.providers.remote.openai_provider import OpenAIProvider
    return OpenAIProvider(api_key=key, model=model or "gpt-4o-mini")


def _make_anthropic(key: str, model: str | None) -> VLMProvider:
    from vlm_labeling.providers.remote.anthropic_provider import AnthropicProvider
    return AnthropicProvider(api_key=key, model=model or "claude-3-5-haiku-20241022")


def _make_google(key: str, model: str | None) -> VLMProvider:
    from vlm_labeling.providers.remote.google_provider import GoogleProvider
    return GoogleProvider(api_key=key, model=model or "gemini-1.5-flash")


def _make_together(key: str, model: str | None) -> VLMProvider:
    from vlm_labeling.providers.remote.together_provider import TogetherProvider
    return TogetherProvider(
        api_key=key,
        model=model or "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
    )


def _make_groq(key: str, model: str | None) -> VLMProvider:
    from vlm_labeling.providers.remote.groq_provider import GroqProvider
    return GroqProvider(api_key=key, model=model or "llama-3.2-11b-vision-preview")


def _make_hf(key: str, model: str | None) -> VLMProvider:
    from vlm_labeling.providers.remote.hf_provider import HuggingFaceProvider
    return HuggingFaceProvider(api_key=key, model=model or "vikhyatk/moondream2")


def _make_moondream(_key: str, _model: str | None) -> VLMProvider:
    from vlm_labeling.providers.local.moondream_provider import MoondreamProvider
    return MoondreamProvider()


def _make_qwen(_key: str, _model: str | None) -> VLMProvider:
    from vlm_labeling.providers.local.qwen_provider import QwenProvider
    return QwenProvider()


_FACTORY: dict[str, Any] = {
    "openai":       _make_openai,
    "anthropic":    _make_anthropic,
    "google":       _make_google,
    "together":     _make_together,
    "groq":         _make_groq,
    "huggingface":  _make_hf,
    "moondream":    _make_moondream,
    "qwen2vl":      _make_qwen,
}


# ── Convenience shortcut ──────────────────────────────────────────────────────
_registry: ProviderRegistry | None = None


def get_registry() -> ProviderRegistry:
    global _registry
    if _registry is None:
        _registry = ProviderRegistry()
    return _registry


def get_active_provider() -> VLMProvider:
    """
    Get the currently configured active VLM provider from settings.

    Falls back to the following priority chain if configured provider is unavailable:
        groq → google → moondream
    """
    try:
        from backend.config import get_settings
        settings = get_settings()
        provider_id = getattr(settings, "active_vlm_provider", "moondream")
        model = getattr(settings, "active_vlm_model", None)
    except Exception:
        provider_id = os.getenv("ACTIVE_VLM_PROVIDER", "moondream")
        model = os.getenv("ACTIVE_VLM_MODEL", None)

    registry = get_registry()

    try:
        return registry.get(provider_id, model=model)
    except ValueError:
        pass

    # Fallback chain
    for fallback_id in ["groq", "google", "moondream"]:
        try:
            return registry.get(fallback_id)
        except (ValueError, Exception):
            continue

    raise RuntimeError(
        "No VLM provider available. "
        "Set ACTIVE_VLM_PROVIDER and corresponding API key in .env, "
        "or ensure a local GPU is available for moondream/qwen2vl."
    )
