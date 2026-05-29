"""
tests/unit/test_vlm_providers.py — VLM provider system unit tests.

Covers:
  VLMResponse         — construction, defaults, is_valid flag
  VLMProviderInfo     — required fields, optional fields
  ProviderCapabilities — defaults
  image_to_base64     — valid JPEG output
  image_to_pil        — RGB array → PIL conversion
  OpenAIProvider      — constructor validation, _call mock, label_maturity/assess_quality/label_morphology
  AnthropicProvider   — constructor validation, markdown-fence stripping in _call
  GoogleProvider      — constructor validation, response wrapping
  TogetherProvider    — constructor validation, JSON extraction fallback
  GroqProvider        — constructor validation, JSON extraction fallback
  HuggingFaceProvider — constructor validation, is_available with no token
  ProviderRegistry    — list_providers, configured_providers, get(), get_active_provider fallback
  RemoteComputeRegistry — list_backends, get_compute_backend, unknown-backend error
"""

from __future__ import annotations

import json
import time
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _rgb_image(h: int = 64, w: int = 64) -> np.ndarray:
    """Create a deterministic RGB test image."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


# ──────────────────────────────────────────────────────────────────────────────
# Base models
# ──────────────────────────────────────────────────────────────────────────────

class TestVLMResponse:
    def test_defaults(self):
        from vlm_labeling.providers.base import VLMResponse
        r = VLMResponse(
            raw_response="{}",
            parsed_response={},
            is_valid=True,
            provider_id="test",
            model_id="m",
        )
        assert r.confidence == 0.0
        assert r.error is None
        assert r.input_tokens == 0
        assert r.output_tokens == 0
        assert r.latency_s >= 0.0

    def test_invalid_response(self):
        from vlm_labeling.providers.base import VLMResponse
        r = VLMResponse(
            raw_response="",
            parsed_response=None,
            is_valid=False,
            provider_id="test",
            model_id="m",
            error="api error",
        )
        assert not r.is_valid
        assert r.error == "api error"
        assert r.parsed_response is None

    def test_confidence_stored(self):
        from vlm_labeling.providers.base import VLMResponse
        r = VLMResponse(
            raw_response='{"confidence": 0.85}',
            parsed_response={"confidence": 0.85},
            is_valid=True,
            provider_id="test",
            model_id="m",
            confidence=0.85,
        )
        assert r.confidence == pytest.approx(0.85)


class TestVLMProviderInfo:
    def test_construction(self):
        from vlm_labeling.providers.base import (
            VLMProviderInfo, ProviderKind, ProviderTier
        )
        info = VLMProviderInfo(
            provider_id="test",
            name="Test Provider",
            kind=ProviderKind.REMOTE,
            tier=ProviderTier.FREE,
            models=["model-a"],
            default_model="model-a",
        )
        assert info.provider_id == "test"
        assert info.vram_gb is None
        assert info.rate_limit_rpm is None
        assert info.cost_per_1k_tokens is None

    def test_optional_fields(self):
        from vlm_labeling.providers.base import (
            VLMProviderInfo, ProviderKind, ProviderTier
        )
        info = VLMProviderInfo(
            provider_id="x",
            name="X",
            kind=ProviderKind.LOCAL,
            tier=ProviderTier.PAID,
            models=[],
            default_model="",
            vram_gb=4.0,
            rate_limit_rpm=60,
            cost_per_1k_tokens=0.001,
            free_tier_note="none",
            signup_url="https://example.com",
        )
        assert info.vram_gb == pytest.approx(4.0)
        assert info.rate_limit_rpm == 60


class TestProviderCapabilities:
    def test_defaults(self):
        from vlm_labeling.providers.base import ProviderCapabilities
        caps = ProviderCapabilities()
        # Core labeling capabilities default to True
        assert caps.maturity_labeling
        assert caps.quality_screening
        assert caps.morphology_classification
        # Batch and streaming default to False
        assert not caps.batch_processing
        assert not caps.streaming

    def test_explicit_true(self):
        from vlm_labeling.providers.base import ProviderCapabilities
        caps = ProviderCapabilities(maturity_labeling=True, streaming=True)
        assert caps.maturity_labeling
        assert caps.streaming
        assert not caps.batch_processing


# ──────────────────────────────────────────────────────────────────────────────
# image utilities
# ──────────────────────────────────────────────────────────────────────────────

class TestImageUtils:
    def test_image_to_base64_is_string(self):
        from vlm_labeling.providers.base import image_to_base64
        img = _rgb_image()
        b64 = image_to_base64(img)
        assert isinstance(b64, str)
        assert len(b64) > 0

    def test_image_to_base64_valid_jpeg(self):
        """Decoded base64 must parse as JPEG."""
        import base64
        import io
        from PIL import Image
        from vlm_labeling.providers.base import image_to_base64
        img = _rgb_image()
        b64 = image_to_base64(img)
        raw = base64.b64decode(b64)
        pil = Image.open(io.BytesIO(raw))
        assert pil.format == "JPEG"

    def test_image_to_pil_rgb(self):
        from vlm_labeling.providers.base import image_to_pil
        img = _rgb_image(32, 32)
        pil = image_to_pil(img)
        assert pil.mode == "RGB"
        assert pil.size == (32, 32)


# ──────────────────────────────────────────────────────────────────────────────
# Provider constructors
# ──────────────────────────────────────────────────────────────────────────────

class TestOpenAIProvider:
    def test_requires_api_key(self):
        from vlm_labeling.providers.remote.openai_provider import OpenAIProvider
        with pytest.raises(ValueError, match="API key"):
            OpenAIProvider(api_key="")

    def test_is_available_with_key(self):
        from vlm_labeling.providers.remote.openai_provider import OpenAIProvider
        p = OpenAIProvider(api_key="sk-test")
        assert p.is_available

    def test_not_available_empty_key(self):
        from vlm_labeling.providers.remote.openai_provider import OpenAIProvider
        # Use object.__setattr__ to bypass __init__ validation
        p = object.__new__(OpenAIProvider)
        p._api_key = ""
        p._model = "gpt-4o-mini"
        p._client = None
        assert not p.is_available

    def test_info_provider_id(self):
        from vlm_labeling.providers.remote.openai_provider import OpenAIProvider
        p = OpenAIProvider(api_key="sk-test")
        assert p.info.provider_id == "openai"

    def test_capabilities_maturity(self):
        from vlm_labeling.providers.remote.openai_provider import OpenAIProvider
        p = OpenAIProvider(api_key="sk-test")
        assert p.capabilities.maturity_labeling

    def test_label_maturity_api_failure_returns_invalid(self):
        """When the API call fails, label_maturity must return is_valid=False."""
        from vlm_labeling.providers.remote.openai_provider import OpenAIProvider
        p = OpenAIProvider(api_key="sk-test")
        # Patch the internal _client_ to raise
        with patch.object(p, "_client_", side_effect=Exception("connection refused")):
            resp = p.label_maturity(_rgb_image())
        assert not resp.is_valid
        assert resp.error is not None

    def test_label_maturity_parses_valid_json(self):
        from vlm_labeling.providers.remote.openai_provider import OpenAIProvider
        p = OpenAIProvider(api_key="sk-test")
        fake_json = json.dumps({
            "maturity_stage": "cloudy",
            "confidence": 0.9,
            "amber_fraction_estimate": 0.1,
            "cloudy_fraction_estimate": 0.8,
            "clear_fraction_estimate": 0.1,
            "observations": "mostly cloudy trichomes",
        })
        # Mock the completion object
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 100
        mock_usage.completion_tokens = 50
        mock_choice = MagicMock()
        mock_choice.message.content = fake_json
        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]
        mock_completion.usage = mock_usage
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_completion

        with patch.object(p, "_client_", return_value=mock_client):
            resp = p.label_maturity(_rgb_image())

        assert resp.is_valid
        assert resp.parsed_response["maturity_stage"] == "cloudy"
        assert resp.confidence == pytest.approx(0.9)

    def test_assess_quality_malformed_json_invalid(self):
        from vlm_labeling.providers.remote.openai_provider import OpenAIProvider
        p = OpenAIProvider(api_key="sk-test")
        mock_choice = MagicMock()
        mock_choice.message.content = "not json at all {broken"
        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]
        mock_completion.usage = MagicMock()
        mock_completion.usage.prompt_tokens = 0
        mock_completion.usage.completion_tokens = 0
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_completion

        with patch.object(p, "_client_", return_value=mock_client):
            resp = p.assess_quality(_rgb_image())

        assert not resp.is_valid
        assert resp.parsed_response is None


class TestAnthropicProvider:
    def test_requires_api_key(self):
        from vlm_labeling.providers.remote.anthropic_provider import AnthropicProvider
        with pytest.raises(ValueError):
            AnthropicProvider(api_key="")

    def test_markdown_fence_stripping(self):
        """Anthropic sometimes wraps JSON in ```json ... ``` — verify we strip it."""
        from vlm_labeling.providers.remote.anthropic_provider import AnthropicProvider
        p = AnthropicProvider(api_key="sk-ant-test")
        raw = '```json\n{"maturity_stage": "amber", "confidence": 0.7}\n```'
        mock_content = MagicMock()
        mock_content.text = raw
        mock_message = MagicMock()
        mock_message.content = [mock_content]
        mock_message.usage.input_tokens = 50
        mock_message.usage.output_tokens = 20
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_message

        with patch.object(p, "_client_", return_value=mock_client):
            resp = p.label_maturity(_rgb_image())

        assert resp.is_valid
        assert resp.parsed_response["maturity_stage"] == "amber"

    def test_api_error_propagates(self):
        from vlm_labeling.providers.remote.anthropic_provider import AnthropicProvider
        p = AnthropicProvider(api_key="sk-ant-test")
        with patch.object(p, "_client_", side_effect=Exception("rate limited")):
            resp = p.label_maturity(_rgb_image())
        assert not resp.is_valid
        assert "rate limited" in resp.error


class TestGoogleProvider:
    def test_requires_api_key(self):
        from vlm_labeling.providers.remote.google_provider import GoogleProvider
        with pytest.raises(ValueError):
            GoogleProvider(api_key="")

    def test_info_free_tier(self):
        from vlm_labeling.providers.remote.google_provider import GoogleProvider
        p = GoogleProvider(api_key="AIza-test")
        assert "free" in p.info.free_tier_note.lower()
        assert p.info.rate_limit_rpm == 15

    def test_json_parse_success(self):
        from vlm_labeling.providers.remote.google_provider import GoogleProvider
        p = GoogleProvider(api_key="AIza-test")
        payload = {"maturity_stage": "clear", "confidence": 0.95}
        mock_resp = MagicMock()
        mock_resp.text = json.dumps(payload)
        mock_model = MagicMock()
        mock_model.generate_content.return_value = mock_resp

        with patch.object(p, "_model_", return_value=mock_model):
            resp = p.label_maturity(_rgb_image())

        assert resp.is_valid
        assert resp.parsed_response["maturity_stage"] == "clear"

    def test_strips_markdown_fences(self):
        from vlm_labeling.providers.remote.google_provider import GoogleProvider
        p = GoogleProvider(api_key="AIza-test")
        payload = {"maturity_stage": "mixed", "confidence": 0.6}
        mock_resp = MagicMock()
        mock_resp.text = f"```json\n{json.dumps(payload)}\n```"
        mock_model = MagicMock()
        mock_model.generate_content.return_value = mock_resp

        with patch.object(p, "_model_", return_value=mock_model):
            resp = p.assess_quality(_rgb_image())

        assert resp.is_valid


class TestTogetherProvider:
    def test_requires_api_key(self):
        from vlm_labeling.providers.remote.together_provider import TogetherProvider
        with pytest.raises(ValueError):
            TogetherProvider(api_key="")

    def test_json_extraction_fallback(self):
        """Together models sometimes prefix JSON with prose — regex fallback must work."""
        from vlm_labeling.providers.remote.together_provider import TogetherProvider
        p = TogetherProvider(api_key="ta-test")
        payload = {"dominant_type": "capitate_stalked", "confidence": 0.88}
        prose_wrap = f'Sure! Here is the result: {json.dumps(payload)} That is my analysis.'
        mock_choice = MagicMock()
        mock_choice.message.content = prose_wrap
        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]
        mock_completion.usage.prompt_tokens = 0
        mock_completion.usage.completion_tokens = 0
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_completion

        with patch.object(p, "_client_", return_value=mock_client):
            resp = p.label_morphology(_rgb_image())

        assert resp.is_valid
        assert resp.parsed_response["dominant_type"] == "capitate_stalked"


class TestGroqProvider:
    def test_requires_api_key(self):
        from vlm_labeling.providers.remote.groq_provider import GroqProvider
        with pytest.raises(ValueError):
            GroqProvider(api_key="")

    def test_free_tier_note(self):
        from vlm_labeling.providers.remote.groq_provider import GroqProvider
        p = GroqProvider(api_key="gsk-test")
        assert "free" in p.info.free_tier_note.lower()
        assert p.info.cost_per_1k_tokens is None  # free tier — no cost

    def test_json_extraction_fallback(self):
        """Groq regex fallback: extracts innermost {...} from prose."""
        from vlm_labeling.providers.remote.groq_provider import GroqProvider
        p = GroqProvider(api_key="gsk-test")
        payload = {"maturity_stage": "amber", "confidence": 0.75}
        mock_choice = MagicMock()
        mock_choice.message.content = f"Result: {json.dumps(payload)}"
        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]
        mock_completion.usage.prompt_tokens = 0
        mock_completion.usage.completion_tokens = 0
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_completion

        with patch.object(p, "_client_", return_value=mock_client):
            resp = p.label_maturity(_rgb_image())

        assert resp.is_valid
        assert resp.parsed_response["confidence"] == pytest.approx(0.75)

    def test_api_error_returns_invalid(self):
        from vlm_labeling.providers.remote.groq_provider import GroqProvider
        p = GroqProvider(api_key="gsk-test")
        with patch.object(p, "_client_", side_effect=Exception("503")):
            resp = p.assess_quality(_rgb_image())
        assert not resp.is_valid


class TestHuggingFaceProvider:
    def test_no_key_still_instantiates(self):
        """HF provider allows empty token (anonymous tier, rate-limited)."""
        from vlm_labeling.providers.remote.hf_provider import HuggingFaceProvider
        p = HuggingFaceProvider(api_key="")
        # Anonymous access is supported — provider is always available
        assert p.is_available is True

    def test_info_free_tier(self):
        from vlm_labeling.providers.remote.hf_provider import HuggingFaceProvider
        p = HuggingFaceProvider(api_key="hf-test")
        assert p.info.provider_id == "huggingface"
        assert "free" in p.info.free_tier_note.lower()


# ──────────────────────────────────────────────────────────────────────────────
# ProviderRegistry
# ──────────────────────────────────────────────────────────────────────────────

class TestProviderRegistry:
    def test_list_providers_returns_infos(self):
        """list_providers() returns VLMProviderInfo objects (not dicts)."""
        from vlm_labeling.provider_registry import ProviderRegistry
        from vlm_labeling.providers.base import VLMProviderInfo
        reg = ProviderRegistry()
        providers = reg.list_providers()
        assert isinstance(providers, list)
        assert len(providers) > 0
        for p in providers:
            assert isinstance(p, VLMProviderInfo)
            assert p.provider_id
            assert p.name

    def test_known_provider_ids(self):
        from vlm_labeling.provider_registry import ProviderRegistry
        reg = ProviderRegistry()
        ids = {p.provider_id for p in reg.list_providers()}
        # These must always be listed regardless of API key availability
        assert "groq" in ids
        assert "google" in ids
        assert "openai" in ids
        assert "anthropic" in ids
        assert "together" in ids
        assert "huggingface" in ids
        assert "moondream" in ids

    def test_get_info_returns_correct_info(self):
        from vlm_labeling.provider_registry import ProviderRegistry
        from vlm_labeling.providers.base import VLMProviderInfo
        reg = ProviderRegistry()
        info = reg.get_info("groq")
        assert isinstance(info, VLMProviderInfo)
        assert info.provider_id == "groq"

    def test_get_info_returns_none_for_unknown(self):
        from vlm_labeling.provider_registry import ProviderRegistry
        reg = ProviderRegistry()
        assert reg.get_info("nonexistent_xyz") is None

    def test_configured_providers_remote_no_api_keys(self, monkeypatch):
        """With no env vars set, remote providers have available=False."""
        from vlm_labeling.provider_registry import ProviderRegistry
        for k in [
            "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY",
            "GROQ_API_KEY", "TOGETHER_API_KEY", "HUGGINGFACE_API_KEY",
        ]:
            monkeypatch.delenv(k, raising=False)

        reg = ProviderRegistry()
        configured = reg.configured_providers()
        # configured_providers returns ALL providers with availability flags
        assert isinstance(configured, list)
        assert len(configured) > 0
        # Remote providers without API keys must show available=False
        remote_without_keys = [
            p for p in configured
            if p["provider_id"] in {"openai", "anthropic", "google", "groq", "together"}
        ]
        for p in remote_without_keys:
            assert not p["available"], f"{p['provider_id']} should be unavailable without key"
            assert not p["has_api_key"]

    def test_configured_providers_with_key(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "gsk-testkey")
        from vlm_labeling.provider_registry import ProviderRegistry
        reg = ProviderRegistry()
        configured = reg.configured_providers()
        groq_entry = next((p for p in configured if p["provider_id"] == "groq"), None)
        assert groq_entry is not None
        assert groq_entry["available"]
        assert groq_entry["has_api_key"]

    def test_get_raises_for_unknown(self):
        from vlm_labeling.provider_registry import ProviderRegistry
        reg = ProviderRegistry()
        with pytest.raises((ValueError, KeyError)):
            reg.get("nonexistent_provider_xyz")

    def test_get_groq_with_key(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "gsk-testkey")
        from vlm_labeling.provider_registry import ProviderRegistry
        reg = ProviderRegistry()
        provider = reg.get("groq")
        assert provider is not None
        assert provider.is_available

    def test_get_registry_singleton(self):
        from vlm_labeling.provider_registry import get_registry
        r1 = get_registry()
        r2 = get_registry()
        assert r1 is r2

    def test_get_active_provider_module_function(self, monkeypatch):
        """get_active_provider() is a module-level function (not a method)."""
        for k in ["GROQ_API_KEY", "GOOGLE_API_KEY"]:
            monkeypatch.delenv(k, raising=False)
        monkeypatch.setenv("ACTIVE_VLM_PROVIDER", "moondream")

        from vlm_labeling.provider_registry import get_active_provider
        # Should attempt moondream and raise (local GPU not available in test env)
        # rather than crashing with AttributeError
        try:
            _ = get_active_provider()
        except (RuntimeError, ImportError, Exception) as e:
            # Acceptable in CI/no-GPU environment
            err = str(e).lower()
            assert any(word in err for word in [
                "moondream", "import", "not found", "not configured",
                "no vlm provider", "local", "vram", "model",
            ]), f"Unexpected error: {e}"


# ──────────────────────────────────────────────────────────────────────────────
# Remote compute registry
# ──────────────────────────────────────────────────────────────────────────────

class TestRemoteComputeRegistry:
    def test_list_backends_returns_list(self):
        from services.remote_compute.registry import list_backends
        backends = list_backends()
        assert isinstance(backends, list)
        assert len(backends) >= 2

    def test_backend_ids(self):
        from services.remote_compute.registry import list_backends
        ids = {b["backend_id"] for b in list_backends()}
        assert "modal" in ids
        assert "replicate" in ids

    def test_backend_structure(self):
        from services.remote_compute.registry import list_backends
        for b in list_backends():
            assert "backend_id" in b
            assert "name" in b
            assert "available" in b
            assert "required_env_vars" in b
            assert "gpu_tiers" in b
            assert isinstance(b["gpu_tiers"], list)

    def test_get_modal_backend(self):
        from services.remote_compute.registry import get_compute_backend
        from services.remote_compute.modal_backend import ModalBackend
        b = get_compute_backend("modal")
        assert isinstance(b, ModalBackend)

    def test_get_replicate_backend(self):
        from services.remote_compute.registry import get_compute_backend
        from services.remote_compute.replicate_backend import ReplicateBackend
        b = get_compute_backend("replicate")
        assert isinstance(b, ReplicateBackend)

    def test_get_unknown_raises(self):
        from services.remote_compute.registry import get_compute_backend
        with pytest.raises(ValueError, match="Unknown"):
            get_compute_backend("aws_sagemaker")

    def test_modal_unavailable_without_keys(self, monkeypatch):
        monkeypatch.delenv("MODAL_TOKEN_ID", raising=False)
        monkeypatch.delenv("MODAL_TOKEN_SECRET", raising=False)
        from services.remote_compute.registry import get_compute_backend
        b = get_compute_backend("modal")
        assert not b.is_available

    def test_replicate_unavailable_without_key(self, monkeypatch):
        monkeypatch.delenv("REPLICATE_API_KEY", raising=False)
        from services.remote_compute.registry import get_compute_backend
        b = get_compute_backend("replicate")
        assert not b.is_available

    def test_modal_available_with_keys(self, monkeypatch):
        monkeypatch.setenv("MODAL_TOKEN_ID", "tok-id")
        monkeypatch.setenv("MODAL_TOKEN_SECRET", "tok-secret")
        from services.remote_compute.registry import get_compute_backend
        b = get_compute_backend("modal")
        assert b.is_available

    def test_replicate_available_with_key(self, monkeypatch):
        monkeypatch.setenv("REPLICATE_API_KEY", "r8_test")
        from services.remote_compute.registry import get_compute_backend
        b = get_compute_backend("replicate")
        assert b.is_available

    def test_modal_run_training_not_implemented(self, monkeypatch):
        """Training submission is a TODO — should return failure result, not raise."""
        import asyncio
        monkeypatch.setenv("MODAL_TOKEN_ID", "tok-id")
        monkeypatch.setenv("MODAL_TOKEN_SECRET", "tok-secret")
        from services.remote_compute.registry import get_compute_backend
        b = get_compute_backend("modal")

        result = asyncio.get_event_loop().run_until_complete(
            b.run_training_job(config={}, dataset_path="/tmp/fake")
        )
        assert not result.success
        assert result.error is not None

    def test_replicate_training_not_supported(self, monkeypatch):
        """Replicate backend explicitly rejects custom training."""
        import asyncio
        monkeypatch.setenv("REPLICATE_API_KEY", "r8_test")
        from services.remote_compute.registry import get_compute_backend
        b = get_compute_backend("replicate")

        result = asyncio.get_event_loop().run_until_complete(
            b.run_training_job(config={}, dataset_path="/tmp/fake")
        )
        assert not result.success
        assert "not support" in result.error.lower()

    def test_replicate_vlm_fails_gracefully_without_key(self, monkeypatch):
        """VLM inference on unconfigured Replicate returns failure, doesn't raise."""
        import asyncio
        monkeypatch.delenv("REPLICATE_API_KEY", raising=False)
        from services.remote_compute.registry import get_compute_backend
        b = get_compute_backend("replicate")

        result = asyncio.get_event_loop().run_until_complete(
            b.run_vlm_inference(_rgb_image(), "describe this image")
        )
        assert not result.success
        assert result.error is not None
