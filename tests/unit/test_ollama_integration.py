"""
tests/unit/test_ollama_integration.py — Ollama integration unit tests.

Coverage (30+ tests):
  OllamaConfig                — defaults, custom values
  OllamaProvider.is_available — True on 200, False on connection error, False on non-200
  OllamaProvider.list_models  — parse response, empty list on error, empty list on non-200
  OllamaProvider.build_prompt — style variants, language variants, THC safety filter
  OllamaProvider.generate_narrative — correct HTTP call, returns string, handles timeout,
                                       handles non-200, handles bad JSON structure
  OllamaProvider.pull_model   — streaming response parsed correctly, non-200 raises
  API GET /ollama/status      — available / unavailable states
  API GET /ollama/models      — model list returned, empty on error
  API POST /ollama/narrative  — correct response schema, model field populated,
                                validation errors for bad style / language,
                                503 when Ollama unavailable
  API PUT /ollama/config      — settings updated, cache cleared, 400 on empty body
  THC safety                  — all forbidden field variants stripped
  Narrative generation error handling — aiohttp errors, RuntimeError
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_analysis_result(**overrides: Any) -> dict[str, Any]:
    """Minimal valid analysis_result dict."""
    base: dict[str, Any] = {
        "total_detections": 142,
        "type_distribution": {
            "CAPITATE_STALKED": 78,
            "SESSILE": 41,
            "BULBOUS": 12,
            "NON_GLANDULAR": 11,
        },
        "maturity_distribution": {
            "CLEAR": 15.5,
            "CLOUDY": 72.3,
            "AMBER": 12.2,
        },
        "harvest_recommendation": "Optimal optical maturity window (70%+ cloudy trichomes).",
        "confidence_stats": {
            "mean": 0.87,
            "std": 0.06,
            "min": 0.61,
            "max": 0.97,
        },
        "session_id": "test-session-001",
        "timestamp": "2026-05-29T12:00:00Z",
    }
    base.update(overrides)
    return base


def _make_ollama_chat_response(content: str = "Test narrative text.") -> dict[str, Any]:
    """Minimal Ollama /api/chat response."""
    return {
        "model": "llama3.2:3b",
        "message": {"role": "assistant", "content": content},
        "done": True,
    }


def _make_aiohttp_response(
    status: int = 200,
    json_data: Any = None,
    text_data: str = "",
    lines: list[bytes] | None = None,
) -> MagicMock:
    """Build a MagicMock that mimics aiohttp ClientResponse."""
    resp = MagicMock()
    resp.status = status
    resp.json = AsyncMock(return_value=json_data or {})
    resp.text = AsyncMock(return_value=text_data)

    if lines is not None:
        # Async iterator for streaming
        async def _aiter():
            for line in lines:
                yield line

        resp.content = MagicMock()
        resp.content.__aiter__ = lambda self: _aiter()
    else:
        resp.content = None

    resp.__aenter__ = AsyncMock(return_value=resp)
    resp.__aexit__ = AsyncMock(return_value=False)
    return resp


def _make_session(response: MagicMock) -> MagicMock:
    """Build a MagicMock aiohttp.ClientSession that returns response."""
    session = MagicMock()
    session.get = MagicMock(return_value=response)
    session.post = MagicMock(return_value=response)
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    return session


# ---------------------------------------------------------------------------
# OllamaConfig tests
# ---------------------------------------------------------------------------

class TestOllamaConfig:
    def test_defaults(self):
        from vlm_labeling.providers.local.ollama_provider import OllamaConfig

        cfg = OllamaConfig()
        assert cfg.base_url == "http://localhost:11434"
        assert cfg.model == "llama3.2:3b"
        assert cfg.timeout_s == 30.0
        assert cfg.max_tokens == 1024
        assert cfg.temperature == 0.3
        assert cfg.system_prompt == ""

    def test_custom_values(self):
        from vlm_labeling.providers.local.ollama_provider import OllamaConfig

        cfg = OllamaConfig(
            base_url="http://192.168.1.5:11434",
            model="mistral:7b",
            timeout_s=60.0,
            max_tokens=2048,
            temperature=0.1,
            system_prompt="Custom system prompt.",
        )
        assert cfg.base_url == "http://192.168.1.5:11434"
        assert cfg.model == "mistral:7b"
        assert cfg.timeout_s == 60.0
        assert cfg.max_tokens == 2048
        assert cfg.temperature == 0.1
        assert cfg.system_prompt == "Custom system prompt."


# ---------------------------------------------------------------------------
# OllamaProvider.is_available
# ---------------------------------------------------------------------------

class TestOllamaProviderIsAvailable:
    @pytest.mark.asyncio
    async def test_available_on_200(self):
        from vlm_labeling.providers.local.ollama_provider import OllamaProvider

        resp = _make_aiohttp_response(status=200, json_data={"models": []})
        session = _make_session(resp)

        with patch("aiohttp.ClientSession", return_value=session):
            provider = OllamaProvider()
            result = await provider.is_available()

        assert result is True

    @pytest.mark.asyncio
    async def test_unavailable_on_connection_error(self):
        import aiohttp
        from vlm_labeling.providers.local.ollama_provider import OllamaProvider

        session = MagicMock()
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=False)
        session.get = MagicMock(side_effect=aiohttp.ClientConnectorError(
            MagicMock(), OSError("Connection refused")
        ))

        with patch("aiohttp.ClientSession", return_value=session):
            provider = OllamaProvider()
            result = await provider.is_available()

        assert result is False

    @pytest.mark.asyncio
    async def test_unavailable_on_non_200(self):
        from vlm_labeling.providers.local.ollama_provider import OllamaProvider

        resp = _make_aiohttp_response(status=503)
        session = _make_session(resp)

        with patch("aiohttp.ClientSession", return_value=session):
            provider = OllamaProvider()
            result = await provider.is_available()

        assert result is False

    @pytest.mark.asyncio
    async def test_unavailable_on_generic_exception(self):
        from vlm_labeling.providers.local.ollama_provider import OllamaProvider

        session = MagicMock()
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=False)
        session.get = MagicMock(side_effect=Exception("unexpected"))

        with patch("aiohttp.ClientSession", return_value=session):
            provider = OllamaProvider()
            result = await provider.is_available()

        assert result is False


# ---------------------------------------------------------------------------
# OllamaProvider.list_models
# ---------------------------------------------------------------------------

class TestOllamaProviderListModels:
    @pytest.mark.asyncio
    async def test_parses_models_correctly(self):
        from vlm_labeling.providers.local.ollama_provider import OllamaProvider

        tags_data = {
            "models": [
                {"name": "llama3.2:3b", "size": 2_000_000_000},
                {"name": "mistral:7b", "size": 4_000_000_000},
            ]
        }
        resp = _make_aiohttp_response(status=200, json_data=tags_data)
        session = _make_session(resp)

        with patch("aiohttp.ClientSession", return_value=session):
            provider = OllamaProvider()
            models = await provider.list_models()

        assert models == ["llama3.2:3b", "mistral:7b"]

    @pytest.mark.asyncio
    async def test_empty_list_on_connection_error(self):
        import aiohttp
        from vlm_labeling.providers.local.ollama_provider import OllamaProvider

        session = MagicMock()
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=False)
        session.get = MagicMock(side_effect=aiohttp.ClientConnectorError(
            MagicMock(), OSError("Connection refused")
        ))

        with patch("aiohttp.ClientSession", return_value=session):
            provider = OllamaProvider()
            models = await provider.list_models()

        assert models == []

    @pytest.mark.asyncio
    async def test_empty_list_on_non_200(self):
        from vlm_labeling.providers.local.ollama_provider import OllamaProvider

        resp = _make_aiohttp_response(status=404, json_data={})
        session = _make_session(resp)

        with patch("aiohttp.ClientSession", return_value=session):
            provider = OllamaProvider()
            models = await provider.list_models()

        assert models == []

    @pytest.mark.asyncio
    async def test_empty_list_when_no_models_key(self):
        from vlm_labeling.providers.local.ollama_provider import OllamaProvider

        resp = _make_aiohttp_response(status=200, json_data={})
        session = _make_session(resp)

        with patch("aiohttp.ClientSession", return_value=session):
            provider = OllamaProvider()
            models = await provider.list_models()

        assert models == []


# ---------------------------------------------------------------------------
# OllamaProvider.build_prompt
# ---------------------------------------------------------------------------

class TestOllamaProviderBuildPrompt:
    def test_scientific_style_contains_data(self):
        from vlm_labeling.providers.local.ollama_provider import OllamaProvider

        provider = OllamaProvider()
        result = _make_analysis_result()
        prompt = provider.build_prompt(result, style="scientific", language="en")

        assert "142" in prompt                 # total_detections
        assert "CLOUDY" in prompt              # maturity key
        assert "scientific" in prompt.lower()  # style instruction present
        assert "English" in prompt or "English".lower() in prompt.lower()

    def test_summary_style_instruction_differs(self):
        from vlm_labeling.providers.local.ollama_provider import OllamaProvider

        provider = OllamaProvider()
        result = _make_analysis_result()
        sci_prompt = provider.build_prompt(result, style="scientific", language="en")
        sum_prompt = provider.build_prompt(result, style="summary", language="en")

        # Style instructions differ
        assert sci_prompt != sum_prompt
        assert "brief" in sum_prompt.lower() or "plain" in sum_prompt.lower()

    def test_technical_style_instruction_present(self):
        from vlm_labeling.providers.local.ollama_provider import OllamaProvider

        provider = OllamaProvider()
        result = _make_analysis_result()
        prompt = provider.build_prompt(result, style="technical", language="en")

        assert "technical" in prompt.lower() or "QA" in prompt or "detailed" in prompt.lower()

    def test_language_en(self):
        from vlm_labeling.providers.local.ollama_provider import OllamaProvider

        provider = OllamaProvider()
        prompt = provider.build_prompt(_make_analysis_result(), style="scientific", language="en")
        assert "English" in prompt

    def test_language_de(self):
        from vlm_labeling.providers.local.ollama_provider import OllamaProvider

        provider = OllamaProvider()
        prompt = provider.build_prompt(_make_analysis_result(), style="scientific", language="de")
        assert "German" in prompt or "Deutsch" in prompt

    def test_language_es(self):
        from vlm_labeling.providers.local.ollama_provider import OllamaProvider

        provider = OllamaProvider()
        prompt = provider.build_prompt(_make_analysis_result(), style="scientific", language="es")
        assert "Spanish" in prompt or "Español" in prompt

    def test_thc_percentage_stripped(self):
        from vlm_labeling.providers.local.ollama_provider import OllamaProvider

        provider = OllamaProvider()
        result = _make_analysis_result(thc_percentage=27.5)
        prompt = provider.build_prompt(result, style="scientific", language="en")

        assert '"thc_percentage"' not in prompt
        assert "27.5" not in prompt

    def test_cannabinoid_content_stripped(self):
        from vlm_labeling.providers.local.ollama_provider import OllamaProvider

        provider = OllamaProvider()
        result = _make_analysis_result(cannabinoid_content="high", potency=0.28)
        prompt = provider.build_prompt(result, style="scientific", language="en")

        # The field *keys* must not appear in the JSON data block embedded in the prompt.
        # (Static instruction text may mention the word "potency" as a constraint warning —
        #  what matters is that the field key and its value are not emitted as JSON.)
        assert '"cannabinoid_content"' not in prompt
        assert '"potency"' not in prompt
        assert '"high"' not in prompt  # value of cannabinoid_content stripped
        assert "0.28" not in prompt    # value of potency stripped

    def test_multiple_forbidden_fields_stripped(self):
        from vlm_labeling.providers.local.ollama_provider import OllamaProvider

        forbidden = {
            "thc_percentage": 28.0,
            "thc_content": "high",
            "thc_level": 3,
            "thc_concentration": 280.0,
            "cannabinoid_content": "moderate",
            "cannabinoid_percentage": 0.25,
            "potency_estimate": "high potency",
        }
        provider = OllamaProvider()
        result = _make_analysis_result(**forbidden)
        prompt = provider.build_prompt(result, style="scientific", language="en")

        for key in forbidden:
            assert key not in prompt, f"Forbidden key '{key}' found in prompt"

    def test_non_forbidden_fields_preserved(self):
        from vlm_labeling.providers.local.ollama_provider import OllamaProvider

        provider = OllamaProvider()
        result = _make_analysis_result()
        prompt = provider.build_prompt(result, style="scientific", language="en")

        assert "harvest_recommendation" in prompt
        assert "confidence_stats" in prompt
        assert "session_id" in prompt

    def test_unknown_style_falls_back_to_scientific(self):
        from vlm_labeling.providers.local.ollama_provider import OllamaProvider

        provider = OllamaProvider()
        prompt_unknown = provider.build_prompt(
            _make_analysis_result(), style="unknown_style", language="en"
        )
        prompt_sci = provider.build_prompt(
            _make_analysis_result(), style="scientific", language="en"
        )
        # Should use the same style instruction as scientific
        assert prompt_unknown == prompt_sci


# ---------------------------------------------------------------------------
# OllamaProvider.generate_narrative
# ---------------------------------------------------------------------------

class TestOllamaProviderGenerateNarrative:
    @pytest.mark.asyncio
    async def test_returns_narrative_string(self):
        from vlm_labeling.providers.local.ollama_provider import OllamaProvider

        expected_narrative = "The trichome sample exhibits predominantly cloudy morphology."
        resp = _make_aiohttp_response(
            status=200,
            json_data=_make_ollama_chat_response(content=expected_narrative),
        )
        session = _make_session(resp)

        with patch("aiohttp.ClientSession", return_value=session):
            provider = OllamaProvider()
            narrative = await provider.generate_narrative(
                analysis_result=_make_analysis_result(),
                style="scientific",
                language="en",
            )

        assert narrative == expected_narrative

    @pytest.mark.asyncio
    async def test_posts_to_api_chat(self):
        from vlm_labeling.providers.local.ollama_provider import OllamaProvider

        resp = _make_aiohttp_response(
            status=200,
            json_data=_make_ollama_chat_response(),
        )
        session = _make_session(resp)

        with patch("aiohttp.ClientSession", return_value=session):
            provider = OllamaProvider()
            await provider.generate_narrative(
                analysis_result=_make_analysis_result(),
                style="scientific",
                language="en",
            )

        session.post.assert_called_once()
        call_url = session.post.call_args[0][0]
        assert "/api/chat" in call_url

    @pytest.mark.asyncio
    async def test_raises_runtime_error_on_non_200(self):
        from vlm_labeling.providers.local.ollama_provider import OllamaProvider

        resp = _make_aiohttp_response(status=500, text_data="Internal server error")
        session = _make_session(resp)

        with patch("aiohttp.ClientSession", return_value=session):
            provider = OllamaProvider()
            with pytest.raises(RuntimeError, match="HTTP 500"):
                await provider.generate_narrative(
                    analysis_result=_make_analysis_result(),
                    style="scientific",
                    language="en",
                )

    @pytest.mark.asyncio
    async def test_raises_runtime_error_on_bad_response_structure(self):
        from vlm_labeling.providers.local.ollama_provider import OllamaProvider

        # Missing "message" key
        resp = _make_aiohttp_response(status=200, json_data={"done": True})
        session = _make_session(resp)

        with patch("aiohttp.ClientSession", return_value=session):
            provider = OllamaProvider()
            with pytest.raises(RuntimeError, match="Unexpected Ollama response"):
                await provider.generate_narrative(
                    analysis_result=_make_analysis_result(),
                    style="scientific",
                    language="en",
                )

    @pytest.mark.asyncio
    async def test_handles_timeout(self):
        import aiohttp
        from vlm_labeling.providers.local.ollama_provider import OllamaConfig, OllamaProvider

        session = MagicMock()
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=False)
        session.post = MagicMock(side_effect=asyncio.TimeoutError())

        with patch("aiohttp.ClientSession", return_value=session):
            provider = OllamaProvider(config=OllamaConfig(timeout_s=1.0))
            with pytest.raises(asyncio.TimeoutError):
                await provider.generate_narrative(
                    analysis_result=_make_analysis_result(),
                    style="scientific",
                    language="en",
                )

    @pytest.mark.asyncio
    async def test_uses_model_from_config(self):
        from vlm_labeling.providers.local.ollama_provider import OllamaConfig, OllamaProvider

        resp = _make_aiohttp_response(
            status=200,
            json_data=_make_ollama_chat_response(),
        )
        session = _make_session(resp)

        with patch("aiohttp.ClientSession", return_value=session):
            cfg = OllamaConfig(model="mistral:7b")
            provider = OllamaProvider(config=cfg)
            await provider.generate_narrative(
                analysis_result=_make_analysis_result(),
                style="summary",
                language="en",
            )

        call_kwargs = session.post.call_args[1]
        payload = call_kwargs.get("json", {})
        assert payload.get("model") == "mistral:7b"

    @pytest.mark.asyncio
    async def test_uses_custom_system_prompt(self):
        from vlm_labeling.providers.local.ollama_provider import OllamaConfig, OllamaProvider

        custom_system = "You are a very specific assistant."
        resp = _make_aiohttp_response(
            status=200,
            json_data=_make_ollama_chat_response(),
        )
        session = _make_session(resp)

        with patch("aiohttp.ClientSession", return_value=session):
            cfg = OllamaConfig(system_prompt=custom_system)
            provider = OllamaProvider(config=cfg)
            await provider.generate_narrative(
                analysis_result=_make_analysis_result(),
                style="scientific",
                language="en",
            )

        call_kwargs = session.post.call_args[1]
        payload = call_kwargs.get("json", {})
        messages = payload.get("messages", [])
        system_messages = [m for m in messages if m.get("role") == "system"]
        assert len(system_messages) == 1
        assert system_messages[0]["content"] == custom_system


# ---------------------------------------------------------------------------
# OllamaProvider.pull_model
# ---------------------------------------------------------------------------

class TestOllamaProviderPullModel:
    @pytest.mark.asyncio
    async def test_yields_progress_chunks(self):
        from vlm_labeling.providers.local.ollama_provider import OllamaProvider

        lines = [
            json.dumps({"status": "downloading", "completed": 100, "total": 1000}).encode(),
            json.dumps({"status": "downloading", "completed": 500, "total": 1000}).encode(),
            json.dumps({"status": "success"}).encode(),
        ]
        resp = _make_aiohttp_response(status=200, lines=lines)
        session = _make_session(resp)

        with patch("aiohttp.ClientSession", return_value=session):
            provider = OllamaProvider()
            chunks = [c async for c in provider.pull_model("llama3.2:3b")]

        assert len(chunks) == 3
        assert chunks[0]["status"] == "downloading"
        assert chunks[0]["completed"] == 100
        assert chunks[2]["status"] == "success"

    @pytest.mark.asyncio
    async def test_raises_on_non_200(self):
        from vlm_labeling.providers.local.ollama_provider import OllamaProvider

        resp = _make_aiohttp_response(status=404, text_data="model not found")
        session = _make_session(resp)

        with patch("aiohttp.ClientSession", return_value=session):
            provider = OllamaProvider()
            with pytest.raises(RuntimeError, match="HTTP 404"):
                async for _ in provider.pull_model("nonexistent:model"):
                    pass

    @pytest.mark.asyncio
    async def test_skips_empty_lines(self):
        from vlm_labeling.providers.local.ollama_provider import OllamaProvider

        lines = [
            b"",           # empty line should be skipped
            b"  ",         # whitespace-only should be skipped
            json.dumps({"status": "success"}).encode(),
        ]
        resp = _make_aiohttp_response(status=200, lines=lines)
        session = _make_session(resp)

        with patch("aiohttp.ClientSession", return_value=session):
            provider = OllamaProvider()
            chunks = [c async for c in provider.pull_model("llama3.2:3b")]

        assert len(chunks) == 1
        assert chunks[0]["status"] == "success"


# ---------------------------------------------------------------------------
# API tests via TestClient
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    """FastAPI TestClient with the full app."""
    from backend.main import create_app
    app = create_app()
    return TestClient(app, raise_server_exceptions=False)


class TestApiOllamaStatus:
    def test_status_available(self, client):
        with patch(
            "vlm_labeling.providers.local.ollama_provider.OllamaProvider.is_available",
            new_callable=AsyncMock,
            return_value=True,
        ), patch(
            "vlm_labeling.providers.local.ollama_provider.OllamaProvider.list_models",
            new_callable=AsyncMock,
            return_value=["llama3.2:3b", "mistral:7b"],
        ):
            response = client.get("/api/v1/ollama/status")

        assert response.status_code == 200
        data = response.json()
        assert data["available"] is True
        assert "llama3.2:3b" in data["installed_models"]
        assert "base_url" in data
        assert "current_model" in data

    def test_status_unavailable(self, client):
        with patch(
            "vlm_labeling.providers.local.ollama_provider.OllamaProvider.is_available",
            new_callable=AsyncMock,
            return_value=False,
        ):
            response = client.get("/api/v1/ollama/status")

        assert response.status_code == 200
        data = response.json()
        assert data["available"] is False
        assert data["installed_models"] == []

    def test_status_schema_fields_present(self, client):
        with patch(
            "vlm_labeling.providers.local.ollama_provider.OllamaProvider.is_available",
            new_callable=AsyncMock,
            return_value=False,
        ):
            response = client.get("/api/v1/ollama/status")

        data = response.json()
        assert set(data.keys()) >= {"available", "base_url", "installed_models", "current_model"}


class TestApiOllamaModels:
    def test_models_returned(self, client):
        import aiohttp

        tags_data = {
            "models": [
                {"name": "llama3.2:3b", "size": 2_000_000_000, "modified_at": "2025-01-01"},
                {"name": "mistral:7b", "size": 4_100_000_000, "modified_at": "2025-01-02"},
            ]
        }
        resp = _make_aiohttp_response(status=200, json_data=tags_data)
        session = _make_session(resp)

        with patch("aiohttp.ClientSession", return_value=session):
            response = client.get("/api/v1/ollama/models")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        names = [m["name"] for m in data]
        assert "llama3.2:3b" in names

    def test_models_empty_on_ollama_down(self, client):
        import aiohttp

        session = MagicMock()
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=False)
        session.get = MagicMock(side_effect=aiohttp.ClientConnectorError(
            MagicMock(), OSError("refused")
        ))

        with patch("aiohttp.ClientSession", return_value=session):
            response = client.get("/api/v1/ollama/models")

        assert response.status_code == 200
        assert response.json() == []


class TestApiOllamaNarrative:
    def test_narrative_correct_response_schema(self, client):
        with patch(
            "vlm_labeling.providers.local.ollama_provider.OllamaProvider.is_available",
            new_callable=AsyncMock,
            return_value=True,
        ), patch(
            "vlm_labeling.providers.local.ollama_provider.OllamaProvider.generate_narrative",
            new_callable=AsyncMock,
            return_value="Sample scientific narrative text about trichomes.",
        ):
            response = client.post(
                "/api/v1/ollama/narrative",
                json={
                    "analysis_result": _make_analysis_result(),
                    "style": "scientific",
                    "language": "en",
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert "narrative" in data
        assert "model" in data
        assert "style" in data
        assert "language" in data
        assert "generation_time_ms" in data
        assert data["narrative"] == "Sample scientific narrative text about trichomes."
        assert data["style"] == "scientific"
        assert data["language"] == "en"

    def test_narrative_model_field_populated(self, client):
        with patch(
            "vlm_labeling.providers.local.ollama_provider.OllamaProvider.is_available",
            new_callable=AsyncMock,
            return_value=True,
        ), patch(
            "vlm_labeling.providers.local.ollama_provider.OllamaProvider.generate_narrative",
            new_callable=AsyncMock,
            return_value="Narrative.",
        ):
            response = client.post(
                "/api/v1/ollama/narrative",
                json={"analysis_result": _make_analysis_result()},
            )

        assert response.status_code == 200
        data = response.json()
        # model should be the configured default (non-empty string)
        assert isinstance(data["model"], str)
        assert len(data["model"]) > 0

    def test_narrative_invalid_style_returns_422(self, client):
        response = client.post(
            "/api/v1/ollama/narrative",
            json={
                "analysis_result": _make_analysis_result(),
                "style": "haiku",  # invalid
                "language": "en",
            },
        )
        assert response.status_code == 422

    def test_narrative_invalid_language_returns_422(self, client):
        response = client.post(
            "/api/v1/ollama/narrative",
            json={
                "analysis_result": _make_analysis_result(),
                "style": "scientific",
                "language": "klingon",  # invalid
            },
        )
        assert response.status_code == 422

    def test_narrative_503_when_ollama_unavailable(self, client):
        with patch(
            "vlm_labeling.providers.local.ollama_provider.OllamaProvider.is_available",
            new_callable=AsyncMock,
            return_value=False,
        ):
            response = client.post(
                "/api/v1/ollama/narrative",
                json={"analysis_result": _make_analysis_result()},
            )

        assert response.status_code == 503

    def test_narrative_generation_time_non_negative(self, client):
        with patch(
            "vlm_labeling.providers.local.ollama_provider.OllamaProvider.is_available",
            new_callable=AsyncMock,
            return_value=True,
        ), patch(
            "vlm_labeling.providers.local.ollama_provider.OllamaProvider.generate_narrative",
            new_callable=AsyncMock,
            return_value="Narrative text.",
        ):
            response = client.post(
                "/api/v1/ollama/narrative",
                json={"analysis_result": _make_analysis_result()},
            )

        assert response.status_code == 200
        assert response.json()["generation_time_ms"] >= 0.0


class TestApiOllamaConfig:
    def test_update_model(self, client):
        with patch("backend.utils.env_file.write_env_keys") as mock_write:
            response = client.put(
                "/api/v1/ollama/config",
                json={"model": "mistral:7b"},
            )

        assert response.status_code == 200
        data = response.json()
        assert "model" in data
        assert "temperature" in data
        assert "max_tokens" in data
        assert "base_url" in data
        # Env write was called
        mock_write.assert_called_once()

    def test_update_temperature(self, client):
        with patch("backend.utils.env_file.write_env_keys"):
            response = client.put(
                "/api/v1/ollama/config",
                json={"temperature": 0.7},
            )

        assert response.status_code == 200

    def test_update_max_tokens(self, client):
        with patch("backend.utils.env_file.write_env_keys"):
            response = client.put(
                "/api/v1/ollama/config",
                json={"max_tokens": 512},
            )

        assert response.status_code == 200

    def test_update_base_url(self, client):
        with patch("backend.utils.env_file.write_env_keys"):
            response = client.put(
                "/api/v1/ollama/config",
                json={"base_url": "http://192.168.1.10:11434"},
            )

        assert response.status_code == 200

    def test_empty_body_returns_400(self, client):
        response = client.put("/api/v1/ollama/config", json={})
        assert response.status_code == 400

    def test_cache_cleared_after_update(self, client):
        from backend.config import get_settings

        with patch("backend.utils.env_file.write_env_keys"), \
             patch.object(get_settings, "cache_clear") as mock_clear:
            client.put("/api/v1/ollama/config", json={"model": "llama3.2:3b"})

        mock_clear.assert_called_once()


# ---------------------------------------------------------------------------
# THC safety — standalone unit tests (no API)
# ---------------------------------------------------------------------------

class TestTHCSafety:
    """Verify that all forbidden THC/cannabinoid fields are stripped."""

    _ALL_FORBIDDEN = [
        "thc_percentage",
        "thc_content",
        "thc_level",
        "thc_concentration",
        "thc_estimate",
        "cannabinoid_content",
        "cannabinoid_percentage",
        "cannabinoid_concentration",
        "cannabinoid_level",
        "cannabinoid_estimate",
        "potency",
        "potency_estimate",
        "potency_percentage",
        "terpene_content",
        "terpene_percentage",
    ]

    @pytest.mark.parametrize("forbidden_key", _ALL_FORBIDDEN)
    def test_forbidden_key_not_in_prompt(self, forbidden_key: str):
        """
        Verify that the JSON-serialised field key (e.g. '"potency"') does not appear
        in the prompt's data block.  We check for the quoted JSON key form to avoid
        false positives from static instruction text that may contain the word as part
        of a warning sentence (e.g. "Do NOT claim pharmaceutical strength…").
        """
        from vlm_labeling.providers.local.ollama_provider import OllamaProvider

        provider = OllamaProvider()
        result = _make_analysis_result(**{forbidden_key: 99.9})
        prompt = provider.build_prompt(result, style="scientific", language="en")

        # JSON key form: the field must not appear as a quoted JSON property name
        json_key = f'"{forbidden_key}"'
        assert json_key not in prompt, (
            f"Forbidden field key {json_key} found in generated prompt JSON block — "
            "THC safety filter not working."
        )

    def test_forbidden_key_value_not_in_prompt(self):
        """Ensure the value associated with a forbidden key does not leak either."""
        from vlm_labeling.providers.local.ollama_provider import OllamaProvider

        provider = OllamaProvider()
        # Use a very distinctive value unlikely to appear by coincidence
        result = _make_analysis_result(thc_percentage=99.12345)
        prompt = provider.build_prompt(result, style="scientific", language="en")

        assert "99.12345" not in prompt

    def test_safe_result_unmodified_original_dict(self):
        """build_prompt must not mutate the original analysis_result dict."""
        from vlm_labeling.providers.local.ollama_provider import OllamaProvider

        provider = OllamaProvider()
        result = _make_analysis_result(thc_percentage=25.0, potency="high")
        original_keys = set(result.keys())
        provider.build_prompt(result, style="scientific", language="en")

        assert set(result.keys()) == original_keys, (
            "build_prompt mutated the original analysis_result dict."
        )
