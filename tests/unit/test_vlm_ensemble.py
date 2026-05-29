"""
tests/unit/test_vlm_ensemble.py — VLM Ensemble API unit tests.

Coverage:
  - _compute_consensus: majority vote, all-agree, all-disagree, ties, single provider
  - Agreement score calculation (all formulae)
  - Agreement level thresholds: high (>=0.8), medium (>=0.6), low (<0.6)
  - _agreement_level helper
  - _extract_label_from_response across all task types
  - _task_label_key mapping
  - Provider error handling: missing API key → skip, not HTTP 500
  - Empty providers list → 422
  - Unknown task type → 422
  - Invalid base64 → 422
  - Ensemble endpoint: full round-trip with mocked providers
  - GET /vlm/prompts — lists all templates
  - GET /vlm/prompts/{name} — returns specific template
  - GET /vlm/prompts/{unknown} → 404
  - POST /vlm/prompts/validate — valid prompt accepted
  - POST /vlm/prompts/validate — empty system → invalid
  - POST /vlm/prompts/validate — empty user → invalid
  - POST /vlm/prompts/validate — forbidden THC claim → invalid
  - POST /vlm/prompts/validate — bad output_schema → invalid
  - POST /vlm/prompts/validate — good schema accepted
  - Dissenting providers computed correctly
  - Consensus when all providers error
  - Ensemble with mix of successful and errored providers
"""

from __future__ import annotations

import asyncio
import base64
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _rgb_image(h: int = 64, w: int = 64) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def _encode_image_b64(img: np.ndarray) -> str:
    """Encode a numpy RGB image to base64 JPEG string."""
    import cv2
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _make_provider_result(
    provider_id: str,
    maturity_stage: str | None = "cloudy",
    confidence: float = 0.9,
    error: str | None = None,
    dominant_type: str | None = None,
    overall_quality: str | None = None,
) -> "ProviderResult":
    from backend.api.v1.vlm_ensemble import ProviderResult

    return ProviderResult(
        provider_id=provider_id,
        model="test-model",
        maturity_stage=maturity_stage,
        dominant_type=dominant_type,
        overall_quality=overall_quality,
        confidence=confidence,
        elapsed_ms=100.0,
        error=error,
    )


# ──────────────────────────────────────────────────────────────────────────────
# FastAPI test client fixture
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client() -> TestClient:
    """Minimal FastAPI app with just the vlm_ensemble router."""
    from fastapi import FastAPI
    from backend.api.v1.vlm_ensemble import router

    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    return TestClient(app)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Unit tests: _compute_consensus
# ══════════════════════════════════════════════════════════════════════════════

class TestComputeConsensus:
    """Consensus logic — pure function tests, no I/O."""

    def _consensus(self, results, task="maturity"):
        from backend.api.v1.vlm_ensemble import _compute_consensus
        return _compute_consensus(results, task)

    # ── All providers agree ────────────────────────────────────────────────
    def test_all_agree_score_is_1(self):
        results = [
            _make_provider_result("groq", maturity_stage="cloudy"),
            _make_provider_result("together", maturity_stage="cloudy"),
            _make_provider_result("openai", maturity_stage="cloudy"),
        ]
        c = self._consensus(results)
        assert c.maturity_stage == "cloudy"
        assert c.agreement_score == 1.0
        assert c.agreement_level == "high"
        assert c.dissenting_providers == []

    # ── Majority vote (2/3) ────────────────────────────────────────────────
    def test_majority_vote_two_thirds(self):
        results = [
            _make_provider_result("groq", maturity_stage="cloudy"),
            _make_provider_result("together", maturity_stage="cloudy"),
            _make_provider_result("openai", maturity_stage="amber"),
        ]
        c = self._consensus(results)
        assert c.maturity_stage == "cloudy"
        assert c.agreement_score == pytest.approx(2 / 3, rel=1e-4)
        assert c.agreement_level == "medium"
        assert "openai" in c.dissenting_providers

    # ── Tie-breaking is alphabetically first ──────────────────────────────
    def test_tie_breaking_alphabetical(self):
        results = [
            _make_provider_result("groq", maturity_stage="amber"),
            _make_provider_result("together", maturity_stage="cloudy"),
        ]
        c = self._consensus(results)
        # "amber" < "cloudy" alphabetically
        assert c.maturity_stage == "amber"
        assert c.agreement_score == 0.5

    # ── All disagree (3 different labels) ─────────────────────────────────
    def test_all_disagree_score_one_third(self):
        results = [
            _make_provider_result("groq", maturity_stage="clear"),
            _make_provider_result("together", maturity_stage="cloudy"),
            _make_provider_result("openai", maturity_stage="amber"),
        ]
        c = self._consensus(results)
        assert c.agreement_score == pytest.approx(1 / 3, rel=1e-4)
        assert c.agreement_level == "low"

    # ── Single provider ────────────────────────────────────────────────────
    def test_single_provider_score_is_1(self):
        results = [_make_provider_result("groq", maturity_stage="amber")]
        c = self._consensus(results)
        assert c.maturity_stage == "amber"
        assert c.agreement_score == 1.0
        assert c.agreement_level == "high"

    # ── All providers errored ──────────────────────────────────────────────
    def test_all_providers_errored(self):
        results = [
            _make_provider_result("groq", maturity_stage=None, error="API key not configured"),
            _make_provider_result("together", maturity_stage=None, error="timeout"),
        ]
        c = self._consensus(results)
        assert c.agreement_score == 0.0
        assert c.agreement_level == "low"
        assert c.participating_providers == []

    # ── Dissenting providers correctly identified ──────────────────────────
    def test_dissenting_providers_correct(self):
        results = [
            _make_provider_result("groq", maturity_stage="cloudy"),
            _make_provider_result("together", maturity_stage="cloudy"),
            _make_provider_result("openai", maturity_stage="amber"),
            _make_provider_result("anthropic", maturity_stage="amber"),
            _make_provider_result("google", maturity_stage="cloudy"),
        ]
        c = self._consensus(results)
        assert c.maturity_stage == "cloudy"
        assert sorted(c.dissenting_providers) == ["anthropic", "openai"]
        assert sorted(c.participating_providers) == sorted(
            ["groq", "together", "openai", "anthropic", "google"]
        )

    # ── Mix of errored and successful providers ────────────────────────────
    def test_mixed_error_and_success(self):
        results = [
            _make_provider_result("groq", maturity_stage="cloudy"),
            _make_provider_result("together", maturity_stage=None, error="API key not configured"),
            _make_provider_result("openai", maturity_stage="cloudy"),
        ]
        c = self._consensus(results)
        assert c.maturity_stage == "cloudy"
        # Only 2 providers participated (together errored)
        assert c.agreement_score == 1.0
        assert sorted(c.participating_providers) == ["groq", "openai"]

    # ── Morphology task ────────────────────────────────────────────────────
    def test_morphology_task_consensus(self):
        results = [
            _make_provider_result("groq", maturity_stage=None, dominant_type="capitate_stalked"),
            _make_provider_result("together", maturity_stage=None, dominant_type="capitate_stalked"),
            _make_provider_result("openai", maturity_stage=None, dominant_type="bulbous"),
        ]
        c = self._consensus(results, task="morphology")
        assert c.dominant_type == "capitate_stalked"
        assert c.agreement_score == pytest.approx(2 / 3, rel=1e-4)

    # ── Quality screen task ────────────────────────────────────────────────
    def test_quality_screen_task_consensus(self):
        results = [
            _make_provider_result("groq", maturity_stage=None, overall_quality="good"),
            _make_provider_result("together", maturity_stage=None, overall_quality="poor"),
            _make_provider_result("openai", maturity_stage=None, overall_quality="good"),
        ]
        c = self._consensus(results, task="quality_screen")
        assert c.overall_quality == "good"
        assert c.agreement_score == pytest.approx(2 / 3, rel=1e-4)


# ══════════════════════════════════════════════════════════════════════════════
# 2. Unit tests: Agreement level thresholds
# ══════════════════════════════════════════════════════════════════════════════

class TestAgreementLevel:
    def _level(self, score: float) -> str:
        from backend.api.v1.vlm_ensemble import _agreement_level
        return _agreement_level(score)

    def test_high_at_exact_0_8(self):
        assert self._level(0.8) == "high"

    def test_high_above_0_8(self):
        assert self._level(1.0) == "high"
        assert self._level(0.9) == "high"

    def test_medium_at_exact_0_6(self):
        assert self._level(0.6) == "medium"

    def test_medium_between_0_6_and_0_8(self):
        assert self._level(0.7) == "medium"
        assert self._level(0.75) == "medium"
        assert self._level(0.799) == "medium"

    def test_low_below_0_6(self):
        assert self._level(0.5) == "low"
        assert self._level(0.0) == "low"
        assert self._level(0.333) == "low"


# ══════════════════════════════════════════════════════════════════════════════
# 3. Unit tests: label extraction and task key mapping
# ══════════════════════════════════════════════════════════════════════════════

class TestHelpers:
    def test_task_label_key_maturity(self):
        from backend.api.v1.vlm_ensemble import _task_label_key
        assert _task_label_key("maturity") == "maturity_stage"

    def test_task_label_key_morphology(self):
        from backend.api.v1.vlm_ensemble import _task_label_key
        assert _task_label_key("morphology") == "dominant_type"

    def test_task_label_key_quality_screen(self):
        from backend.api.v1.vlm_ensemble import _task_label_key
        assert _task_label_key("quality_screen") == "overall_quality"

    def test_extract_label_maturity(self):
        from backend.api.v1.vlm_ensemble import _extract_label_from_response
        parsed = {"maturity_stage": "cloudy", "confidence": 0.9}
        assert _extract_label_from_response(parsed, "maturity") == "cloudy"

    def test_extract_label_morphology(self):
        from backend.api.v1.vlm_ensemble import _extract_label_from_response
        parsed = {"dominant_type": "capitate_stalked", "confidence": 0.8}
        assert _extract_label_from_response(parsed, "morphology") == "capitate_stalked"

    def test_extract_label_quality_screen(self):
        from backend.api.v1.vlm_ensemble import _extract_label_from_response
        parsed = {"overall_quality": "good", "confidence": 0.7}
        assert _extract_label_from_response(parsed, "quality_screen") == "good"

    def test_extract_label_none_parsed(self):
        from backend.api.v1.vlm_ensemble import _extract_label_from_response
        assert _extract_label_from_response(None, "maturity") is None

    def test_extract_label_missing_key(self):
        from backend.api.v1.vlm_ensemble import _extract_label_from_response
        assert _extract_label_from_response({"confidence": 0.5}, "maturity") is None


# ══════════════════════════════════════════════════════════════════════════════
# 4. HTTP endpoint tests
# ══════════════════════════════════════════════════════════════════════════════

class TestEnsembleLabelEndpoint:
    """Tests for POST /api/v1/vlm/ensemble/label."""

    def _b64_image(self) -> str:
        return _encode_image_b64(_rgb_image())

    def test_empty_providers_returns_422(self, client: TestClient):
        payload = {
            "image_base64": self._b64_image(),
            "providers": [],
            "task": "maturity",
        }
        resp = client.post("/api/v1/vlm/ensemble/label", json=payload)
        assert resp.status_code == 422

    def test_unknown_task_returns_422(self, client: TestClient):
        payload = {
            "image_base64": self._b64_image(),
            "providers": ["groq"],
            "task": "predict_thc",  # invalid
        }
        resp = client.post("/api/v1/vlm/ensemble/label", json=payload)
        assert resp.status_code == 422

    def test_invalid_base64_returns_422(self, client: TestClient):
        payload = {
            "image_base64": "NOT_VALID_BASE64!!!",
            "providers": ["groq"],
            "task": "maturity",
        }
        resp = client.post("/api/v1/vlm/ensemble/label", json=payload)
        assert resp.status_code == 422

    def test_missing_api_key_provider_skipped_not_500(self, client: TestClient):
        """
        Providers whose API key is not set should return error in results,
        not cause HTTP 500.
        """
        from vlm_labeling.providers.base import VLMResponse

        mock_response = VLMResponse(
            raw_response='{"maturity_stage": "cloudy", "confidence": 0.9}',
            parsed_response={"maturity_stage": "cloudy", "confidence": 0.9},
            is_valid=True,
            confidence=0.9,
            provider_id="groq",
            model_id="llama-3.2-11b-vision-preview",
            latency_s=0.4,
        )

        def mock_get(provider_id, model=None):
            if provider_id == "together":
                raise ValueError("Together AI API key is required")
            m = MagicMock()
            m.info = MagicMock()
            m.info.default_model = "llama-3.2-11b-vision-preview"
            m._model = "llama-3.2-11b-vision-preview"
            m.label_maturity = MagicMock(return_value=mock_response)
            return m

        with patch("backend.api.v1.vlm_ensemble.get_registry") as mock_reg_fn:
            registry = MagicMock()
            registry.get = mock_get
            mock_reg_fn.return_value = registry

            payload = {
                "image_base64": self._b64_image(),
                "providers": ["groq", "together"],
                "task": "maturity",
            }
            resp = client.post("/api/v1/vlm/ensemble/label", json=payload)

        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data
        results_by_id = {r["provider_id"]: r for r in data["results"]}
        assert "together" in results_by_id
        assert results_by_id["together"]["error"] == "API key not configured"
        assert "groq" in results_by_id
        assert results_by_id["groq"]["error"] is None

    def test_successful_ensemble_response_structure(self, client: TestClient):
        """Full ensemble with mocked providers — check response structure."""
        from vlm_labeling.providers.base import VLMResponse

        def _make_vlm_resp(provider_id: str, stage: str, conf: float) -> VLMResponse:
            return VLMResponse(
                raw_response=json.dumps({"maturity_stage": stage, "confidence": conf}),
                parsed_response={"maturity_stage": stage, "confidence": conf},
                is_valid=True,
                confidence=conf,
                provider_id=provider_id,
                model_id="test-model",
                latency_s=0.3,
            )

        def mock_get(provider_id, model=None):
            m = MagicMock()
            m.info = MagicMock()
            m.info.default_model = "test-model"
            m._model = "test-model"
            if provider_id == "groq":
                m.label_maturity = MagicMock(
                    return_value=_make_vlm_resp("groq", "cloudy", 0.9)
                )
            elif provider_id == "together":
                m.label_maturity = MagicMock(
                    return_value=_make_vlm_resp("together", "cloudy", 0.85)
                )
            elif provider_id == "openai":
                m.label_maturity = MagicMock(
                    return_value=_make_vlm_resp("openai", "amber", 0.7)
                )
            return m

        with patch("backend.api.v1.vlm_ensemble.get_registry") as mock_reg_fn:
            registry = MagicMock()
            registry.get = mock_get
            mock_reg_fn.return_value = registry

            payload = {
                "image_base64": self._b64_image(),
                "providers": ["groq", "together", "openai"],
                "task": "maturity",
            }
            resp = client.post("/api/v1/vlm/ensemble/label", json=payload)

        assert resp.status_code == 200
        data = resp.json()

        # Check top-level structure
        assert "results" in data
        assert "consensus" in data
        assert "scientific_caveat" in data
        assert len(data["results"]) == 3

        # Check consensus
        consensus = data["consensus"]
        assert consensus["maturity_stage"] == "cloudy"
        assert pytest.approx(consensus["agreement_score"], rel=1e-3) == 2 / 3
        assert consensus["agreement_level"] == "medium"
        assert "openai" in consensus["dissenting_providers"]

        # Check scientific caveat is present and non-empty
        assert len(data["scientific_caveat"]) > 50

    def test_model_overrides_passed_to_provider(self, client: TestClient):
        """model_overrides should be forwarded to registry.get() as the model kwarg."""
        from vlm_labeling.providers.base import VLMResponse

        captured_model: dict[str, str | None] = {}

        def mock_get(provider_id, model=None):
            captured_model[provider_id] = model
            m = MagicMock()
            m.info = MagicMock()
            m.info.default_model = model or "default-model"
            m._model = model or "default-model"
            m.label_maturity = MagicMock(
                return_value=VLMResponse(
                    raw_response='{"maturity_stage": "clear", "confidence": 0.8}',
                    parsed_response={"maturity_stage": "clear", "confidence": 0.8},
                    is_valid=True,
                    confidence=0.8,
                    provider_id=provider_id,
                    model_id=model or "default-model",
                    latency_s=0.2,
                )
            )
            return m

        with patch("backend.api.v1.vlm_ensemble.get_registry") as mock_reg_fn:
            registry = MagicMock()
            registry.get = mock_get
            mock_reg_fn.return_value = registry

            payload = {
                "image_base64": self._b64_image(),
                "providers": ["together"],
                "task": "maturity",
                "model_overrides": {"together": "Qwen/Qwen2-VL-72B-Instruct"},
            }
            client.post("/api/v1/vlm/ensemble/label", json=payload)

        assert captured_model.get("together") == "Qwen/Qwen2-VL-72B-Instruct"


# ══════════════════════════════════════════════════════════════════════════════
# 5. Prompt template endpoints
# ══════════════════════════════════════════════════════════════════════════════

class TestPromptEndpoints:
    def test_list_prompts_returns_all(self, client: TestClient):
        resp = client.get("/api/v1/vlm/prompts")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) >= 4  # maturity, morphology, quality, count at minimum
        names = [p["name"] for p in data]
        assert "maturity_classification" in names
        assert "morphology_classification" in names
        assert "trichome_count" in names
        assert "image_quality" in names

    def test_list_prompts_structure(self, client: TestClient):
        resp = client.get("/api/v1/vlm/prompts")
        assert resp.status_code == 200
        for tmpl in resp.json():
            assert "name" in tmpl
            assert "system_prompt" in tmpl
            assert "user_prompt_template" in tmpl
            assert "output_schema" in tmpl
            assert "model_compatibility" in tmpl
            assert isinstance(tmpl["model_compatibility"], list)

    def test_get_specific_prompt_maturity(self, client: TestClient):
        resp = client.get("/api/v1/vlm/prompts/maturity_classification")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "maturity_classification"
        assert "maturity_stage" in data["user_prompt_template"]
        assert "required" in data["output_schema"]

    def test_get_specific_prompt_morphology(self, client: TestClient):
        resp = client.get("/api/v1/vlm/prompts/morphology_classification")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "morphology_classification"

    def test_get_unknown_prompt_returns_404(self, client: TestClient):
        resp = client.get("/api/v1/vlm/prompts/does_not_exist_xyz")
        assert resp.status_code == 404
        detail = resp.json()["detail"]
        assert "does_not_exist_xyz" in detail

    def test_get_prompt_404_lists_available(self, client: TestClient):
        resp = client.get("/api/v1/vlm/prompts/nope")
        assert resp.status_code == 404
        detail = resp.json()["detail"]
        # Should list available prompts in the error message
        assert "Available" in detail or "available" in detail


# ══════════════════════════════════════════════════════════════════════════════
# 6. Prompt validation endpoint
# ══════════════════════════════════════════════════════════════════════════════

class TestPromptValidation:
    def _post(self, client: TestClient, payload: dict) -> dict:
        resp = client.post("/api/v1/vlm/prompts/validate", json=payload)
        assert resp.status_code == 200
        return resp.json()

    def test_valid_prompt_accepted(self, client: TestClient):
        result = self._post(client, {
            "system": "You are a trichome expert.",
            "user": "Classify the maturity stage of the trichomes visible.",
        })
        assert result["valid"] is True
        assert result["errors"] == []

    def test_empty_system_is_invalid(self, client: TestClient):
        result = self._post(client, {
            "system": "",
            "user": "Classify the trichomes.",
        })
        assert result["valid"] is False
        assert any("system" in e for e in result["errors"])

    def test_empty_user_is_invalid(self, client: TestClient):
        result = self._post(client, {
            "system": "You are an expert.",
            "user": "   ",  # whitespace only
        })
        assert result["valid"] is False
        assert any("user" in e for e in result["errors"])

    def test_forbidden_thc_claim_rejected(self, client: TestClient):
        result = self._post(client, {
            "system": "Predict THC content from this image.",
            "user": "What is the THC content percentage?",
        })
        assert result["valid"] is False
        assert any("THC" in e for e in result["errors"])

    def test_forbidden_potency_claim_rejected(self, client: TestClient):
        result = self._post(client, {
            "system": "You are an expert.",
            "user": "Estimate the potency of this plant.",
        })
        assert result["valid"] is False
        assert any("potency" in e.lower() for e in result["errors"])

    def test_valid_schema_accepted(self, client: TestClient):
        result = self._post(client, {
            "system": "You are an expert.",
            "user": "Classify maturity.",
            "output_schema": {
                "required": ["maturity_stage"],
                "properties": {
                    "maturity_stage": {"type": "string"},
                },
            },
        })
        assert result["valid"] is True
        assert result["errors"] == []

    def test_bad_schema_missing_keys_invalid(self, client: TestClient):
        result = self._post(client, {
            "system": "You are an expert.",
            "user": "Classify.",
            "output_schema": {"description": "some schema without required/properties/type"},
        })
        assert result["valid"] is False
        assert len(result["errors"]) >= 1
