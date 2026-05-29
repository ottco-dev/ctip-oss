"""
vlm_labeling.providers.remote.openai_provider — OpenAI GPT-4o/GPT-4V provider.

Models: gpt-4o, gpt-4o-mini, gpt-4-turbo
Free tier: None (pay-per-token from first request)
Signup: https://platform.openai.com/api-keys
Cost (2025): gpt-4o $2.50/1M input, $10/1M output
             gpt-4o-mini $0.15/1M input, $0.60/1M output
"""

from __future__ import annotations

import json
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

from vlm_labeling.providers.base import (
    VLMProvider,
    VLMProviderInfo,
    VLMResponse,
    ProviderKind,
    ProviderTier,
    ProviderCapabilities,
    image_to_base64,
)
from vlm_labeling.prompts.trichome_prompts import PROMPT_REGISTRY, PromptTemplate
from shared.logging.logger import get_logger


def _get_user_prompt(key: str, fallback: str) -> str:
    """Safely retrieve user_prompt_template from PROMPT_REGISTRY, falling back gracefully."""
    tmpl = PROMPT_REGISTRY.get(key)
    if isinstance(tmpl, PromptTemplate):
        return tmpl.user_prompt_template
    return fallback

logger = get_logger(__name__)

_INFO = VLMProviderInfo(
    provider_id="openai",
    name="OpenAI GPT-4o",
    kind=ProviderKind.REMOTE,
    tier=ProviderTier.PAID,
    models=["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
    default_model="gpt-4o-mini",
    vram_gb=None,
    cost_per_1k_tokens=0.00015,   # gpt-4o-mini input
    rate_limit_rpm=500,
    free_tier_note="No free tier. Trial credit available on new accounts.",
    signup_url="https://platform.openai.com/api-keys",
)

_CAPABILITIES = ProviderCapabilities(
    maturity_labeling=True,
    quality_screening=True,
    morphology_classification=True,
    batch_processing=True,
    streaming=True,
)

class OpenAIProvider(VLMProvider):
    """
    OpenAI GPT-4o vision provider.

    Sends base64-encoded JPEG images via the Chat Completions API.
    Structured JSON output is requested via response_format=json_object.
    """

    def __init__(self, api_key: str, model: str = "gpt-4o-mini") -> None:
        if not api_key:
            raise ValueError("OpenAI API key is required")
        self._api_key = api_key
        self._model = model
        self._client: Any = None

    @property
    def info(self) -> VLMProviderInfo:
        return _INFO

    @property
    def capabilities(self) -> ProviderCapabilities:
        return _CAPABILITIES

    @property
    def is_available(self) -> bool:
        return bool(self._api_key)

    def _client_(self):  # type: ignore[return]
        """Lazy-init OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self._api_key)
            except ImportError:
                raise ImportError(
                    "openai package not installed. Run: uv pip install openai"
                )
        return self._client

    def _call(
        self,
        prompt: str,
        image: NDArray[np.uint8],
        system: str = "",
    ) -> VLMResponse:
        t0 = time.perf_counter()
        b64 = image_to_base64(image)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{b64}",
                        "detail": "high",
                    },
                },
            ],
        })

        try:
            client = self._client_()
            completion = client.chat.completions.create(
                model=self._model,
                messages=messages,
                response_format={"type": "json_object"},
                max_tokens=512,
                temperature=0.1,
            )
            raw = completion.choices[0].message.content or ""
            input_tokens = completion.usage.prompt_tokens if completion.usage else 0
            output_tokens = completion.usage.completion_tokens if completion.usage else 0
        except Exception as e:
            logger.error("OpenAI API call failed", error=str(e), model=self._model)
            return VLMResponse(
                raw_response="",
                parsed_response=None,
                is_valid=False,
                provider_id="openai",
                model_id=self._model,
                latency_s=time.perf_counter() - t0,
                error=str(e),
            )

        latency = time.perf_counter() - t0
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = None

        return VLMResponse(
            raw_response=raw,
            parsed_response=parsed,
            is_valid=parsed is not None,
            confidence=float(parsed.get("confidence", 0.0)) if parsed else 0.0,
            provider_id="openai",
            model_id=self._model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_s=latency,
        )

    def label_maturity(self, image: NDArray[np.uint8]) -> VLMResponse:
        prompt = _get_user_prompt(
            "maturity_classification",
            "Analyze this trichome image. Return JSON with: maturity_stage, confidence, "
            "amber_fraction_estimate, cloudy_fraction_estimate, clear_fraction_estimate, observations.",
        )
        system = (
            "You are an expert cannabis microscopy analyst. "
            "Analyze trichome maturity based solely on visual morphology. "
            "IMPORTANT: Do NOT make claims about THC, CBD, or any cannabinoid content — "
            "optical maturity only. Return strictly valid JSON."
        )
        return self._call(prompt, image, system)

    def assess_quality(self, image: NDArray[np.uint8]) -> VLMResponse:
        prompt = _get_user_prompt(
            "image_quality",
            "Assess the quality of this microscopy image. Return JSON with: "
            "overall_quality (excellent/good/poor/unusable), focus_quality, "
            "lighting_quality, analyzable (bool), reject_reason, confidence.",
        )
        return self._call(prompt, image)

    def label_morphology(self, image: NDArray[np.uint8]) -> VLMResponse:
        prompt = _get_user_prompt(
            "morphology_classification",
            "Classify trichome morphology. Return JSON with: dominant_type "
            "(capitate_stalked/capitate_sessile/bulbous), confidence, stalk_visible, "
            "head_shape, mixed_types_present.",
        )
        return self._call(prompt, image)
