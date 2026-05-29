"""
vlm_labeling.providers.remote.together_provider — Together AI provider.

Together AI hosts open-source vision models via API (OpenAI-compatible endpoint).
Notable vision models: Llama-3.2-Vision-11B/90B, Qwen2-VL-72B, InternVL

Free tier: $1 free credit on signup — enough for ~3,000 images with Llama-3.2-Vision-11B
Pricing: Llama-3.2-Vision-11B ~$0.18/1M tokens (cheapest large VLM)
Signup: https://api.together.xyz/settings/api-keys
Docs:   https://docs.together.ai/docs/vision
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
from shared.logging.logger import get_logger

logger = get_logger(__name__)

_INFO = VLMProviderInfo(
    provider_id="together",
    name="Together AI",
    kind=ProviderKind.REMOTE,
    tier=ProviderTier.FREEMIUM,
    models=[
        "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
        "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
        "Qwen/Qwen2-VL-72B-Instruct",
    ],
    default_model="meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
    vram_gb=None,
    cost_per_1k_tokens=0.00018,   # Llama-3.2-11B-Vision
    free_tier_note=(
        "$1 free credit on signup (~3,000 images at Llama-3.2-11B rates). "
        "Cheapest option after Groq for production volume."
    ),
    signup_url="https://api.together.xyz/settings/api-keys",
)

_CAPABILITIES = ProviderCapabilities(
    maturity_labeling=True,
    quality_screening=True,
    morphology_classification=True,
    batch_processing=False,
    streaming=True,
)

class TogetherProvider(VLMProvider):
    """
    Together AI vision provider using OpenAI-compatible SDK interface.

    Together AI is the most cost-effective option for high-volume production use.
    Models: Llama-3.2-Vision, Qwen2-VL, InternVL (all open-source, all cheap).
    """

    def __init__(
        self,
        api_key: str,
        model: str = "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
    ) -> None:
        if not api_key:
            raise ValueError("Together AI API key is required")
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

    def _client_(self) -> Any:
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=self._api_key,
                    base_url="https://api.together.xyz/v1",
                )
            except ImportError:
                raise ImportError(
                    "openai package required for Together AI provider. "
                    "Run: uv pip install openai"
                )
        return self._client

    def _call(self, prompt: str, image: NDArray[np.uint8], system: str = "") -> VLMResponse:
        t0 = time.perf_counter()
        b64 = image_to_base64(image)

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                },
                {"type": "text", "text": prompt},
            ],
        })

        try:
            client = self._client_()
            completion = client.chat.completions.create(
                model=self._model,
                messages=messages,
                max_tokens=512,
                temperature=0.1,
            )
            raw = completion.choices[0].message.content or ""
            input_tokens = completion.usage.prompt_tokens if completion.usage else 0
            output_tokens = completion.usage.completion_tokens if completion.usage else 0
        except Exception as e:
            logger.error("Together AI call failed", error=str(e), model=self._model)
            return VLMResponse(
                raw_response="",
                parsed_response=None,
                is_valid=False,
                provider_id="together",
                model_id=self._model,
                latency_s=time.perf_counter() - t0,
                error=str(e),
            )

        latency = time.perf_counter() - t0
        clean = raw.strip()
        # Strip markdown code blocks if present
        if clean.startswith("```"):
            lines = clean.split("\n")
            clean = "\n".join(lines[1:]).rstrip("```").strip()

        try:
            parsed = json.loads(clean)
        except json.JSONDecodeError:
            # Try to extract first JSON object
            import re
            match = re.search(r"\{.*\}", clean, re.DOTALL)
            parsed = json.loads(match.group()) if match else None

        return VLMResponse(
            raw_response=raw,
            parsed_response=parsed,
            is_valid=parsed is not None,
            confidence=float(parsed.get("confidence", 0.0)) if parsed else 0.0,
            provider_id="together",
            model_id=self._model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_s=latency,
        )

    def label_maturity(self, image: NDArray[np.uint8]) -> VLMResponse:
        system = (
            "You are a cannabis trichome microscopy expert. "
            "Analyze visual morphology only — never claim cannabinoid content. "
            "Return ONLY valid JSON, no markdown."
        )
        prompt = (
            "Analyze this trichome image maturity. Return JSON only: "
            "{\"maturity_stage\": \"clear|cloudy|amber|mixed\", \"confidence\": 0.0, "
            "\"amber_fraction_estimate\": 0.0, \"cloudy_fraction_estimate\": 0.0, "
            "\"clear_fraction_estimate\": 0.0, \"observations\": \"\"}"
        )
        return self._call(prompt, image, system)

    def assess_quality(self, image: NDArray[np.uint8]) -> VLMResponse:
        prompt = (
            "Assess microscopy image quality. Return JSON only: "
            "{\"overall_quality\": \"excellent|good|poor|unusable\", "
            "\"focus_quality\": \"\", \"lighting_quality\": \"\", "
            "\"analyzable\": true, \"reject_reason\": null, \"confidence\": 0.0}"
        )
        return self._call(prompt, image)

    def label_morphology(self, image: NDArray[np.uint8]) -> VLMResponse:
        prompt = (
            "Classify trichome morphology. Return JSON only: "
            "{\"dominant_type\": \"capitate_stalked|capitate_sessile|bulbous\", "
            "\"confidence\": 0.0, \"stalk_visible\": true, "
            "\"head_shape\": \"\", \"mixed_types_present\": false}"
        )
        return self._call(prompt, image)
