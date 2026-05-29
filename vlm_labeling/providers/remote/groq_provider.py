"""
vlm_labeling.providers.remote.groq_provider — Groq vision provider.

Groq offers ultra-fast inference via LPU hardware. The free tier is the
most generous of all major providers for development and testing.

Free tier: YES — llama-3.2-11b-vision-preview: 7000 tokens/min, 14400 RPD
           No credit card required for free tier.
Models: llama-3.2-11b-vision-preview, llama-3.2-90b-vision-preview
Signup: https://console.groq.com/keys
Docs:   https://console.groq.com/docs/vision

NOTE: Groq free tier has low token/min limits — use for testing, not production.
For production, use Together AI (cheaper) or Google (more generous free tier).
"""

from __future__ import annotations

import json
import re
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
    provider_id="groq",
    name="Groq (Ultra-Fast)",
    kind=ProviderKind.REMOTE,
    tier=ProviderTier.FREEMIUM,
    models=[
        "llama-3.2-11b-vision-preview",
        "llama-3.2-90b-vision-preview",
    ],
    default_model="llama-3.2-11b-vision-preview",
    vram_gb=None,
    cost_per_1k_tokens=None,   # Free tier
    rate_limit_rpm=30,          # Free tier: ~30 RPM effective (token limited)
    free_tier_note=(
        "Free: 7000 tokens/min, 14400 requests/day. No credit card needed. "
        "Best for development/testing. Very fast (<1s latency)."
    ),
    signup_url="https://console.groq.com/keys",
)

_CAPABILITIES = ProviderCapabilities(
    maturity_labeling=True,
    quality_screening=True,
    morphology_classification=True,
    batch_processing=False,
    streaming=True,
)

class GroqProvider(VLMProvider):
    """
    Groq vision provider.

    Groq's LPU hardware delivers <1s latency even for 11B parameter models.
    Ideal for interactive development and rapid experimentation.
    Free tier is the most accessible entry point for new users.

    Uses the groq Python SDK which mirrors the OpenAI interface.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "llama-3.2-11b-vision-preview",
    ) -> None:
        if not api_key:
            raise ValueError("Groq API key is required")
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
                from groq import Groq
                self._client = Groq(api_key=self._api_key)
            except ImportError:
                raise ImportError(
                    "groq package not installed. Run: uv pip install groq"
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
            logger.error("Groq API call failed", error=str(e), model=self._model)
            return VLMResponse(
                raw_response="",
                parsed_response=None,
                is_valid=False,
                provider_id="groq",
                model_id=self._model,
                latency_s=time.perf_counter() - t0,
                error=str(e),
            )

        latency = time.perf_counter() - t0

        # Parse JSON from response
        clean = raw.strip()
        if clean.startswith("```"):
            clean = "\n".join(clean.split("\n")[1:]).rstrip("```").strip()
        try:
            parsed = json.loads(clean)
        except json.JSONDecodeError:
            match = re.search(r"\{[^{}]*\}", clean, re.DOTALL)
            parsed = json.loads(match.group()) if match else None

        return VLMResponse(
            raw_response=raw,
            parsed_response=parsed,
            is_valid=parsed is not None,
            confidence=float(parsed.get("confidence", 0.0)) if parsed else 0.0,
            provider_id="groq",
            model_id=self._model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_s=latency,
        )

    def label_maturity(self, image: NDArray[np.uint8]) -> VLMResponse:
        system = (
            "You are a cannabis trichome expert. Analyze morphology only — "
            "no cannabinoid content claims. Return ONLY valid JSON."
        )
        prompt = (
            "Classify trichome maturity. Return ONLY this JSON structure:\n"
            '{"maturity_stage": "clear|cloudy|amber|mixed", "confidence": 0.0, '
            '"amber_fraction_estimate": 0.0, "cloudy_fraction_estimate": 0.0, '
            '"clear_fraction_estimate": 0.0, "observations": ""}'
        )
        return self._call(prompt, image, system)

    def assess_quality(self, image: NDArray[np.uint8]) -> VLMResponse:
        prompt = (
            "Rate this microscopy image quality. Return ONLY this JSON:\n"
            '{"overall_quality": "excellent|good|poor|unusable", '
            '"focus_quality": "", "lighting_quality": "", '
            '"analyzable": true, "reject_reason": null, "confidence": 0.0}'
        )
        return self._call(prompt, image)

    def label_morphology(self, image: NDArray[np.uint8]) -> VLMResponse:
        prompt = (
            "Classify trichome morphology type. Return ONLY this JSON:\n"
            '{"dominant_type": "capitate_stalked|capitate_sessile|bulbous", '
            '"confidence": 0.0, "stalk_visible": true, '
            '"head_shape": "", "mixed_types_present": false}'
        )
        return self._call(prompt, image)
