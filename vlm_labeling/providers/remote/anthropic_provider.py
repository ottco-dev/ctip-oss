"""
vlm_labeling.providers.remote.anthropic_provider — Anthropic Claude provider.

Models: claude-3-5-sonnet-20241022, claude-3-5-haiku-20241022, claude-3-opus-20240229
Free tier: None (pay-per-token)
Signup: https://console.anthropic.com/account/keys
Cost: claude-3-5-haiku $0.80/1M input, $4/1M output (best cost-per-quality)
"""

from __future__ import annotations

import base64
import json
import time

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
from vlm_labeling.prompts.trichome_prompts import PROMPT_REGISTRY
from shared.logging.logger import get_logger

logger = get_logger(__name__)

_INFO = VLMProviderInfo(
    provider_id="anthropic",
    name="Anthropic Claude",
    kind=ProviderKind.REMOTE,
    tier=ProviderTier.PAID,
    models=[
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022",
        "claude-3-opus-20240229",
    ],
    default_model="claude-3-5-haiku-20241022",
    vram_gb=None,
    cost_per_1k_tokens=0.0008,   # claude-3-5-haiku input $/1k tokens
    free_tier_note="No free tier. $5 trial credit on new accounts.",
    signup_url="https://console.anthropic.com/account/keys",
)

_CAPABILITIES = ProviderCapabilities(
    maturity_labeling=True,
    quality_screening=True,
    morphology_classification=True,
    batch_processing=False,
    streaming=True,
)

class AnthropicProvider(VLMProvider):
    """
    Anthropic Claude vision provider.

    Uses the Messages API with base64-encoded image content blocks.
    Claude is strongly instructed to return valid JSON via system prompt.
    """

    def __init__(self, api_key: str, model: str = "claude-3-5-haiku-20241022") -> None:
        if not api_key:
            raise ValueError("Anthropic API key is required")
        self._api_key = api_key
        self._model = model
        self._client = None

    @property
    def info(self) -> VLMProviderInfo:
        return _INFO

    @property
    def capabilities(self) -> ProviderCapabilities:
        return _CAPABILITIES

    @property
    def is_available(self) -> bool:
        return bool(self._api_key)

    def _client_(self):
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self._api_key)
            except ImportError:
                raise ImportError(
                    "anthropic package not installed. Run: uv pip install anthropic"
                )
        return self._client

    def _call(self, prompt: str, image: NDArray[np.uint8], task: str = "") -> VLMResponse:
        t0 = time.perf_counter()
        b64 = image_to_base64(image)
        system = (
            "You are an expert cannabis trichome microscopy analyst. "
            "Analyze images based on visual morphology only. "
            "Do NOT make claims about THC, CBD, or cannabinoid content — optical maturity only. "
            "Always respond with strictly valid JSON and nothing else."
        )
        try:
            client = self._client_()
            message = client.messages.create(
                model=self._model,
                max_tokens=512,
                system=system,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": b64,
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
            )
            raw = message.content[0].text if message.content else ""
            input_tokens = message.usage.input_tokens if message.usage else 0
            output_tokens = message.usage.output_tokens if message.usage else 0
        except Exception as e:
            logger.error("Anthropic API call failed", error=str(e), model=self._model)
            return VLMResponse(
                raw_response="",
                parsed_response=None,
                is_valid=False,
                provider_id="anthropic",
                model_id=self._model,
                latency_s=time.perf_counter() - t0,
                error=str(e),
            )

        latency = time.perf_counter() - t0
        # Claude sometimes wraps JSON in ```json ... ``` blocks
        clean = raw.strip()
        if clean.startswith("```"):
            clean = "\n".join(clean.split("\n")[1:])
            clean = clean.rsplit("```", 1)[0].strip()

        try:
            parsed = json.loads(clean)
        except json.JSONDecodeError:
            parsed = None

        return VLMResponse(
            raw_response=raw,
            parsed_response=parsed,
            is_valid=parsed is not None,
            confidence=float(parsed.get("confidence", 0.0)) if parsed else 0.0,
            provider_id="anthropic",
            model_id=self._model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_s=latency,
        )

    def label_maturity(self, image: NDArray[np.uint8]) -> VLMResponse:
        prompt = (
            "Analyze this trichome microscopy image. Return ONLY valid JSON with keys: "
            "maturity_stage (clear/cloudy/amber/mixed), confidence (0.0-1.0), "
            "amber_fraction_estimate (0.0-1.0), cloudy_fraction_estimate (0.0-1.0), "
            "clear_fraction_estimate (0.0-1.0), observations (string)."
        )
        return self._call(prompt, image, "maturity")

    def assess_quality(self, image: NDArray[np.uint8]) -> VLMResponse:
        prompt = (
            "Assess the quality of this microscopy image. Return ONLY valid JSON with keys: "
            "overall_quality (excellent/good/poor/unusable), focus_quality (string), "
            "lighting_quality (string), analyzable (boolean), "
            "reject_reason (string or null), confidence (0.0-1.0)."
        )
        return self._call(prompt, image, "quality")

    def label_morphology(self, image: NDArray[np.uint8]) -> VLMResponse:
        prompt = (
            "Classify the trichome morphology in this image. Return ONLY valid JSON with keys: "
            "dominant_type (capitate_stalked/capitate_sessile/bulbous), confidence (0.0-1.0), "
            "stalk_visible (boolean), head_shape (string), mixed_types_present (boolean)."
        )
        return self._call(prompt, image, "morphology")
