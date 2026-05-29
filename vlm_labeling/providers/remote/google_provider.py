"""
vlm_labeling.providers.remote.google_provider — Google Gemini provider.

Models: gemini-1.5-flash, gemini-1.5-pro, gemini-2.0-flash
Free tier: YES — gemini-1.5-flash: 15 RPM, 1M tokens/day, 1500 RPD (no billing)
Signup: https://aistudio.google.com/app/apikey
Cost (paid): gemini-1.5-flash $0.075/1M input tokens (images: ~$0.002/image)
             gemini-1.5-pro   $3.50/1M input tokens
"""

from __future__ import annotations

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
    image_to_pil,
)
from shared.logging.logger import get_logger

logger = get_logger(__name__)

_INFO = VLMProviderInfo(
    provider_id="google",
    name="Google Gemini",
    kind=ProviderKind.REMOTE,
    tier=ProviderTier.FREEMIUM,
    models=["gemini-1.5-flash", "gemini-2.0-flash", "gemini-1.5-pro"],
    default_model="gemini-1.5-flash",
    vram_gb=None,
    cost_per_1k_tokens=0.000075,   # gemini-1.5-flash paid tier
    rate_limit_rpm=15,              # free tier
    free_tier_note=(
        "Free: 15 RPM, 1500 RPD, 1M tokens/day on gemini-1.5-flash. "
        "No credit card required. Best free option for trichome analysis."
    ),
    signup_url="https://aistudio.google.com/app/apikey",
)

_CAPABILITIES = ProviderCapabilities(
    maturity_labeling=True,
    quality_screening=True,
    morphology_classification=True,
    batch_processing=False,
    streaming=True,
)

class GoogleProvider(VLMProvider):
    """
    Google Gemini vision provider via google-generativeai SDK.

    Recommended for free-tier usage: gemini-1.5-flash has generous limits.
    Uses PIL image objects (not base64) for the Gemini SDK.
    """

    def __init__(self, api_key: str, model: str = "gemini-1.5-flash") -> None:
        if not api_key:
            raise ValueError("Google API key is required")
        self._api_key = api_key
        self._model_name = model
        self._model = None

    @property
    def info(self) -> VLMProviderInfo:
        return _INFO

    @property
    def capabilities(self) -> ProviderCapabilities:
        return _CAPABILITIES

    @property
    def is_available(self) -> bool:
        return bool(self._api_key)

    def _model_(self):
        if self._model is None:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self._api_key)
                self._model = genai.GenerativeModel(
                    model_name=self._model_name,
                    generation_config={
                        "response_mime_type": "application/json",
                        "temperature": 0.1,
                        "max_output_tokens": 512,
                    },
                    system_instruction=(
                        "You are a cannabis trichome microscopy expert. "
                        "Analyze visual morphology only. "
                        "Do NOT claim any cannabinoid content from optical analysis. "
                        "Always return strictly valid JSON."
                    ),
                )
            except ImportError:
                raise ImportError(
                    "google-generativeai package not installed. "
                    "Run: uv pip install google-generativeai"
                )
        return self._model

    def _call(self, prompt: str, image: NDArray[np.uint8]) -> VLMResponse:
        t0 = time.perf_counter()
        pil_img = image_to_pil(image)

        try:
            model = self._model_()
            response = model.generate_content([pil_img, prompt])
            raw = response.text or ""

            # Try to get usage (not always available in all SDK versions)
            input_tokens = output_tokens = 0
            try:
                if hasattr(response, "usage_metadata"):
                    input_tokens = response.usage_metadata.prompt_token_count or 0
                    output_tokens = response.usage_metadata.candidates_token_count or 0
            except Exception:
                pass

        except Exception as e:
            logger.error("Google Gemini API call failed", error=str(e), model=self._model_name)
            return VLMResponse(
                raw_response="",
                parsed_response=None,
                is_valid=False,
                provider_id="google",
                model_id=self._model_name,
                latency_s=time.perf_counter() - t0,
                error=str(e),
            )

        latency = time.perf_counter() - t0
        clean = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        try:
            parsed = json.loads(clean)
        except json.JSONDecodeError:
            parsed = None

        return VLMResponse(
            raw_response=raw,
            parsed_response=parsed,
            is_valid=parsed is not None,
            confidence=float(parsed.get("confidence", 0.0)) if parsed else 0.0,
            provider_id="google",
            model_id=self._model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_s=latency,
        )

    def label_maturity(self, image: NDArray[np.uint8]) -> VLMResponse:
        prompt = (
            "Analyze this trichome image for maturity. "
            "Return JSON: {maturity_stage: clear|cloudy|amber|mixed, confidence: float, "
            "amber_fraction_estimate: float, cloudy_fraction_estimate: float, "
            "clear_fraction_estimate: float, observations: string}"
        )
        return self._call(prompt, image)

    def assess_quality(self, image: NDArray[np.uint8]) -> VLMResponse:
        prompt = (
            "Assess this microscopy image quality. "
            "Return JSON: {overall_quality: excellent|good|poor|unusable, "
            "focus_quality: string, lighting_quality: string, analyzable: bool, "
            "reject_reason: string|null, confidence: float}"
        )
        return self._call(prompt, image)

    def label_morphology(self, image: NDArray[np.uint8]) -> VLMResponse:
        prompt = (
            "Classify trichome morphology in this image. "
            "Return JSON: {dominant_type: capitate_stalked|capitate_sessile|bulbous, "
            "confidence: float, stalk_visible: bool, head_shape: string, "
            "mixed_types_present: bool}"
        )
        return self._call(prompt, image)
