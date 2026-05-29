"""
vlm_labeling.providers.remote.hf_provider — Hugging Face Inference API provider.

Supports hosted models via HF Inference API (Serverless and Dedicated Endpoints).

Free tier: YES — Serverless Inference API: ~1000 RPD per model on free tier
           Models: Idefics2, LLaVA, Moondream (hosted), InternVL, Phi-3.5-Vision
           NOTE: Free tier has cold starts (can take 10-60s on first request).
Signup: https://huggingface.co/settings/tokens
Docs:   https://huggingface.co/docs/api-inference/
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
    provider_id="huggingface",
    name="Hugging Face Inference",
    kind=ProviderKind.REMOTE,
    tier=ProviderTier.FREEMIUM,
    models=[
        "vikhyatk/moondream2",                          # Moondream (hosted)
        "HuggingFaceM4/Idefics3-8B-Llama3",             # Idefics3
        "microsoft/Phi-3.5-vision-instruct",             # Phi-3.5-Vision
        "llava-hf/llava-1.5-7b-hf",                     # LLaVA 1.5
    ],
    default_model="vikhyatk/moondream2",
    vram_gb=None,
    cost_per_1k_tokens=None,     # Free tier
    rate_limit_rpm=10,            # Free tier conservative estimate
    free_tier_note=(
        "Free Serverless Inference: ~1000 RPD. Cold starts possible (10-60s first request). "
        "Upgrade to PRO ($9/mo) for dedicated endpoints with guaranteed latency."
    ),
    signup_url="https://huggingface.co/settings/tokens",
)

_CAPABILITIES = ProviderCapabilities(
    maturity_labeling=True,
    quality_screening=True,
    morphology_classification=True,
    batch_processing=False,
    streaming=False,
)

class HuggingFaceProvider(VLMProvider):
    """
    Hugging Face Inference API provider.

    Uses the InferenceClient from huggingface_hub for vision-language models.
    Best option when you want to avoid third-party commercial API keys.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "vikhyatk/moondream2",
    ) -> None:
        # api_key is optional — HF Inference API allows anonymous requests for
        # public models (rate-limited). A token unlocks higher rate limits.
        self._api_key = api_key or ""
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
        # Anonymous access is allowed (rate-limited); any state is "available"
        return True

    def _client_(self) -> Any:
        if self._client is None:
            try:
                from huggingface_hub import InferenceClient
                self._client = InferenceClient(
                    model=self._model,
                    token=self._api_key,
                )
            except ImportError:
                raise ImportError(
                    "huggingface_hub package not installed. "
                    "Run: uv pip install huggingface_hub"
                )
        return self._client

    def _call(self, prompt: str, image: NDArray[np.uint8]) -> VLMResponse:
        t0 = time.perf_counter()
        from PIL import Image as PILImage
        pil_img = PILImage.fromarray(image)

        try:
            client = self._client_()
            # Use visual question answering or chat completion depending on model
            response = client.visual_question_answering(
                image=pil_img,
                question=prompt,
            )
            raw = response.answer if hasattr(response, "answer") else str(response)
        except Exception:
            # Fallback to chat completion style
            try:
                import io
                buf = io.BytesIO()
                PILImage.fromarray(image).save(buf, format="JPEG")
                buf.seek(0)
                response = client.chat_completion(
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_to_base64(image)}"}},
                            {"type": "text", "text": prompt},
                        ],
                    }],
                    max_tokens=512,
                )
                raw = response.choices[0].message.content or ""
            except Exception as e2:
                logger.error("HF Inference API call failed", error=str(e2), model=self._model)
                return VLMResponse(
                    raw_response="",
                    parsed_response=None,
                    is_valid=False,
                    provider_id="huggingface",
                    model_id=self._model,
                    latency_s=time.perf_counter() - t0,
                    error=str(e2),
                )

        latency = time.perf_counter() - t0
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
            provider_id="huggingface",
            model_id=self._model,
            latency_s=latency,
        )

    def label_maturity(self, image: NDArray[np.uint8]) -> VLMResponse:
        prompt = (
            "You are a trichome microscopy expert. Analyze maturity morphology only. "
            "Return ONLY JSON: {\"maturity_stage\": \"clear|cloudy|amber|mixed\", "
            "\"confidence\": 0.0, \"amber_fraction_estimate\": 0.0, "
            "\"cloudy_fraction_estimate\": 0.0, \"clear_fraction_estimate\": 0.0, "
            "\"observations\": \"\"}"
        )
        return self._call(prompt, image)

    def assess_quality(self, image: NDArray[np.uint8]) -> VLMResponse:
        prompt = (
            "Rate this microscopy image. Return ONLY JSON: "
            "{\"overall_quality\": \"excellent|good|poor|unusable\", "
            "\"focus_quality\": \"\", \"lighting_quality\": \"\", "
            "\"analyzable\": true, \"reject_reason\": null, \"confidence\": 0.0}"
        )
        return self._call(prompt, image)

    def label_morphology(self, image: NDArray[np.uint8]) -> VLMResponse:
        prompt = (
            "Classify trichome type. Return ONLY JSON: "
            "{\"dominant_type\": \"capitate_stalked|capitate_sessile|bulbous\", "
            "\"confidence\": 0.0, \"stalk_visible\": true, "
            "\"head_shape\": \"\", \"mixed_types_present\": false}"
        )
        return self._call(prompt, image)
