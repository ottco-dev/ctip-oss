"""vlm_labeling.providers.local.qwen_provider — Qwen2-VL-7B local provider."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from vlm_labeling.providers.base import (
    VLMProvider, VLMProviderInfo, VLMResponse,
    ProviderKind, ProviderTier, ProviderCapabilities,
)

_INFO = VLMProviderInfo(
    provider_id="qwen2vl",
    name="Qwen2-VL-7B (local, 4-bit)",
    kind=ProviderKind.LOCAL,
    tier=ProviderTier.FREE,
    models=["Qwen/Qwen2-VL-7B-Instruct"],
    default_model="Qwen/Qwen2-VL-7B-Instruct",
    vram_gb=5.5,
    cost_per_1k_tokens=None,
    free_tier_note="Fully local. Best quality local model. Requires 5.5 GB VRAM (4-bit).",
    signup_url="",
)

_CAPABILITIES = ProviderCapabilities(
    maturity_labeling=True,
    quality_screening=True,
    morphology_classification=True,
    batch_processing=False,
    streaming=False,
)


class QwenProvider(VLMProvider):
    """Thin adapter: QwenVLLabeler → VLMProvider interface."""

    def __init__(self) -> None:
        self._labeler = None

    @property
    def info(self) -> VLMProviderInfo:
        return _INFO

    @property
    def capabilities(self) -> ProviderCapabilities:
        return _CAPABILITIES

    @property
    def is_available(self) -> bool:
        try:
            import torch
            return torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory >= 5_500_000_000
        except ImportError:
            return False

    def load(self) -> None:
        from vlm_labeling.qwen2vl.qwen_labeler import QwenVLLabeler
        self._labeler = QwenVLLabeler()
        self._labeler.load()

    def unload(self) -> None:
        if self._labeler:
            self._labeler.unload()
            self._labeler = None

    def _ensure_loaded(self) -> None:
        if self._labeler is None:
            self.load()

    def _adapt(self, result) -> VLMResponse:
        return VLMResponse(
            raw_response=result.raw_response,
            parsed_response=result.parsed_response,
            is_valid=result.is_valid,
            confidence=result.confidence,
            provider_id="qwen2vl",
            model_id="Qwen/Qwen2-VL-7B-Instruct",
        )

    def label_maturity(self, image: NDArray[np.uint8]) -> VLMResponse:
        self._ensure_loaded()
        return self._adapt(self._labeler.label_maturity(image))

    def assess_quality(self, image: NDArray[np.uint8]) -> VLMResponse:
        self._ensure_loaded()
        return self._adapt(self._labeler.assess_quality(image))

    def label_morphology(self, image: NDArray[np.uint8]) -> VLMResponse:
        self._ensure_loaded()
        return self._adapt(self._labeler.label_morphology(image))
