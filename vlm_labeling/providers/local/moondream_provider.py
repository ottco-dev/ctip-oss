"""
vlm_labeling.providers.local.moondream_provider — Moondream-2B local provider.

Wraps the existing MoondreamLabeler to conform to the VLMProvider interface.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from vlm_labeling.providers.base import (
    VLMProvider,
    VLMProviderInfo,
    VLMResponse,
    ProviderKind,
    ProviderTier,
    ProviderCapabilities,
)

_INFO = VLMProviderInfo(
    provider_id="moondream",
    name="Moondream-2B (local, 4-bit)",
    kind=ProviderKind.LOCAL,
    tier=ProviderTier.FREE,
    models=["vikhyatk/moondream2"],
    default_model="vikhyatk/moondream2",
    vram_gb=2.1,
    cost_per_1k_tokens=None,
    free_tier_note="Fully local. No API key required. Requires 2.1 GB VRAM.",
    signup_url="",
)

_CAPABILITIES = ProviderCapabilities(
    maturity_labeling=True,
    quality_screening=True,
    morphology_classification=True,
    batch_processing=False,
    streaming=False,
)


class MoondreamProvider(VLMProvider):
    """Thin adapter: MoondreamLabeler → VLMProvider interface."""

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
            return torch.cuda.is_available()
        except ImportError:
            return False

    def load(self) -> None:
        from vlm_labeling.moondream.moondream_labeler import MoondreamLabeler
        self._labeler = MoondreamLabeler()
        self._labeler.load()

    def unload(self) -> None:
        if self._labeler:
            self._labeler.unload()
            self._labeler = None

    def _ensure_loaded(self) -> None:
        if self._labeler is None:
            self.load()

    def _adapt(self, result) -> VLMResponse:  # type: ignore[return]
        """Convert MoondreamLabeler result to VLMResponse."""
        return VLMResponse(
            raw_response=result.raw_response,
            parsed_response=result.parsed_response,
            is_valid=result.is_valid,
            confidence=result.confidence,
            provider_id="moondream",
            model_id="vikhyatk/moondream2",
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
