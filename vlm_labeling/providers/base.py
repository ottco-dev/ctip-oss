"""
vlm_labeling.providers.base — Abstract base for all VLM providers.

Every provider (local or remote) implements this interface so the pipeline
can switch backends without changing calling code.

HITL INVARIANT (enforced here, not by providers):
All VLM outputs are AnnotationSource.VLM_AUTO. Providers never write to the
training dataset. That gate is enforced upstream in AutoLabelPipeline.
"""

from __future__ import annotations

import abc
import base64
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


class ProviderKind(str, Enum):
    """Whether the provider runs locally or via remote API."""
    LOCAL = "local"
    REMOTE = "remote"


class ProviderTier(str, Enum):
    """Cost tier for capacity planning / UI display."""
    FREE = "free"           # Always free (e.g., Groq free tier, HF free)
    FREEMIUM = "freemium"   # Free with limits, paid above
    PAID = "paid"           # Always costs money per request


@dataclass
class VLMProviderInfo:
    """
    Static metadata about a VLM provider.

    Shown in the UI provider selector and docs.
    """
    provider_id: str
    """Unique identifier, e.g. 'openai', 'groq', 'moondream'."""

    name: str
    """Human-readable name."""

    kind: ProviderKind
    tier: ProviderTier

    models: list[str]
    """Model IDs available under this provider."""

    default_model: str

    vram_gb: float | None = None
    """Required VRAM (local only). None = no local GPU needed."""

    cost_per_1k_tokens: float | None = None
    """Approximate USD cost per 1k input tokens (remote only). None = free."""

    rate_limit_rpm: int | None = None
    """Requests per minute limit (free tiers)."""

    free_tier_note: str = ""
    """Description of free tier terms."""

    signup_url: str = ""
    """Where to get an API key."""

    supports_vision: bool = True


@dataclass
class VLMResponse:
    """
    Standardised response from any VLM provider.

    Providers parse their raw API/model output into this structure.
    """
    raw_response: str
    """Original text output from the model/API."""

    parsed_response: dict[str, Any] | None
    """Structured fields extracted from raw_response via schema enforcement."""

    is_valid: bool
    """True if parsed_response meets minimum schema requirements."""

    confidence: float = 0.0
    """Provider-reported or parsed confidence [0, 1]."""

    provider_id: str = ""
    model_id: str = ""

    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0

    error: str | None = None
    """Set if inference failed; is_valid will be False."""

    @property
    def is_analyzable(self) -> bool:
        """Shorthand: can we extract a label from this response."""
        return self.is_valid and self.parsed_response is not None


@dataclass
class ProviderCapabilities:
    """What a provider can do (not all endpoints apply to every provider)."""
    maturity_labeling: bool = True
    quality_screening: bool = True
    morphology_classification: bool = True
    batch_processing: bool = False
    streaming: bool = False


def image_to_base64(image: NDArray[np.uint8]) -> str:
    """Convert uint8 RGB array → base64 JPEG string for API upload."""
    import cv2

    success, buf = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                                [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not success:
        raise ValueError("Failed to encode image to JPEG")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def image_to_pil(image: NDArray[np.uint8]):  # type: ignore[return]
    """Convert uint8 RGB array → PIL Image."""
    from PIL import Image
    return Image.fromarray(image)


class VLMProvider(abc.ABC):
    """
    Abstract base for all VLM labeling providers.

    Subclasses implement the three core labeling methods. The framework
    handles JSON schema enforcement, hallucination filtering, and HITL routing.

    Local providers:
        - Manage model loading/unloading lifecycle
        - Respect GPU semaphore
        - Run in-process

    Remote providers:
        - Accept API key at construction
        - Are stateless (no load/unload)
        - Report token usage for cost tracking
    """

    @property
    @abc.abstractmethod
    def info(self) -> VLMProviderInfo:
        """Static metadata about this provider."""
        ...

    @property
    @abc.abstractmethod
    def capabilities(self) -> ProviderCapabilities:
        ...

    def load(self) -> None:
        """Load model into memory (local providers only). No-op for remote."""

    def unload(self) -> None:
        """Unload model from memory (local providers only). No-op for remote."""

    @property
    def is_available(self) -> bool:
        """
        Check if the provider can currently be used.

        Local: checks CUDA availability and model files.
        Remote: checks if API key is set.
        """
        return True

    @abc.abstractmethod
    def label_maturity(
        self,
        image: NDArray[np.uint8],
    ) -> VLMResponse:
        """
        Classify trichome maturity from an image.

        Expected parsed_response keys:
            maturity_stage: str  # "clear" | "cloudy" | "amber" | "mixed"
            confidence: float
            amber_fraction_estimate: float | None
            cloudy_fraction_estimate: float | None
            clear_fraction_estimate: float | None
            observations: str | None
        """
        ...

    @abc.abstractmethod
    def assess_quality(
        self,
        image: NDArray[np.uint8],
    ) -> VLMResponse:
        """
        Assess microscopy image quality.

        Expected parsed_response keys:
            overall_quality: str  # "excellent" | "good" | "poor" | "unusable"
            focus_quality: str
            lighting_quality: str
            analyzable: bool
            reject_reason: str | None
            confidence: float
        """
        ...

    @abc.abstractmethod
    def label_morphology(
        self,
        image: NDArray[np.uint8],
    ) -> VLMResponse:
        """
        Classify trichome morphology type.

        Expected parsed_response keys:
            dominant_type: str  # "capitate_stalked" | "capitate_sessile" | "bulbous"
            confidence: float
            stalk_visible: bool | None
            head_shape: str | None
            mixed_types_present: bool | None
        """
        ...

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float | None:
        """
        Estimate USD cost for a request with given token counts.

        Returns None for local/free-tier providers.
        """
        info = self.info
        if info.cost_per_1k_tokens is None:
            return None
        return (input_tokens + output_tokens) / 1000 * info.cost_per_1k_tokens
