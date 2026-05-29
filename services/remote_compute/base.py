"""
services.remote_compute.base — Abstract remote compute backend.

All backends implement the same interface for consistency.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray


class ComputeBackendKind(str, Enum):
    MODAL = "modal"
    REPLICATE = "replicate"
    HF_SPACES = "hf_spaces"
    LOCAL = "local"


class GpuTier(str, Enum):
    """GPU class for capacity/cost planning."""
    T4 = "t4"           # ~$0.35/hr
    A10G = "a10g"       # ~$1.10/hr (Modal default)
    A100_40 = "a100_40" # ~$3.50/hr
    A100_80 = "a100_80" # ~$4.50/hr
    H100 = "h100"       # ~$8/hr


@dataclass
class ComputeBackendInfo:
    backend_id: str
    name: str
    kind: ComputeBackendKind
    gpu_tiers: list[GpuTier]
    free_tier: bool
    free_tier_note: str
    signup_url: str
    cost_per_hour: dict[str, float] = field(default_factory=dict)
    """Approximate USD/hr per GPU tier."""


@dataclass
class RemoteTaskResult:
    success: bool
    output: Any | None
    error: str | None = None
    latency_s: float = 0.0
    gpu_tier: str = ""
    cost_usd: float | None = None
    task_id: str = ""


class RemoteComputeBackend(abc.ABC):
    """Abstract base for remote GPU compute backends."""

    @property
    @abc.abstractmethod
    def info(self) -> ComputeBackendInfo: ...

    @property
    def is_available(self) -> bool:
        """True if API key is set and backend can be used."""
        return True

    @abc.abstractmethod
    async def run_vlm_inference(
        self,
        image: NDArray[np.uint8],
        prompt: str,
        model_id: str,
        gpu_tier: GpuTier = GpuTier.T4,
    ) -> RemoteTaskResult:
        """Run VLM inference on a remote GPU."""
        ...

    @abc.abstractmethod
    async def run_training_job(
        self,
        config: dict[str, Any],
        dataset_path: str,
        model_id: str,
        gpu_tier: GpuTier = GpuTier.A10G,
    ) -> RemoteTaskResult:
        """Submit a training job to a remote GPU cluster."""
        ...

    def estimate_cost(
        self,
        task_type: str,
        estimated_duration_hours: float,
        gpu_tier: GpuTier = GpuTier.A10G,
    ) -> float | None:
        """Estimate USD cost for a task."""
        cost_map = self.info.cost_per_hour
        gpu_cost = cost_map.get(gpu_tier.value)
        if gpu_cost is None:
            return None
        return gpu_cost * estimated_duration_hours
