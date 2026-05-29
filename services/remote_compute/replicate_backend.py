"""
services.remote_compute.replicate_backend — Replicate hosted models backend.

Replicate provides hosted open-source models as API endpoints.
Pay-per-prediction (no idle cost, no setup).

NOTABLE MODELS:
    Vision/VLM:
        - meta/meta-llama-3-2-11b-vision-instruct (~$0.0016/prediction)
        - yorickvp/llava-13b (~$0.0045/prediction)

    Segmentation:
        - cjwbw/segment-anything (SAM1, free to try)
        - meta/sam-2 (SAM2)
        - adirik/grounding-dino (GroundingDINO)

    Detection:
        - ultralytics/yolov8 (free tier)

FREE TIER: New accounts get some free predictions. Not a sustained free tier.

Signup: https://replicate.com/account/api-tokens
Docs:   https://replicate.com/docs
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

from services.remote_compute.base import (
    RemoteComputeBackend,
    ComputeBackendInfo,
    ComputeBackendKind,
    GpuTier,
    RemoteTaskResult,
)
from shared.logging.logger import get_logger

logger = get_logger(__name__)

_INFO = ComputeBackendInfo(
    backend_id="replicate",
    name="Replicate (Hosted Models)",
    kind=ComputeBackendKind.REPLICATE,
    gpu_tiers=[GpuTier.T4, GpuTier.A100_40],
    free_tier=False,
    free_tier_note=(
        "No sustained free tier. New accounts get trial credits. "
        "Pay-per-prediction: ~$0.0016-$0.005 per image depending on model."
    ),
    signup_url="https://replicate.com/account/api-tokens",
    cost_per_hour={
        "t4": 0.50,
        "a100_40": 2.30,
    },
)

# Curated model IDs for our use cases
REPLICATE_MODELS = {
    "sam2": "meta/sam-2",
    "sam1": "cjwbw/segment-anything",
    "grounding_dino": "adirik/grounding-dino",
    "yolov8": "ultralytics/yolov8",
    "llava": "yorickvp/llava-13b",
    "llama_vision": "meta/meta-llama-3-2-11b-vision-instruct",
}


class ReplicateBackend(RemoteComputeBackend):
    """
    Replicate hosted model backend.

    Uses the replicate Python client to run models via API.
    Best for: SAM2 inference, GroundingDINO, and VLMs without local GPU.
    """

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or os.getenv("REPLICATE_API_KEY", "")

    @property
    def info(self) -> ComputeBackendInfo:
        return _INFO

    @property
    def is_available(self) -> bool:
        return bool(self._api_key)

    def _client(self):
        try:
            import replicate
            return replicate.Client(api_token=self._api_key)
        except ImportError:
            raise ImportError(
                "replicate package not installed. Run: uv pip install replicate"
            )

    async def run_vlm_inference(
        self,
        image: NDArray[np.uint8],
        prompt: str,
        model_id: str = "meta/meta-llama-3-2-11b-vision-instruct",
        gpu_tier: GpuTier = GpuTier.T4,
    ) -> RemoteTaskResult:
        """Run VLM inference via Replicate API."""
        t0 = time.perf_counter()

        if not self.is_available:
            return RemoteTaskResult(
                success=False,
                output=None,
                error="Replicate not configured. Set REPLICATE_API_KEY in .env.",
            )

        try:
            import base64
            import io
            import cv2
            from PIL import Image as PILImage

            # Convert to PNG for Replicate (no base64 overhead with file)
            pil = PILImage.fromarray(image)
            buf = io.BytesIO()
            pil.save(buf, format="PNG")
            buf.seek(0)

            client = self._client()
            output = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: client.run(
                    model_id,
                    input={
                        "image": buf,
                        "prompt": prompt,
                        "max_new_tokens": 256,
                    },
                ),
            )

            result_text = "".join(output) if hasattr(output, "__iter__") else str(output)

            return RemoteTaskResult(
                success=True,
                output={"text": result_text},
                latency_s=time.perf_counter() - t0,
                gpu_tier=gpu_tier.value,
            )

        except Exception as e:
            logger.error("Replicate inference failed", error=str(e), model=model_id)
            return RemoteTaskResult(
                success=False,
                output=None,
                error=str(e),
                latency_s=time.perf_counter() - t0,
            )

    async def run_sam2_inference(
        self,
        image: NDArray[np.uint8],
        point_coords: list[list[int]] | None = None,
        boxes: list[list[float]] | None = None,
    ) -> RemoteTaskResult:
        """
        Run SAM2 segmentation via Replicate.

        Use when local VRAM is insufficient for SAM2-large,
        or for batch jobs without blocking the local GPU.
        """
        t0 = time.perf_counter()

        if not self.is_available:
            return RemoteTaskResult(
                success=False,
                output=None,
                error="Replicate not configured.",
            )

        try:
            import io
            from PIL import Image as PILImage

            pil = PILImage.fromarray(image)
            buf = io.BytesIO()
            pil.save(buf, format="PNG")
            buf.seek(0)

            client = self._client()
            inputs: dict[str, Any] = {"image": buf}
            if point_coords:
                inputs["point_coords"] = point_coords
            if boxes:
                inputs["box"] = boxes[0] if boxes else None

            output = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: client.run(REPLICATE_MODELS["sam2"], input=inputs),
            )

            return RemoteTaskResult(
                success=True,
                output=output,
                latency_s=time.perf_counter() - t0,
                gpu_tier="a100_40",
            )

        except Exception as e:
            logger.error("Replicate SAM2 failed", error=str(e))
            return RemoteTaskResult(
                success=False,
                output=None,
                error=str(e),
                latency_s=time.perf_counter() - t0,
            )

    async def run_training_job(
        self,
        config: dict[str, Any],
        dataset_path: str,
        model_id: str = "ultralytics/yolov8",
        gpu_tier: GpuTier = GpuTier.A100_40,
    ) -> RemoteTaskResult:
        """Replicate does not support custom training jobs via API."""
        return RemoteTaskResult(
            success=False,
            output=None,
            error=(
                "Replicate does not support custom training. "
                "Use Modal for training outsourcing."
            ),
        )
