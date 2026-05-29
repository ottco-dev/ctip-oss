"""
services.remote_compute.modal_backend — Modal serverless GPU backend.

Modal is the recommended remote compute backend for this platform.

WHY MODAL:
    - Free $30/month credit (enough for ~27 hours of A10G or ~85 hours of T4)
    - Serverless: pay only when running (no idle costs)
    - Fastest cold start in the industry (~1-3s for containers)
    - Native Python — define GPU functions as regular Python code
    - Supports all our model families (SAM2, YOLO, Qwen2-VL)

SETUP:
    1. pip install modal
    2. modal token new  # opens browser for auth
    3. Set MODAL_TOKEN_ID and MODAL_TOKEN_SECRET in .env

USE CASES:
    - Training on A100 when local VRAM not enough
    - Large model inference (SAM2-large, Qwen2-VL-72B)
    - Parallel batch processing (multiple containers)
    - Experiment hyperparameter sweeps

COST EXAMPLES:
    - T4 (16GB):  $0.000164/s  → 85h/month free
    - A10G (24GB): $0.000306/s → 27h/month free
    - A100 (40GB): $0.000583/s → 14h/month free

Docs: https://modal.com/docs/guide
"""

from __future__ import annotations

import asyncio
import os
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
    backend_id="modal",
    name="Modal (Serverless GPU)",
    kind=ComputeBackendKind.MODAL,
    gpu_tiers=[GpuTier.T4, GpuTier.A10G, GpuTier.A100_40, GpuTier.A100_80, GpuTier.H100],
    free_tier=True,
    free_tier_note=(
        "$30/month free credit. ~27h A10G or ~85h T4. "
        "No credit card required for free tier. "
        "Best for training jobs and large model inference."
    ),
    signup_url="https://modal.com/",
    cost_per_hour={
        "t4": 0.59,
        "a10g": 1.10,
        "a100_40": 2.10,
        "a100_80": 3.50,
        "h100": 7.50,
    },
)


class ModalBackend(RemoteComputeBackend):
    """
    Modal serverless GPU backend.

    Uses Modal's Python SDK to define and run GPU functions.
    The trichome inference/training logic runs inside Modal containers
    that are pre-built with the same dependencies as our Docker image.
    """

    def __init__(
        self,
        token_id: str | None = None,
        token_secret: str | None = None,
    ) -> None:
        self._token_id = token_id or os.getenv("MODAL_TOKEN_ID", "")
        self._token_secret = token_secret or os.getenv("MODAL_TOKEN_SECRET", "")

    @property
    def info(self) -> ComputeBackendInfo:
        return _INFO

    @property
    def is_available(self) -> bool:
        return bool(self._token_id and self._token_secret)

    def _modal_app(self):
        """Build a Modal app with the trichome image."""
        try:
            import modal
        except ImportError:
            raise ImportError(
                "modal package not installed. Run: uv pip install modal\n"
                "Then authenticate: modal token new"
            )

        app = modal.App("trichome-remote")
        image = (
            modal.Image.debian_slim(python_version="3.12")
            .apt_install(["libgl1", "libglib2.0-0", "libsm6", "libxext6"])
            .pip_install([
                "torch", "torchvision", "ultralytics",
                "opencv-python-headless", "numpy", "pillow",
                "transformers", "accelerate",
            ])
        )
        return app, image

    async def run_vlm_inference(
        self,
        image: NDArray[np.uint8],
        prompt: str,
        model_id: str = "vikhyatk/moondream2",
        gpu_tier: GpuTier = GpuTier.T4,
    ) -> RemoteTaskResult:
        """
        Run VLM inference on a Modal GPU.

        The image is serialized, sent to Modal, inference runs in a container,
        and results are returned. Cold starts take ~30s; warm containers ~2s.
        """
        import time
        t0 = time.perf_counter()

        if not self.is_available:
            return RemoteTaskResult(
                success=False,
                output=None,
                error="Modal not configured. Set MODAL_TOKEN_ID and MODAL_TOKEN_SECRET in .env.",
            )

        try:
            import modal
            import base64
            import cv2

            # Serialize image
            _, buf = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            img_b64 = base64.b64encode(buf.tobytes()).decode()

            # Define remote function
            app, modal_image = self._modal_app()
            gpu_map = {
                GpuTier.T4: modal.gpu.T4(),
                GpuTier.A10G: modal.gpu.A10G(),
                GpuTier.A100_40: modal.gpu.A100(size="40GB"),
                GpuTier.A100_80: modal.gpu.A100(size="80GB"),
            }
            gpu = gpu_map.get(gpu_tier, modal.gpu.T4())

            @app.function(image=modal_image, gpu=gpu, timeout=300)
            def _remote_infer(image_b64: str, prompt: str, model_id: str) -> dict:
                import base64
                import json
                import numpy as np
                import cv2
                from PIL import Image

                # Decode image
                buf = base64.b64decode(image_b64)
                arr = np.frombuffer(buf, np.uint8)
                img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

                # Run model
                if "moondream" in model_id:
                    from transformers import AutoModelForCausalLM, AutoTokenizer
                    model = AutoModelForCausalLM.from_pretrained(
                        model_id, trust_remote_code=True,
                        load_in_4bit=True, device_map="auto",
                    )
                    tokenizer = AutoTokenizer.from_pretrained(model_id)
                    pil = Image.fromarray(img_rgb)
                    enc = model.encode_image(pil)
                    answer = model.answer_question(enc, prompt, tokenizer)
                    return {"raw": answer, "model": model_id}
                else:
                    return {"error": f"Model {model_id} not yet supported in Modal backend"}

            with app.run():
                result = _remote_infer.remote(img_b64, prompt, model_id)

            return RemoteTaskResult(
                success="error" not in result,
                output=result,
                latency_s=time.perf_counter() - t0,
                gpu_tier=gpu_tier.value,
                task_id="modal-sync",
            )

        except Exception as e:
            logger.error("Modal inference failed", error=str(e))
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
        model_id: str = "yolov9c.pt",
        gpu_tier: GpuTier = GpuTier.A10G,
    ) -> RemoteTaskResult:
        """
        Submit a YOLO training job to Modal.

        For large training runs that exceed local 8GB VRAM.
        Results (weights) are saved to Modal Volumes and downloadable.
        """
        import time
        t0 = time.perf_counter()

        if not self.is_available:
            return RemoteTaskResult(
                success=False,
                output=None,
                error="Modal not configured. Set MODAL_TOKEN_ID and MODAL_TOKEN_SECRET.",
            )

        logger.info(
            "Submitting training job to Modal",
            model=model_id,
            gpu=gpu_tier.value,
            config=config,
        )

        # TODO: Implement full training submission with Modal Volumes
        # This requires dataset upload to Modal storage first
        return RemoteTaskResult(
            success=False,
            output=None,
            error=(
                "Modal training submission requires dataset upload to Modal storage. "
                "See docs/remote_compute.md for setup instructions."
            ),
        )
