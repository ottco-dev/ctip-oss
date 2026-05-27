"""
inference.local.runner — Local PyTorch inference runner.

Wraps model loading and inference in a stateful runner that:
1. Loads model once, keeps in memory for repeated inference
2. Handles batch inference efficiently (no repeated H→D transfers)
3. Supports FP16 mixed precision (RTX 4060 optimized)
4. Provides warm-up run to avoid first-inference latency spikes
5. Tracks per-run latency and throughput statistics

For production serving:
  - Use inference/onnx_runtime/runner.py for CPU-optimized throughput
  - Use inference/tensorrt_engine/runner.py for maximum GPU throughput

This runner is for development/research use where flexibility matters.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class LocalRunnerConfig:
    """Configuration for local PyTorch inference."""

    model_path: str
    """Path to .pt model file or model name (e.g. 'yolo11s.pt')."""

    device: str = "cuda"
    """Device: 'cuda', 'cuda:0', 'cpu'."""

    half_precision: bool = True
    """Use FP16 (requires CUDA). ~2× faster on RTX 4060."""

    warmup_runs: int = 2
    """Number of warmup forward passes before timing."""

    conf_threshold: float = 0.35
    iou_threshold: float = 0.45
    imgsz: int = 1280

    # Batch settings
    max_batch_size: int = 8
    """Maximum images per batch. Limited by VRAM."""

    # Augmented inference
    augment: bool = False
    """TTA (test-time augmentation). Slower but more accurate."""


# ---------------------------------------------------------------------------
# Latency tracker
# ---------------------------------------------------------------------------

@dataclass
class LatencyStats:
    """Running latency statistics."""

    runs: int = 0
    total_ms: float = 0.0
    min_ms: float = float("inf")
    max_ms: float = 0.0
    _recent: list[float] = field(default_factory=list)

    def update(self, ms: float) -> None:
        self.runs += 1
        self.total_ms += ms
        self.min_ms = min(self.min_ms, ms)
        self.max_ms = max(self.max_ms, ms)
        self._recent.append(ms)
        if len(self._recent) > 50:
            self._recent.pop(0)

    @property
    def mean_ms(self) -> float:
        return self.total_ms / max(self.runs, 1)

    @property
    def p95_ms(self) -> float:
        if not self._recent:
            return 0.0
        return float(np.percentile(self._recent, 95))

    def to_dict(self) -> dict:
        return {
            "runs": self.runs,
            "mean_ms": round(self.mean_ms, 2),
            "min_ms": round(self.min_ms, 2) if self.min_ms != float("inf") else 0,
            "max_ms": round(self.max_ms, 2),
            "p95_ms": round(self.p95_ms, 2),
        }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class LocalPyTorchRunner:
    """
    Stateful local PyTorch inference runner.

    Loads a YOLO model once and provides fast repeated inference
    with latency tracking.

    Usage::

        runner = LocalPyTorchRunner(config)
        runner.load()

        result = runner.infer(image)
        batch_results = runner.infer_batch([img1, img2, img3])

        print(runner.latency_stats.to_dict())
        runner.unload()
    """

    def __init__(self, config: LocalRunnerConfig) -> None:
        self.config = config
        self._model: Any | None = None
        self._is_loaded: bool = False
        self.latency_stats = LatencyStats()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load model into GPU memory with warm-up."""
        if self._is_loaded:
            return

        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                "ultralytics not installed. Install: pip install ultralytics"
            ) from e

        import torch

        logger.info("Loading model: %s", self.config.model_path)
        t0 = time.monotonic()

        device = self.config.device
        if device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            device = "cpu"

        self._model = YOLO(self.config.model_path)

        # Half precision
        if self.config.half_precision and device != "cpu":
            self._model.model.half()
            logger.debug("Model set to FP16")

        # Warm-up
        if self.config.warmup_runs > 0:
            dummy = np.zeros(
                (self.config.imgsz, self.config.imgsz, 3), dtype=np.uint8
            )
            for _ in range(self.config.warmup_runs):
                self._model(
                    dummy,
                    device=device,
                    half=self.config.half_precision and device != "cpu",
                    verbose=False,
                    conf=self.config.conf_threshold,
                    iou=self.config.iou_threshold,
                    imgsz=self.config.imgsz,
                )
            logger.debug("Warmup complete (%d runs)", self.config.warmup_runs)

        self._device = device
        self._is_loaded = True
        elapsed = time.monotonic() - t0
        logger.info("Model loaded in %.1fs on %s", elapsed, device)

    def unload(self) -> None:
        """Release model and free GPU memory."""
        if not self._is_loaded:
            return

        del self._model
        self._model = None

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        import gc
        gc.collect()
        self._is_loaded = False
        logger.info("Model unloaded")

    def __enter__(self) -> "LocalPyTorchRunner":
        self.load()
        return self

    def __exit__(self, *args: Any) -> None:
        self.unload()

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def infer(
        self,
        image: NDArray[np.uint8],
        conf_threshold: float | None = None,
        iou_threshold: float | None = None,
    ) -> list[dict]:
        """
        Run inference on a single image.

        Args:
            image: HWC uint8 numpy array (RGB).
            conf_threshold: Override config threshold.
            iou_threshold: Override config threshold.

        Returns:
            List of detection dicts: {x1, y1, x2, y2, confidence, class_id, class_name}
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call .load() first.")

        conf = conf_threshold or self.config.conf_threshold
        iou = iou_threshold or self.config.iou_threshold

        t0 = time.perf_counter()

        results = self._model(
            image,
            device=self._device,
            half=self.config.half_precision and self._device != "cpu",
            verbose=False,
            conf=conf,
            iou=iou,
            imgsz=self.config.imgsz,
            augment=self.config.augment,
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000
        self.latency_stats.update(elapsed_ms)

        return self._parse_results(results[0])

    def infer_batch(
        self,
        images: list[NDArray[np.uint8]],
        conf_threshold: float | None = None,
    ) -> list[list[dict]]:
        """
        Run inference on a batch of images.

        Args:
            images: List of HWC uint8 numpy arrays.
            conf_threshold: Confidence threshold.

        Returns:
            List of detection lists, one per image.
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded.")

        if len(images) > self.config.max_batch_size:
            logger.warning(
                "Batch size %d > max_batch_size %d. Processing in sub-batches.",
                len(images),
                self.config.max_batch_size,
            )

        all_results = []
        batch_size = self.config.max_batch_size
        conf = conf_threshold or self.config.conf_threshold

        for i in range(0, len(images), batch_size):
            batch = images[i: i + batch_size]

            t0 = time.perf_counter()
            results = self._model(
                batch,
                device=self._device,
                half=self.config.half_precision and self._device != "cpu",
                verbose=False,
                conf=conf,
                iou=self.config.iou_threshold,
                imgsz=self.config.imgsz,
                stream=True,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000
            self.latency_stats.update(elapsed_ms / len(batch))

            for res in results:
                all_results.append(self._parse_results(res))

        return all_results

    # ------------------------------------------------------------------
    # Result parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_results(result: Any) -> list[dict]:
        """Parse Ultralytics result object to detection dicts."""
        detections = []

        if result.boxes is None:
            return detections

        boxes = result.boxes
        xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else boxes.xyxy
        confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else boxes.conf
        cls = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else boxes.cls
        names = result.names or {}

        for i in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[i].tolist()
            class_id = int(cls[i])
            detections.append({
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
                "confidence": float(confs[i]),
                "class_id": class_id,
                "class_name": str(names.get(class_id, f"class_{class_id}")),
            })

        return detections

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def get_vram_usage_mb(self) -> float | None:
        """Return current VRAM usage in MB."""
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / (1024 ** 2)
        except Exception:
            pass
        return None

    def __repr__(self) -> str:
        status = "loaded" if self._is_loaded else "not loaded"
        return (
            f"LocalPyTorchRunner("
            f"model={self.config.model_path!r}, "
            f"device={self.config.device!r}, "
            f"status={status})"
        )
