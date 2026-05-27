"""
segmentation.infrastructure.mobile_sam — MobileSAM fallback backend.

MobileSAM (Zhang et al. 2023): distilled SAM variant, ~10× smaller.
- Encoder: TinyViT-5M (vs SAM's ViT-H 636M)
- Decoder: identical to original SAM
- VRAM: ~38 MB GPU / ~150 MB CPU (works without GPU)
- Speed: ~12ms/image on RTX 4060 vs SAM's ~50ms

Use cases:
  - CPU-only machines
  - When RTX 4060 VRAM is occupied by other models
  - Real-time preview (lower accuracy acceptable)

Tradeoff vs SAM2:
  - MobileSAM: faster, less accurate, works on CPU
  - SAM2-tiny: slower, more accurate, requires GPU

Reference:
  Zhang, C. et al. (2023). Faster Segment Anything: Towards Lightweight
  SAM for Mobile Applications. arXiv:2306.14289.

Install:
    pip install mobile-sam
"""

from __future__ import annotations

import gc
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from segmentation.domain.segmentor import (
    BaseSegmentor,
    BatchSegmentationResult,
    BoxPrompt,
    PointPrompt,
    SegmentationResult,
    SegmentorConfig,
)

logger = logging.getLogger(__name__)

MOBILE_SAM_CHECKPOINT_URL = (
    "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt"
)


class MobileSAMBackend(BaseSegmentor):
    """
    MobileSAM segmentation backend.

    Compatible interface with SAM2TinyBackend.
    Falls back to CPU if CUDA unavailable.

    Usage::

        backend = MobileSAMBackend(checkpoint_path="weights/mobile_sam.pt")
        backend.load()
        result = backend.segment_with_boxes(image, boxes)
        backend.unload()
    """

    def __init__(
        self,
        config: SegmentorConfig | None = None,
        checkpoint_path: str = "weights/mobile_sam.pt",
    ) -> None:
        super().__init__(config)
        self.checkpoint_path = checkpoint_path
        self._sam: Any | None = None
        self._predictor: Any | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load MobileSAM into memory."""
        if self._is_loaded:
            return

        logger.info("Loading MobileSAM from %s", self.checkpoint_path)
        t0 = time.monotonic()

        try:
            from mobile_sam import sam_model_registry, SamPredictor
        except ImportError as e:
            raise ImportError(
                "MobileSAM not installed. Install: pip install mobile-sam"
            ) from e

        import torch

        ckpt_path = Path(self.checkpoint_path)
        if not ckpt_path.exists():
            self._download_checkpoint(ckpt_path)

        device = self.config.device
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, using CPU for MobileSAM")
            device = "cpu"

        self._sam = sam_model_registry["vit_t"](checkpoint=str(ckpt_path))
        self._sam.to(device=device)
        self._sam.eval()

        self._predictor = SamPredictor(self._sam)
        self._device = device

        self._is_loaded = True
        elapsed = time.monotonic() - t0
        logger.info("MobileSAM loaded in %.1fs on %s", elapsed, device)

    def unload(self) -> None:
        """Release MobileSAM and free memory."""
        if not self._is_loaded:
            return

        del self._predictor
        del self._sam
        self._predictor = None
        self._sam = None

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        gc.collect()
        self._is_loaded = False
        logger.info("MobileSAM unloaded")

    def __enter__(self) -> "MobileSAMBackend":
        self.load()
        return self

    def __exit__(self, *args: Any) -> None:
        self.unload()

    # ------------------------------------------------------------------
    # Segmentation
    # ------------------------------------------------------------------

    def segment_with_boxes(
        self,
        image: NDArray[np.uint8],
        boxes: list[BoxPrompt],
    ) -> BatchSegmentationResult:
        """Segment trichomes using bounding box prompts (MobileSAM)."""
        if not self._is_loaded:
            raise RuntimeError("MobileSAM not loaded. Call .load() first.")

        if len(boxes) == 0:
            return BatchSegmentationResult(
                masks=[],
                image_height=image.shape[0],
                image_width=image.shape[1],
                backend="mobile_sam",
            )

        import torch

        t0 = time.perf_counter()
        h, w = image.shape[:2]

        self._predictor.set_image(image)

        results = []
        for box in boxes:
            box_arr = np.array([box.x1, box.y1, box.x2, box.y2], dtype=np.float32)

            with torch.no_grad():
                masks, scores, _ = self._predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=box_arr[np.newaxis],  # SAM expects (1, 4)
                    multimask_output=self.config.multimask_output,
                )

            mask_list = [masks[i].astype(bool) for i in range(len(masks))]
            score_list = scores.tolist()

            best_mask, best_score = self._select_best_mask(mask_list, score_list)
            clean_mask = self._postprocess_mask(best_mask, h, w)

            results.append(
                SegmentationResult(
                    mask=clean_mask,
                    score=float(best_score),
                    source_prompt=box,
                )
            )

        elapsed_ms = (time.perf_counter() - t0) * 1000
        return BatchSegmentationResult(
            masks=results,
            image_height=h,
            image_width=w,
            backend="mobile_sam",
            inference_time_ms=elapsed_ms,
        )

    def segment_with_points(
        self,
        image: NDArray[np.uint8],
        point_sets: list[list[PointPrompt]],
    ) -> BatchSegmentationResult:
        """Segment instances using foreground/background point prompts."""
        if not self._is_loaded:
            raise RuntimeError("MobileSAM not loaded. Call .load() first.")

        if len(point_sets) == 0:
            return BatchSegmentationResult(
                masks=[],
                image_height=image.shape[0],
                image_width=image.shape[1],
                backend="mobile_sam",
            )

        import torch

        t0 = time.perf_counter()
        h, w = image.shape[:2]

        self._predictor.set_image(image)

        results = []
        for points in point_sets:
            coords = np.array([[p.x, p.y] for p in points], dtype=np.float32)
            labels = np.array([p.label for p in points], dtype=np.int32)

            with torch.no_grad():
                masks, scores, _ = self._predictor.predict(
                    point_coords=coords,
                    point_labels=labels,
                    multimask_output=self.config.multimask_output,
                )

            mask_list = [masks[i].astype(bool) for i in range(len(masks))]
            score_list = scores.tolist()

            best_mask, best_score = self._select_best_mask(mask_list, score_list)
            clean_mask = self._postprocess_mask(best_mask, h, w)

            results.append(SegmentationResult(mask=clean_mask, score=float(best_score)))

        elapsed_ms = (time.perf_counter() - t0) * 1000
        return BatchSegmentationResult(
            masks=results,
            image_height=h,
            image_width=w,
            backend="mobile_sam",
            inference_time_ms=elapsed_ms,
        )

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    @staticmethod
    def _download_checkpoint(path: Path) -> None:
        """Download MobileSAM weights."""
        import urllib.request

        path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading MobileSAM from %s → %s", MOBILE_SAM_CHECKPOINT_URL, path)
        urllib.request.urlretrieve(MOBILE_SAM_CHECKPOINT_URL, str(path))
        logger.info("MobileSAM download complete: %.1f MB", path.stat().st_size / 1e6)

    @property
    def vram_required_gb(self) -> float:
        return 0.05  # ~38 MB, essentially free
