"""
segmentation.infrastructure.sam2_backend — SAM2-tiny segmentation backend.

Uses the Segment Anything Model 2 (SAM2) from Meta AI.
Default model: sam2-hiera-tiny (~38.9M parameters, 3.8 GB VRAM).

Install:
    pip install 'git+https://github.com/facebookresearch/segment-anything-2.git'

Reference:
    Ravi, N. et al. (2024). SAM 2: Segment Anything in Images and Videos.
    arXiv:2408.00714.
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


# SAM2 model registry
SAM2_MODELS = {
    "tiny": {
        "config": "sam2_hiera_tiny.yaml",
        "checkpoint": "sam2_hiera_tiny.pt",
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt",
        "vram_gb": 3.8,
        "params_m": 38.9,
    },
    "small": {
        "config": "sam2_hiera_small.yaml",
        "checkpoint": "sam2_hiera_small.pt",
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt",
        "vram_gb": 4.5,
        "params_m": 46.0,
    },
    "base_plus": {
        "config": "sam2_hiera_b+.yaml",
        "checkpoint": "sam2_hiera_b+.pt",
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_b+.pt",
        "vram_gb": 5.6,
        "params_m": 80.8,
    },
}


class SAM2TinyBackend(BaseSegmentor):
    """
    SAM2-tiny segmentation backend.

    Recommended for RTX 4060 (8 GB VRAM):
    - SAM2-tiny: 3.8 GB VRAM — leaves room for detection model (~1.2 GB)
    - Total typical: ~5.0 GB (detection + segmentation)

    Usage::

        backend = SAM2TinyBackend(config)
        backend.load()
        result = backend.segment_with_boxes(image, boxes)
        backend.unload()
    """

    def __init__(
        self,
        config: SegmentorConfig | None = None,
        model_variant: str = "tiny",
        checkpoint_path: str | None = None,
    ) -> None:
        super().__init__(config)
        self.model_variant = model_variant
        self.checkpoint_path = checkpoint_path
        self._predictor: Any | None = None
        self._model_info = SAM2_MODELS.get(model_variant, SAM2_MODELS["tiny"])

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load SAM2 model into GPU memory."""
        if self._is_loaded:
            return

        logger.info(
            "Loading SAM2-%s (%s GB VRAM)",
            self.model_variant,
            self._model_info["vram_gb"],
        )
        t0 = time.monotonic()

        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError as e:
            raise ImportError(
                "SAM2 not installed. Install: "
                "pip install 'git+https://github.com/facebookresearch/segment-anything-2.git'"
            ) from e

        import torch

        ckpt = self.checkpoint_path or self._model_info["checkpoint"]
        cfg = self._model_info["config"]

        # Download if checkpoint not present
        ckpt_path = Path(ckpt)
        if not ckpt_path.exists():
            self._download_checkpoint(ckpt_path)

        device = self.config.device
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU (will be slow)")
            device = "cpu"

        sam2_model = build_sam2(cfg, str(ckpt_path), device=device)
        self._predictor = SAM2ImagePredictor(sam2_model)

        self._is_loaded = True
        elapsed = time.monotonic() - t0
        logger.info("SAM2-%s loaded in %.1fs", self.model_variant, elapsed)

    def unload(self) -> None:
        """Release SAM2 and free GPU memory."""
        if not self._is_loaded:
            return

        del self._predictor
        self._predictor = None

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        gc.collect()
        self._is_loaded = False
        logger.info("SAM2 unloaded")

    def __enter__(self) -> "SAM2TinyBackend":
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
        """
        Segment trichome instances using bounding box prompts.

        Args:
            image: HWC uint8 RGB numpy array.
            boxes: Bounding boxes from detector (one per trichome).

        Returns:
            BatchSegmentationResult with binary masks.
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call .load() first.")

        if len(boxes) == 0:
            return BatchSegmentationResult(
                masks=[],
                image_height=image.shape[0],
                image_width=image.shape[1],
                backend="sam2_tiny",
            )

        import torch

        t0 = time.perf_counter()
        h, w = image.shape[:2]

        # Set image in predictor
        self._predictor.set_image(image)

        results = []

        # Process in batches of 16 boxes for memory efficiency
        BATCH_SIZE = 16
        box_arrays = np.array(
            [[b.x1, b.y1, b.x2, b.y2] for b in boxes], dtype=np.float32
        )

        for batch_start in range(0, len(boxes), BATCH_SIZE):
            batch_boxes = box_arrays[batch_start : batch_start + BATCH_SIZE]
            batch_prompts = boxes[batch_start : batch_start + BATCH_SIZE]

            with torch.no_grad():
                masks, scores, _ = self._predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=batch_boxes,
                    multimask_output=self.config.multimask_output,
                )

            # masks shape: (N, num_masks_per_prompt, H, W) or (num_masks_per_prompt, H, W)
            if masks.ndim == 3:
                # Single box was passed — add batch dim
                masks = masks[np.newaxis]
                scores = scores[np.newaxis]

            for i, (box_masks, box_scores) in enumerate(zip(masks, scores)):
                # box_masks: (num_masks, H, W)
                # box_scores: (num_masks,)
                mask_list = [box_masks[j].astype(bool) for j in range(len(box_masks))]
                score_list = box_scores.tolist()

                best_mask, best_score = self._select_best_mask(mask_list, score_list)

                # Post-process
                clean_mask = self._postprocess_mask(best_mask, h, w)

                results.append(
                    SegmentationResult(
                        mask=clean_mask,
                        score=float(best_score),
                        source_prompt=batch_prompts[i],
                    )
                )

        elapsed_ms = (time.perf_counter() - t0) * 1000
        return BatchSegmentationResult(
            masks=results,
            image_height=h,
            image_width=w,
            backend="sam2_tiny",
            inference_time_ms=elapsed_ms,
        )

    def segment_with_points(
        self,
        image: NDArray[np.uint8],
        point_sets: list[list[PointPrompt]],
    ) -> BatchSegmentationResult:
        """
        Segment instances using foreground/background point prompts.

        Args:
            image: HWC uint8 RGB numpy array.
            point_sets: List of point lists (one per object).

        Returns:
            BatchSegmentationResult with binary masks.
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call .load() first.")

        if len(point_sets) == 0:
            return BatchSegmentationResult(
                masks=[],
                image_height=image.shape[0],
                image_width=image.shape[1],
                backend="sam2_tiny",
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

            mask_list = [masks[j].astype(bool) for j in range(len(masks))]
            score_list = scores.tolist()

            best_mask, best_score = self._select_best_mask(mask_list, score_list)
            clean_mask = self._postprocess_mask(best_mask, h, w)

            results.append(
                SegmentationResult(
                    mask=clean_mask,
                    score=float(best_score),
                )
            )

        elapsed_ms = (time.perf_counter() - t0) * 1000
        return BatchSegmentationResult(
            masks=results,
            image_height=h,
            image_width=w,
            backend="sam2_tiny",
            inference_time_ms=elapsed_ms,
        )

    def segment_everything(
        self,
        image: NDArray[np.uint8],
        points_per_side: int = 32,
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.95,
    ) -> BatchSegmentationResult:
        """
        Automatic segmentation without prompts (grid-based).

        Useful for: discovering all trichomes without a detection step.
        Slower than prompted segmentation (~2-4s per image on RTX 4060).

        Args:
            image: HWC uint8 RGB numpy array.
            points_per_side: Grid density (32 = 1024 points per image).
            pred_iou_thresh: SAM2 IoU prediction threshold.
            stability_score_thresh: Mask stability threshold.
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call .load() first.")

        try:
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        except ImportError as e:
            raise ImportError("SAM2 automatic mask generator not available") from e

        t0 = time.perf_counter()
        h, w = image.shape[:2]

        mask_gen = SAM2AutomaticMaskGenerator(
            model=self._predictor.model,
            points_per_side=points_per_side,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
        )

        raw_masks = mask_gen.generate(image)

        results = []
        for mask_data in raw_masks:
            mask = mask_data["segmentation"].astype(bool)
            score = float(mask_data.get("predicted_iou", mask_data.get("stability_score", 0.5)))
            clean_mask = self._postprocess_mask(mask, h, w)
            if clean_mask.any():
                results.append(SegmentationResult(mask=clean_mask, score=score))

        elapsed_ms = (time.perf_counter() - t0) * 1000
        return BatchSegmentationResult(
            masks=results,
            image_height=h,
            image_width=w,
            backend="sam2_tiny_auto",
            inference_time_ms=elapsed_ms,
        )

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    @staticmethod
    def _download_checkpoint(path: Path) -> None:
        """Download SAM2 checkpoint from Meta AI servers."""
        import urllib.request

        variant = "tiny"  # default
        for v, info in SAM2_MODELS.items():
            if info["checkpoint"] in str(path):
                variant = v
                break

        url = SAM2_MODELS[variant]["url"]
        path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Downloading SAM2 checkpoint from %s → %s", url, path)
        urllib.request.urlretrieve(url, str(path))
        logger.info("Download complete: %s (%.1f MB)", path, path.stat().st_size / 1e6)

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    @property
    def vram_required_gb(self) -> float:
        return float(self._model_info.get("vram_gb", 3.8))
