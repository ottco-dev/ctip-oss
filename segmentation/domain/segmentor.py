"""
segmentation.domain.segmentor — Abstract segmentor protocol and base class.

Defines the interface that all segmentation backends must implement.
Currently supported backends:
  - SAM2-tiny (primary): facebook/sam2-hiera-tiny — 3.8 GB VRAM
  - MobileSAM (fallback): ChaoningZhang/MobileSAM — 38 MB, CPU-capable

Prompt types:
  - Point prompts: (x, y, label) tuples, label=1 foreground, 0 background
  - Box prompts: (x1, y1, x2, y2) bounding boxes from detector
  - Automatic (SAM2): grid-based everything segmentation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class PointPrompt:
    """Foreground/background point prompt for SAM."""

    x: float
    y: float
    label: int  # 1 = foreground, 0 = background

    def as_array(self) -> NDArray[np.float32]:
        return np.array([[self.x, self.y]], dtype=np.float32)


@dataclass
class BoxPrompt:
    """Bounding box prompt (x1, y1, x2, y2)."""

    x1: float
    y1: float
    x2: float
    y2: float

    def as_array(self) -> NDArray[np.float32]:
        return np.array([self.x1, self.y1, self.x2, self.y2], dtype=np.float32)

    def center_point(self) -> PointPrompt:
        return PointPrompt(
            x=(self.x1 + self.x2) / 2,
            y=(self.y1 + self.y2) / 2,
            label=1,
        )


@dataclass
class SegmentationResult:
    """Result from a single segmentation prediction."""

    mask: NDArray[np.bool_]
    """Binary mask (H, W) for the segmented object."""

    score: float
    """Predicted quality score [0, 1]."""

    logits: NDArray[np.float32] | None = None
    """Low-res logits for refinement (optional, SAM2 only)."""

    source_prompt: BoxPrompt | PointPrompt | None = None
    """The prompt that generated this mask."""


@dataclass
class BatchSegmentationResult:
    """Collection of masks from all prompts."""

    masks: list[SegmentationResult]
    """One per detected instance."""

    image_height: int
    image_width: int

    backend: str = "unknown"
    """Backend used: 'sam2_tiny', 'mobile_sam', etc."""

    inference_time_ms: float = 0.0


@dataclass
class SegmentorConfig:
    """Shared configuration for segmentation backends."""

    device: str = "cuda"
    """Device for model inference."""

    score_threshold: float = 0.50
    """Minimum predicted quality score to accept a mask."""

    multimask_output: bool = True
    """Whether to return multiple masks per prompt (SAM2 feature)."""

    use_postprocessing: bool = True
    """Apply morphological cleanup after segmentation."""

    # Morphological post-processing
    erosion_radius: int = 1
    dilation_radius: int = 2
    min_mask_area_px: int = 25
    max_mask_area_fraction: float = 0.5


# ---------------------------------------------------------------------------
# Segmentor Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class Segmentor(Protocol):
    """
    Protocol defining the interface for all segmentation backends.

    All backends must implement:
    - load()
    - unload()
    - segment_with_boxes(image, boxes) -> BatchSegmentationResult
    - segment_with_points(image, points) -> BatchSegmentationResult

    Optional:
    - segment_everything(image) -> BatchSegmentationResult
    """

    def load(self) -> None:
        """Load model weights into memory (GPU or CPU)."""
        ...

    def unload(self) -> None:
        """Release model weights and free memory."""
        ...

    @property
    def is_loaded(self) -> bool:
        """True if model is in memory and ready for inference."""
        ...

    def segment_with_boxes(
        self,
        image: NDArray[np.uint8],
        boxes: list[BoxPrompt],
    ) -> BatchSegmentationResult:
        """
        Segment each object defined by a bounding box.

        Args:
            image: HWC uint8 numpy array (RGB).
            boxes: List of BoxPrompt objects from detector.

        Returns:
            BatchSegmentationResult with one SegmentationResult per box.
        """
        ...

    def segment_with_points(
        self,
        image: NDArray[np.uint8],
        point_sets: list[list[PointPrompt]],
    ) -> BatchSegmentationResult:
        """
        Segment each object defined by point prompts.

        Args:
            image: HWC uint8 numpy array (RGB).
            point_sets: List of point lists, one per object.

        Returns:
            BatchSegmentationResult with one SegmentationResult per point set.
        """
        ...


# ---------------------------------------------------------------------------
# Base segmentor (shared utilities)
# ---------------------------------------------------------------------------

class BaseSegmentor:
    """
    Base class providing shared post-processing utilities.

    Backends inherit from this to get mask cleanup for free.
    """

    def __init__(self, config: SegmentorConfig | None = None) -> None:
        self.config = config or SegmentorConfig()
        self._is_loaded: bool = False

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def _postprocess_mask(
        self,
        mask: NDArray[np.bool_],
        image_height: int,
        image_width: int,
    ) -> NDArray[np.bool_]:
        """
        Apply morphological cleanup to binary mask.

        Operations:
        1. Erosion: removes thin connections and noise
        2. Dilation: restores boundary after erosion
        3. Fill holes: closes interior gaps
        4. Remove tiny components: < min_mask_area_px
        5. Remove oversized masks: > max_mask_area_fraction of image
        """
        if not self.config.use_postprocessing:
            return mask

        import cv2

        mask_u8 = mask.astype(np.uint8) * 255

        # Erosion + dilation (opening)
        if self.config.erosion_radius > 0:
            kernel_e = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (self.config.erosion_radius * 2 + 1, self.config.erosion_radius * 2 + 1),
            )
            mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel_e)

        if self.config.dilation_radius > 0:
            kernel_d = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (self.config.dilation_radius * 2 + 1, self.config.dilation_radius * 2 + 1),
            )
            mask_u8 = cv2.dilate(mask_u8, kernel_d)

        # Fill holes
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(mask_u8)
        cv2.drawContours(filled, contours, -1, 255, cv2.FILLED)
        mask_u8 = filled

        result = mask_u8 > 127

        # Size gates
        area = result.sum()
        if area < self.config.min_mask_area_px:
            return np.zeros((image_height, image_width), dtype=bool)

        max_area = image_height * image_width * self.config.max_mask_area_fraction
        if area > max_area:
            return np.zeros((image_height, image_width), dtype=bool)

        return result

    def _select_best_mask(
        self,
        masks: list[NDArray[np.bool_]],
        scores: list[float],
    ) -> tuple[NDArray[np.bool_], float]:
        """
        Select the best mask from multimask SAM2 output.

        Heuristic: prefer mask with highest score AND reasonable area.
        """
        if len(masks) == 0:
            raise ValueError("No masks to select from")
        if len(masks) == 1:
            return masks[0], scores[0]

        # Filter by score threshold
        valid = [
            (m, s)
            for m, s in zip(masks, scores)
            if s >= self.config.score_threshold
        ]
        if not valid:
            # Fall back to best score overall
            best_idx = int(np.argmax(scores))
            return masks[best_idx], scores[best_idx]

        # Among valid, pick best score
        valid.sort(key=lambda x: x[1], reverse=True)
        return valid[0]
