"""
detection.domain.detector — Abstract base detector interface.

Design:
All concrete detector implementations (YOLO, RTMDet, ensemble, etc.)
must implement TrichomeDetector. This allows pipeline components to
depend on the abstraction, not the implementation (Dependency Inversion).

Scientific context:
Trichome detection is a small-object detection problem.
Standard models trained on COCO (mean object size ~50×50px on 640px input)
underperform significantly on trichome data where:
- Target objects are 8-80px on 1920×1080 inputs
- Background similarity is high
- Negative space is cluttered (leaf tissue, hairs, debris)

Mitigation strategies implemented here:
1. Tiled inference: divide image into overlapping tiles
2. Multi-scale: run inference at multiple input resolutions
3. Hard example mining: identify missed small objects
4. Focal loss + small object anchors: in training

References:
  Zhu, C. et al. (2019). "Feature Selective Anchor-Free Module for
  Single-Shot Object Detection." CVPR 2019.

  Yang, F. et al. (2022). "QueryDet: Cascaded Sparse Query for
  Accelerating High-Resolution Small Object Detection." CVPR 2022.
"""

from __future__ import annotations

import abc
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from shared.core.entities import Detection
from shared.core.enums import ModelBackend, TrichomeType
from shared.core.value_objects import BoundingBox, Confidence


@dataclass
class DetectionConfig:
    """
    Configuration for a single detector run.

    All thresholds are purposefully conservative (low confidence threshold).
    Reason: In scientific contexts, we prefer false positives that humans
    can review over missed detections. False negatives (missed trichomes)
    distort density and maturity statistics more than false positives
    (which can be filtered in post-processing).
    """

    confidence_threshold: float = 0.25
    """
    Minimum confidence to include a detection.

    SCIENTIFIC NOTE: Lower = more detections = more false positives.
    Higher = fewer detections = more missed trichomes (false negatives).
    0.25 is a starting point; calibrate per microscope+lens combination.
    """

    iou_threshold: float = 0.45
    """
    IoU threshold for NMS. Lower = more aggressive suppression.

    For dense trichome fields, consider increasing to 0.55-0.65
    to preserve detections of overlapping/touching trichomes.
    """

    max_detections: int = 5000
    """Maximum detections per image. Safety cap for crowded images."""

    input_size: tuple[int, int] = (1280, 1280)
    """Model input resolution. Higher = slower but better small object detection."""

    backend: ModelBackend = ModelBackend.PYTORCH
    use_fp16: bool = True
    """Half-precision inference. ~2× speedup on modern GPUs, minimal accuracy loss."""

    tiled: bool = True
    """
    Enable tiled inference.
    RECOMMENDED for images larger than 2× the input_size.
    Essential for full-resolution microscopy images (4K+).
    """

    tile_size: int = 1280
    tile_overlap: float = 0.2
    """
    Tile overlap as fraction [0, 1].
    0.2 = 20% overlap prevents edge artifacts.
    Higher overlap = better boundary handling, higher compute cost.
    """

    augment: bool = False
    """
    Test-Time Augmentation (TTA).
    Runs inference on flipped/rotated versions and merges.
    Improves recall ~2-5% at ~3× compute cost.
    Only use for final evaluation, not live inference.
    """

    num_tta_variants: int = 4
    """Number of TTA augmentation variants (if augment=True)."""

    device: str = "cuda:0"
    warmup_runs: int = 3
    """Number of warmup inference runs (for reliable timing benchmarks)."""

    def __post_init__(self) -> None:
        if not (0.0 < self.confidence_threshold < 1.0):
            raise ValueError(
                f"confidence_threshold must be in (0, 1), got {self.confidence_threshold}"
            )
        if not (0.0 < self.iou_threshold < 1.0):
            raise ValueError(
                f"iou_threshold must be in (0, 1), got {self.iou_threshold}"
            )
        if not (0.0 <= self.tile_overlap < 0.5):
            raise ValueError(
                f"tile_overlap must be in [0, 0.5), got {self.tile_overlap}"
            )


@dataclass
class DetectionResult:
    """
    Complete result from a detector run.

    Contains not just the detections but also performance metrics,
    metadata, and diagnostics needed for scientific evaluation.
    """

    detections: list[Detection]
    image_id: str
    model_id: str
    inference_time_ms: float
    image_shape: tuple[int, int, int]

    # Preprocessing metadata
    preprocessing_time_ms: float = 0.0
    nms_time_ms: float = 0.0
    num_raw_detections: int = 0  # Before NMS
    num_tiles: int | None = None  # If tiled inference was used

    # Quality flags
    was_tiled: bool = False
    was_augmented: bool = False
    confidence_threshold_used: float = 0.25
    iou_threshold_used: float = 0.45

    # Optional per-tile diagnostics
    tile_results: list[dict[str, Any]] = field(default_factory=list)

    @property
    def num_detections(self) -> int:
        return len(self.detections)

    @property
    def total_time_ms(self) -> float:
        return self.preprocessing_time_ms + self.inference_time_ms + self.nms_time_ms

    @property
    def fps(self) -> float:
        return 1000.0 / self.total_time_ms if self.total_time_ms > 0 else 0.0

    @property
    def mean_confidence(self) -> float:
        if not self.detections:
            return 0.0
        return float(np.mean([float(d.confidence) for d in self.detections]))

    @property
    def high_confidence_count(self) -> int:
        return sum(1 for d in self.detections if d.is_high_confidence)

    @property
    def uncertain_count(self) -> int:
        return sum(1 for d in self.detections if d.is_uncertain)

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_id": self.image_id,
            "model_id": self.model_id,
            "num_detections": self.num_detections,
            "num_raw_detections": self.num_raw_detections,
            "mean_confidence": self.mean_confidence,
            "high_confidence_count": self.high_confidence_count,
            "uncertain_count": self.uncertain_count,
            "timing": {
                "preprocessing_ms": self.preprocessing_time_ms,
                "inference_ms": self.inference_time_ms,
                "nms_ms": self.nms_time_ms,
                "total_ms": self.total_time_ms,
                "fps": self.fps,
            },
            "config": {
                "was_tiled": self.was_tiled,
                "was_augmented": self.was_augmented,
                "num_tiles": self.num_tiles,
                "confidence_threshold": self.confidence_threshold_used,
                "iou_threshold": self.iou_threshold_used,
            },
            "detections": [d.to_dict() for d in self.detections],
        }


@runtime_checkable
class TrichomeDetector(Protocol):
    """
    Protocol (structural interface) for all trichome detectors.

    Any class implementing these methods is a valid detector,
    without requiring explicit inheritance.
    """

    @property
    def model_id(self) -> str:
        """Unique identifier for this model version."""
        ...

    @property
    def is_loaded(self) -> bool:
        """Whether the model weights are loaded into memory."""
        ...

    def load(self) -> None:
        """Load model weights. May be called lazily."""
        ...

    def unload(self) -> None:
        """Release GPU/CPU memory. Call when switching models."""
        ...

    def detect(
        self,
        image: NDArray[np.uint8],
        config: DetectionConfig | None = None,
    ) -> DetectionResult:
        """
        Run detection on a single image.

        Args:
            image: RGB image as uint8 numpy array, shape (H, W, 3)
            config: Detection configuration. Uses defaults if None.

        Returns:
            DetectionResult with all detections and metadata.
        """
        ...

    def detect_batch(
        self,
        images: list[NDArray[np.uint8]],
        config: DetectionConfig | None = None,
    ) -> list[DetectionResult]:
        """
        Run detection on a batch of images.

        Batch processing is significantly faster than sequential single-image
        inference due to GPU parallelism (typically 3-8× speedup for batch_size=8).
        """
        ...


class BaseDetector(abc.ABC):
    """
    Abstract base class for concrete detector implementations.

    Provides:
    - Timing utilities
    - Confidence calibration integration
    - Uncertainty estimation
    - Pre/post processing hooks
    - Logging and diagnostics

    Concrete classes must implement _run_inference().
    """

    def __init__(
        self,
        model_id: str,
        weights_path: Path,
        device: str = "cuda:0",
    ) -> None:
        self._model_id = model_id
        self._weights_path = weights_path
        self._device = device
        self._is_loaded = False
        self._default_config = DetectionConfig()
        self._calibrator: Any | None = None  # ConfidenceCalibrator

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    def device(self) -> str:
        return self._device

    @abc.abstractmethod
    def load(self) -> None:
        """Load model weights into memory."""
        ...

    @abc.abstractmethod
    def unload(self) -> None:
        """Release model from memory."""
        ...

    @abc.abstractmethod
    def _run_inference(
        self,
        image: NDArray[np.uint8],
        config: DetectionConfig,
    ) -> tuple[list[Detection], int]:
        """
        Run raw inference. Returns (detections_after_nms, num_raw_before_nms).

        Implementations MUST handle NMS internally.
        """
        ...

    def detect(
        self,
        image: NDArray[np.uint8],
        config: DetectionConfig | None = None,
    ) -> DetectionResult:
        """
        Full detection pipeline: preprocess → inference → postprocess.
        """
        if not self._is_loaded:
            self.load()

        cfg = config or self._default_config
        image_shape = image.shape if image.ndim == 3 else (*image.shape, 1)

        # Preprocessing
        t_pre_start = time.perf_counter()
        preprocessed = self._preprocess(image)
        t_pre_end = time.perf_counter()

        # Inference
        t_inf_start = time.perf_counter()
        detections, num_raw = self._run_inference(preprocessed, cfg)
        t_inf_end = time.perf_counter()

        # Post-processing (calibration, uncertainty)
        if self._calibrator is not None:
            detections = self._apply_calibration(detections)

        return DetectionResult(
            detections=detections,
            image_id="",  # Set by caller
            model_id=self._model_id,
            inference_time_ms=(t_inf_end - t_inf_start) * 1000,
            preprocessing_time_ms=(t_pre_end - t_pre_start) * 1000,
            image_shape=image_shape,
            num_raw_detections=num_raw,
            confidence_threshold_used=cfg.confidence_threshold,
            iou_threshold_used=cfg.iou_threshold,
        )

    def detect_batch(
        self,
        images: list[NDArray[np.uint8]],
        config: DetectionConfig | None = None,
    ) -> list[DetectionResult]:
        """Default batch implementation: sequential. Override for true batching."""
        return [self.detect(img, config) for img in images]

    def _preprocess(self, image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """
        Standard preprocessing pipeline.

        Steps:
        1. Validate input
        2. Convert to RGB if needed
        3. Denoising (conservative — we don't want to remove fine trichome detail)
        4. Contrast enhancement (CLAHE on L channel in LAB space)

        Note: We apply CLAHE to improve detection of low-contrast transparent
        trichomes without introducing artifacts. Parameters are tuned for
        microscopy images — use caution if adapting for other domains.
        """
        import cv2

        if image.ndim == 2:
            # Grayscale → RGB
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            # RGBA → RGB
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

        # Mild denoising — preserve fine structure
        # Bilateral filter preserves edges better than Gaussian for trichome detail
        denoised = cv2.bilateralFilter(image, d=5, sigmaColor=20, sigmaSpace=5)

        # CLAHE on L channel (LAB space) for uniform contrast enhancement
        lab = cv2.cvtColor(denoised, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l_channel)

        enhanced_lab = cv2.merge([l_enhanced, a_channel, b_channel])
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)

        return enhanced

    def _apply_calibration(self, detections: list[Detection]) -> list[Detection]:
        """Apply confidence calibration if calibrator is set."""
        if self._calibrator is None:
            return detections
        # Calibration is applied per-detection
        for det in detections:
            if det.raw_logit is not None:
                calibrated_score = self._calibrator.calibrate(det.raw_logit)
                det.calibrated_confidence = Confidence(calibrated_score)
        return detections

    def warmup(self, config: DetectionConfig | None = None) -> None:
        """
        Run warmup inference to initialize CUDA kernels.

        Essential for accurate timing benchmarks. First inference
        is always slower due to kernel compilation and memory allocation.
        """
        cfg = config or self._default_config
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        for _ in range(cfg.warmup_runs):
            self._run_inference(dummy, cfg)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"model_id={self._model_id}, "
            f"device={self._device}, "
            f"loaded={self._is_loaded})"
        )
