"""
vlm_labeling.application.auto_label_pipeline — Auto-labeling orchestration pipeline.

PIPELINE FLOW:
    Input images (unlabeled)
    → Quality screening (fast VLM/rule check)
    → Auto-labeling (VLM inference)
    → Hallucination filtering
    → Confidence scoring
    → Review queue (human-in-loop)

HUMAN-IN-LOOP INVARIANT:
VLM pseudo-labels NEVER directly enter the training dataset.
They always go through the human review queue first.
This is architecturally enforced by:
  1. AnnotationSource.VLM_AUTO flag on all VLM outputs
  2. Review queue requiring explicit human approval before promotion
  3. Training data loader rejecting VLM_AUTO unless reviewed=True

THROUGHPUT ESTIMATES (RTX 4060, Moondream-2B 4-bit):
- Quality screening:   ~1.2 images/s
- Maturity labeling:   ~1.1 images/s
- Both tasks:          ~0.6 images/s (sequential due to GPU constraint)
- 1000 images:         ~28 minutes (single GPU, no parallelism)
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from numpy.typing import NDArray

from shared.core.enums import AnnotationSource, MaturityStage
from shared.logging.logger import get_logger
from vlm_labeling.prompts.trichome_prompts import PROMPT_REGISTRY
from vlm_labeling.filtering.hallucination import (
    HallucinationFilter,
    HallucinationFilterConfig,
    FilterResult,
)

logger = get_logger(__name__)


class PseudoLabelStatus(str, Enum):
    """Status of an auto-generated label."""

    PENDING_REVIEW = "pending_review"
    """VLM labeled successfully, awaiting human review."""

    REVIEW_APPROVED = "review_approved"
    """Human approved — can enter training dataset."""

    REVIEW_REJECTED = "review_rejected"
    """Human rejected — discarded."""

    REVIEW_CORRECTED = "review_corrected"
    """Human corrected the label before approving."""

    FAILED_QUALITY = "failed_quality"
    """Image quality too poor for labeling."""

    FAILED_INFERENCE = "failed_inference"
    """VLM inference failed (model error)."""

    FLAGGED_HALLUCINATION = "flagged_hallucination"
    """Hallucination filter raised concern — high-priority review."""


@dataclass
class PseudoLabel:
    """
    A VLM-generated pseudo-label for a single image.

    This is the unit of work in the auto-label pipeline.
    All fields are populated by the pipeline; review fields
    are filled in by the human annotation interface.
    """

    label_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Source
    image_path: str = ""
    image_id: str = ""

    # VLM outputs
    prompt_used: str = ""
    raw_vlm_response: str = ""
    parsed_vlm_response: dict[str, Any] | None = None
    vlm_model: str = ""
    vlm_confidence: float = 0.0

    # Filter result
    filter_result: FilterResult | None = None
    hallucination_flags: list[str] = field(default_factory=list)

    # Derived labels
    maturity_stage: str | None = None
    """Extracted from parsed_vlm_response."""

    dominant_morphology: str | None = None
    image_quality: str | None = None
    amber_fraction: float | None = None
    cloudy_fraction: float | None = None
    clear_fraction: float | None = None

    # Status tracking
    status: PseudoLabelStatus = PseudoLabelStatus.PENDING_REVIEW
    annotation_source: str = AnnotationSource.VLM_AUTO.value

    # Timing
    inference_time_s: float = 0.0
    created_at: float = field(default_factory=time.time)

    # Human review
    reviewed: bool = False
    reviewer_id: str | None = None
    review_comment: str | None = None
    corrected_maturity: str | None = None
    """If reviewer corrects the label, this field holds the correction."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "label_id": self.label_id,
            "image_path": self.image_path,
            "image_id": self.image_id,
            "prompt_used": self.prompt_used,
            "vlm_model": self.vlm_model,
            "vlm_confidence": self.vlm_confidence,
            "maturity_stage": self.maturity_stage,
            "dominant_morphology": self.dominant_morphology,
            "image_quality": self.image_quality,
            "amber_fraction": self.amber_fraction,
            "cloudy_fraction": self.cloudy_fraction,
            "clear_fraction": self.clear_fraction,
            "hallucination_flags": self.hallucination_flags,
            "status": self.status.value,
            "annotation_source": self.annotation_source,
            "reviewed": self.reviewed,
            "inference_time_s": self.inference_time_s,
            "created_at": self.created_at,
        }

    @property
    def final_maturity(self) -> str | None:
        """Effective maturity stage: corrected if reviewed, original otherwise."""
        return self.corrected_maturity or self.maturity_stage

    @property
    def is_ready_for_training(self) -> bool:
        """True only if reviewed and approved."""
        return (
            self.reviewed
            and self.status in (
                PseudoLabelStatus.REVIEW_APPROVED,
                PseudoLabelStatus.REVIEW_CORRECTED,
            )
        )


@dataclass
class AutoLabelPipelineConfig:
    """Configuration for the auto-labeling pipeline."""

    # VLM settings
    vlm_backend: str = "moondream"
    """Which VLM to use: 'moondream', 'florence2', 'qwen2vl'"""

    run_quality_screen: bool = True
    """
    Run image quality screening before maturity classification.
    Adds ~1s per image but skips poor-quality images before expensive inference.
    """

    skip_unusable_images: bool = True
    """If quality screen flags 'unusable', skip maturity labeling entirely."""

    run_morphology_classification: bool = False
    """
    Also classify morphology type. Doubles inference time.
    Recommended: run separately in dedicated morphology labeling pass.
    """

    # Filtering
    min_vlm_confidence: float = 0.40
    """Labels with VLM confidence below this get LOW confidence flag."""

    enable_hallucination_filter: bool = True
    """Run hallucination detection on all VLM outputs."""

    # Output
    output_dir: str | None = None
    """Directory to save pseudo-labels JSON. None = memory only."""

    batch_size: int = 1
    """
    Images to process between saving results.
    Larger = fewer I/O operations, more memory.
    """

    # Performance
    max_images: int | None = None
    """Limit total images processed (useful for testing)."""


@dataclass
class PipelineStats:
    """Aggregate statistics from auto-label pipeline run."""

    total_processed: int = 0
    quality_failed: int = 0
    inference_failed: int = 0
    hallucination_flagged: int = 0
    pending_review: int = 0
    mean_confidence: float = 0.0
    total_time_s: float = 0.0

    @property
    def success_rate(self) -> float:
        if self.total_processed == 0:
            return 0.0
        failed = self.quality_failed + self.inference_failed
        return (self.total_processed - failed) / self.total_processed

    @property
    def throughput_per_minute(self) -> float:
        if self.total_time_s == 0:
            return 0.0
        return self.total_processed / (self.total_time_s / 60)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_processed": self.total_processed,
            "quality_failed": self.quality_failed,
            "inference_failed": self.inference_failed,
            "hallucination_flagged": self.hallucination_flagged,
            "pending_review": self.pending_review,
            "success_rate": self.success_rate,
            "mean_confidence": self.mean_confidence,
            "total_time_s": self.total_time_s,
            "throughput_per_minute": self.throughput_per_minute,
        }


class AutoLabelPipeline:
    """
    Full auto-labeling pipeline: image → VLM → filter → review queue.

    Usage:
        pipeline = AutoLabelPipeline(
            config=AutoLabelPipelineConfig(vlm_backend="moondream"),
        )
        pipeline.load()

        results, stats = pipeline.run(image_paths)

        # All results start as PENDING_REVIEW
        for label in results:
            print(label.maturity_stage, label.vlm_confidence, label.status)

        pipeline.unload()
    """

    def __init__(
        self,
        config: AutoLabelPipelineConfig | None = None,
    ) -> None:
        self._config = config or AutoLabelPipelineConfig()
        self._vlm_labeler: Any = None
        self._hallucination_filter: HallucinationFilter | None = None
        self._is_loaded = False

    def load(self) -> None:
        """Load VLM backend and initialize filters."""
        if self._is_loaded:
            return

        logger.info(
            "Loading auto-label pipeline",
            backend=self._config.vlm_backend,
        )

        self._vlm_labeler = self._create_vlm_labeler()
        self._vlm_labeler.load()

        if self._config.enable_hallucination_filter:
            self._hallucination_filter = HallucinationFilter(
                HallucinationFilterConfig(
                    min_confidence=self._config.min_vlm_confidence,
                )
            )

        self._is_loaded = True
        logger.info("Auto-label pipeline loaded")

    def unload(self) -> None:
        """Unload VLM and free VRAM."""
        if self._vlm_labeler is not None:
            self._vlm_labeler.unload()
        self._is_loaded = False

    def run(
        self,
        image_paths: list[Path | str],
        image_arrays: list[NDArray[np.uint8]] | None = None,
    ) -> tuple[list[PseudoLabel], PipelineStats]:
        """
        Process a list of images through the full auto-label pipeline.

        Args:
            image_paths: List of image file paths.
            image_arrays: Optional pre-loaded image arrays (avoids disk I/O).
                         Must match image_paths length if provided.

        Returns:
            (list of PseudoLabel, PipelineStats)
        """
        if not self._is_loaded:
            raise RuntimeError("Pipeline not loaded. Call load() first.")

        from shared.utils.image_utils import load_image

        all_labels: list[PseudoLabel] = []
        confidences: list[float] = []
        t_start = time.perf_counter()

        max_images = self._config.max_images
        paths_to_process = image_paths[:max_images] if max_images else image_paths

        logger.info(
            "Starting auto-labeling",
            num_images=len(paths_to_process),
            backend=self._config.vlm_backend,
        )

        for i, img_path in enumerate(paths_to_process):
            img_path = Path(img_path)

            # Load image
            if image_arrays is not None and i < len(image_arrays):
                image = image_arrays[i]
            else:
                try:
                    image = load_image(str(img_path))
                except Exception as e:
                    logger.warning("Failed to load image", path=str(img_path), error=str(e))
                    label = PseudoLabel(
                        image_path=str(img_path),
                        image_id=img_path.stem,
                        status=PseudoLabelStatus.FAILED_INFERENCE,
                    )
                    all_labels.append(label)
                    continue

            # Process single image
            label = self._process_image(image, img_path)
            all_labels.append(label)

            if label.vlm_confidence > 0:
                confidences.append(label.vlm_confidence)

            # Log progress
            if (i + 1) % 50 == 0:
                elapsed = time.perf_counter() - t_start
                logger.info(
                    "Progress",
                    processed=i + 1,
                    total=len(paths_to_process),
                    elapsed_s=f"{elapsed:.0f}",
                )

        # Save results if output_dir configured
        if self._config.output_dir:
            self._save_results(all_labels, Path(self._config.output_dir))

        t_end = time.perf_counter()

        stats = PipelineStats(
            total_processed=len(all_labels),
            quality_failed=sum(1 for l in all_labels if l.status == PseudoLabelStatus.FAILED_QUALITY),
            inference_failed=sum(1 for l in all_labels if l.status == PseudoLabelStatus.FAILED_INFERENCE),
            hallucination_flagged=sum(1 for l in all_labels if l.status == PseudoLabelStatus.FLAGGED_HALLUCINATION),
            pending_review=sum(1 for l in all_labels if l.status == PseudoLabelStatus.PENDING_REVIEW),
            mean_confidence=float(np.mean(confidences)) if confidences else 0.0,
            total_time_s=t_end - t_start,
        )

        logger.info(
            "Auto-labeling complete",
            **stats.to_dict(),
        )

        return all_labels, stats

    def _process_image(
        self,
        image: NDArray[np.uint8],
        image_path: Path,
    ) -> PseudoLabel:
        """Process a single image through the full pipeline."""
        label = PseudoLabel(
            image_path=str(image_path),
            image_id=image_path.stem,
            vlm_model=self._config.vlm_backend,
        )

        # Step 1: Quality screening
        if self._config.run_quality_screen:
            quality_result = self._vlm_labeler.assess_quality(image)
            label.raw_vlm_response = quality_result.raw_response

            if quality_result.is_valid and quality_result.parsed_response:
                label.image_quality = quality_result.overall_quality

                if (
                    self._config.skip_unusable_images
                    and quality_result.overall_quality == "unusable"
                ):
                    label.status = PseudoLabelStatus.FAILED_QUALITY
                    logger.debug(
                        "Image flagged unusable, skipping",
                        path=str(image_path),
                    )
                    return label

                if not quality_result.is_analyzable:
                    label.status = PseudoLabelStatus.FAILED_QUALITY
                    return label

        # Step 2: Maturity classification
        t_infer_start = time.perf_counter()
        maturity_result = self._vlm_labeler.label_maturity(image)
        label.inference_time_s = time.perf_counter() - t_infer_start
        label.prompt_used = "maturity_classification"
        label.raw_vlm_response = maturity_result.raw_response

        if not maturity_result.is_valid or maturity_result.parsed_response is None:
            label.status = PseudoLabelStatus.FAILED_INFERENCE
            logger.debug("VLM inference failed", path=str(image_path))
            return label

        # Extract labels from parsed response
        resp = maturity_result.parsed_response
        label.parsed_vlm_response = resp
        label.maturity_stage = resp.get("maturity_stage")
        label.vlm_confidence = float(resp.get("confidence", 0.0))
        label.amber_fraction = resp.get("amber_fraction_estimate")
        label.cloudy_fraction = resp.get("cloudy_fraction_estimate")
        label.clear_fraction = resp.get("clear_fraction_estimate")

        # Step 3: Optional morphology classification
        if self._config.run_morphology_classification:
            morph_result = self._vlm_labeler.label_morphology(image)
            if morph_result.is_valid and morph_result.parsed_response:
                label.dominant_morphology = morph_result.parsed_response.get("dominant_type")

        # Step 4: Hallucination filtering
        if self._config.enable_hallucination_filter and self._hallucination_filter:
            filter_result = self._hallucination_filter.filter_maturity(resp)
            label.filter_result = filter_result
            label.hallucination_flags = filter_result.flag_names

            # Adjust confidence
            label.vlm_confidence = filter_result.adjusted_confidence

            if not filter_result.passed:
                label.status = PseudoLabelStatus.FLAGGED_HALLUCINATION
                logger.debug(
                    "Hallucination flags raised",
                    path=str(image_path),
                    flags=filter_result.flag_names,
                    priority=filter_result.review_priority,
                )
                return label

        # Step 5: Add to review queue with PENDING status
        label.status = PseudoLabelStatus.PENDING_REVIEW
        return label

    def _create_vlm_labeler(self) -> Any:
        """Instantiate the configured VLM backend."""
        backend = self._config.vlm_backend.lower()

        if backend == "moondream":
            from vlm_labeling.moondream.moondream_labeler import MoondreamLabeler
            return MoondreamLabeler()

        elif backend == "florence2":
            try:
                from vlm_labeling.florence2.florence_labeler import FlorenceLabeler
                return FlorenceLabeler()
            except ImportError:
                logger.warning("Florence-2 not available, falling back to Moondream")
                from vlm_labeling.moondream.moondream_labeler import MoondreamLabeler
                return MoondreamLabeler()

        elif backend == "qwen2vl":
            try:
                from vlm_labeling.qwen2vl.qwen_labeler import QwenVLLabeler
                return QwenVLLabeler()
            except ImportError:
                logger.warning("Qwen2-VL not available, falling back to Moondream")
                from vlm_labeling.moondream.moondream_labeler import MoondreamLabeler
                return MoondreamLabeler()

        else:
            raise ValueError(
                f"Unknown VLM backend: '{backend}'. "
                "Choose from: moondream, florence2, qwen2vl"
            )

    def _save_results(
        self,
        labels: list[PseudoLabel],
        output_dir: Path,
    ) -> None:
        """Save pseudo-labels to disk as JSON."""
        import json

        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"pseudo_labels_{timestamp}.json"

        data = {
            "vlm_backend": self._config.vlm_backend,
            "timestamp": timestamp,
            "count": len(labels),
            "labels": [l.to_dict() for l in labels],
        }

        with open(output_file, "w") as f:
            json.dump(data, f, indent=2, default=str)

        logger.info("Pseudo-labels saved", path=str(output_file), count=len(labels))


def get_review_queue(
    labels: list[PseudoLabel],
    sort_by_priority: bool = True,
) -> list[PseudoLabel]:
    """
    Filter labels to those requiring human review, sorted by priority.

    Args:
        labels: All generated pseudo-labels.
        sort_by_priority: If True, highest priority items first.

    Returns:
        Labels in PENDING_REVIEW or FLAGGED_HALLUCINATION status.
    """
    needs_review = [
        l for l in labels
        if l.status in (
            PseudoLabelStatus.PENDING_REVIEW,
            PseudoLabelStatus.FLAGGED_HALLUCINATION,
        )
    ]

    if sort_by_priority and needs_review:
        needs_review.sort(
            key=lambda l: (
                -(l.filter_result.review_priority if l.filter_result else 0),
                l.vlm_confidence,  # Lower confidence = review first
            )
        )

    return needs_review
