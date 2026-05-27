"""
video_pipeline.application.video_pipeline — Complete video analysis pipeline.

Pipeline:
    Input video
    → Frame extraction (adaptive or fixed rate)
    → Quality scoring (blur + exposure + noise)
    → Temporal deduplication (perceptual hash)
    → Stabilization metadata
    → Best frame selection
    → Per-frame analysis (detection + optional segmentation)
    → Temporal aggregation
    → Output: best frames + analysis results

DESIGN FOR RTX 4060 (8 GB VRAM):
- Frame extraction: CPU-only (ffmpeg)
- Quality scoring: CPU-only (OpenCV)
- Detection: GPU, one frame at a time or small batches
- No temporal super-resolution (too much VRAM)
- Max recommended: 4K input, extract 1080p for analysis

MEMORY MANAGEMENT:
Never load all frames into memory simultaneously.
Use a streaming approach: extract → score → select → analyze → discard.
For a 2-minute 4K video at 30fps = 3600 frames × 8MB = 28 GB (impossible on 16GB RAM).
Use frame-by-frame streaming with a rolling buffer.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generator

import cv2
import numpy as np
from numpy.typing import NDArray

from focus.metrics.composite import compute_focus_score, FocusScoreResult, rank_frames_by_focus
from shared.logging.logger import get_logger

logger = get_logger(__name__)


@dataclass
class FrameMetadata:
    """Metadata for a single extracted video frame."""

    frame_index: int
    timestamp_ms: float
    """Position in video (milliseconds)."""

    focus_score: FocusScoreResult | None = None
    quality_score: float = 0.0
    """Combined quality score [0,1] combining focus + exposure + noise."""

    perceptual_hash: str = ""
    """For near-duplicate detection."""

    is_keyframe: bool = False
    is_selected: bool = False
    """True if this frame is selected as a best frame."""

    exposure_score: float = 0.0
    """Exposure quality: 0=over/underexposed, 1=well-exposed."""

    motion_score: float = 0.0
    """Estimated motion blur score. Lower = more motion blur."""

    def compute_quality(self) -> float:
        """Compute combined quality score from all metrics."""
        if self.focus_score is None:
            return 0.0
        focus = self.focus_score.composite
        exposure = self.exposure_score
        motion = 1.0 - min(self.motion_score, 1.0)
        self.quality_score = float(0.50 * focus + 0.30 * exposure + 0.20 * motion)
        return self.quality_score


@dataclass
class VideoPipelineConfig:
    """Configuration for the video analysis pipeline."""

    # Extraction
    extraction_fps: float | None = None
    """
    Target extraction rate. None = extract all frames.
    For quality selection, extracting 5-10 fps is usually sufficient.
    Full 30fps extraction for fine-grained analysis.
    """

    max_frames: int = 5000
    """Maximum frames to extract (safety limit for memory)."""

    resize_to: tuple[int, int] | None = (1920, 1080)
    """Resize extracted frames. None = keep original resolution."""

    # Quality filtering
    min_focus_score: float = 0.30
    """Discard frames with composite focus score below this."""

    min_exposure_score: float = 0.25
    """Discard severely over/underexposed frames."""

    # Deduplication
    dedup_hash_threshold: int = 8
    """
    Maximum Hamming distance between perceptual hashes to consider frames duplicates.
    0 = exact duplicates only
    8 = near-duplicates (typical for slowly moving microscope)
    """

    # Best frame selection
    num_best_frames: int = 20
    """Number of best frames to select for detailed analysis."""

    temporal_diversity_min_gap_ms: float = 1000.0
    """
    Minimum time gap between selected best frames (milliseconds).
    Prevents selecting N similar frames from the same moment.
    """

    # Processing
    batch_size: int = 8
    """Number of frames to process per batch for GPU inference."""

    device: str = "cuda:0"


@dataclass
class VideoPipelineResult:
    """Complete result from video analysis pipeline."""

    video_path: Path
    total_frames: int
    extracted_frames: int
    quality_passed_frames: int
    deduplicated_frames: int
    selected_frames: int

    frame_metadata: list[FrameMetadata] = field(default_factory=list)
    best_frame_paths: list[Path] = field(default_factory=list)
    best_frame_indices: list[int] = field(default_factory=list)

    # Statistics
    mean_focus_score: float = 0.0
    mean_quality_score: float = 0.0
    processing_time_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "video": str(self.video_path),
            "frames": {
                "total": self.total_frames,
                "extracted": self.extracted_frames,
                "quality_passed": self.quality_passed_frames,
                "deduplicated": self.deduplicated_frames,
                "selected": self.selected_frames,
            },
            "quality": {
                "mean_focus_score": self.mean_focus_score,
                "mean_quality_score": self.mean_quality_score,
            },
            "best_frames": self.best_frame_indices,
            "processing_time_s": self.processing_time_s,
        }


class VideoPipeline:
    """
    Complete video analysis pipeline.

    Memory-efficient streaming implementation — never holds all frames in RAM.
    Processes frames in chunks, keeping only the top-N best frames.
    """

    def __init__(self, config: VideoPipelineConfig | None = None) -> None:
        self._config = config or VideoPipelineConfig()

    def run(
        self,
        video_path: Path | str,
        output_dir: Path | str | None = None,
    ) -> VideoPipelineResult:
        """
        Run full video analysis pipeline.

        Args:
            video_path: Path to input video file.
            output_dir: Directory to save best frames. None = don't save.

        Returns:
            VideoPipelineResult with selected frames and statistics.
        """
        import time

        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        t_start = time.perf_counter()
        logger.info("Starting video pipeline", video=str(video_path))

        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        native_fps = cap.get(cv2.CAP_PROP_FPS)
        logger.info(
            "Video info",
            total_frames=total_frames,
            fps=native_fps,
        )

        # Compute frame step for extraction rate
        if self._config.extraction_fps and native_fps > 0:
            frame_step = max(1, int(native_fps / self._config.extraction_fps))
        else:
            frame_step = 1

        # Stream through frames
        all_metadata: list[FrameMetadata] = []
        seen_hashes: set[str] = set()
        extracted = 0
        quality_passed = 0
        deduped = 0

        frame_idx = 0
        while extracted < self._config.max_frames:
            # Skip frames according to frame_step
            if frame_step > 1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx * frame_step)

            ret, frame_bgr = cap.read()
            if not ret:
                break

            frame_idx += 1
            actual_frame_idx = (frame_idx - 1) * frame_step
            timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)

            # Resize if needed
            if self._config.resize_to:
                frame_bgr = cv2.resize(frame_bgr, self._config.resize_to)

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            extracted += 1

            # Quality scoring
            meta = FrameMetadata(
                frame_index=actual_frame_idx,
                timestamp_ms=timestamp_ms,
            )

            # Focus scoring
            focus_result = compute_focus_score(frame_rgb)
            meta.focus_score = focus_result

            # Exposure scoring
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            meta.exposure_score = self._compute_exposure_score(gray)

            # Quality gate
            if focus_result.composite < self._config.min_focus_score:
                continue
            if meta.exposure_score < self._config.min_exposure_score:
                continue
            quality_passed += 1

            # Perceptual hash for deduplication
            phash = self._perceptual_hash(gray)
            meta.perceptual_hash = phash

            # Deduplication check
            if self._is_duplicate(phash, seen_hashes):
                continue
            seen_hashes.add(phash)
            deduped += 1

            meta.compute_quality()
            all_metadata.append(meta)

        cap.release()

        # Select best frames with temporal diversity
        best_metadata = self._select_best_frames(all_metadata)

        # Save best frames
        best_paths: list[Path] = []
        if output_dir and best_metadata:
            cap2 = cv2.VideoCapture(str(video_path))
            for meta in best_metadata:
                cap2.set(cv2.CAP_PROP_POS_FRAMES, meta.frame_index)
                ret, frame = cap2.read()
                if ret:
                    if self._config.resize_to:
                        frame = cv2.resize(frame, self._config.resize_to)
                    out_path = output_dir / f"best_frame_{meta.frame_index:06d}.jpg"
                    cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    best_paths.append(out_path)
                    meta.is_selected = True
            cap2.release()

        t_end = time.perf_counter()

        # Compute statistics
        quality_scores = [m.quality_score for m in all_metadata]
        focus_scores = [m.focus_score.composite for m in all_metadata if m.focus_score]

        result = VideoPipelineResult(
            video_path=video_path,
            total_frames=total_frames,
            extracted_frames=extracted,
            quality_passed_frames=quality_passed,
            deduplicated_frames=deduped,
            selected_frames=len(best_metadata),
            frame_metadata=all_metadata,
            best_frame_paths=best_paths,
            best_frame_indices=[m.frame_index for m in best_metadata],
            mean_focus_score=float(np.mean(focus_scores)) if focus_scores else 0.0,
            mean_quality_score=float(np.mean(quality_scores)) if quality_scores else 0.0,
            processing_time_s=t_end - t_start,
        )

        logger.info(
            "Video pipeline complete",
            extracted=extracted,
            quality_passed=quality_passed,
            deduped=deduped,
            selected=len(best_metadata),
            time_s=f"{t_end - t_start:.1f}",
        )

        return result

    def _select_best_frames(
        self,
        metadata: list[FrameMetadata],
    ) -> list[FrameMetadata]:
        """
        Select top-N frames with temporal diversity.

        Prevents selecting many similar frames from a single sharp moment.
        Ensures temporal spread across the video.
        """
        if not metadata:
            return []

        # Sort by quality score
        sorted_meta = sorted(metadata, key=lambda m: m.quality_score, reverse=True)

        selected: list[FrameMetadata] = []
        selected_times: list[float] = []
        min_gap = self._config.temporal_diversity_min_gap_ms

        for meta in sorted_meta:
            if len(selected) >= self._config.num_best_frames:
                break

            # Check temporal diversity
            too_close = any(
                abs(meta.timestamp_ms - t) < min_gap
                for t in selected_times
            )
            if too_close and len(selected) > 0:
                continue

            selected.append(meta)
            selected_times.append(meta.timestamp_ms)

        # Sort selected by timestamp for chronological output
        return sorted(selected, key=lambda m: m.timestamp_ms)

    @staticmethod
    def _compute_exposure_score(gray: NDArray[np.uint8]) -> float:
        """
        Score exposure quality.

        Penalizes over/underexposed images.
        Optimal: mean intensity 80-180 (on 0-255 scale).
        """
        mean = float(gray.mean())
        p2 = float(np.percentile(gray, 2))
        p98 = float(np.percentile(gray, 98))

        # Penalize if too dark or too bright
        brightness_score = float(np.clip(1 - abs(mean - 128) / 128, 0, 1))

        # Penalize if dynamic range is clipped
        clipping_penalty = 0.0
        if p2 < 5:
            clipping_penalty += 0.3  # Underexposed (clipped shadows)
        if p98 > 250:
            clipping_penalty += 0.3  # Overexposed (blown highlights)

        return float(np.clip(brightness_score - clipping_penalty, 0, 1))

    def _perceptual_hash(self, gray: NDArray[np.uint8]) -> str:
        """
        Compute 64-bit average perceptual hash.

        Two frames with Hamming distance < threshold are considered duplicates.
        Much faster than pixel-wise comparison.
        """
        small = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA)
        mean = small.mean()
        bits = (small > mean).flatten()
        hash_val = 0
        for bit in bits:
            hash_val = (hash_val << 1) | int(bit)
        return format(hash_val, "016x")

    def _is_duplicate(self, phash: str, seen: set[str]) -> bool:
        """Check if hash is within Hamming distance threshold of any seen hash."""
        threshold = self._config.dedup_hash_threshold
        if threshold == 0:
            return phash in seen

        new_val = int(phash, 16)
        for seen_hash in seen:
            seen_val = int(seen_hash, 16)
            hamming = bin(new_val ^ seen_val).count("1")
            if hamming <= threshold:
                return True
        return False
