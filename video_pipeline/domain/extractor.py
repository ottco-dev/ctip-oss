"""
video_pipeline.domain.extractor — Streaming video frame extractor.

Extracts frames from video files using OpenCV without loading the entire
video into memory. Implements a configurable frame selection strategy.

EXTRACTION MODES:
1. Fixed rate:    Extract every N frames (e.g., every 30 = 1/sec at 30fps)
2. Adaptive:      Focus-aware extraction — keep extracting until N good frames
3. Keyframe only: Extract only scene-change keyframes (I-frames)
4. Time range:    Extract only frames in a specific time window

MEMORY STRATEGY:
Frames are yielded one at a time (generator pattern). The caller decides
whether to keep or discard each frame. Never holds more than one frame
in memory at extraction time.

HARDWARE ACCELERATION:
When CUDA is available, OpenCV's VideoCapture can use CUDA decoding.
Falls back to CPU automatically if unavailable.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, List, Optional, Tuple

import cv2
import numpy as np
from numpy.typing import NDArray


@dataclass
class FrameInfo:
    """Metadata for a single extracted frame."""

    frame_index: int
    """0-based frame index in the video."""

    timestamp_ms: float
    """Position in video in milliseconds."""

    timestamp_s: float
    """Position in video in seconds."""

    width: int
    height: int
    """Frame dimensions in pixels."""

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height if self.height > 0 else 1.0


@dataclass
class VideoInfo:
    """Metadata about a video file."""

    path: Path
    total_frames: int
    fps: float
    width: int
    height: int
    duration_s: float
    codec: str = ""

    @property
    def duration_ms(self) -> float:
        return self.duration_s * 1000.0

    @property
    def is_4k(self) -> bool:
        return self.width >= 3840

    @property
    def is_hd(self) -> bool:
        return self.width >= 1920

    def frames_for_interval(self, interval_s: float) -> int:
        """Number of frames to skip for a given time interval."""
        if self.fps <= 0:
            return 1
        return max(1, int(self.fps * interval_s))


def get_video_info(video_path: str | Path) -> VideoInfo:
    """
    Extract metadata from a video file.

    Args:
        video_path: Path to the video file.

    Returns:
        VideoInfo with all available metadata.

    Raises:
        FileNotFoundError: If the video file does not exist.
        ValueError: If OpenCV cannot open the file.
    """
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {path}")

    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_s = total_frames / fps if fps > 0 else 0.0
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join(chr((fourcc >> (8 * i)) & 0xFF) for i in range(4)).strip()

        return VideoInfo(
            path=path,
            total_frames=total_frames,
            fps=fps,
            width=width,
            height=height,
            duration_s=duration_s,
            codec=codec,
        )
    finally:
        cap.release()


def extract_frames_fixed_rate(
    video_path: str | Path,
    *,
    every_n_frames: int = 1,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    max_dimension: Optional[int] = None,
) -> Generator[Tuple[NDArray[np.uint8], FrameInfo], None, None]:
    """
    Extract frames at a fixed rate (every N frames).

    Generator yields (frame_rgb, frame_info) pairs.
    Memory usage: one frame at a time.

    Args:
        video_path:     Path to video file.
        every_n_frames: Extract 1 frame per N frames. Default=1 (every frame).
        start_frame:    First frame index to consider.
        end_frame:      Last frame index (exclusive). None = until end.
        max_dimension:  If set, resize frames so the largest dimension ≤ this.

    Yields:
        (frame_rgb, FrameInfo) tuples.

    Raises:
        FileNotFoundError, ValueError: See get_video_info.
    """
    path = Path(video_path)
    info = get_video_info(path)
    fps = info.fps

    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {path}")

        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frame_idx = start_frame
        _end = end_frame if end_frame is not None else info.total_frames

        while frame_idx < _end:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            if (frame_idx - start_frame) % every_n_frames == 0:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                if max_dimension is not None:
                    h, w = frame_rgb.shape[:2]
                    if max(h, w) > max_dimension:
                        scale = max_dimension / max(h, w)
                        new_w, new_h = int(w * scale), int(h * scale)
                        frame_rgb = cv2.resize(frame_rgb, (new_w, new_h))

                ts_ms = (frame_idx / fps * 1000.0) if fps > 0 else 0.0
                fi = FrameInfo(
                    frame_index=frame_idx,
                    timestamp_ms=ts_ms,
                    timestamp_s=ts_ms / 1000.0,
                    width=frame_rgb.shape[1],
                    height=frame_rgb.shape[0],
                )
                yield frame_rgb, fi

            frame_idx += 1

    finally:
        cap.release()


def extract_frames_by_timestamps(
    video_path: str | Path,
    timestamps_s: List[float],
    *,
    max_dimension: Optional[int] = None,
) -> Generator[Tuple[NDArray[np.uint8], FrameInfo], None, None]:
    """
    Extract frames at specific timestamps.

    Args:
        video_path:    Path to video file.
        timestamps_s:  List of timestamps in seconds to extract.
        max_dimension: Optional resize limit.

    Yields:
        (frame_rgb, FrameInfo) pairs in timestamp order.
    """
    path = Path(video_path)
    info = get_video_info(path)
    cap = cv2.VideoCapture(str(path))

    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {path}")

        for ts in sorted(timestamps_s):
            if ts > info.duration_s:
                continue

            ts_ms = ts * 1000.0
            cap.set(cv2.CAP_PROP_POS_MSEC, ts_ms)
            ret, frame_bgr = cap.read()
            if not ret:
                continue

            frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            if max_dimension is not None:
                h, w = frame_rgb.shape[:2]
                if max(h, w) > max_dimension:
                    scale = max_dimension / max(h, w)
                    frame_rgb = cv2.resize(
                        frame_rgb, (int(w * scale), int(h * scale))
                    )

            fi = FrameInfo(
                frame_index=frame_idx,
                timestamp_ms=ts_ms,
                timestamp_s=ts,
                width=frame_rgb.shape[1],
                height=frame_rgb.shape[0],
            )
            yield frame_rgb, fi

    finally:
        cap.release()
