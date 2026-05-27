"""
video_pipeline.api.router — FastAPI endpoints for video analysis.

Endpoints:
  POST /video/info           — Get video metadata
  POST /video/score-frames   — Score all frames (returns JSON quality list)
  POST /video/best-frames    — Extract and return N best frames as ZIP
  POST /video/analyze        — Full analysis pipeline (async)
  GET  /video/health         — Health check
"""

from __future__ import annotations

import io
import os
import tempfile
import time
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, List, Optional

import cv2
import numpy as np
from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from fastapi.responses import Response, StreamingResponse

from video_pipeline.domain.extractor import get_video_info, extract_frames_fixed_rate
from video_pipeline.domain.scorer import score_frame
from video_pipeline.domain.hasher import perceptual_hash, deduplicate_frames
from video_pipeline.domain.ranker import RankedFrame, rank_top_n, rank_diverse_n, rank_adaptive
from video_pipeline.schemas.schemas import (
    VideoInfoSchema,
    FrameQualitySchema,
    RankedFrameSchema,
    VideoAnalysisRequest,
    VideoAnalysisResponse,
)

router = APIRouter(prefix="/video", tags=["Video Pipeline"])


@contextmanager
def _temp_video(file: UploadFile, content: bytes) -> Generator[str, None, None]:
    """
    Context manager for uploaded video temp files.

    Guarantees cleanup even on process-kill (os.unlink in finally).
    Fixes TDB-005: bare try/finally could skip cleanup on exception.
    """
    suffix = Path(file.filename or "video.mp4").suffix
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    try:
        os.write(fd, content)
        os.close(fd)
        yield tmp_path
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


@router.get("/health")
async def health() -> dict:
    return {"status": "ok", "module": "video_pipeline"}


@router.post("/info", response_model=VideoInfoSchema, summary="Get video file metadata")
async def video_info(
    file: UploadFile = File(..., description="Video file to analyze"),
) -> VideoInfoSchema:
    """Return metadata (fps, dimensions, duration, codec) for an uploaded video."""
    content = await file.read()
    with _temp_video(file, content) as tmp_path:
        try:
            info = get_video_info(tmp_path)
        except Exception as e:
            raise HTTPException(422, f"Cannot read video: {e}")

    return VideoInfoSchema(
        path=file.filename or "",
        total_frames=info.total_frames,
        fps=info.fps,
        width=info.width,
        height=info.height,
        duration_s=info.duration_s,
        codec=info.codec,
        is_4k=info.is_4k,
        is_hd=info.is_hd,
    )


@router.post(
    "/best-frames",
    summary="Extract N best frames from video as ZIP",
)
async def extract_best_frames(
    file: UploadFile = File(..., description="Video file"),
    n_frames: int = Form(default=10, ge=1, le=100),
    strategy: str = Form(default="adaptive"),
    min_focus: float = Form(default=0.25, ge=0, le=1),
    every_n: int = Form(default=5, ge=1, le=300),
    max_dimension: int = Form(default=1920, ge=64, le=7680),
    dedup_threshold: int = Form(default=8, ge=0, le=64),
) -> Response:
    """
    Extract the N best-quality frames from a video and return them as a ZIP.

    Frames are selected based on focus, exposure, and noise quality.
    Strategy options: 'top_n', 'diverse', 'adaptive'.
    """
    content = await file.read()
    with _temp_video(file, content) as tmp_path:
        try:
            frames_and_metadata: List[tuple] = []  # (frame_rgb, score, frame_info)
            hashes: List[int] = []

            # Stream frames and score
            for frame_rgb, fi in extract_frames_fixed_rate(
                tmp_path,
                every_n_frames=every_n,
                max_dimension=max_dimension,
            ):
                score = score_frame(frame_rgb, use_focus_composite=False)
                ph = perceptual_hash(frame_rgb)
                frames_and_metadata.append((frame_rgb, score, fi))
                hashes.append(ph)

            if not frames_and_metadata:
                raise HTTPException(422, "No frames could be extracted from video")

            # Deduplication
            keep_indices = deduplicate_frames(hashes, threshold=dedup_threshold)

            ranked = [
                RankedFrame(
                    frame_index=frames_and_metadata[i][2].frame_index,
                    timestamp_s=frames_and_metadata[i][2].timestamp_s,
                    quality=frames_and_metadata[i][1],
                    phash=hashes[i],
                )
                for i in keep_indices
            ]

            # Rank
            if strategy == "top_n":
                selected = rank_top_n(ranked, n_frames, min_score=min_focus)
            elif strategy == "diverse":
                selected = rank_diverse_n(ranked, n_frames, min_score=min_focus)
            else:  # adaptive (default)
                selected = rank_adaptive(ranked, n_frames, min_focus=min_focus)

            # Build index for fast lookup
            fi_to_idx = {
                frames_and_metadata[i][2].frame_index: i
                for i in keep_indices
            }

            # Pack into ZIP
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                for rank_pos, rf in enumerate(selected):
                    orig_idx = fi_to_idx.get(rf.frame_index)
                    if orig_idx is None:
                        continue
                    frame_rgb = frames_and_metadata[orig_idx][0]
                    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                    _, jpg = cv2.imencode(
                        ".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92]
                    )
                    name = (
                        f"frame_{rank_pos+1:03d}_"
                        f"t{rf.timestamp_s:.2f}s_"
                        f"q{rf.quality.composite:.2f}.jpg"
                    )
                    zf.writestr(name, jpg.tobytes())

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(500, f"Video processing failed: {e}")

    zip_buffer.seek(0)
    return Response(
        content=zip_buffer.read(),
        media_type="application/zip",
        headers={
            "Content-Disposition": "attachment; filename=best_frames.zip"
        },
    )


@router.post(
    "/analyze",
    response_model=VideoAnalysisResponse,
    summary="Full video analysis: score all frames, select best",
)
async def analyze_video(body: VideoAnalysisRequest) -> VideoAnalysisResponse:
    """
    Run the full video analysis pipeline on a video at a given path.

    Returns quality scores, selected frames, and optional motion analysis.
    Output frames are saved to output_dir if specified.
    """
    video_path = Path(body.video_path)
    if not video_path.exists():
        raise HTTPException(404, f"Video not found: {body.video_path}")

    start_time = time.time()

    try:
        info = get_video_info(video_path)
    except Exception as e:
        raise HTTPException(422, f"Cannot read video: {e}")

    frames_data: list = []
    hashes: list = []
    prev_frame: Optional[np.ndarray] = None
    motions = []

    for frame_rgb, fi in extract_frames_fixed_rate(
        video_path,
        every_n_frames=body.every_n_frames,
        max_dimension=body.max_dimension,
    ):
        score = score_frame(frame_rgb)
        ph = perceptual_hash(frame_rgb)

        if body.compute_motion and prev_frame is not None:
            from video_pipeline.domain.motion import estimate_motion
            m = estimate_motion(prev_frame, frame_rgb)
            motions.append(m)

        frames_data.append((frame_rgb, score, fi))
        hashes.append(ph)
        prev_frame = frame_rgb

    if not frames_data:
        raise HTTPException(422, "No frames extracted")

    # Deduplicate
    keep_idxs = deduplicate_frames(hashes, threshold=8)

    ranked = [
        RankedFrame(
            frame_index=frames_data[i][2].frame_index,
            timestamp_s=frames_data[i][2].timestamp_s,
            quality=frames_data[i][1],
        )
        for i in keep_idxs
    ]

    # Select
    if body.strategy == "top_n":
        selected = rank_top_n(ranked, body.n_best_frames, min_score=body.min_focus_score)
    elif body.strategy == "diverse":
        selected = rank_diverse_n(ranked, body.n_best_frames, min_score=body.min_focus_score)
    else:
        selected = rank_adaptive(ranked, body.n_best_frames, min_focus=body.min_focus_score)

    # Optionally save frames
    output_dir = Path(body.output_dir) if body.output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    fi_to_idx = {frames_data[i][2].frame_index: i for i in keep_idxs}

    selected_schemas = []
    for rank_pos, rf in enumerate(selected):
        out_path = None
        orig_idx = fi_to_idx.get(rf.frame_index)
        if output_dir and orig_idx is not None:
            frame_bgr = cv2.cvtColor(frames_data[orig_idx][0], cv2.COLOR_RGB2BGR)
            out_path = str(
                output_dir
                / f"frame_{rank_pos+1:03d}_t{rf.timestamp_s:.2f}s.jpg"
            )
            cv2.imwrite(out_path, frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])

        selected_schemas.append(
            RankedFrameSchema(
                frame_index=rf.frame_index,
                timestamp_s=rf.timestamp_s,
                quality=FrameQualitySchema(
                    composite=rf.quality.composite,
                    focus=rf.quality.focus,
                    exposure=rf.quality.exposure,
                    noise=rf.quality.noise,
                    quality_label=rf.quality.quality_label,
                    is_usable=rf.quality.is_usable,
                ),
                output_path=out_path,
            )
        )

    motion_summary = None
    if motions:
        from video_pipeline.domain.motion import classify_motion_sequence
        motion_summary = classify_motion_sequence(motions)

    return VideoAnalysisResponse(
        video_info=VideoInfoSchema(
            path=str(video_path),
            total_frames=info.total_frames,
            fps=info.fps,
            width=info.width,
            height=info.height,
            duration_s=info.duration_s,
            codec=info.codec,
            is_4k=info.is_4k,
            is_hd=info.is_hd,
        ),
        total_frames_analyzed=len(frames_data),
        n_selected=len(selected),
        selected_frames=selected_schemas,
        motion_summary=motion_summary,
        processing_time_s=time.time() - start_time,
    )
