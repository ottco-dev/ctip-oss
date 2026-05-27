"""
backend/api/v1/video.py — Video upload and analysis endpoints.

Routes:
  POST /video/upload        — upload video file, store to disk
  POST /video/analyze       — run video pipeline (extract frames, rank quality)
  GET  /video/jobs/{id}     — job status + results
  GET  /video/frames/{id}   — stream best frames as JSON
  GET  /video/thumbnail/{id}/{frame} — serve a specific frame image
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
import uuid
from datetime import datetime
from pathlib import Path

import cv2
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlmodel import Session, select

from backend.config import get_settings
from backend.database import get_session
from backend.models.job import BackgroundJob, JobStatus

router = APIRouter(prefix="/video", tags=["video"])
settings = get_settings()

# ---------------------------------------------------------------------------
# Storage helpers
# ---------------------------------------------------------------------------

VIDEO_STORE = Path("/data/videos") if Path("/data").exists() else Path("/tmp/trichome/videos")
FRAME_STORE = Path("/data/frames") if Path("/data").exists() else Path("/tmp/trichome/frames")
VIDEO_STORE.mkdir(parents=True, exist_ok=True)
FRAME_STORE.mkdir(parents=True, exist_ok=True)

# In-memory video registry (replace with DB table)
_VIDEOS: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class VideoInfo(BaseModel):
    id: str
    filename: str
    path: str
    size_bytes: int
    duration_s: float | None = None
    fps: float | None = None
    frame_count: int | None = None
    width: int | None = None
    height: int | None = None
    uploaded_at: datetime = Field(default_factory=datetime.utcnow)
    sha256: str = ""


class AnalyzeRequest(BaseModel):
    video_id: str
    max_frames: int = Field(default=50, ge=1, le=500)
    min_sharpness: float = Field(default=80.0, ge=0.0)
    scene_change_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    deduplicate: bool = True
    dedup_threshold: float = Field(default=0.95, ge=0.0, le=1.0)


class FrameResult(BaseModel):
    frame_index: int
    timestamp_s: float
    sharpness: float
    exposure_ok: bool
    path: str
    selected: bool


class VideoAnalysisResult(BaseModel):
    video_id: str
    total_frames: int
    analyzed_frames: int
    selected_frames: int
    best_sharpness: float
    mean_sharpness: float
    frames: list[FrameResult]
    duration_s: float
    processing_time_s: float


# ---------------------------------------------------------------------------
# Background task
# ---------------------------------------------------------------------------


def _analyze_video(job_id: str, request: AnalyzeRequest, db_url: str):
    """Run video analysis pipeline in background thread."""
    from sqlmodel import Session, create_engine

    engine = create_engine(db_url)
    start_time = time.time()

    try:
        if request.video_id not in _VIDEOS:
            raise ValueError(f"Video {request.video_id} not found")

        video_info = _VIDEOS[request.video_id]
        video_path = video_info["path"]

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        duration_s = total_frames / fps

        # Sample frames evenly
        step = max(1, total_frames // (request.max_frames * 3))
        frame_dir = FRAME_STORE / request.video_id
        frame_dir.mkdir(parents=True, exist_ok=True)

        frame_results: list[FrameResult] = []
        frame_idx = 0

        while True:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            # Laplacian sharpness
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

            # Exposure check: mean brightness in [30, 220]
            mean_brightness = float(gray.mean())
            exposure_ok = 30 < mean_brightness < 220

            timestamp_s = frame_idx / fps
            frame_path = frame_dir / f"frame_{frame_idx:06d}.jpg"

            if sharpness >= request.min_sharpness and exposure_ok:
                cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                frame_results.append(
                    FrameResult(
                        frame_index=frame_idx,
                        timestamp_s=round(timestamp_s, 3),
                        sharpness=round(sharpness, 2),
                        exposure_ok=exposure_ok,
                        path=str(frame_path),
                        selected=False,
                    )
                )

            frame_idx += step
            if frame_idx >= total_frames:
                break

        cap.release()

        # Select top-N by sharpness
        frame_results.sort(key=lambda x: x.sharpness, reverse=True)
        selected = frame_results[: request.max_frames]
        for f in selected:
            f.selected = True

        all_sharpness = [f.sharpness for f in frame_results]

        result = VideoAnalysisResult(
            video_id=request.video_id,
            total_frames=total_frames,
            analyzed_frames=len(frame_results),
            selected_frames=len(selected),
            best_sharpness=max(all_sharpness) if all_sharpness else 0.0,
            mean_sharpness=sum(all_sharpness) / len(all_sharpness) if all_sharpness else 0.0,
            frames=frame_results,
            duration_s=round(duration_s, 2),
            processing_time_s=round(time.time() - start_time, 2),
        )

        with Session(engine) as db:
            job = db.get(BackgroundJob, job_id)
            if job:
                job.status = JobStatus.COMPLETED
                job.progress = 100
                job.finished_at = datetime.utcnow()
                job.result_json = result.model_dump_json()
                db.commit()

    except Exception as exc:
        with Session(engine) as db:
            job = db.get(BackgroundJob, job_id)
            if job:
                job.status = JobStatus.FAILED
                job.error_message = str(exc)
                job.finished_at = datetime.utcnow()
                db.commit()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/upload", response_model=VideoInfo, status_code=201)
async def upload_video(file: UploadFile):
    """Upload a video file for analysis. Streams directly to disk."""
    if not file.filename:
        raise HTTPException(400, "No filename provided")

    video_id = str(uuid.uuid4())
    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
        raise HTTPException(415, f"Unsupported video format: {suffix}")

    dest_path = VIDEO_STORE / f"{video_id}{suffix}"

    # Stream to disk
    hasher = hashlib.sha256()
    with open(dest_path, "wb") as out:
        while chunk := await file.read(1024 * 1024):  # 1 MB chunks
            out.write(chunk)
            hasher.update(chunk)

    sha256 = hasher.hexdigest()
    size_bytes = dest_path.stat().st_size

    # Read video metadata
    cap = cv2.VideoCapture(str(dest_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_s = frame_count / fps if fps > 0 else None
    cap.release()

    info = VideoInfo(
        id=video_id,
        filename=file.filename,
        path=str(dest_path),
        size_bytes=size_bytes,
        duration_s=round(duration_s, 2) if duration_s else None,
        fps=round(fps, 2) if fps else None,
        frame_count=frame_count,
        width=width,
        height=height,
        sha256=sha256,
    )
    _VIDEOS[video_id] = info.model_dump()
    return info


@router.post("/analyze", status_code=202)
def analyze_video(
    request: AnalyzeRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_session),
):
    """Start video analysis job. Returns job_id for polling."""
    if request.video_id not in _VIDEOS:
        raise HTTPException(404, f"Video {request.video_id} not found")

    job = BackgroundJob(
        job_type="video_analyze",
        status=JobStatus.PENDING,
        params_json=request.model_dump_json(),
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    db_url = str(db.bind.url) if db.bind else "sqlite:////app/data/trichome.db"
    background_tasks.add_task(_analyze_video, str(job.id), request, db_url)

    return {
        "job_id": str(job.id),
        "status": "queued",
        "video_id": request.video_id,
    }


@router.get("/jobs/{job_id}")
def get_video_job(job_id: str, db: Session = Depends(get_session)):
    """Poll video analysis job status and results."""
    job = db.get(BackgroundJob, job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    response: dict = {
        "id": str(job.id),
        "status": job.status.value if hasattr(job.status, "value") else str(job.status),
        "progress": job.progress,
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "finished_at": job.finished_at.isoformat() if job.finished_at else None,
        "error_message": job.error_message,
    }

    if job.result_json:
        response["result"] = json.loads(job.result_json)

    return response


@router.get("/videos")
def list_videos():
    """List all uploaded videos."""
    return list(_VIDEOS.values())


@router.get("/videos/{video_id}")
def get_video_info(video_id: str):
    if video_id not in _VIDEOS:
        raise HTTPException(404, "Video not found")
    return _VIDEOS[video_id]


@router.get("/thumbnail/{video_id}/{frame_index}")
def get_frame_thumbnail(video_id: str, frame_index: int):
    """Serve a specific extracted frame as JPEG."""
    frame_dir = FRAME_STORE / video_id
    frame_path = frame_dir / f"frame_{frame_index:06d}.jpg"
    if not frame_path.exists():
        raise HTTPException(404, f"Frame {frame_index} not found for video {video_id}")
    return FileResponse(str(frame_path), media_type="image/jpeg")


@router.delete("/videos/{video_id}", status_code=204)
def delete_video(video_id: str):
    """Delete uploaded video and extracted frames."""
    if video_id not in _VIDEOS:
        raise HTTPException(404, "Video not found")

    info = _VIDEOS.pop(video_id)
    video_path = Path(info["path"])
    if video_path.exists():
        video_path.unlink()

    frame_dir = FRAME_STORE / video_id
    if frame_dir.exists():
        shutil.rmtree(frame_dir)
