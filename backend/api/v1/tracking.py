"""
backend.api.v1.tracking — Temporal trichome tracking API endpoints.

Routes:
  POST   /video/tracking/start                     — Start a tracking session for a video.
  GET    /video/tracking/{session_id}/status        — Poll session status.
  GET    /video/tracking/{session_id}/summary       — Retrieve tracking summary.
  GET    /video/tracking/{session_id}/trajectories  — Retrieve trajectory data.
  DELETE /video/tracking/{session_id}               — Clean up session.

Architecture:
    Sessions are managed in-process using a dict keyed by UUID session IDs.
    Tracking runs as a FastAPI BackgroundTask, simulating frame-by-frame
    processing using synthetic detections seeded per video_id.

    In production this would:
    1. Load real detections from the database per (video_id, frame_idx).
    2. Drive the TrackingSession across all extracted video frames.
    3. Persist trajectory results to the DB for frontend consumption.

    The synthetic detection generator provides deterministic, reproducible
    test data that exercises all tracker codepaths without requiring a real
    detection pipeline.
"""

from __future__ import annotations

import asyncio
import random
import time
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

from shared.core.entities import Detection
from shared.core.enums import TrichomeType
from shared.core.value_objects import BoundingBox, Confidence
from shared.logging.logger import get_logger
from video_pipeline.tracking import (
    TrackingSession,
    TrackingSessionConfig,
    TrackingSummary,
)

router = APIRouter(prefix="/video/tracking", tags=["tracking"])
logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# In-memory session registry
# ---------------------------------------------------------------------------

class _SessionRecord:
    """Internal session record holding state and results."""

    def __init__(self, session_id: str, video_id: str, config: TrackingSessionConfig) -> None:
        self.session_id = session_id
        self.video_id = video_id
        self.config = config
        self.session = TrackingSession(config)
        self.status: str = "running"  # "running" | "complete" | "error"
        self.frames_processed: int = 0
        self.track_count: int = 0
        self.error_message: Optional[str] = None
        self.summary: Optional[TrackingSummary] = None
        self.trajectories: Optional[List[Dict[str, Any]]] = None
        self.created_at: float = time.time()

    def to_status_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "video_id": self.video_id,
            "status": self.status,
            "frames_processed": self.frames_processed,
            "track_count": self.track_count,
            "error": self.error_message,
        }


_SESSIONS: Dict[str, _SessionRecord] = {}


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class TrackingConfigSchema(BaseModel):
    """Request schema mirroring TrackingSessionConfig."""
    max_age: int = Field(default=3, ge=1, le=30, description="Max frames without detection before track deletion.")
    min_hits: int = Field(default=2, ge=1, le=20, description="Consecutive frames to confirm a track.")
    iou_threshold: float = Field(default=0.3, gt=0.0, lt=1.0, description="Min IoU for detection→track assignment.")
    min_track_length: int = Field(default=3, ge=1, description="Minimum track length included in summary.")
    export_trajectories: bool = Field(default=True, description="Export full trajectory data.")


class StartTrackingRequest(BaseModel):
    video_id: str = Field(..., description="Video identifier (used to look up detections).")
    config: TrackingConfigSchema = Field(default_factory=TrackingConfigSchema)
    n_synthetic_frames: int = Field(
        default=30,
        ge=1,
        le=1000,
        description="Number of synthetic frames to generate (for mock mode).",
    )


class StartTrackingResponse(BaseModel):
    session_id: str
    video_id: str
    message: str


class SessionStatusResponse(BaseModel):
    session_id: str
    video_id: str
    status: str
    frames_processed: int
    track_count: int
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Synthetic detection generator (mock DB backend)
# ---------------------------------------------------------------------------

_TRICHOME_TYPES = [
    TrichomeType.CAPITATE_STALKED,
    TrichomeType.CAPITATE_SESSILE,
    TrichomeType.BULBOUS,
]


def _generate_synthetic_detections(
    frame_idx: int,
    n_objects: int,
    rng: random.Random,
    img_w: int = 1280,
    img_h: int = 720,
) -> List[Detection]:
    """
    Generate deterministic synthetic detections for a single frame.

    Objects have slight positional jitter per frame to simulate microscope
    drift. Some objects appear/disappear to exercise track creation/deletion.

    Args:
        frame_idx:  Current frame index (used for drift calculation).
        n_objects:  Number of objects in this frame.
        rng:        Seeded random generator for reproducibility.
        img_w/h:    Image dimensions (pixels).

    Returns:
        List of Detection objects.
    """
    detections = []

    # Base positions seeded per-object (stable across frames with small drift)
    for i in range(n_objects):
        # Base position changes slowly with frame (simulate drift)
        base_x = (i * 137 + 50) % (img_w - 100) + 50.0
        base_y = (i * 89 + 40) % (img_h - 60) + 30.0

        drift_x = rng.uniform(-3.0, 3.0) * (frame_idx % 5)
        drift_y = rng.uniform(-2.0, 2.0) * (frame_idx % 3)

        cx = max(40.0, min(img_w - 40.0, base_x + drift_x))
        cy = max(30.0, min(img_h - 30.0, base_y + drift_y))

        w = rng.uniform(20.0, 60.0)
        h = rng.uniform(20.0, 60.0)

        x_min = max(0.0, cx - w / 2)
        y_min = max(0.0, cy - h / 2)
        x_max = min(float(img_w), cx + w / 2)
        y_max = min(float(img_h), cy + h / 2)

        # Ensure box is valid
        if x_max <= x_min + 1:
            x_max = x_min + 2.0
        if y_max <= y_min + 1:
            y_max = y_min + 2.0

        trichome_type = _TRICHOME_TYPES[i % len(_TRICHOME_TYPES)]
        conf_val = round(rng.uniform(0.65, 0.98), 3)

        det = Detection(
            bounding_box=BoundingBox(x_min, y_min, x_max, y_max),
            confidence=Confidence(conf_val),
            trichome_type=trichome_type,
            frame_index=frame_idx,
        )
        detections.append(det)

    return detections


# ---------------------------------------------------------------------------
# Background task
# ---------------------------------------------------------------------------

def _run_tracking_session(
    record: _SessionRecord,
    n_frames: int,
    seed: int,
) -> None:
    """
    Run the tracking session synchronously across synthetic frames.

    This function is called from a BackgroundTasks callback.
    In production, replace the synthetic generator with DB queries.
    """
    try:
        rng = random.Random(seed)
        # Vary the number of objects per frame (simulate entry/exit)
        n_base_objects = max(3, rng.randint(5, 12))

        for frame_idx in range(n_frames):
            # Simulate some objects entering/leaving the frame
            drop_prob = rng.random()
            n_obj = n_base_objects
            if drop_prob < 0.15:
                n_obj = max(2, n_base_objects - rng.randint(1, 2))
            elif drop_prob < 0.25:
                n_obj = n_base_objects + rng.randint(1, 2)

            dets = _generate_synthetic_detections(frame_idx, n_obj, rng)
            active = record.session.process_frame(dets, frame_idx)

            record.frames_processed = frame_idx + 1
            record.track_count = len(active)

        # Generate summary after all frames processed
        summary = record.session.get_summary()
        record.summary = summary
        record.trajectories = record.session.export_trajectories()
        record.status = "complete"

        logger.info(
            "Tracking session complete",
            session_id=record.session_id,
            frames=n_frames,
            confirmed_tracks=summary.confirmed_tracks,
        )

    except Exception as exc:
        record.status = "error"
        record.error_message = str(exc)
        logger.error(
            "Tracking session error",
            session_id=record.session_id,
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/start", response_model=StartTrackingResponse, status_code=202)
def start_tracking(
    body: StartTrackingRequest,
    background_tasks: BackgroundTasks,
) -> StartTrackingResponse:
    """
    Start a tracking session for the given video.

    Creates a new session, schedules frame-by-frame tracking as a background
    task, and returns the session ID immediately.

    The session processes frames sequentially using the configured SORT tracker.
    Poll /status to check progress; retrieve results via /summary or /trajectories.
    """
    session_id = str(uuid.uuid4())
    cfg = TrackingSessionConfig(
        max_age=body.config.max_age,
        min_hits=body.config.min_hits,
        iou_threshold=body.config.iou_threshold,
        min_track_length=body.config.min_track_length,
        export_trajectories=body.config.export_trajectories,
    )

    record = _SessionRecord(
        session_id=session_id,
        video_id=body.video_id,
        config=cfg,
    )
    _SESSIONS[session_id] = record

    # Deterministic seed derived from video_id for reproducibility
    seed = abs(hash(body.video_id)) % (2 ** 31)

    background_tasks.add_task(
        _run_tracking_session,
        record=record,
        n_frames=body.n_synthetic_frames,
        seed=seed,
    )

    logger.info(
        "Tracking session started",
        session_id=session_id,
        video_id=body.video_id,
        n_frames=body.n_synthetic_frames,
    )

    return StartTrackingResponse(
        session_id=session_id,
        video_id=body.video_id,
        message=f"Tracking session {session_id} started for video {body.video_id}.",
    )


@router.get("/{session_id}/status", response_model=SessionStatusResponse)
def get_tracking_status(session_id: str) -> SessionStatusResponse:
    """
    Return current status of a tracking session.

    Returns:
        status: "running" | "complete" | "error"
        frames_processed: frames completed so far
        track_count: currently active tracks (last frame)
    """
    record = _get_session_or_404(session_id)
    return SessionStatusResponse(**record.to_status_dict())


@router.get("/{session_id}/summary")
def get_tracking_summary(session_id: str) -> Dict[str, Any]:
    """
    Return the tracking summary for a completed session.

    Returns TrackingSummary as JSON including:
    - total_tracks, confirmed_tracks, avg_track_length
    - type_distribution: per-type counts
    - trajectory_data: full per-track trajectory (if export_trajectories=True)

    Raises 400 if the session is not yet complete.
    """
    record = _get_session_or_404(session_id)

    if record.status == "running":
        raise HTTPException(
            status_code=400,
            detail=f"Session {session_id} is still running ({record.frames_processed} frames processed). Poll /status first.",
        )

    if record.status == "error":
        raise HTTPException(
            status_code=500,
            detail=f"Session {session_id} failed: {record.error_message}",
        )

    if record.summary is None:
        raise HTTPException(
            status_code=500,
            detail=f"Session {session_id} has no summary — internal error.",
        )

    return record.summary.to_dict()


@router.get("/{session_id}/trajectories")
def get_trajectories(session_id: str) -> Dict[str, Any]:
    """
    Return full trajectory data for frontend overlay rendering.

    Each trajectory contains:
    - id: integer track ID
    - type: trichome type string
    - frames: list of frame indices
    - positions: [[cx, cy], ...] per-frame centre points
    - bboxes: [[x_min, y_min, x_max, y_max], ...] per frame
    - confidence: float
    - track_length: number of frames

    Raises 400 if the session is not yet complete.
    """
    record = _get_session_or_404(session_id)

    if record.status == "running":
        raise HTTPException(
            status_code=400,
            detail=f"Session {session_id} is still running. Poll /status first.",
        )

    if record.status == "error":
        raise HTTPException(
            status_code=500,
            detail=f"Session {session_id} failed: {record.error_message}",
        )

    trajectories = record.trajectories or []
    return {
        "session_id": session_id,
        "video_id": record.video_id,
        "track_count": len(trajectories),
        "trajectories": trajectories,
    }


@router.delete("/{session_id}", status_code=204)
def delete_tracking_session(session_id: str) -> None:
    """
    Delete a tracking session and free its memory.

    Safe to call on running sessions — the background task will continue
    briefly but results will be discarded.
    """
    if session_id not in _SESSIONS:
        raise HTTPException(
            status_code=404,
            detail=f"Session {session_id} not found.",
        )

    record = _SESSIONS.pop(session_id)
    record.session.reset()

    logger.info("Tracking session deleted", session_id=session_id)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_session_or_404(session_id: str) -> _SessionRecord:
    """Retrieve session record or raise HTTP 404."""
    if session_id not in _SESSIONS:
        raise HTTPException(
            status_code=404,
            detail=f"Tracking session '{session_id}' not found.",
        )
    return _SESSIONS[session_id]
