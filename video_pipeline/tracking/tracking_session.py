"""
video_pipeline.tracking.tracking_session — High-level tracking session manager.

Wraps SORTTracker with session lifecycle management, summary generation,
and JSON-serialisable trajectory export for frontend visualisation overlays.

A TrackingSession is designed to process one complete video sequence.
Each call to process_frame() feeds detections from a single frame and
receives the current set of active TrichomeTracks.

Design decisions:
- All state is held in memory (not persisted to DB here — that's the API layer's job).
- TrackingSummary is generated lazily from the complete track history.
- Trajectory export filters out very short tracks (< min_track_length) to
  eliminate noise from false detections.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from shared.core.entities import Detection
from shared.logging.logger import get_logger
from video_pipeline.tracking.sort_tracker import (
    SORTTracker,
    TrackState,
    TrichomeTrack,
)

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TrackingSessionConfig:
    """
    Configuration for a single tracking session.

    Attributes:
        max_age:           Frames a track persists without a matched detection.
        min_hits:          Consecutive frames before a track is confirmed.
        iou_threshold:     Minimum IoU for detection→track assignment.
        min_track_length:  Minimum number of frames for a track to appear
                           in the summary (filters ephemeral false positives).
        export_trajectories: Whether to compute full trajectory data.
    """
    max_age: int = 3
    min_hits: int = 2
    iou_threshold: float = 0.3
    min_track_length: int = 3
    export_trajectories: bool = True


# ---------------------------------------------------------------------------
# Summary output
# ---------------------------------------------------------------------------

@dataclass
class TrackingSummary:
    """
    Aggregate statistics for a completed tracking session.

    Attributes:
        total_tracks:      Total number of unique track IDs created (including
                           very short/tentative tracks).
        confirmed_tracks:  Tracks that reached CONFIRMED state at least once.
        avg_track_length:  Mean number of frames across confirmed tracks.
        type_distribution: Count of confirmed tracks per trichome type.
        trajectory_data:   Per-track trajectory list (empty if
                           export_trajectories is False).
    """
    total_tracks: int
    confirmed_tracks: int
    avg_track_length: float
    type_distribution: Dict[str, int]
    trajectory_data: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_tracks": self.total_tracks,
            "confirmed_tracks": self.confirmed_tracks,
            "avg_track_length": self.avg_track_length,
            "type_distribution": self.type_distribution,
            "trajectory_data": self.trajectory_data,
        }


# ---------------------------------------------------------------------------
# TrackingSession
# ---------------------------------------------------------------------------

class TrackingSession:
    """
    Manages a complete SORT tracking session across a video sequence.

    Usage::

        session = TrackingSession(config)
        for frame_idx, detections in enumerate(frame_detections):
            active_tracks = session.process_frame(detections, frame_idx)

        summary = session.get_summary()
        trajectories = session.export_trajectories()

    Thread safety:
        Not thread-safe. Use one TrackingSession per concurrent video.
    """

    def __init__(self, config: Optional[TrackingSessionConfig] = None) -> None:
        self.config = config or TrackingSessionConfig()
        self._tracker = SORTTracker(
            max_age=self.config.max_age,
            min_hits=self.config.min_hits,
            iou_threshold=self.config.iou_threshold,
        )
        self._frames_processed: int = 0
        # Final snapshot of all-time track metadata (id → TrichomeTrack)
        # keyed by track_id, continuously updated
        self._all_tracks: Dict[int, TrichomeTrack] = {}

        logger.debug(
            "TrackingSession initialised",
            max_age=self.config.max_age,
            min_hits=self.config.min_hits,
            iou_threshold=self.config.iou_threshold,
        )

    # --- Frame processing ---------------------------------------------------

    def process_frame(
        self,
        detections: List[Detection],
        frame_idx: int,
    ) -> List[TrichomeTrack]:
        """
        Process one video frame, updating all active tracks.

        Args:
            detections: Trichome detections from the current frame.
            frame_idx:  Zero-based frame index.

        Returns:
            Active (CONFIRMED + TENTATIVE) tracks after this update.
        """
        active = self._tracker.update(detections, frame_idx)
        self._frames_processed += 1

        # Snapshot track state into all_tracks archive
        for track in active:
            self._all_tracks[track.track_id] = track

        # Also capture any tracks that just got deleted (they're still in meta)
        for tid, meta in self._tracker._track_meta.items():
            if tid not in self._all_tracks:
                self._all_tracks[tid] = meta
            else:
                # Overwrite with latest (may be DELETED now)
                self._all_tracks[tid] = meta

        logger.debug(
            "Frame processed",
            frame_idx=frame_idx,
            detections=len(detections),
            active_tracks=len(active),
        )

        return active

    # --- Summary generation -------------------------------------------------

    def get_summary(self) -> TrackingSummary:
        """
        Generate aggregate statistics from the complete tracking history.

        Returns:
            TrackingSummary with counts, averages, and type distributions.
        """
        all_track_list = list(self._all_tracks.values())
        total_tracks = len(all_track_list)

        # A track is "confirmed" if it ever had hit_streak >= min_hits
        # or was explicitly set to CONFIRMED.
        confirmed = [
            t for t in all_track_list
            if t.state == TrackState.CONFIRMED
            or t.hits >= self.config.min_hits
        ]

        confirmed_count = len(confirmed)

        lengths = [t.track_length for t in confirmed]
        avg_length = float(sum(lengths) / len(lengths)) if lengths else 0.0

        type_counts: Counter = Counter()
        for t in confirmed:
            type_name = t.trichome_type or "UNKNOWN"
            type_counts[type_name] += 1

        traj_data: List[Dict[str, Any]] = []
        if self.config.export_trajectories:
            traj_data = self._build_trajectory_data(confirmed)

        summary = TrackingSummary(
            total_tracks=total_tracks,
            confirmed_tracks=confirmed_count,
            avg_track_length=avg_length,
            type_distribution=dict(type_counts),
            trajectory_data=traj_data,
        )

        logger.debug(
            "Summary generated",
            total=total_tracks,
            confirmed=confirmed_count,
            avg_length=avg_length,
        )

        return summary

    # --- Trajectory export --------------------------------------------------

    def export_trajectories(self) -> List[Dict[str, Any]]:
        """
        Export all confirmed track trajectories as JSON-serialisable dicts.

        Filters out tracks shorter than ``config.min_track_length`` to
        suppress ephemeral detection noise.

        Returns:
            List of trajectory dicts, each containing:
            - id: track integer ID
            - type: trichome type string
            - frames: list of frame indices
            - positions: list of [cx, cy] centre points
            - bboxes: list of [x_min, y_min, x_max, y_max] per frame
            - confidence: float confidence of the last matched detection
            - track_length: number of frames
        """
        all_track_list = list(self._all_tracks.values())
        confirmed = [
            t for t in all_track_list
            if (t.state == TrackState.CONFIRMED or t.hits >= self.config.min_hits)
        ]
        return self._build_trajectory_data(confirmed)

    def reset(self) -> None:
        """
        Reset the session, clearing all tracks and frame history.

        After reset, the session is ready to process a new video sequence.
        """
        self._tracker.reset()
        self._all_tracks.clear()
        self._frames_processed = 0
        logger.debug("TrackingSession reset")

    # --- Properties ---------------------------------------------------------

    @property
    def frames_processed(self) -> int:
        """Number of frames processed so far in this session."""
        return self._frames_processed

    @property
    def active_track_count(self) -> int:
        """Number of currently active (non-deleted) tracks."""
        return len(self._tracker._trackers)

    # --- Private helpers ----------------------------------------------------

    def _build_trajectory_data(
        self,
        tracks: List[TrichomeTrack],
    ) -> List[Dict[str, Any]]:
        """Build serialisable trajectory list from a list of TrichomeTracks."""
        result = []
        for t in tracks:
            if t.track_length < self.config.min_track_length:
                continue

            positions = []
            bboxes = []
            for bbox in t.history:
                cx = (bbox.x_min + bbox.x_max) / 2.0
                cy = (bbox.y_min + bbox.y_max) / 2.0
                positions.append([round(cx, 2), round(cy, 2)])
                bboxes.append([
                    round(bbox.x_min, 2),
                    round(bbox.y_min, 2),
                    round(bbox.x_max, 2),
                    round(bbox.y_max, 2),
                ])

            result.append({
                "id": t.track_id,
                "type": t.trichome_type or "UNKNOWN",
                "frames": list(t.frame_indices),
                "positions": positions,
                "bboxes": bboxes,
                "confidence": round(t.confidence, 4),
                "track_length": t.track_length,
            })

        # Sort by track ID for deterministic output
        result.sort(key=lambda d: d["id"])
        return result

    def __repr__(self) -> str:
        return (
            f"TrackingSession("
            f"frames={self._frames_processed}, "
            f"active={self.active_track_count}, "
            f"total_seen={len(self._all_tracks)})"
        )
