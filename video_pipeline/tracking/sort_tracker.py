"""
video_pipeline.tracking.sort_tracker — SORT multi-object tracking.

Implements the SORT algorithm (Bewley et al., 2016) adapted for trichome
tracking in microscopy video sequences.

State vector for each track:
    x = [cx, cy, s, r, vx, vy, vs]^T

where:
    cx, cy = bounding box center coordinates (pixels)
    s      = box area (pixels²)
    r      = aspect ratio (width / height) — treated as constant
    vx, vy = velocity in x and y directions (pixels/frame)
    vs     = rate of change of area (pixels²/frame)

Measurement vector:
    z = [cx, cy, s, r]^T

The constant-velocity kinematic model is appropriate for trichomes because:
1. Microscope drift is typically smooth and directional between frames.
2. Trichomes do not self-propel — any apparent motion is from the stage.
3. Area changes are gradual (focus plane drift or zoom changes).

Scientific note:
    This tracker assigns persistent integer IDs to trichomes across frames,
    enabling population-level temporal analysis (e.g., maturity progression
    over time in the same field of view).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from scipy.linalg import inv
from scipy.optimize import linear_sum_assignment

from shared.core.entities import Detection
from shared.core.value_objects import BoundingBox
from shared.logging.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Track state enumeration
# ---------------------------------------------------------------------------

class TrackState:
    """
    Lifecycle state for a tracked object.

    Transitions:
        (new detection)  → TENTATIVE
        (min_hits reached) → CONFIRMED
        (max_age exceeded) → DELETED
    """
    TENTATIVE: int = 1
    """Track created but not yet seen enough consecutive frames."""
    CONFIRMED: int = 2
    """Track has been reliably observed for >= min_hits consecutive frames."""
    DELETED: int = 3
    """Track has exceeded max_age frames without a matching detection."""


# ---------------------------------------------------------------------------
# TrichomeTrack — track entity
# ---------------------------------------------------------------------------

@dataclass
class TrichomeTrack:
    """
    A persistent trichome track across video frames.

    Each track maintains its full spatial history and the frame indices
    at which it was observed, enabling downstream temporal analysis.

    Attributes:
        track_id:       Unique monotonically-increasing integer identifier.
        state:          Current lifecycle state (TrackState.*).
        bbox:           Current estimated bounding box from Kalman filter.
        confidence:     Confidence of the most recent matched detection.
        trichome_type:  Morphological type string from the most recent detection.
        hits:           Number of consecutive frames with a matched detection.
        age:            Number of frames since this track was last matched
                        (0 if matched in the current frame, incremented otherwise).
        history:        Ordered list of estimated positions (one per frame update).
        frame_indices:  Ordered list of frame indices at which this track appeared.
    """
    track_id: int
    state: int                          # TrackState constant
    bbox: BoundingBox
    confidence: float
    trichome_type: Optional[str]
    hits: int
    age: int
    history: List[BoundingBox] = field(default_factory=list)
    frame_indices: List[int] = field(default_factory=list)

    @property
    def is_confirmed(self) -> bool:
        return self.state == TrackState.CONFIRMED

    @property
    def is_tentative(self) -> bool:
        return self.state == TrackState.TENTATIVE

    @property
    def is_deleted(self) -> bool:
        return self.state == TrackState.DELETED

    @property
    def track_length(self) -> int:
        """Number of frames in this track's history."""
        return len(self.frame_indices)

    def to_dict(self) -> dict:
        """JSON-serialisable representation for API responses."""
        return {
            "track_id": self.track_id,
            "state": self.state,
            "state_name": (
                "TENTATIVE" if self.state == TrackState.TENTATIVE
                else "CONFIRMED" if self.state == TrackState.CONFIRMED
                else "DELETED"
            ),
            "bbox": list(self.bbox.to_xyxy()),
            "confidence": self.confidence,
            "trichome_type": self.trichome_type,
            "hits": self.hits,
            "age": self.age,
            "track_length": self.track_length,
            "frame_indices": list(self.frame_indices),
        }


# ---------------------------------------------------------------------------
# KalmanBoxTracker — constant-velocity Kalman filter for bounding boxes
# ---------------------------------------------------------------------------

class KalmanBoxTracker:
    """
    Constant-velocity Kalman filter for a single bounding box track.

    State:  x = [cx, cy, s, r, vx, vy, vs]^T  (7-dimensional)
    Measurement: z = [cx, cy, s, r]^T          (4-dimensional)

    The aspect ratio r is modelled as constant (vs. velocity model on cx/cy/s).
    This is the same convention used in the original SORT paper.

    Noise parameters are tuned for microscopy:
    - Process noise (Q): relatively small; microscope drift is smooth.
    - Measurement noise (R): moderate; YOLO detections have bbox jitter.

    Matrix operations use scipy.linalg to avoid filterpy dependency.
    """

    _count: int = 0  # class-level monotonic ID counter

    def __init__(self, bbox: BoundingBox) -> None:
        """
        Initialise tracker from the first detected bounding box.

        Args:
            bbox: Initial bounding box observation.
        """
        # --- State transition matrix F (7×7 constant velocity model) ---
        # x_k+1 = F x_k + noise
        self.F = np.eye(7, dtype=np.float64)
        # Velocity coupling: position += velocity per frame
        for i in range(4):
            self.F[i, i + 3] = 1.0

        # --- Observation matrix H (4×7) — maps state to measurement ---
        self.H = np.zeros((4, 7), dtype=np.float64)
        self.H[:4, :4] = np.eye(4)

        # --- Process noise covariance Q (7×7) ---
        # Higher noise on velocity components → more uncertainty in dynamics.
        self.Q = np.diag([
            1.0, 1.0, 10.0, 1.0,   # positional + area + aspect uncertainty
            0.01, 0.01, 0.001,      # velocity uncertainty (tight — smooth drift)
        ]).astype(np.float64)

        # --- Measurement noise covariance R (4×4) ---
        # YOLO detection noise: ~1px position error, ~10% area error.
        self.R = np.diag([1.0, 1.0, 10.0, 1.0]).astype(np.float64)

        # --- Initial state covariance P (7×7) ---
        # High uncertainty on velocities at initialisation.
        self.P = np.diag([
            10.0, 10.0, 10.0, 10.0,
            1e4, 1e4, 1e4,
        ]).astype(np.float64)

        # --- Initial state x (7×1 column vector) ---
        cx, cy, s, r = self._bbox_to_z(bbox)
        self.x = np.array([[cx], [cy], [s], [r], [0.0], [0.0], [0.0]], dtype=np.float64)

        # Track metadata
        KalmanBoxTracker._count += 1
        self.track_id: int = KalmanBoxTracker._count
        self.time_since_update: int = 0
        # hits and hit_streak start at 0; the first matched detection in a
        # subsequent frame increments them via update().  Confirmation in
        # SORTTracker uses (hit_streak >= min_hits - 1), which means a track
        # created in frame N is confirmed in frame N+min_hits-1 after
        # min_hits-1 consecutive matched frames (creation itself counts as
        # the first "frame seen").
        self.hit_streak: int = 0
        self.hits: int = 0
        self.age: int = 0

    # --- Kalman predict step -------------------------------------------------

    def predict(self) -> BoundingBox:
        """
        Advance the Kalman filter by one time step.

        Propagates the state forward using the constant-velocity model:
            x_k|k-1 = F x_k-1|k-1
            P_k|k-1 = F P_k-1|k-1 F^T + Q

        Returns:
            Predicted bounding box (may contain negative coordinates —
            callers should validate before using for display).
        """
        # Prevent area going negative before multiply
        if (self.x[2, 0] + self.x[6, 0]) < 0:
            self.x[6, 0] = 0.0

        # Predict
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1

        return self._x_to_bbox(self.x)

    # --- Kalman update step --------------------------------------------------

    def update(self, bbox: BoundingBox) -> None:
        """
        Update the Kalman filter with a new detection measurement.

        Runs the standard Kalman update equations:
            y   = z - H x_k|k-1             (innovation)
            S   = H P H^T + R                (innovation covariance)
            K   = P H^T S^{-1}               (Kalman gain)
            x   = x_k|k-1 + K y             (posterior state)
            P   = (I - K H) P_k|k-1         (posterior covariance)

        Args:
            bbox: New bounding box measurement from the detector.
        """
        z = np.array(self._bbox_to_z(bbox), dtype=np.float64).reshape(4, 1)

        # Innovation
        y = z - self.H @ self.x

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        K = self.P @ self.H.T @ inv(S)

        # State update
        self.x = self.x + K @ y

        # Joseph form covariance update (numerically stable)
        I_KH = np.eye(7) - K @ self.H
        self.P = I_KH @ self.P

        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1

    # --- Validity check ------------------------------------------------------

    @property
    def is_valid(self) -> bool:
        """
        True if the current state estimate is geometrically valid.

        A state is invalid if the predicted area or aspect ratio becomes
        degenerate (≤0), which can occur when tracks drift off-screen.
        """
        area = float(self.x[2, 0])
        ratio = float(self.x[3, 0])
        return area > 0 and ratio > 0

    # --- Conversion utilities ------------------------------------------------

    @staticmethod
    def _bbox_to_z(bbox: BoundingBox) -> Tuple[float, float, float, float]:
        """Convert BoundingBox to [cx, cy, area, aspect_ratio]."""
        cx = (bbox.x_min + bbox.x_max) / 2.0
        cy = (bbox.y_min + bbox.y_max) / 2.0
        w = bbox.x_max - bbox.x_min
        h = bbox.y_max - bbox.y_min
        s = w * h
        r = w / h if h > 0 else 1.0
        return cx, cy, s, r

    @staticmethod
    def _x_to_bbox(x: np.ndarray) -> BoundingBox:
        """
        Convert state vector [cx, cy, s, r, ...] to BoundingBox.

        Clamps to minimum size 1×1 at position (0,0) to ensure validity.
        """
        cx = float(x[0, 0])
        cy = float(x[1, 0])
        s = max(float(x[2, 0]), 1.0)   # area must be positive
        r = max(float(x[3, 0]), 0.01)  # aspect ratio must be positive

        # w = sqrt(s * r), h = sqrt(s / r)
        w = math.sqrt(s * r)
        h = math.sqrt(s / r) if r > 0 else math.sqrt(s)

        x_min = max(0.0, cx - w / 2.0)
        y_min = max(0.0, cy - h / 2.0)
        x_max = cx + w / 2.0
        y_max = cy + h / 2.0

        # Enforce minimum 1×1 box
        if x_max <= x_min:
            x_max = x_min + 1.0
        if y_max <= y_min:
            y_max = y_min + 1.0

        return BoundingBox(
            x_min=x_min,
            y_min=y_min,
            x_max=x_max,
            y_max=y_max,
        )

    def get_state(self) -> BoundingBox:
        """Return current estimated bounding box without advancing time."""
        return self._x_to_bbox(self.x)


# ---------------------------------------------------------------------------
# SORTTracker — Multi-object tracker
# ---------------------------------------------------------------------------

class SORTTracker:
    """
    SORT (Simple Online Realtime Tracking) multi-object tracker.

    Manages a pool of KalmanBoxTracker instances, one per active track.
    Each call to ``update()`` performs:
        1. Predict all existing tracks forward one frame.
        2. Compute IoU cost matrix between predictions and new detections.
        3. Hungarian assignment to match detections to tracks.
        4. Update matched tracks; mark unmatched tracks as aged.
        5. Initialise new tracks for unmatched detections.
        6. Delete tracks that have exceeded ``max_age`` without a match.

    Track lifecycle:
        - New detection → TENTATIVE track (not yet reliable)
        - After ``min_hits`` consecutive matches → CONFIRMED
        - After ``max_age`` frames without match → DELETED and removed

    Args:
        max_age:       Frames a track survives without a detection (default 3).
                       Higher values handle brief occlusions but increase ID
                       switches in dense fields.
        min_hits:      Consecutive frames required to confirm a track (default 2).
                       Filters out spurious single-frame detections.
        iou_threshold: Minimum IoU for a detection→track assignment (default 0.3).
                       Lower values handle larger inter-frame displacements
                       (e.g., fast stage movement) but increase false matches.
    """

    def __init__(
        self,
        max_age: int = 3,
        min_hits: int = 2,
        iou_threshold: float = 0.3,
    ) -> None:
        if max_age < 1:
            raise ValueError(f"max_age must be >= 1, got {max_age}")
        if min_hits < 1:
            raise ValueError(f"min_hits must be >= 1, got {min_hits}")
        if not (0.0 < iou_threshold < 1.0):
            raise ValueError(f"iou_threshold must be in (0, 1), got {iou_threshold}")

        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold

        self._trackers: List[KalmanBoxTracker] = []
        self._track_meta: dict[int, TrichomeTrack] = {}  # track_id → TrichomeTrack
        self._frame_count: int = 0

    # --- Public API ----------------------------------------------------------

    def update(
        self,
        detections: List[Detection],
        frame_idx: int,
    ) -> List[TrichomeTrack]:
        """
        Update all active tracks with detections from one frame.

        Args:
            detections: List of Detection objects from the current frame.
            frame_idx:  Zero-based index of the current frame in the video.

        Returns:
            All currently active tracks (CONFIRMED + TENTATIVE).
            DELETED tracks are removed from internal state before returning.
        """
        self._frame_count += 1

        # ── Step 1: Predict all existing trackers one step forward ──────────
        predicted_boxes: List[Optional[BoundingBox]] = []
        for trk in self._trackers:
            pred = trk.predict()
            predicted_boxes.append(pred if trk.is_valid else None)

        # ── Step 2: Build cost matrix and run Hungarian assignment ───────────
        det_boxes = [d.bounding_box for d in detections]
        valid_indices = [i for i, b in enumerate(predicted_boxes) if b is not None]
        valid_pred_boxes = [predicted_boxes[i] for i in valid_indices]

        matched_dets: List[int]
        matched_trks: List[int]
        unmatched_dets: List[int]
        unmatched_trks: List[int]

        if det_boxes and valid_pred_boxes:
            iou_mat = self.iou_matrix(det_boxes, valid_pred_boxes)  # type: ignore[arg-type]
            cost_mat = 1.0 - iou_mat

            matched_local, unmatched_dets, unmatched_local_trks = self.hungarian_assignment(
                cost_mat
            )

            # Filter matches below IoU threshold
            final_matched_dets: List[int] = []
            final_matched_trks: List[int] = []
            for di, ti_local in matched_local:
                if iou_mat[di, ti_local] >= self.iou_threshold:
                    final_matched_dets.append(di)
                    final_matched_trks.append(valid_indices[ti_local])
                else:
                    unmatched_dets.append(di)
                    unmatched_local_trks.append(ti_local)

            matched_dets = final_matched_dets
            matched_trks = final_matched_trks
            # Unmatched real tracker indices
            unmatched_trks = [valid_indices[i] for i in unmatched_local_trks]
            # Also add invalid trackers (NaN predictions) as unmatched
            invalid_trk_indices = [i for i in range(len(self._trackers)) if i not in valid_indices]
            unmatched_trks.extend(invalid_trk_indices)

        elif det_boxes and not valid_pred_boxes:
            # No existing tracks — all detections are unmatched
            matched_dets, matched_trks = [], []
            unmatched_dets = list(range(len(det_boxes)))
            unmatched_trks = list(range(len(self._trackers)))
        else:
            # No detections — all tracks are unmatched
            matched_dets, matched_trks = [], []
            unmatched_dets = []
            unmatched_trks = list(range(len(self._trackers)))

        # ── Step 3: Update matched tracks ────────────────────────────────────
        for di, ti in zip(matched_dets, matched_trks):
            det = detections[di]
            trk = self._trackers[ti]
            trk.update(det.bounding_box)
            trk.time_since_update = 0

            # Update TrichomeTrack metadata
            meta = self._track_meta[trk.track_id]
            meta.bbox = trk.get_state()
            meta.confidence = float(det.effective_confidence)
            meta.trichome_type = det.trichome_type.value
            meta.hits = trk.hits          # total hits (not just current streak)
            meta.age = trk.time_since_update
            meta.history.append(meta.bbox)
            meta.frame_indices.append(frame_idx)

            # State transition: TENTATIVE → CONFIRMED
            # hit_streak starts at 0 at creation; each matched frame adds 1.
            # Confirm when hit_streak >= min_hits - 1:
            #   min_hits=2 → confirms after 1st explicit update (streak=1 ≥ 1)
            #   min_hits=3 → confirms after 2nd explicit update (streak=2 ≥ 2)
            # This preserves the semantics: "confirmed after min_hits total
            # frames observed" (creation + (min_hits-1) matched updates).
            if trk.hit_streak >= max(1, self.min_hits - 1):
                meta.state = TrackState.CONFIRMED
            else:
                meta.state = TrackState.TENTATIVE

        # ── Step 4: Age out unmatched tracks ─────────────────────────────────
        for ti in unmatched_trks:
            trk = self._trackers[ti]
            # time_since_update was already incremented in predict()
            meta = self._track_meta[trk.track_id]
            meta.age = trk.time_since_update
            meta.hits = trk.hits          # total hits
            if trk.time_since_update > self.max_age:
                meta.state = TrackState.DELETED

        # ── Step 5: Initialise new tracks for unmatched detections ───────────
        for di in unmatched_dets:
            det = detections[di]
            new_trk = KalmanBoxTracker(det.bounding_box)
            self._trackers.append(new_trk)

            new_meta = TrichomeTrack(
                track_id=new_trk.track_id,
                state=TrackState.TENTATIVE,
                bbox=new_trk.get_state(),
                confidence=float(det.effective_confidence),
                trichome_type=det.trichome_type.value,
                hits=1,
                age=0,
                history=[new_trk.get_state()],
                frame_indices=[frame_idx],
            )
            self._track_meta[new_trk.track_id] = new_meta

        # ── Step 6: Remove DELETED tracks from active pool ───────────────────
        active_trackers = []
        for trk in self._trackers:
            meta = self._track_meta[trk.track_id]
            if meta.state != TrackState.DELETED:
                active_trackers.append(trk)

        self._trackers = active_trackers

        logger.debug(
            "SORT update complete",
            frame=frame_idx,
            detections=len(detections),
            active_tracks=len(self._trackers),
            confirmed=sum(
                1 for m in self._track_meta.values()
                if m.state == TrackState.CONFIRMED
            ),
        )

        return self._get_active_tracks()

    def get_confirmed_tracks(self) -> List[TrichomeTrack]:
        """Return only CONFIRMED tracks (appeared >= min_hits consecutive frames)."""
        return [
            m for m in self._track_meta.values()
            if m.state == TrackState.CONFIRMED
        ]

    def reset(self) -> None:
        """
        Clear all active tracks and reset the frame counter.

        Call when starting a new video or new tracking session.
        Does NOT reset the global KalmanBoxTracker ID counter (track IDs remain
        unique across sessions within the same process).
        """
        self._trackers.clear()
        self._track_meta.clear()
        self._frame_count = 0
        logger.debug("SORTTracker reset — all tracks cleared")

    # --- Static utilities ----------------------------------------------------

    @staticmethod
    def iou_matrix(
        bboxes_a: List[BoundingBox],
        bboxes_b: List[BoundingBox],
    ) -> np.ndarray:
        """
        Compute N×M IoU matrix between two lists of bounding boxes.

        Args:
            bboxes_a: N bounding boxes (rows).
            bboxes_b: M bounding boxes (columns).

        Returns:
            ndarray of shape (N, M) with float64 IoU values in [0, 1].
        """
        n = len(bboxes_a)
        m = len(bboxes_b)
        if n == 0 or m == 0:
            return np.zeros((n, m), dtype=np.float64)

        # Vectorised IoU: broadcast NxM comparisons
        # Convert to arrays [x_min, y_min, x_max, y_max]
        a = np.array([[b.x_min, b.y_min, b.x_max, b.y_max] for b in bboxes_a], dtype=np.float64)
        b = np.array([[b.x_min, b.y_min, b.x_max, b.y_max] for b in bboxes_b], dtype=np.float64)

        # Intersection
        inter_x_min = np.maximum(a[:, 0:1], b[:, 0])   # (N, M)
        inter_y_min = np.maximum(a[:, 1:2], b[:, 1])
        inter_x_max = np.minimum(a[:, 2:3], b[:, 2])
        inter_y_max = np.minimum(a[:, 3:4], b[:, 3])

        inter_w = np.maximum(0.0, inter_x_max - inter_x_min)
        inter_h = np.maximum(0.0, inter_y_max - inter_y_min)
        intersection = inter_w * inter_h  # (N, M)

        # Union
        area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])  # (N,)
        area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])  # (M,)
        union = area_a[:, None] + area_b[None, :] - intersection

        iou = np.where(union > 0, intersection / union, 0.0)
        return iou.astype(np.float64)

    @staticmethod
    def hungarian_assignment(
        cost_matrix: np.ndarray,
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Run Hungarian (linear sum) assignment on a cost matrix.

        Wraps scipy.optimize.linear_sum_assignment.

        Args:
            cost_matrix: (N, M) cost matrix — N detections × M tracks.
                         Typically 1 - IoU so that lower cost = better match.

        Returns:
            Tuple of:
            - matched_indices: list of (det_idx, trk_idx) pairs
            - unmatched_dets:  list of detection indices with no match
            - unmatched_trks:  list of track indices with no match
        """
        if cost_matrix.size == 0:
            return (
                [],
                list(range(cost_matrix.shape[0])),
                list(range(cost_matrix.shape[1])),
            )

        n_dets, n_trks = cost_matrix.shape
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched: List[Tuple[int, int]] = list(zip(row_ind.tolist(), col_ind.tolist()))

        matched_det_set = set(row_ind.tolist())
        matched_trk_set = set(col_ind.tolist())

        unmatched_dets = [i for i in range(n_dets) if i not in matched_det_set]
        unmatched_trks = [j for j in range(n_trks) if j not in matched_trk_set]

        return matched, unmatched_dets, unmatched_trks

    # --- Private helpers -----------------------------------------------------

    def _get_active_tracks(self) -> List[TrichomeTrack]:
        """Return all non-DELETED tracks in the current internal state."""
        active_ids = {trk.track_id for trk in self._trackers}
        return [
            meta for tid, meta in self._track_meta.items()
            if tid in active_ids
        ]

    def __repr__(self) -> str:
        n_active = len(self._trackers)
        n_confirmed = sum(
            1 for trk in self._trackers
            if self._track_meta[trk.track_id].state == TrackState.CONFIRMED
        )
        return (
            f"SORTTracker("
            f"active={n_active}, confirmed={n_confirmed}, "
            f"max_age={self.max_age}, min_hits={self.min_hits}, "
            f"iou_threshold={self.iou_threshold})"
        )
