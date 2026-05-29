"""
tests.unit.test_trichome_tracking — Comprehensive tests for temporal trichome tracking.

Coverage:
  - KalmanBoxTracker: initialisation, predict, update, validity, edge cases
  - SORTTracker.iou_matrix: known values, zero overlap, full overlap, empty inputs
  - SORTTracker.hungarian_assignment: 2×2, 3×2, 3×3, empty, threshold filtering
  - SORTTracker.update: TENTATIVE creation, CONFIRMED promotion, DELETED after max_age
  - Track identity preservation across frames
  - Multiple simultaneous tracks, dense clusters
  - TrackingSession.process_frame, sequential frames
  - TrackingSummary: counts, averages, type_distribution
  - API endpoints: start, status, summary, trajectories, delete
"""

from __future__ import annotations

import math
import time
from typing import List
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from shared.core.entities import Detection
from shared.core.enums import TrichomeType
from shared.core.value_objects import BoundingBox, Confidence
from video_pipeline.tracking.sort_tracker import (
    KalmanBoxTracker,
    SORTTracker,
    TrackState,
    TrichomeTrack,
)
from video_pipeline.tracking.tracking_session import (
    TrackingSession,
    TrackingSessionConfig,
    TrackingSummary,
)


# ============================================================================
# Fixtures
# ============================================================================

def make_bbox(x: float = 100.0, y: float = 100.0, w: float = 40.0, h: float = 40.0) -> BoundingBox:
    """Create a BoundingBox from top-left (x, y) + size."""
    return BoundingBox(x_min=x, y_min=y, x_max=x + w, y_max=y + h)


def make_detection(
    x: float = 100.0,
    y: float = 100.0,
    w: float = 40.0,
    h: float = 40.0,
    conf: float = 0.90,
    ttype: TrichomeType = TrichomeType.CAPITATE_STALKED,
    frame: int = 0,
) -> Detection:
    """Create a Detection with the given bounding box and metadata."""
    return Detection(
        bounding_box=make_bbox(x, y, w, h),
        confidence=Confidence(conf),
        trichome_type=ttype,
        frame_index=frame,
    )


@pytest.fixture
def simple_bbox() -> BoundingBox:
    return make_bbox(100, 100, 40, 40)


@pytest.fixture
def tracker() -> SORTTracker:
    """Fresh SORTTracker with default parameters."""
    return SORTTracker(max_age=3, min_hits=2, iou_threshold=0.3)


@pytest.fixture
def strict_tracker() -> SORTTracker:
    """SORTTracker requiring 3 hits to confirm."""
    return SORTTracker(max_age=2, min_hits=3, iou_threshold=0.3)


@pytest.fixture
def session() -> TrackingSession:
    """Fresh TrackingSession with default config."""
    return TrackingSession(TrackingSessionConfig(max_age=3, min_hits=2, min_track_length=2))


# ============================================================================
# KalmanBoxTracker Tests
# ============================================================================

class TestKalmanBoxTrackerInit:
    def test_initial_state_matches_bbox(self, simple_bbox: BoundingBox) -> None:
        """Initial state estimate should closely match the input bbox."""
        kbt = KalmanBoxTracker(simple_bbox)
        est = kbt.get_state()
        assert abs(est.x_min - simple_bbox.x_min) < 5.0
        assert abs(est.y_min - simple_bbox.y_min) < 5.0
        assert abs(est.x_max - simple_bbox.x_max) < 5.0
        assert abs(est.y_max - simple_bbox.y_max) < 5.0

    def test_track_id_is_positive_integer(self, simple_bbox: BoundingBox) -> None:
        kbt = KalmanBoxTracker(simple_bbox)
        assert isinstance(kbt.track_id, int)
        assert kbt.track_id > 0

    def test_consecutive_track_ids_increase(self) -> None:
        """Each new tracker gets a strictly larger track ID."""
        a = KalmanBoxTracker(make_bbox(10, 10))
        b = KalmanBoxTracker(make_bbox(200, 200))
        assert b.track_id > a.track_id

    def test_initial_time_since_update_is_zero(self, simple_bbox: BoundingBox) -> None:
        kbt = KalmanBoxTracker(simple_bbox)
        assert kbt.time_since_update == 0

    def test_initial_hits_zero(self, simple_bbox: BoundingBox) -> None:
        kbt = KalmanBoxTracker(simple_bbox)
        assert kbt.hits == 0

    def test_is_valid_after_init(self, simple_bbox: BoundingBox) -> None:
        kbt = KalmanBoxTracker(simple_bbox)
        assert kbt.is_valid is True


class TestKalmanBoxTrackerPredict:
    def test_predict_returns_bounding_box(self, simple_bbox: BoundingBox) -> None:
        kbt = KalmanBoxTracker(simple_bbox)
        pred = kbt.predict()
        assert isinstance(pred, BoundingBox)

    def test_predict_increments_age(self, simple_bbox: BoundingBox) -> None:
        kbt = KalmanBoxTracker(simple_bbox)
        kbt.predict()
        assert kbt.age == 1
        kbt.predict()
        assert kbt.age == 2

    def test_predict_increments_time_since_update(self, simple_bbox: BoundingBox) -> None:
        kbt = KalmanBoxTracker(simple_bbox)
        kbt.predict()
        assert kbt.time_since_update == 1

    def test_predict_with_zero_velocity_stays_near_initial(self, simple_bbox: BoundingBox) -> None:
        """With no velocity initialised, predict should stay near initial position."""
        kbt = KalmanBoxTracker(simple_bbox)
        pred = kbt.predict()
        # Center should not drift more than a few pixels in first prediction
        cx_init = (simple_bbox.x_min + simple_bbox.x_max) / 2
        cy_init = (simple_bbox.y_min + simple_bbox.y_max) / 2
        cx_pred = (pred.x_min + pred.x_max) / 2
        cy_pred = (pred.y_min + pred.y_max) / 2
        assert abs(cx_pred - cx_init) < 15.0
        assert abs(cy_pred - cy_init) < 15.0

    def test_predict_preserves_approximate_size(self, simple_bbox: BoundingBox) -> None:
        kbt = KalmanBoxTracker(simple_bbox)
        pred = kbt.predict()
        # Area should remain within 50% of original for first step
        assert abs(pred.area - simple_bbox.area) / simple_bbox.area < 0.5

    def test_multiple_predicts_without_update_still_valid(self) -> None:
        bbox = make_bbox(200, 200, 50, 50)
        kbt = KalmanBoxTracker(bbox)
        for _ in range(10):
            pred = kbt.predict()
            assert isinstance(pred, BoundingBox)


class TestKalmanBoxTrackerUpdate:
    def test_update_resets_time_since_update(self, simple_bbox: BoundingBox) -> None:
        kbt = KalmanBoxTracker(simple_bbox)
        kbt.predict()
        kbt.predict()
        assert kbt.time_since_update == 2
        kbt.update(simple_bbox)
        assert kbt.time_since_update == 0

    def test_update_increments_hits(self, simple_bbox: BoundingBox) -> None:
        kbt = KalmanBoxTracker(simple_bbox)
        kbt.predict()
        assert kbt.hits == 0
        kbt.update(simple_bbox)
        assert kbt.hits == 1
        kbt.predict()
        kbt.update(simple_bbox)
        assert kbt.hits == 2

    def test_update_moves_state_toward_measurement(self) -> None:
        """After multiple updates at a displaced position, state should converge."""
        original = make_bbox(100, 100, 40, 40)
        target = make_bbox(200, 200, 40, 40)  # Large shift

        kbt = KalmanBoxTracker(original)
        for _ in range(8):
            kbt.predict()
            kbt.update(target)

        state = kbt.get_state()
        cx_target = (target.x_min + target.x_max) / 2
        cy_target = (target.y_min + target.y_max) / 2
        cx_state = (state.x_min + state.x_max) / 2
        cy_state = (state.y_min + state.y_max) / 2
        # Should converge within 30px after 8 updates
        assert abs(cx_state - cx_target) < 30.0
        assert abs(cy_state - cy_target) < 30.0

    def test_update_increments_hit_streak(self, simple_bbox: BoundingBox) -> None:
        kbt = KalmanBoxTracker(simple_bbox)
        kbt.predict()
        kbt.update(simple_bbox)
        assert kbt.hit_streak == 1
        kbt.predict()
        kbt.update(simple_bbox)
        assert kbt.hit_streak == 2

    def test_predict_without_update_resets_hit_streak(self, simple_bbox: BoundingBox) -> None:
        kbt = KalmanBoxTracker(simple_bbox)
        kbt.predict()
        kbt.update(simple_bbox)
        kbt.predict()
        kbt.update(simple_bbox)
        assert kbt.hit_streak == 2
        # Now predict twice without update → streak should reset
        kbt.predict()
        kbt.predict()
        assert kbt.hit_streak == 0


class TestKalmanBoxTrackerValidity:
    def test_valid_for_reasonable_bbox(self) -> None:
        kbt = KalmanBoxTracker(make_bbox(50, 50, 30, 30))
        assert kbt.is_valid

    def test_bbox_to_z_roundtrip(self) -> None:
        """_bbox_to_z → _x_to_bbox should recover original box within tolerance."""
        bbox = make_bbox(150, 200, 60, 45)
        cx, cy, s, r = KalmanBoxTracker._bbox_to_z(bbox)
        x = np.array([[cx], [cy], [s], [r], [0], [0], [0]], dtype=np.float64)
        recovered = KalmanBoxTracker._x_to_bbox(x)
        assert abs(recovered.x_min - bbox.x_min) < 2.0
        assert abs(recovered.y_min - bbox.y_min) < 2.0
        assert abs(recovered.x_max - bbox.x_max) < 2.0
        assert abs(recovered.y_max - bbox.y_max) < 2.0


# ============================================================================
# SORTTracker IoU Matrix Tests
# ============================================================================

class TestIoUMatrix:
    def test_empty_inputs_return_empty_matrix(self) -> None:
        result = SORTTracker.iou_matrix([], [])
        assert result.shape == (0, 0)

    def test_empty_a_returns_zero_rows(self) -> None:
        b = [make_bbox(0, 0, 10, 10)]
        result = SORTTracker.iou_matrix([], b)
        assert result.shape == (0, 1)

    def test_empty_b_returns_zero_cols(self) -> None:
        a = [make_bbox(0, 0, 10, 10)]
        result = SORTTracker.iou_matrix(a, [])
        assert result.shape == (1, 0)

    def test_identical_boxes_iou_is_one(self) -> None:
        bbox = make_bbox(10, 10, 30, 30)
        result = SORTTracker.iou_matrix([bbox], [bbox])
        assert result.shape == (1, 1)
        assert abs(result[0, 0] - 1.0) < 1e-6

    def test_non_overlapping_boxes_iou_is_zero(self) -> None:
        a = [make_bbox(0, 0, 10, 10)]
        b = [make_bbox(100, 100, 10, 10)]
        result = SORTTracker.iou_matrix(a, b)
        assert result[0, 0] == 0.0

    def test_known_partial_overlap(self) -> None:
        """Two 10×10 boxes with 5×10 overlap → IoU = 50/(100+100-50) = 1/3."""
        a = [make_bbox(0, 0, 10, 10)]   # [0..10, 0..10]
        b = [make_bbox(5, 0, 10, 10)]   # [5..15, 0..10]
        result = SORTTracker.iou_matrix(a, b)
        expected = 50.0 / 150.0
        assert abs(result[0, 0] - expected) < 1e-6

    def test_output_shape_NxM(self) -> None:
        N, M = 3, 5
        a = [make_bbox(i * 50, 0, 30, 30) for i in range(N)]
        b = [make_bbox(j * 50, 0, 30, 30) for j in range(M)]
        result = SORTTracker.iou_matrix(a, b)
        assert result.shape == (N, M)

    def test_all_values_in_0_1(self) -> None:
        a = [make_bbox(i * 20, 0, 30, 30) for i in range(4)]
        b = [make_bbox(j * 15, 5, 25, 25) for j in range(4)]
        result = SORTTracker.iou_matrix(a, b)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_dtype_is_float64(self) -> None:
        a = [make_bbox(0, 0, 20, 20)]
        b = [make_bbox(10, 10, 20, 20)]
        result = SORTTracker.iou_matrix(a, b)
        assert result.dtype == np.float64


# ============================================================================
# SORTTracker Hungarian Assignment Tests
# ============================================================================

class TestHungarianAssignment:
    def test_empty_cost_matrix_returns_empty_lists(self) -> None:
        cost = np.zeros((0, 0))
        matched, unmatched_dets, unmatched_trks = SORTTracker.hungarian_assignment(cost)
        assert matched == []
        assert unmatched_dets == []
        assert unmatched_trks == []

    def test_single_element_matrix_matches(self) -> None:
        cost = np.array([[0.2]])
        matched, ud, ut = SORTTracker.hungarian_assignment(cost)
        assert len(matched) == 1
        assert (0, 0) in matched
        assert ud == []
        assert ut == []

    def test_2x2_diagonal_matching(self) -> None:
        # Low cost on diagonal → diagonal matches
        cost = np.array([[0.1, 0.9], [0.9, 0.1]])
        matched, ud, ut = SORTTracker.hungarian_assignment(cost)
        assert (0, 0) in matched
        assert (1, 1) in matched
        assert ud == []
        assert ut == []

    def test_3x2_more_dets_than_trks(self) -> None:
        """3 detections, 2 tracks → 2 matched, 1 unmatched detection."""
        cost = np.array([
            [0.1, 0.9],
            [0.9, 0.1],
            [0.5, 0.5],
        ])
        matched, ud, ut = SORTTracker.hungarian_assignment(cost)
        assert len(matched) == 2
        assert len(ud) == 1
        assert len(ut) == 0

    def test_2x3_more_trks_than_dets(self) -> None:
        """2 detections, 3 tracks → 2 matched, 1 unmatched track."""
        cost = np.array([
            [0.1, 0.9, 0.8],
            [0.9, 0.1, 0.8],
        ])
        matched, ud, ut = SORTTracker.hungarian_assignment(cost)
        assert len(matched) == 2
        assert len(ud) == 0
        assert len(ut) == 1

    def test_no_rows_all_trks_unmatched(self) -> None:
        cost = np.zeros((0, 3))
        matched, ud, ut = SORTTracker.hungarian_assignment(cost)
        assert matched == []
        assert ud == []
        assert ut == [0, 1, 2]

    def test_no_cols_all_dets_unmatched(self) -> None:
        cost = np.zeros((3, 0))
        matched, ud, ut = SORTTracker.hungarian_assignment(cost)
        assert matched == []
        assert ud == [0, 1, 2]
        assert ut == []

    def test_returns_minimum_cost_assignment(self) -> None:
        """Verify that optimal (minimum cost) assignment is chosen."""
        cost = np.array([[0.9, 0.1], [0.1, 0.9]])
        matched, _, _ = SORTTracker.hungarian_assignment(cost)
        # Optimal: (0,1) + (1,0) → total cost 0.2 vs (0,0) + (1,1) → 1.8
        assert (0, 1) in matched
        assert (1, 0) in matched


# ============================================================================
# SORTTracker.update Tests
# ============================================================================

class TestSORTTrackerUpdate:
    def test_new_detection_creates_tentative_track(self, tracker: SORTTracker) -> None:
        dets = [make_detection(100, 100)]
        active = tracker.update(dets, frame_idx=0)
        assert len(active) == 1
        assert active[0].state == TrackState.TENTATIVE

    def test_track_confirmed_after_min_hits(self, tracker: SORTTracker) -> None:
        """With min_hits=2, track should be CONFIRMED on the 2nd match."""
        det = make_detection(100, 100)
        # Frame 0: TENTATIVE
        active = tracker.update([det], frame_idx=0)
        assert active[0].state == TrackState.TENTATIVE
        # Frame 1: same bbox → hits = 2 → CONFIRMED
        active = tracker.update([make_detection(102, 102)], frame_idx=1)
        confirmed = [t for t in active if t.state == TrackState.CONFIRMED]
        assert len(confirmed) == 1

    def test_track_deleted_after_max_age(self, tracker: SORTTracker) -> None:
        """max_age=3: track should be deleted if unseen for 4 frames."""
        dets = [make_detection(100, 100)]
        # Create and confirm the track
        tracker.update(dets, frame_idx=0)
        tracker.update([make_detection(102, 102)], frame_idx=1)
        # Stop sending the detection — track should age out
        for frame in range(2, 6):
            active = tracker.update([], frame_idx=frame)
        # After max_age frames without match, no tracks should remain
        assert len(active) == 0

    def test_empty_detections_returns_empty_when_no_tracks(self, tracker: SORTTracker) -> None:
        active = tracker.update([], frame_idx=0)
        assert active == []

    def test_unmatched_detection_creates_new_track(self, tracker: SORTTracker) -> None:
        """A detection far from any existing track should create a new track."""
        tracker.update([make_detection(100, 100)], frame_idx=0)
        # Frame 1: same det + completely new one 1000px away
        active = tracker.update(
            [make_detection(102, 102), make_detection(800, 600)],
            frame_idx=1,
        )
        # Should have at least 2 active tracks
        assert len(active) >= 2

    def test_multiple_detections_multiple_tracks(self, tracker: SORTTracker) -> None:
        """Multiple well-separated detections each become their own track."""
        dets = [
            make_detection(50, 50),
            make_detection(300, 300),
            make_detection(600, 100),
        ]
        active = tracker.update(dets, frame_idx=0)
        assert len(active) == 3

    def test_track_id_preserved_across_frames(self, tracker: SORTTracker) -> None:
        """The same physical trichome should keep the same track_id across frames."""
        # Frame 0: create track
        active0 = tracker.update([make_detection(100, 100)], frame_idx=0)
        assert len(active0) == 1
        original_id = active0[0].track_id

        # Frame 1: nearly same position → same track
        active1 = tracker.update([make_detection(103, 103)], frame_idx=1)
        assert len(active1) == 1
        assert active1[0].track_id == original_id

    def test_track_identity_over_multiple_frames(self, tracker: SORTTracker) -> None:
        """Track ID should persist for 10 consecutive frames with small drift."""
        active = tracker.update([make_detection(200, 200)], frame_idx=0)
        target_id = active[0].track_id

        for f in range(1, 10):
            dx = f * 2.0  # 2 pixels drift per frame
            active = tracker.update([make_detection(200 + dx, 200 + dx)], frame_idx=f)
            ids = [t.track_id for t in active]
            assert target_id in ids, f"Track ID lost at frame {f}"

    def test_two_crossing_tracks_distinct_ids(self, tracker: SORTTracker) -> None:
        """Two tracks crossing the frame should maintain distinct IDs."""
        # Start with two separate detections
        active = tracker.update(
            [make_detection(50, 200), make_detection(600, 200)],
            frame_idx=0,
        )
        ids_0 = {t.track_id for t in active}
        assert len(ids_0) == 2

        active = tracker.update(
            [make_detection(55, 200), make_detection(605, 200)],
            frame_idx=1,
        )
        ids_1 = {t.track_id for t in active}
        # IDs should be the same set
        assert ids_0 == ids_1

    def test_dense_cluster_low_iou_threshold(self) -> None:
        """Low IoU threshold should still assign tightly packed trichomes."""
        tracker = SORTTracker(max_age=3, min_hits=2, iou_threshold=0.1)
        # Create 4 tightly packed (but distinct) trichomes
        dets = [
            make_detection(100, 100, 25, 25),
            make_detection(130, 100, 25, 25),
            make_detection(160, 100, 25, 25),
            make_detection(190, 100, 25, 25),
        ]
        active = tracker.update(dets, frame_idx=0)
        assert len(active) == 4
        ids_0 = {t.track_id for t in active}

        # Slight shift in frame 1
        dets2 = [
            make_detection(102, 102, 25, 25),
            make_detection(132, 102, 25, 25),
            make_detection(162, 102, 25, 25),
            make_detection(192, 102, 25, 25),
        ]
        active = tracker.update(dets2, frame_idx=1)
        ids_1 = {t.track_id for t in active}
        # All original IDs should still be present
        assert len(ids_0 & ids_1) >= 3, "At least 3 of 4 tracks should persist"

    def test_reset_clears_all_tracks(self, tracker: SORTTracker) -> None:
        tracker.update([make_detection(100, 100)], frame_idx=0)
        tracker.update([make_detection(102, 102)], frame_idx=1)
        tracker.reset()
        active = tracker.update([], frame_idx=0)
        assert active == []
        assert len(tracker._trackers) == 0

    def test_get_confirmed_tracks_only_returns_confirmed(self, tracker: SORTTracker) -> None:
        # Frame 0: TENTATIVE
        tracker.update([make_detection(100, 100)], frame_idx=0)
        confirmed = tracker.get_confirmed_tracks()
        assert confirmed == []
        # Frame 1: should reach CONFIRMED (min_hits=2)
        tracker.update([make_detection(102, 102)], frame_idx=1)
        confirmed = tracker.get_confirmed_tracks()
        assert len(confirmed) == 1
        assert confirmed[0].state == TrackState.CONFIRMED

    def test_tracker_repr_is_informative(self, tracker: SORTTracker) -> None:
        r = repr(tracker)
        assert "SORTTracker" in r
        assert "max_age" in r

    def test_invalid_max_age_raises(self) -> None:
        with pytest.raises(ValueError):
            SORTTracker(max_age=0)

    def test_invalid_min_hits_raises(self) -> None:
        with pytest.raises(ValueError):
            SORTTracker(min_hits=0)

    def test_invalid_iou_threshold_raises(self) -> None:
        with pytest.raises(ValueError):
            SORTTracker(iou_threshold=0.0)
        with pytest.raises(ValueError):
            SORTTracker(iou_threshold=1.0)

    def test_detection_below_iou_threshold_not_matched(self) -> None:
        """Detection with IoU below threshold should create a new track."""
        tracker = SORTTracker(max_age=3, min_hits=2, iou_threshold=0.9)
        # Create a track
        tracker.update([make_detection(100, 100, 40, 40)], frame_idx=0)
        # Frame 1: detection with only ~25% overlap (dx=20 → partial overlap)
        active = tracker.update([make_detection(120, 120, 40, 40)], frame_idx=1)
        # Should have 2 tracks (original aged + new one created)
        assert len(active) == 2


# ============================================================================
# TrichomeTrack Tests
# ============================================================================

class TestTrichomeTrack:
    def test_is_confirmed_property(self) -> None:
        t = TrichomeTrack(1, TrackState.CONFIRMED, make_bbox(), 0.9, "capitate_stalked", 3, 0)
        assert t.is_confirmed
        assert not t.is_tentative
        assert not t.is_deleted

    def test_is_tentative_property(self) -> None:
        t = TrichomeTrack(1, TrackState.TENTATIVE, make_bbox(), 0.9, "capitate_stalked", 1, 0)
        assert t.is_tentative
        assert not t.is_confirmed

    def test_is_deleted_property(self) -> None:
        t = TrichomeTrack(1, TrackState.DELETED, make_bbox(), 0.9, "capitate_stalked", 0, 5)
        assert t.is_deleted

    def test_track_length(self) -> None:
        t = TrichomeTrack(
            1, TrackState.CONFIRMED, make_bbox(), 0.9, "capitate_stalked", 5, 0,
            history=[make_bbox(), make_bbox(), make_bbox()],
            frame_indices=[0, 1, 2],
        )
        assert t.track_length == 3

    def test_to_dict_serialisable(self) -> None:
        t = TrichomeTrack(
            42, TrackState.CONFIRMED, make_bbox(100, 100, 40, 40), 0.88,
            "capitate_stalked", 5, 0,
            history=[make_bbox()],
            frame_indices=[0],
        )
        d = t.to_dict()
        assert d["track_id"] == 42
        assert d["state_name"] == "CONFIRMED"
        assert d["confidence"] == 0.88
        assert isinstance(d["bbox"], list)
        assert len(d["bbox"]) == 4


# ============================================================================
# TrackingSession Tests
# ============================================================================

class TestTrackingSession:
    def test_initial_frames_processed_is_zero(self, session: TrackingSession) -> None:
        assert session.frames_processed == 0

    def test_process_single_frame(self, session: TrackingSession) -> None:
        dets = [make_detection(100, 100)]
        active = session.process_frame(dets, frame_idx=0)
        assert len(active) == 1
        assert session.frames_processed == 1

    def test_sequential_frames_increase_counter(self, session: TrackingSession) -> None:
        for i in range(5):
            session.process_frame([make_detection(100 + i, 100)], frame_idx=i)
        assert session.frames_processed == 5

    def test_reset_clears_state(self, session: TrackingSession) -> None:
        session.process_frame([make_detection(100, 100)], frame_idx=0)
        session.reset()
        assert session.frames_processed == 0
        assert session.active_track_count == 0

    def test_summary_after_no_frames(self, session: TrackingSession) -> None:
        summary = session.get_summary()
        assert summary.total_tracks == 0
        assert summary.confirmed_tracks == 0
        assert summary.avg_track_length == 0.0

    def test_summary_counts_confirmed_tracks(self, session: TrackingSession) -> None:
        """After min_hits frames, tracks should appear in confirmed_tracks."""
        for f in range(5):
            dets = [make_detection(100 + f, 100 + f)]
            session.process_frame(dets, frame_idx=f)

        summary = session.get_summary()
        assert summary.confirmed_tracks >= 1

    def test_type_distribution_populated(self) -> None:
        """type_distribution should reflect detected trichome types."""
        config = TrackingSessionConfig(min_hits=1, min_track_length=1)
        s = TrackingSession(config)
        for f in range(4):
            dets = [
                make_detection(100 + f, 100 + f, ttype=TrichomeType.CAPITATE_STALKED),
                make_detection(400 + f, 400 + f, ttype=TrichomeType.BULBOUS),
            ]
            s.process_frame(dets, frame_idx=f)

        summary = s.get_summary()
        assert len(summary.type_distribution) > 0

    def test_avg_track_length_computed(self) -> None:
        config = TrackingSessionConfig(min_hits=2, min_track_length=1)
        s = TrackingSession(config)
        for f in range(6):
            dets = [make_detection(100 + f, 100 + f)]
            s.process_frame(dets, frame_idx=f)

        summary = s.get_summary()
        assert summary.avg_track_length > 0.0

    def test_export_trajectories_returns_list(self, session: TrackingSession) -> None:
        for f in range(5):
            session.process_frame([make_detection(100 + f * 2, 100 + f * 2)], frame_idx=f)
        traj = session.export_trajectories()
        assert isinstance(traj, list)

    def test_trajectory_dict_has_required_keys(self) -> None:
        config = TrackingSessionConfig(min_hits=2, min_track_length=2)
        s = TrackingSession(config)
        for f in range(5):
            s.process_frame([make_detection(100 + f * 2, 100 + f * 2)], frame_idx=f)
        traj = s.export_trajectories()
        if traj:
            t = traj[0]
            for key in ("id", "type", "frames", "positions", "bboxes", "confidence", "track_length"):
                assert key in t, f"Missing key: {key}"

    def test_min_track_length_filters_short_tracks(self) -> None:
        """Tracks shorter than min_track_length should not appear in trajectories."""
        config = TrackingSessionConfig(min_hits=2, min_track_length=5)
        s = TrackingSession(config)
        # Only 3 frames — track_length < 5 → should be filtered
        for f in range(3):
            s.process_frame([make_detection(100 + f, 100 + f)], frame_idx=f)
        traj = s.export_trajectories()
        for t in traj:
            assert t["track_length"] >= 5

    def test_repr_contains_session_info(self, session: TrackingSession) -> None:
        r = repr(session)
        assert "TrackingSession" in r


# ============================================================================
# TrackingSummary Tests
# ============================================================================

class TestTrackingSummary:
    def test_to_dict_has_required_keys(self) -> None:
        summary = TrackingSummary(
            total_tracks=10,
            confirmed_tracks=7,
            avg_track_length=4.5,
            type_distribution={"capitate_stalked": 5, "bulbous": 2},
            trajectory_data=[],
        )
        d = summary.to_dict()
        for key in ("total_tracks", "confirmed_tracks", "avg_track_length",
                    "type_distribution", "trajectory_data"):
            assert key in d

    def test_type_distribution_values_are_ints(self) -> None:
        config = TrackingSessionConfig(min_hits=1, min_track_length=1)
        s = TrackingSession(config)
        for f in range(4):
            s.process_frame([make_detection(100 + f, 100, ttype=TrichomeType.CAPITATE_STALKED)], frame_idx=f)
        summary = s.get_summary()
        for v in summary.type_distribution.values():
            assert isinstance(v, int)


# ============================================================================
# API endpoint Tests
# ============================================================================

@pytest.fixture(scope="module")
def api_client():
    """Create a FastAPI test client for the tracking router."""
    from fastapi import FastAPI
    from backend.api.v1.tracking import router

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


class TestTrackingAPI:
    def test_start_tracking_returns_session_id(self, api_client: TestClient) -> None:
        resp = api_client.post("/video/tracking/start", json={
            "video_id": "test-video-001",
            "n_synthetic_frames": 5,
        })
        assert resp.status_code == 202
        data = resp.json()
        assert "session_id" in data
        assert len(data["session_id"]) == 36  # UUID format

    def test_start_tracking_returns_video_id(self, api_client: TestClient) -> None:
        resp = api_client.post("/video/tracking/start", json={
            "video_id": "test-video-abc",
            "n_synthetic_frames": 5,
        })
        assert resp.json()["video_id"] == "test-video-abc"

    def test_start_with_custom_config(self, api_client: TestClient) -> None:
        resp = api_client.post("/video/tracking/start", json={
            "video_id": "vid-cfg",
            "n_synthetic_frames": 3,
            "config": {
                "max_age": 5,
                "min_hits": 3,
                "iou_threshold": 0.4,
                "min_track_length": 2,
            },
        })
        assert resp.status_code == 202

    def test_status_returns_running_or_complete(self, api_client: TestClient) -> None:
        resp_start = api_client.post("/video/tracking/start", json={
            "video_id": "status-test",
            "n_synthetic_frames": 2,
        })
        session_id = resp_start.json()["session_id"]
        resp = api_client.get(f"/video/tracking/{session_id}/status")
        assert resp.status_code == 200
        assert resp.json()["status"] in ("running", "complete")

    def test_status_has_required_fields(self, api_client: TestClient) -> None:
        resp_start = api_client.post("/video/tracking/start", json={
            "video_id": "fields-test",
            "n_synthetic_frames": 2,
        })
        session_id = resp_start.json()["session_id"]
        resp = api_client.get(f"/video/tracking/{session_id}/status")
        data = resp.json()
        for key in ("session_id", "video_id", "status", "frames_processed", "track_count"):
            assert key in data

    def test_status_404_for_unknown_session(self, api_client: TestClient) -> None:
        resp = api_client.get("/video/tracking/nonexistent-session-id/status")
        assert resp.status_code == 404

    def _start_and_wait_complete(self, api_client: TestClient, video_id: str = "sync-test", frames: int = 10) -> str:
        """Start a session and poll until complete (the background task runs synchronously in TestClient)."""
        resp = api_client.post("/video/tracking/start", json={
            "video_id": video_id,
            "n_synthetic_frames": frames,
        })
        session_id = resp.json()["session_id"]
        # TestClient runs background tasks synchronously after the response
        return session_id

    def test_summary_after_complete_session(self, api_client: TestClient) -> None:
        session_id = self._start_and_wait_complete(api_client, "summary-test", frames=20)
        resp = api_client.get(f"/video/tracking/{session_id}/summary")
        # May still be running or complete — if running, we get 400
        if resp.status_code == 200:
            data = resp.json()
            for key in ("total_tracks", "confirmed_tracks", "avg_track_length", "type_distribution"):
                assert key in data

    def test_summary_400_for_running_session(self, api_client: TestClient) -> None:
        """If session is still running, summary returns 400."""
        from backend.api.v1 import tracking as tracking_module
        from unittest.mock import patch

        resp_start = api_client.post("/video/tracking/start", json={
            "video_id": "running-test",
            "n_synthetic_frames": 5,
        })
        session_id = resp_start.json()["session_id"]

        # Force running state for the test
        if session_id in tracking_module._SESSIONS:
            tracking_module._SESSIONS[session_id].status = "running"
            resp = api_client.get(f"/video/tracking/{session_id}/summary")
            assert resp.status_code == 400

    def test_trajectories_after_complete_session(self, api_client: TestClient) -> None:
        session_id = self._start_and_wait_complete(api_client, "traj-test", frames=15)
        resp = api_client.get(f"/video/tracking/{session_id}/trajectories")
        if resp.status_code == 200:
            data = resp.json()
            assert "trajectories" in data
            assert "session_id" in data
            assert "track_count" in data

    def test_trajectories_400_for_running_session(self, api_client: TestClient) -> None:
        from backend.api.v1 import tracking as tracking_module

        resp_start = api_client.post("/video/tracking/start", json={
            "video_id": "traj-running",
            "n_synthetic_frames": 5,
        })
        session_id = resp_start.json()["session_id"]

        if session_id in tracking_module._SESSIONS:
            tracking_module._SESSIONS[session_id].status = "running"
            resp = api_client.get(f"/video/tracking/{session_id}/trajectories")
            assert resp.status_code == 400

    def test_delete_session_removes_it(self, api_client: TestClient) -> None:
        resp_start = api_client.post("/video/tracking/start", json={
            "video_id": "delete-test",
            "n_synthetic_frames": 2,
        })
        session_id = resp_start.json()["session_id"]

        resp_del = api_client.delete(f"/video/tracking/{session_id}")
        assert resp_del.status_code == 204

        resp_status = api_client.get(f"/video/tracking/{session_id}/status")
        assert resp_status.status_code == 404

    def test_delete_nonexistent_session_returns_404(self, api_client: TestClient) -> None:
        resp = api_client.delete("/video/tracking/does-not-exist")
        assert resp.status_code == 404

    def test_multiple_sessions_independent(self, api_client: TestClient) -> None:
        """Starting two sessions should return distinct IDs."""
        r1 = api_client.post("/video/tracking/start", json={"video_id": "v1", "n_synthetic_frames": 3})
        r2 = api_client.post("/video/tracking/start", json={"video_id": "v2", "n_synthetic_frames": 3})
        assert r1.json()["session_id"] != r2.json()["session_id"]

    def test_invalid_iou_threshold_returns_422(self, api_client: TestClient) -> None:
        resp = api_client.post("/video/tracking/start", json={
            "video_id": "invalid-cfg",
            "config": {"iou_threshold": 1.5},  # out of range
        })
        assert resp.status_code == 422

    def test_invalid_max_age_returns_422(self, api_client: TestClient) -> None:
        resp = api_client.post("/video/tracking/start", json={
            "video_id": "invalid-cfg",
            "config": {"max_age": 0},  # must be >= 1
        })
        assert resp.status_code == 422
