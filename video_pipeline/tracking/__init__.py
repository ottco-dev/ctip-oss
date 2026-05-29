"""
video_pipeline.tracking — Temporal trichome tracking across video frames.

Implements SORT (Simple Online Realtime Tracking) with Kalman filtering
and Hungarian assignment for robust multi-object tracking in microscopy
video streams.

Designed for trichome-specific challenges:
- Dense clusters with low inter-object IoU
- Microscope stage drift between frames
- Trichomes entering and leaving the field of view
- Occlusion by overlapping structures

References:
  Bewley, A. et al. (2016). Simple Online and Realtime Tracking.
  IEEE ICIP 2016. https://arxiv.org/abs/1602.00763

  Kalman, R.E. (1960). A New Approach to Linear Filtering and Prediction.
  Journal of Basic Engineering, 82(1), 35-45.
"""

from video_pipeline.tracking.sort_tracker import (
    TrackState,
    TrichomeTrack,
    KalmanBoxTracker,
    SORTTracker,
)
from video_pipeline.tracking.tracking_session import (
    TrackingSessionConfig,
    TrackingSummary,
    TrackingSession,
)

__all__ = [
    "TrackState",
    "TrichomeTrack",
    "KalmanBoxTracker",
    "SORTTracker",
    "TrackingSessionConfig",
    "TrackingSummary",
    "TrackingSession",
]
