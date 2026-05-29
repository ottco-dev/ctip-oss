## Temporal Tracking

Track individual trichomes across consecutive video frames using SORT (Simple Online and Realtime Tracking): a Kalman filter predicts the next position of each track and the Hungarian algorithm assigns incoming detections to existing tracks.

---

## How it works

```
Video frame N
    │
    ▼
YOLO11s detection (tiled, per frame)
    │
    ▼
SORT tracker
    │  Kalman filter: predict next bounding box position
    │  Hungarian algorithm: match detections → active tracks (IoU cost matrix)
    │
    ▼
Track state update
    │  TENTATIVE  → confirmed after min_hits consecutive matches
    │  CONFIRMED  → active track carrying trajectory history
    │  DELETED    → removed after max_age consecutive misses
    │
    ▼
Trajectory data (per-track positions over time)
    │
    ▼
Summary: count, movement stats, development monitoring
```

---

## Configuration

| Parameter | Default | Description |
|---|---|---|
| `max_age` | 3 | Frames a track survives without a matching detection before deletion |
| `min_hits` | 2 | Consecutive matched frames required to confirm a track |
| `iou_threshold` | 0.3 | Minimum IoU for a detection–track assignment to be accepted |
| `min_track_length` | 3 | Minimum frames a track must span to appear in the summary |

---

## API reference

### Start a tracking session

```bash
POST /api/v1/video/tracking/start
Content-Type: application/json

{
  "video_path": "data/raw/videos/session_01.mp4",
  "max_age": 3,
  "min_hits": 2,
  "iou_threshold": 0.3,
  "min_track_length": 3,
  "model": "yolo11s"
}
```

Response:
```json
{
  "session_id": "trk_abc123",
  "status": "queued",
  "frame_count": 240
}
```

### Check session status

```bash
GET /api/v1/video/tracking/{session_id}/status
```

```json
{
  "session_id": "trk_abc123",
  "status": "running",
  "frames_processed": 120,
  "active_tracks": 18
}
```

### Get session summary

```bash
GET /api/v1/video/tracking/{session_id}/summary
```

```json
{
  "session_id": "trk_abc123",
  "total_tracks": 34,
  "confirmed_tracks": 29,
  "mean_track_length_frames": 47.3,
  "mean_displacement_px": 2.1,
  "frames_analyzed": 240
}
```

### Get trajectory data

```bash
GET /api/v1/video/tracking/{session_id}/trajectories
```

Returns an array of `TrichomeTrack` objects with per-frame bounding box positions, suitable for SVG overlay rendering.

### Delete session

```bash
DELETE /api/v1/video/tracking/{session_id}
```

---

## TrichomeTrack schema

```json
{
  "track_id": 7,
  "state": "CONFIRMED",
  "first_frame": 4,
  "last_frame": 198,
  "trajectory_data": [
    { "frame": 4,  "bbox": [112, 88, 145, 121], "confidence": 0.91 },
    { "frame": 5,  "bbox": [113, 89, 146, 122], "confidence": 0.89 }
  ],
  "mean_confidence": 0.87,
  "total_displacement_px": 4.3
}
```

Track states:

| State | Meaning |
|---|---|
| `TENTATIVE` | Fewer than `min_hits` consecutive matches; not yet reported |
| `CONFIRMED` | Sufficient matches; included in summary and trajectories |
| `DELETED` | Exceeded `max_age` without a match; archived |

---

## Frontend: Video page → Tracking tab

1. **Session setup** — configure `max_age`, `min_hits`, `iou_threshold`, `min_track_length`, select video file.
2. **Live progress** — frames processed / total, active track count (WebSocket `/ws/jobs`).
3. **Trajectory table** — one row per confirmed track: ID, length (frames), mean confidence, displacement.
4. **SVG bar chart** — track lifetimes visualised as horizontal bars across the frame timeline.
5. **Overlay export** — trajectory data can be exported for external annotation overlay tools.

---

## Use cases

- **Development monitoring**: follow the same trichome instances across a time-lapse to observe maturity progression.
- **Movement artefact detection**: high `mean_displacement_px` indicates camera shake or sample drift that may invalidate measurements.
- **Population dynamics**: track entry/exit of trichomes across the field of view during slow panning acquisitions.

---

## Notes

- Tracking operates on the video pipeline's frame cache; frames are processed in sequence, not batched.
- GPU semaphore (`asyncio.Semaphore(1)`) is held for the duration of the tracking job — no concurrent GPU tasks will run.
- For long videos, use `min_track_length` ≥ 5 to suppress spurious short tracks caused by false-positive detections.
