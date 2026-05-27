> Full interactive API documentation: **http://localhost:8000/docs** (Swagger UI)
> ReDoc alternative: **http://localhost:8000/redoc**

All endpoints are under `/api/v1/`.

---

## Setup endpoints

### `GET /setup/status`

Check if setup is complete.

```json
{ "completed": true, "env_exists": true, "configured_keys": ["CUDA_DEVICE", "..."] }
```

### `GET /setup/config`

Read current `.env` configuration. Sensitive keys are redacted.

### `POST /setup/configure`

Write settings to `.env`.

```json
{
  "settings": {
    "CUDA_DEVICE": "cuda:0",
    "VRAM_LIMIT_GB": "8.0",
    "ENVIRONMENT": "development"
  },
  "mark_setup_complete": true
}
```

### `GET /setup/system-check`

Run dependency check. Returns list of check items.

### `GET /setup/docker/status`

Docker availability, group membership, running containers.

### `POST /setup/docker/start-annotation`

Start annotation container stack.

```json
{ "profile": "annotation" }
```

### `GET /setup/models/status`

Model catalog with present/missing status. Returns `list[ModelInfo]`.

### `POST /setup/models/download`

Start background model download. Returns task ID.

```json
{ "model_id": "yolo11s" }
```

### `GET /setup/models/download/{task_id}`

Poll download progress.

```json
{
  "task_id": "abc-123",
  "status": "downloading",
  "progress": 67,
  "filename": "yolo11s.pt",
  "size_mb": 18.4,
  "downloaded_mb": 12.3,
  "detail": "Downloading… 12.3 / 18.4 MB"
}
```

### `POST /setup/label-studio/create-account`

Create Label Studio account.

```json
{ "url": "http://localhost:3005", "email": "admin@example.com", "password": "secret" }
```

Response:
```json
{ "ok": true, "token": "abc123...", "already_existed": false }
```

### `POST /setup/label-studio/test`

Test Label Studio connection and authentication.

### `POST /setup/label-studio/create-project`

Create trichome annotation project with pre-built label config.

### `GET /setup/verification`

Live health check of all configured services. Returns latency + HTTP status per service.

---

## Detection endpoints

### `POST /detection/analyze`

Analyze an uploaded image.

```bash
curl -X POST http://localhost:8000/api/v1/detection/analyze \\
  -F "image=@IMG_0001.tif" \\
  -F "model=yolo11s" \\
  -F "tiled=true" \\
  -F "conf=0.25"
```

Response:
```json
{
  "job_id": "xyz-456",
  "status": "queued",
  "estimated_seconds": 8
}
```

### `GET /detection/jobs/{job_id}`

Poll detection job status.

### `GET /detection/results/{job_id}`

Retrieve detection results (JSON).

---

## Training endpoints

### `POST /training/start`

```json
{ "config": "yolo11s_detection.yaml", "dataset": "v1" }
```

### `GET /training/jobs`

List all training jobs with status.

### `DELETE /training/jobs/{job_id}`

Stop a running training job.

---

## System endpoints

### `GET /system/health`

```json
{ "status": "ok", "gpu_available": true, "gpu_busy": false }
```

### `GET /system/gpu`

```json
{
  "name": "NVIDIA GeForce RTX 4060",
  "vram_total_gb": 8.0,
  "vram_used_gb": 2.3,
  "vram_free_gb": 5.7,
  "utilization_pct": 0
}
```

---

## WebSocket endpoints

All WebSockets at `ws://localhost:8000/ws/`:

| Endpoint | Data stream |
|---------|------------|
| `/ws/training` | Live training metrics (loss, mAP, epoch) |
| `/ws/system` | GPU/RAM usage every 2s |
| `/ws/jobs` | Background job progress updates |
| `/ws/logs` | Live log stream |

```javascript
// Example: live GPU monitoring
const ws = new WebSocket('ws://localhost:8000/ws/system');
ws.onmessage = (e) => {
  const { vram_used_gb, utilization_pct } = JSON.parse(e.data);
  console.log(vram_used_gb, utilization_pct);
};
```
