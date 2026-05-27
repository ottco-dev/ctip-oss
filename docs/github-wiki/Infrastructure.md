## Service map

```
Internet / Browser
    │
    ▼
nginx (:3001)  ←── entry point for all traffic
    │
    ├── /api/v1/*  ────────────► FastAPI (:8000)
    │   /ws/*                         │
    │                         ┌───────┴────────┐
    │                    SQLite DB        GPU semaphore
    │                    (trichome.db)    (max 1 concurrent)
    │
    └── /*  ────────────────► Next.js (:3000)
                                    │
                             React components
                             @tanstack/react-query
                             zustand state
                             native WebSockets

Docker services (optional — annotation profile):
    ├── Label Studio (:3005)
    ├── CVAT         (:3006)
    └── PostgreSQL   (:5432, internal only)

MLflow (:3004)  ←── started by dev-start.sh (uvicorn, not Docker)
```

---

## nginx

### Local development (nginx-local/)

CTIP ships a user-space nginx config at `nginx-local/nginx.conf`.
No root required — runs as your user, PID stored in `nginx-local/nginx.pid`.

```nginx
server {
    listen 3001;
    server_name your-domain.com localhost _;

    # API proxy
    location /api/v1/ {
        limit_req zone=api burst=60 nodelay;
        proxy_pass http://127.0.0.1:8000;
        proxy_read_timeout 300s;
    }

    # WebSocket proxy
    location ~ ^/(ws|api/v1/ws)/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection $connection_upgrade;
        proxy_read_timeout 3600s;
    }

    # Frontend (Next.js with HMR support)
    location / {
        proxy_pass http://127.0.0.1:3000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection $connection_upgrade;
        proxy_read_timeout 3600s;
    }
}
```

### Public domain configuration

To expose CTIP on a public domain:

1. Set in `.env`: `PUBLIC_DOMAIN=mylab.ddns.net`
2. In nginx config, change `server_name` to your domain
3. Add SSL (Let's Encrypt recommended):

```bash
sudo certbot --nginx -d mylab.ddns.net
```

4. Forward port 3001 in your router to the machine running CTIP

### nginx management

```bash
# Start
nginx -c "$(pwd)/nginx-local/nginx.conf"

# Reload config without downtime
kill -HUP $(cat nginx-local/nginx.pid)

# Stop
kill $(cat nginx-local/nginx.pid)

# Check config syntax
nginx -t -c "$(pwd)/nginx-local/nginx.conf"
```

---

## FastAPI Backend (:8000)

Entry point: `backend/main.py`

```python
# Startup sequence (lifespan)
1. Directories created (DATA_ROOT, MODELS_DIR, OUTPUTS_DIR)
2. Database initialized (SQLite via SQLModel)
3. GPU broadcast loop started (WebSocket heartbeat)
4. asyncio.Semaphore(1) created (GPU guard)

# Router structure
/api/v1/
    setup/          # Installation wizard endpoints
    detection/      # YOLO inference
    segmentation/   # SAM2 inference
    maturity/       # Trichome maturity classification
    training/       # Training job management
    annotation/     # Label Studio integration
    analytics/      # Aggregated statistics
    system/         # Health, GPU status

/ws/
    training        # Live training metrics
    system          # GPU/RAM usage stream
    jobs            # Background job progress
    logs            # Live log stream
```

### GPU guard (asyncio.Semaphore)

```python
# backend/middleware/gpu_guard.py
_GPU_SEMAPHORE = asyncio.Semaphore(1)

async def with_gpu(fn, *args):
    async with _GPU_SEMAPHORE:
        return await fn(*args)
```

Only one GPU task runs at a time. Subsequent requests wait in queue.
This is intentional for RTX 4060 (8 GB VRAM) — prevents OOM crashes.

---

## Docker services

### docker-compose profiles

```bash
# Annotation stack (Label Studio + CVAT + PostgreSQL)
docker compose --profile annotation up -d

# Training stack (GPU YOLO trainer)
docker compose -f docker-compose.yml -f docker-compose.training.yml up -d

# Inference-only (no frontend, no annotation)
docker compose -f docker-compose.inference.yml up -d
```

### Port mapping

| Container | Host port | Container port | Service |
|-----------|-----------|----------------|---------|
| nginx | 3001 | 80 | Reverse proxy |
| backend | 3002 | 8000 | FastAPI |
| frontend | 3003 | 3000 | Next.js |
| mlflow | 3004 | 5000 | MLflow |
| label-studio | 3005 | 8080 | Label Studio |
| cvat | 3006 | 8080 | CVAT |
| postgres | internal | 5432 | PostgreSQL |

**Note**: In local dev mode (dev-start.sh), backend is on :8000 directly, not :3002.

### Label Studio Docker

```yaml
# docker/docker-compose.yml (annotation profile)
label-studio:
  image: heartexlabs/label-studio:1.13.1
  ports:
    - "3005:8080"
  volumes:
    - ls-data:/label-studio/data
  environment:
    - LABEL_STUDIO_HOST=http://localhost:3005
```

Label Studio stores all projects, tasks, and annotations in the `ls-data` volume.
**Backup this volume** before updates.

---

## Database (SQLite)

```bash
# Location
trichome.db   # in REPO_ROOT (created on startup)

# Schema managed by SQLModel (auto-migration on startup)
# Tables:
#   annotation_task   — pending/approved/rejected VLM annotations
#   training_run      — training job records
#   inference_result  — stored detection results
#   calibration       — µm/px scale factors per microscope
```

For production with multiple users, switch to PostgreSQL:
```bash
DATABASE_URL="postgresql://user:pass@localhost:5432/ctip"
```

---

## MLflow (:3004)

Started by `dev-start.sh` using:
```bash
mlflow ui --host 0.0.0.0 --port 3004 --backend-store-uri ./mlruns
```

All training runs automatically log to MLflow:
- Hyperparameters
- mAP@0.5, mAP@0.5:0.95
- Loss curves (box, cls, dfl)
- Model artifacts

Access: http://localhost:3004

---

## dev-start.sh

```bash
./scripts/dev-start.sh           # start all
./scripts/dev-start.sh stop      # stop all
./scripts/dev-start.sh restart   # stop + start
./scripts/dev-start.sh status    # show running services

# Logs
tail -f logs/backend.log
tail -f logs/frontend.log
tail -f logs/mlflow.log
tail -f logs/nginx-error.log
```
