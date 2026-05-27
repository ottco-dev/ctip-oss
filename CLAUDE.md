# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Development Commands

### Backend (FastAPI)

```bash
# Activate virtualenv (uv-managed, Python 3.12)
source .venv/bin/activate

# Install / sync dependencies
uv pip install -e ".[dev]"          # core + dev extras
uv pip install -e ".[vlm,sam]"      # add VLM and SAM2 extras
uv pip install -e ".[all]"          # everything

# Run backend API (port 8000, hot reload)
uvicorn backend.main:app --reload --port 8000

# Lint (ruff, black, mypy)
ruff check .
black --check .
mypy .
```

### Frontend (Next.js 14)

```bash
cd frontend
npm install
npm run dev          # development server (port 3000)
npm run build        # production build
npm run type-check   # TypeScript check (no emit)
npm run lint         # ESLint
```

### Testing

```bash
# Full test suite (most tests run without GPU)
pytest tests/ -v

# Skip GPU and integration tests (fastest)
pytest tests/ -m "not gpu and not slow and not integration" -v

# Run single test file
pytest tests/unit/test_detection_metrics.py -v
pytest tests/unit/test_maturity.py -v
pytest tests/unit/test_segmentation.py -v
pytest tests/unit/test_vlm_schema.py -v

# With coverage
pytest tests/ --cov=. --cov-report=html

# Skip GPU tests explicitly
pytest tests/ --no-gpu -v
```

### CLI (installed via pyproject.toml scripts)

```bash
trichome detect   --input image.tif --tiled --tile-size 1280
trichome segment  --input image.tif --model sam2-tiny
trichome maturity --input image.tif --features color,texture,translucency
trichome train detection --config configs/training/yolo11s_detection.yaml
trichome benchmark detection --split test --model yolo11s
```

### Docker

```bash
# Working directory: docker/
cd docker

# Core stack (nginx + backend + frontend + MLflow)
docker compose up -d

# Core + annotation tools (Label Studio + CVAT + PostgreSQL)
docker compose --profile annotation up -d

# Core + training stack (GPU YOLO trainer + MLflow override)
docker compose -f docker-compose.yml -f docker-compose.training.yml up -d

# Inference-only stack (no frontend, no annotation — lighter footprint)
# Requires external volumes created by the main stack:
#   docker volume create trichome-models && docker volume create trichome-mlflow
docker compose -f docker-compose.inference.yml up -d

# Port layout (host:container):
#   3001 → nginx     (PUBLIC: http://ottco.ddns.net:3001)
#   3002 → backend   (:8000)  http://localhost:3002/api/v1
#   3003 → frontend  (:3000)  http://localhost:3003
#   3004 → MLflow    (:5000)  http://localhost:3004
#   3005 → Label Studio (:8080)  http://localhost:3005
#   3006 → CVAT      (:8080)  http://localhost:3006
#   3007 → PostgreSQL (:5432) internal only
```

---

## Architecture

### Module Structure and Design Pattern

Every scientific module follows **Domain-Driven Design (DDD)**:

```
<module>/
  domain/          # Pure business logic, no framework deps
  application/     # Orchestrates domain objects (pipelines)
  infrastructure/  # Model backends, file I/O, external APIs
  api/             # FastAPI router for this module
  schemas/         # Pydantic request/response models
```

Modules: `detection/`, `segmentation/`, `maturity/`, `morphology/`, `measurement/`, `focus/`, `vlm_labeling/`, `annotation/`, `active_learning/`, `training/`, `inference/`, `video_pipeline/`, `analytics/`

### Shared Domain Types (`shared/`)

All modules import from `shared/` — never define domain types locally:

- `shared/core/entities.py` — `Detection`, `Instance`, `MaturityLabel`, `MorphologyType`, `TrichomeRegion`
- `shared/core/value_objects.py` — `BoundingBox`, `Confidence`, `Mask`, `BoundingBox`, `Micrometer`, `CalibrationScale`
- `shared/core/enums.py` — `TrichomeType`, `MaturityStage`, `AnnotationSource`
- `shared/metrics/` — detection (mAP, IoU), segmentation, calibration (ECE)
- `shared/logging/logger.py` — Loguru wrapper (`get_logger(__name__)`)

### CV Pipeline Flow

```
Image → YOLO v11s (tiled, 1280px, 20% overlap)
     → Confidence calibration (Platt/temperature scaling)
     → [Optional] RTMDet ensemble
     → SAM2-tiny (prompted with YOLO boxes → instance masks)
     → Mask refinement (fill holes, smooth)
     → Morphology classifier (bulbous / sessile / stalked)
     → Maturity classifier (clear → cloudy → amber) [HSV + LAB + LBP/GLCM/Gabor]
     → Measurement (px→µm via CalibrationScale)
     → Report (PDF / JSON / CSV)
```

### VLM Auto-Labeling (human-in-loop enforced)

```
Dataset images → VLM backend (Moondream-2B / Florence-2 / Qwen2-VL, 4-bit quantized)
              → Hallucination filter (confidence gate + cross-model agreement)
              → pending_review queue   ← HUMAN MUST APPROVE
              → Training dataset
```
VLM outputs are **never** written directly to training data. This is a hard architectural invariant.

### Backend (`backend/`)

- `backend/main.py` — FastAPI app factory (`create_app()`), lifespan for DB init + GPU broadcast loop
- `backend/config.py` — `Settings` via `pydantic-settings` (`.env` → env vars). Access via `get_settings()` (LRU-cached singleton)
- `backend/database.py` — SQLModel + SQLAlchemy, SQLite by default (`trichome.db`)
- `backend/api/v1/router.py` — aggregates all v1 routers under `/api/v1`
- `backend/websocket/router.py` — WebSocket endpoints: `/ws/training`, `/ws/system`, `/ws/jobs`, `/ws/logs`
- `backend/tasks/task_router.py` — background GPU task management (asyncio, no Celery/Redis)
- `backend/middleware/gpu_guard.py` — blocks requests when VRAM budget is exceeded

**GPU concurrency**: `asyncio.Semaphore(1)` enforces exactly one GPU task at a time. This is intentional for RTX 4060 (8 GB VRAM).

### Frontend (`frontend/`)

Next.js 14 App Router. Pages live in `frontend/src/app/<page>/page.tsx`. Uses:
- `@tanstack/react-query` for server state
- `zustand` for client state
- `recharts` for training/metric visualizations
- `@radix-ui/*` for headless components
- Tailwind CSS (dark theme)
- Native WebSocket in page components for live data (`/ws/*`)

### Configuration

Settings are in `backend/config.py` (`Settings` class). The `.env` file (copy from `.env.example`) controls paths, GPU, Label Studio, CVAT, MLflow/W&B connections. Key settings:
- `DATABASE_URL` — defaults to `sqlite:///./trichome.db`
- `DATA_ROOT`, `MODELS_DIR` — storage roots
- `CUDA_VISIBLE_DEVICES`, `VRAM_LIMIT_GB=8.0`
- `LABEL_STUDIO_URL`, `LABEL_STUDIO_API_KEY`

### Progress Tracking

Maintain `docs/progress/` files after every major change:
- `phase_status.md`, `current_focus.md`, `implementation_log.md`
- `pending.md`, `technical_debt.md`, `known_issues.md`
- `completed.md`, `benchmark_history.md`

### Scientific Constraints

- **Never claim THC/cannabinoid concentration** — maturity = optical observation only
- Every prediction must carry calibrated confidence and uncertainty estimates
- Use `GLOBAL_SEED=42` for reproducibility
- Target hardware: RTX 4060 8 GB / i5-13400F / 16 GB RAM
- Tiled inference is required for images larger than 1280px
- All VLM models run 4-bit quantized to fit in 8 GB VRAM
