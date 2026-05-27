## Repository structure

```
ctip-oss/
├── backend/                    # FastAPI application
│   ├── main.py                 # App factory, lifespan, CORS
│   ├── config.py               # Settings (pydantic-settings, .env)
│   ├── database.py             # SQLite via SQLModel
│   ├── api/v1/                 # REST API routers
│   │   ├── router.py           # Aggregate all v1 routers
│   │   ├── setup.py            # Installation wizard (16 endpoints)
│   │   ├── detection.py        # YOLO inference
│   │   ├── segmentation.py     # SAM2 inference
│   │   ├── maturity.py         # Maturity classification
│   │   ├── training.py         # Training job management
│   │   ├── annotation.py       # Label Studio integration
│   │   └── analytics.py        # Statistics & aggregation
│   ├── websocket/router.py     # WebSocket endpoints
│   ├── middleware/gpu_guard.py # asyncio.Semaphore(1) — GPU concurrency
│   └── tasks/task_router.py   # Background GPU task management
│
├── frontend/                   # Next.js 14 App Router
│   └── src/
│       ├── app/                # Pages (file-system routing)
│       │   ├── layout.tsx      # Root layout + SetupGuard
│       │   ├── setup/page.tsx  # 11-step installer wizard
│       │   ├── wiki/           # This wiki
│       │   └── [other pages]
│       ├── components/
│       │   ├── layout/         # Sidebar, topbar, SetupGuard
│       │   └── ui/             # Shared UI primitives
│       ├── content/wiki/       # Wiki content (multilingual TS)
│       └── lib/api.ts          # axios instance (/api/v1 base URL)
│
├── detection/                  # YOLO detection domain
│   ├── domain/                 # Pure business logic
│   ├── application/            # Pipelines
│   ├── infrastructure/         # Model backends, YOLO adapter
│   └── api/                    # Detection router
│
├── segmentation/               # SAM2 segmentation domain
├── maturity/                   # Maturity classifier domain
├── morphology/                 # Morphology classifier domain
├── measurement/                # µm measurement domain
├── vlm_labeling/               # VLM auto-labeling (HITL enforced)
├── annotation/                 # Label Studio / CVAT integration
├── training/                   # Training orchestration
├── inference/                  # Batch inference pipeline
├── analytics/                  # Statistics, reporting
│
├── shared/                     # Cross-module types
│   ├── core/entities.py        # Detection, Instance, MaturityLabel
│   ├── core/value_objects.py   # BoundingBox, Confidence, Mask, Micrometer
│   ├── core/enums.py           # TrichomeType, MaturityStage
│   └── metrics/                # mAP, IoU, ECE, calibration
│
├── configs/
│   └── training/               # Training YAML configs
├── docker/                     # Docker Compose stacks
├── nginx-local/                # User-space nginx config
├── scripts/dev-start.sh        # All-in-one dev startup
├── tests/unit/                 # Unit tests (no GPU required)
├── docs/                       # Documentation
└── .env                        # Environment configuration
```

---

## Design pattern: Domain-Driven Design (DDD)

Every scientific module follows the same 4-layer structure:

```
<module>/
  domain/          # Pure business logic — no framework dependencies
                   # No FastAPI, no PyTorch, no SQLModel imports
                   # Only domain entities and value objects from shared/

  application/     # Orchestrates domain objects (pipelines)
                   # Calls infrastructure adapters
                   # No HTTP request/response concepts

  infrastructure/  # Model backends, file I/O, external APIs
                   # Where PyTorch, ultralytics, httpx live
                   # Implements interfaces defined in domain/

  api/             # FastAPI router for this module
                   # HTTP layer only — thin, delegates to application/
```

**Why DDD?**
- Domain logic is testable without GPU, HTTP server, or database
- Infrastructure can be swapped (YOLO → RT-DETR) without touching domain logic
- Application layer is parallelizable and cacheable

---

## Shared types (shared/)

Every module imports from `shared/` — never defines domain types locally:

```python
# shared/core/entities.py
@dataclass
class Detection:
    id: UUID
    bbox: BoundingBox
    confidence: Confidence
    trichome_type: TrichomeType
    mask: Mask | None = None
    maturity: MaturityStage | None = None

@dataclass
class Instance:
    detection: Detection
    mask: Mask
    size_um: Micrometer | None = None

# shared/core/value_objects.py
@dataclass(frozen=True)
class BoundingBox:
    x1: float; y1: float; x2: float; y2: float

    @property
    def area(self) -> float:
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    def iou(self, other: BoundingBox) -> float:
        ...

@dataclass(frozen=True)
class Confidence:
    value: float        # raw model output
    calibrated: float   # after Platt scaling

    def __post_init__(self):
        assert 0 <= self.value <= 1
        assert 0 <= self.calibrated <= 1
```

---

## GPU concurrency model

```python
# backend/middleware/gpu_guard.py
import asyncio
from contextlib import asynccontextmanager

_GPU_SEMAPHORE = asyncio.Semaphore(1)

@asynccontextmanager
async def gpu_context():
    async with _GPU_SEMAPHORE:
        yield

# Usage in any endpoint that needs GPU:
async def detect_trichomes(image: bytes) -> list[Detection]:
    async with gpu_context():
        return await _run_yolo(image)
```

This means:
- Request A enters: acquires semaphore, runs YOLO
- Request B arrives: waits (non-blocking via asyncio)
- Request A finishes: releases semaphore
- Request B proceeds

VRAM is never shared between concurrent inference calls.

---

## Frontend architecture

Next.js 14 **App Router** with strict TypeScript:

```typescript
// State management
@tanstack/react-query  → server state (fetch + cache)
zustand                → client state (UI state, user prefs)

// Key patterns:
// 1. Server components for static content (no client JS)
// 2. Client components ("use client") for interactive UI
// 3. Native WebSocket in useEffect for live data
// 4. api.ts (axios) as single HTTP client — base URL: /api/v1

// lib/api.ts
const api = axios.create({
  baseURL: '/api/v1',
  timeout: 30000,
});
```

---

## VLM auto-labeling architecture

```
Raw image
    │
    ▼
VLM inference (Moondream-2B / Florence-2 / Qwen2-VL)
    │  ← 4-bit quantized to fit 8 GB VRAM
    │  ← hallucination filter (confidence gate + cross-model agreement)
    │
    ▼
pending_review queue in Label Studio   ← HUMAN MUST APPROVE HERE
    │
    ▼  (only after human approval)
training dataset
```

**Hard constraint**: VLM output is NEVER written directly to training data.
This is enforced at the database level — `annotation_task.source` must be
`'human'` for tasks to be eligible for training export.

---

## Configuration system

```python
# backend/config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    data_root: Path = Path("data")
    models_dir: Path = Path("data/models")
    cuda_device: str = "cuda:0"
    vram_limit_gb: float = 8.0
    label_studio_url: str = "http://localhost:3005"
    # ... all from .env

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

# Usage anywhere:
from backend.config import get_settings
settings = get_settings()
```

Changing `.env` → restart backend → new settings loaded.
