## What is CTIP?

CTIP (Cannabis Trichome Intelligence Platform) analyzes cannabis trichomes from microscopy images using computer vision. It detects, classifies, and measures individual trichomes without any manual counting.

**Core pipeline:**
1. YOLO11s detects all trichomes (tiled inference for large images)
2. SAM2-tiny generates pixel-accurate masks per instance
3. Classifier assigns: Stalked / Sessile / Bulbous / Non-glandular
4. Maturity model assigns: Clear → Cloudy → Amber (optical only — no THC claims)
5. Calibrated scale converts pixel size to µm

> **Scientific constraint**: CTIP never predicts cannabinoid concentrations. Maturity = optical observation only.

## Architecture overview

```
Browser (:3001)
    │
nginx (Reverse Proxy)
    ├── /api/v1/*  →  FastAPI Backend (:8000)
    └── /*         →  Next.js Frontend (:3000)

Supporting services:
    ├── Label Studio (:3005)  — annotation
    ├── CVAT        (:3006)  — annotation (alternative)
    └── MLflow      (:3004)  — experiment tracking
```

## Quick start (5 minutes)

```bash
git clone https://github.com/ottco-dev/ctip-oss.git
cd ctip-oss

# Python environment
pip install uv
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[all]"

# Frontend
cd frontend && npm install && cd ..

# Start everything
./scripts/dev-start.sh

# Open browser → setup wizard launches automatically
# http://localhost:3001
```

## Hardware requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | NVIDIA 6 GB VRAM | RTX 4060 8 GB |
| RAM | 8 GB | 16 GB |
| CPU | 4 cores | i5-13400F |
| Storage | 20 GB free | 100 GB SSD |
| Python | 3.11 | 3.12 |
| CUDA | 11.8 | 12.1+ |

CPU-only works for inference (slower). Not recommended for training.

## Service ports at a glance

| Service | Port | Purpose |
|---------|------|---------|
| Main UI (nginx) | 3001 | Entry point for all requests |
| Next.js frontend | 3000 | UI server (proxied through nginx) |
| FastAPI backend | 8000 | REST API + WebSockets |
| MLflow | 3004 | Experiment tracking UI |
| Label Studio | 3005 | Annotation platform (Docker) |
| CVAT | 3006 | Alternative annotation (Docker) |
