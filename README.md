# 🔬 Trichome Analysis System — CTIP

> **Cannabis Trichome Intelligence Platform** — Research-Grade Microscopy AI
> Computer Vision · Instance Segmentation · Active Learning · Scientific Web Platform

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)
[![CUDA 12.x](https://img.shields.io/badge/CUDA-12.x-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![TensorRT 10.x](https://img.shields.io/badge/TensorRT-10.x-76b900.svg)](https://developer.nvidia.com/tensorrt)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688.svg)](https://fastapi.tiangolo.com/)
[![Next.js 14](https://img.shields.io/badge/Next.js-14-black.svg)](https://nextjs.org/)
[![Tests](https://img.shields.io/badge/tests-960%20passing-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

> 📖 **Full documentation in three languages:**
> [🇬🇧 English](#-english-documentation) · [🇩🇪 Deutsch](#-deutsche-dokumentation) · [🇪🇸 Español](#-documentación-en-español)

---

# 🇬🇧 English Documentation

## Table of Contents

1. [What Is This?](#1-what-is-this)
2. [Hardware Requirements](#2-hardware-requirements)
3. [Installation](#3-installation)
4. [First Steps After Install](#4-first-steps-after-install)
5. [Data Collection — Image Tips](#5-data-collection--image-tips)
6. [Labeling Workflow](#6-labeling-workflow)
7. [Training Workflow](#7-training-workflow)
8. [Verification & Benchmarking](#8-verification--benchmarking)
9. [Improvement Loop](#9-improvement-loop)
10. [Docker Deployment](#10-docker-deployment)
11. [All URLs & Pages](#11-all-urls--pages)
12. [API Reference](#12-api-reference)
13. [CLI Reference](#13-cli-reference)
14. [Configuration](#14-configuration)
15. [Architecture](#15-architecture)
16. [Scientific Methodology](#16-scientific-methodology)
17. [Testing](#17-testing)

---

## 1. What Is This?

CTIP is a **full-stack, production-grade research platform** for automated trichome analysis of *Cannabis sativa L.* specimens under digital microscopy. Not a demo or toy — a complete, continuously running system for real scientific work.

### What It Does

| Capability | Method | Target |
|---|---|---|
| Trichome Detection | YOLO v11s + RTMDet ensemble | mAP50 > 0.88 |
| Instance Segmentation | SAM2-tiny + mask refinement | IoU > 0.82 |
| Maturity Classification | HSV + LAB + Texture (LBP/GLCM/Gabor) | F1 > 0.85 |
| Morphology Typing | Geometric + CNN (stalked/sessile/bulbous) | Accuracy > 0.90 |
| Size Measurement | Calibrated px to µm conversion | ±5% error |
| Focus Assessment | Laplacian + Tenengrad + FFT | — |
| Video Analysis | Frame quality ranking + temporal dedup | — |
| VLM Pre-labeling | Moondream-2B / Florence-2 / Qwen2-VL (4-bit) | Human-in-loop |
| Active Learning | Uncertainty + disagreement sampling | — |
| TensorRT Inference | FP16 engine, async v3 API | RTX 4060 optimized |
| Container Management | docker compose background tasks + Browser Notifications | — |
| In-app Documentation | Wiki in EN/DE/ES (14 pages) | — |

### What It Is NOT

- No THC/cannabinoid concentration predictions (optical maturity only)
- No pseudoscience
- VLM outputs never go directly to training data (HITL gate is mandatory)

---

## 2. Hardware Requirements

### Minimum (development)

| Component | Minimum | Recommended |
|---|---|---|
| GPU | NVIDIA GTX 1080 (8 GB VRAM) | RTX 4060 / 3080 (8+ GB) |
| CPU | 6-core modern | i5-13400F or better |
| RAM | 16 GB | 32 GB |
| Storage | 50 GB SSD | 500 GB NVMe |
| CUDA | 11.8+ | 12.6 |

### VRAM Budget (RTX 4060, 8 GB)

| Component | VRAM |
|---|---|
| YOLO v11s inference | ~0.9 GB |
| SAM2-tiny | ~1.8 GB |
| Florence-2 (4-bit) | ~2.1 GB |
| Moondream-2B (4-bit) | ~1.4 GB |
| Qwen2-VL-7B (4-bit) | ~4.8 GB |
| YOLO v11s training (bs=8) | ~5.5 GB |

> Only **one GPU task runs at a time** — enforced by `asyncio.Semaphore(1)`. Intentional for 8 GB VRAM cards.

---

## 3. Installation

### 3.1 Prerequisites

```bash
# Ubuntu 22.04 / 24.04
sudo apt update && sudo apt install -y \
    git curl wget build-essential \
    python3.12 python3.12-venv python3.12-dev \
    ffmpeg libgl1 libglib2.0-0 libsm6 libxext6

# Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# Node.js 20 (for frontend)
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs

# Verify CUDA
nvcc --version && nvidia-smi
```

### 3.2 Clone & Install

```bash
git clone https://github.com/youruser/trichome-analysis.git
cd trichome-analysis

python3.12 -m venv .venv
source .venv/bin/activate

uv pip install -e ".[dev]"       # core + dev
uv pip install -e ".[vlm]"       # + VLM models (Florence-2, Moondream, Qwen2-VL)
uv pip install -e ".[sam]"       # + SAM2 segmentation
uv pip install -e ".[all]"       # everything
```

### 3.3 TensorRT (optional, production inference)

```bash
sudo apt install -y python3-libnvinfer python3-libnvinfer-dev tensorrt tensorrt-dev

export PATH=/usr/local/cuda-12.6/bin:$PATH
pip install pycuda

# Wire system TRT into venv
SITE=$(python -c "import site; print(site.getsitepackages()[0])")
printf "/usr/lib/python3/dist-packages\n/usr/lib/python3.12/dist-packages\n" > "$SITE/system_trt.pth"
echo 'export PATH=/usr/local/cuda-12.6/bin:$PATH' >> .venv/bin/activate

# Verify
python -c "import tensorrt; print(tensorrt.__version__)"
```

### 3.4 Frontend

```bash
cd frontend && npm install && cd ..
```

### 3.5 Environment Config

```bash
cp .env.example .env
```

> **Tip:** Use the built-in **Setup Wizard** instead of editing `.env` by hand — it guides you through every setting interactively (see §4.1).

If you prefer manual configuration, the key settings are:

```env
DATA_ROOT=/mnt/data/trichome          # or ./data for local dev
MODELS_ROOT=/mnt/models/trichome
CUDA_VISIBLE_DEVICES=0
VRAM_LIMIT_GB=8.0
MLFLOW_TRACKING_URI=http://localhost:3004
EXPERIMENT_TRACKER=mlflow             # mlflow | wandb | both | none
LABEL_STUDIO_URL=http://localhost:3005
LABEL_STUDIO_API_KEY=your_key_here
```

---

## 4. First Steps After Install

### 4.1 Start in Dev Mode & First-Time Setup

```bash
# Terminal 1 — Backend API
source .venv/bin/activate
uvicorn backend.main:app --reload --port 8000

# Terminal 2 — Frontend
cd frontend && npm run dev
```

Open **http://localhost:3000** — the **Setup Wizard launches automatically** on first start (when no `.env` is configured yet).

The wizard walks you through 7 steps:

| Step | Configures |
|---|---|
| 🌐 Network | Public domain vs. localhost-only, nginx port |
| ⚙️ Hardware | CUDA device, VRAM budget |
| 💾 Storage | Data root, model dir, outputs dir |
| 🔌 Services | Label Studio API key, MLflow URI, W&B (optional) |
| 🔒 Security | Secret key (auto-generator), API auth token |
| ✅ Review | Summary before saving |
| 🎉 Done | Writes `.env`, shows Docker restart command |

After finishing, `.env` is written automatically — no manual editing needed.
Re-run the wizard anytime from the sidebar: **First-Time Setup**.

- API Docs (Swagger): http://localhost:8000/docs

### 4.2 Verify the System

```bash
source .venv/bin/activate
pytest tests/ -v --tb=short
# Expected: 960 passed, 4 skipped (GPU-only + reportlab guard)

python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
curl http://localhost:8000/api/v1/system/health | python -m json.tool
```

### 4.3 Run Your First Detection

```bash
# CLI
trichome detect --input /path/to/image.jpg --tiled --tile-size 1280

# API
curl -X POST http://localhost:8000/api/v1/detection/infer \
  -F "file=@/path/to/image.jpg" \
  -F "confidence_threshold=0.25" | python -m json.tool

# Frontend: http://localhost:3000/inference — drag & drop image
```

---

## 5. Data Collection — Image Tips

Good data is the single most important factor. This is what actually matters for trichome microscopy.

### 5.1 Equipment

| Setup | Notes |
|---|---|
| Digital microscope | 40x–200x magnification optimal. USB microscopes (Andonstar, Celestron, Jiusion) are fine to start. |
| Phone + clip lens | Acceptable for bulkier trichomes, poor for bulbous/small ones. |
| Stereo microscope | Best optical clarity, hardest to digitize consistently. |

### 5.2 Imaging Protocol (Critical)

```
DO:
  - Consistent magnification every session (e.g. always 100x)
  - Capture in RAW or maximum JPEG quality
  - Shoot a calibration slide with known scale (stage micrometer, e.g. 1 mm)
    This enables px to µm measurement calibration
  - Minimum 1920x1080, ideally 4K
  - Same light source position every time
  - Encode metadata in filename:
    microscope01_100x_20260101_sample42_001.jpg
  - Capture all trichome types in one session (stalked + sessile + bulbous)
  - Include empty background patches (no trichomes) for negative examples

DON'T:
  - Mix magnifications without noting them (ruins calibration)
  - Use auto-exposure (inconsistent brightness)
  - Discard blurry images without logging (the focus scorer filters them)
  - Only photograph perfect trichomes — include partial, overlapping, at-edge cases
  - Use compressed social media images
```

### 5.3 Maturity Stage Coverage

For the maturity classifier to generalize, you need roughly equal coverage across stages:

| Stage | Visual | Target % of dataset |
|---|---|---|
| Clear | Glassy, fully transparent | ~25% |
| Cloudy | White/milky, opaque | ~35% |
| Amber | Golden-orange, degraded | ~25% |
| Mixed | Transition specimens | ~15% |

### 5.4 Organizing Your Data

```
data/
├── raw/                    # Original, unmodified images
│   ├── session_20260101/
│   ├── session_20260115/
│   └── ...
├── calibration/            # Stage micrometer images per microscope+magnification
│   └── microscope01_100x_cal.jpg
├── annotated/              # After labeling (Label Studio exports here)
│   ├── images/
│   └── labels/             # YOLO format .txt files
└── splits/                 # train / val / test — NEVER mix sessions!
    ├── train/
    ├── val/
    └── test/
```

> **Never** put images from the same microscopy session in both train and val/test — that is data leakage.
> Always split by **session**, not by image.

### 5.5 Focus Quality Filter

Use the built-in focus scorer before labeling to discard blurry images:

```bash
trichome focus --input data/raw/session_20260101/ \
               --output data/filtered/ \
               --min-sharpness 80.0 \
               --copy-passing

# Or via API
curl -X POST http://localhost:8000/api/v1/focus/score -F "file=@image.jpg"
```

### 5.6 Minimum Dataset Size

| Phase | Images | Annotations (boxes) |
|---|---|---|
| First working model | 150–300 | 2,000–5,000 |
| Decent generalization | 500–1,000 | 10,000–25,000 |
| Production-grade | 2,000+ | 50,000+ |

Start small, train fast, identify failure cases, collect targeted images. This beats 1,000 random images every time.

---

## 6. Labeling Workflow

### 6.1 Start Label Studio

```bash
# Docker (recommended)
cd docker && docker compose --profile annotation up -d label-studio

# Standalone
pip install label-studio && label-studio start --port 3005
```

Access: http://localhost:3005

### 6.2 Create a Project

1. Click **Create Project** — name it (e.g. "Trichomes Session 20260101")
2. **Labeling Setup** → Object Detection with Bounding Boxes
3. Add these labels (exact spelling required for YOLO export):

```
capitate-stalked    #FF4444  (red)
capitate-sessile    #44FF44  (green)
bulbous             #4444FF  (blue)
non-glandular       #FFAA00  (orange)
```

4. **Import images**: Settings → Cloud Storage → Add Source Storage → Local Files
   Set path to `data/raw/session_XXXXXXXX/`

### 6.3 VLM Pre-Labeling (3–5x Faster)

Before manual annotation, generate candidate boxes with a VLM:

```bash
curl -X POST http://localhost:8000/api/v1/vlm/label \
  -H "Content-Type: application/json" \
  -d '{
    "image_paths": ["data/raw/session_20260101/img001.jpg"],
    "model": "florence2",
    "confidence_threshold": 0.3
  }'
```

These land in a **review queue** — never written to training data directly.
In Label Studio you see pre-filled boxes to correct, add to, or reject.

| Model | VRAM | Speed | Quality |
|---|---|---|---|
| Moondream-2B (4-bit) | ~1.4 GB | Fast | Good for detection |
| Florence-2-large (4-bit) | ~2.1 GB | Medium | Best for complex scenes |
| Qwen2-VL-7B (4-bit) | ~4.8 GB | Slow | Highest quality |

### 6.4 Annotation Standards

```
Box drawing rules:
  YES: Tight box around trichome head (not the stalk)
  YES: Include full head even if partially occluded
  YES: Mark trichomes at image edges
  YES: Label ALL visible trichomes — no selective skipping
  YES: Unsure stalked/sessile? Look for visible neck

  NO: Boxes around bare stalks (no head)
  NO: Label debris or artifacts
  NO: Skip blurry trichomes if they are identifiable
```

### 6.5 Export Annotations

```bash
# Label Studio UI: Project → Export → YOLO format → Download

# Or via API
curl -X POST http://localhost:8000/api/v1/annotation/export \
  -H "Content-Type: application/json" \
  -d '{"project_id": 1, "format": "yolo", "output_dir": "data/annotated/session_20260101"}'
```

### 6.6 Annotation Quality Check

```bash
curl -X POST http://localhost:8000/api/v1/annotation/stats \
  -H "Content-Type: application/json" \
  -d '{"annotation_dir": "data/annotated/session_20260101"}'
```

Returns: class distribution, Cohen's κ (if multiple annotators), box size distribution, suspicious annotation flags.

---

## 7. Training Workflow

### 7.1 Prepare Dataset Split

```bash
# Split by session (not by image!) — prevents data leakage
trichome dataset split \
  --input data/annotated/ --output data/splits/ \
  --train 0.75 --val 0.15 --test 0.10 --split-by session

trichome dataset verify --path data/splits/
```

### 7.2 Configure Training

`configs/training/yolo11s_detection.yaml`:

```yaml
model: yolo11s.pt           # Downloads automatically
task: detect
data: data/splits/dataset.yaml
imgsz: 1280                 # Required for tiled inference
batch: 8                    # RTX 4060 8GB sweet spot
workers: 4
epochs: 100
patience: 20
lr0: 0.01
cos_lr: true

# Microscopy-specific augmentation
degrees: 90.0               # Full rotation — trichomes have no canonical orientation
flipud: 0.5
fliplr: 0.5
hsv_h: 0.015                # Small hue shift — microscope lighting varies
hsv_s: 0.7
mosaic: 0.3                 # Lower mosaic — microscopy context matters

device: 0                   # GPU 0
amp: true                   # FP16 mixed precision
```

### 7.3 Start Training

```bash
# CLI
trichome train detection --config configs/training/yolo11s_detection.yaml

# API (non-blocking, streams via WebSocket)
curl -X POST http://localhost:8000/api/v1/training/start \
  -H "Content-Type: application/json" \
  -d '{"config_path": "configs/training/yolo11s_detection.yaml"}'

# Live dashboard:   http://localhost:3000/training
# WebSocket stream: ws://localhost:8000/ws/training
# MLflow UI:        http://localhost:3004
```

### 7.4 Training Output

```
runs/
└── detect/
    └── trichome_yolo11s_20260101/
        ├── weights/
        │   ├── best.pt          ← use this for inference
        │   └── last.pt
        ├── results.csv
        ├── confusion_matrix.png
        ├── PR_curve.png
        └── val_batch0_pred.jpg
```

---

## 8. Verification & Benchmarking

### 8.1 Evaluate on Test Set

```bash
trichome benchmark detection \
  --weights runs/detect/trichome_yolo11s_20260101/weights/best.pt \
  --split test --data data/splits/dataset.yaml \
  --conf 0.25 --iou 0.5
```

Expected output:

```
Class               P      R      mAP50  mAP50-95
all                 0.887  0.862  0.883  0.512
capitate-stalked    0.921  0.905  0.918  0.561
capitate-sessile    0.873  0.841  0.864  0.498
bulbous             0.841  0.812  0.832  0.445
non-glandular       0.913  0.890  0.918  0.543
```

### 8.2 Confidence Calibration

Raw YOLO confidence scores are not well-calibrated. Fix this before deployment:

```bash
curl -X POST http://localhost:8000/api/v1/detection/calibrate \
  -H "Content-Type: application/json" \
  -d '{
    "weights_path": "runs/.../best.pt",
    "val_data": "data/splits/val/",
    "method": "temperature"
  }'
```

Target: ECE < 0.05. Reliability diagrams are generated automatically.

### 8.3 TensorRT Engine Build (Production)

```bash
# Export YOLO to ONNX
python -c "
from ultralytics import YOLO
YOLO('runs/.../best.pt').export(format='onnx', imgsz=1280, dynamic=True, half=True)
"

# Build FP16 TRT engine
trichome build-engine \
  --onnx runs/.../best.onnx \
  --output models/trichome_yolo11s_fp16.engine \
  --fp16 --imgsz 1280 --workspace-gb 4

# Benchmark TRT vs PyTorch
trichome benchmark inference \
  --engine models/trichome_yolo11s_fp16.engine \
  --pytorch runs/.../best.pt \
  --image data/splits/test/images/ --n 100
```

### 8.4 Tiled Inference Benchmark

```bash
trichome benchmark tiled \
  --weights runs/.../best.pt \
  --image data/splits/test/images/highres_001.jpg \
  --tile-sizes 640 1280 --overlaps 0.1 0.2 0.3
```

---

## 9. Improvement Loop

### 9.1 Active Learning — Find Hard Cases

```bash
curl -X POST http://localhost:8000/api/v1/active_learning/sample \
  -H "Content-Type: application/json" \
  -d '{"strategy": "uncertainty", "n_samples": 50, "unlabeled_dir": "data/raw/new_session/"}'
```

Label the returned images first — they teach the model the most per annotation hour.

### 9.2 Common Failure Cases

| Failure | Cause | Fix |
|---|---|---|
| Missing bulbous trichomes | Too small in training data | Collect targeted close-up images |
| False positives on debris | Debris resembles trichomes | Label debris as non-glandular |
| Stalked/sessile confusion | Short stalk at bad angle | More varied angle images |
| Poor detection at image edges | Padding artifacts | Increase tiled inference overlap |
| Low mAP at high IoU | Loose box drawing | Enforce tighter annotation protocol |

### 9.3 Dataset Improvement Checklist

```
After each training round:
  [ ] Check confusion matrix — which class is confused with which?
  [ ] Examine val_batch*_pred.jpg — where does the model visually fail?
  [ ] Run active learning sampling on new unlabeled data
  [ ] Check class distribution — balanced?
  [ ] Add targeted images for underperforming classes
  [ ] Re-check annotation quality (Cohen kappa >= 0.80)
  [ ] Verify no session overlap in train/val/test splits
  [ ] Re-run calibration after new training run
```

### 9.4 Retraining Triggers

```bash
curl http://localhost:8000/api/v1/active_learning/trigger | python -m json.tool
```

Auto-triggers fire when:
- 100+ new annotated images added since last training
- Mean uncertainty of recent predictions > 0.45
- New class distribution diverges > 15% from training distribution

---

## 10. Docker Deployment

### 10.1 Core Stack (nginx + backend + frontend + MLflow)

```bash
cd docker
docker compose build       # first time only
docker compose up -d
docker compose logs -f
docker compose down
```

### 10.2 With Annotation Tools (Label Studio + CVAT + PostgreSQL)

```bash
docker compose --profile annotation up -d
```

### 10.3 With GPU Training Stack

```bash
docker compose -f docker-compose.yml -f docker-compose.training.yml up -d
docker exec trichome-backend nvidia-smi   # verify GPU access
```

### 10.4 Inference-Only (lightweight, no frontend)

```bash
docker compose -f docker-compose.inference.yml up -d
```

### 10.5 Environment Setup for Docker

```bash
cp .env.example .env
# Use container-internal paths inside Docker:
# DATA_ROOT=/data
# MODELS_ROOT=/models
# MLFLOW_TRACKING_URI=http://mlflow:5000   <- internal Docker DNS
```

### 10.6 Data Volumes

```bash
docker volume ls | grep trichome
# trichome-models        model weights (shared across containers)
# trichome-mlflow        experiment data
# trichome-db            SQLite database
# trichome-label-studio  Label Studio data
```

### 10.7 Update / Rebuild

```bash
cd docker && git pull
docker compose build --no-cache
docker compose up -d
```

---

## 11. All URLs & Pages

### Development Mode (no Docker)

| Service | URL | Purpose |
|---|---|---|
| Frontend | http://localhost:3000 | Main web UI |
| API Swagger | http://localhost:8000/docs | Interactive API docs |
| API ReDoc | http://localhost:8000/redoc | API reference |
| API Base | http://localhost:8000/api/v1 | REST endpoints |
| WS Training | ws://localhost:8000/ws/training | Live training stream |
| WS System | ws://localhost:8000/ws/system | System / GPU stats |
| WS Jobs | ws://localhost:8000/ws/jobs | Background job status |
| WS Logs | ws://localhost:8000/ws/logs | Live log stream |

### Docker Mode

| Service | Local URL | Public (via nginx) |
|---|---|---|
| Nginx gateway | http://localhost:3001 | http://your-domain.com:3001 |
| Backend API | http://localhost:3002/api/v1 | http://your-domain.com:3001/api/v1/ |
| API Docs | http://localhost:3002/docs | http://your-domain.com:3001/docs |
| Frontend | http://localhost:3003 | http://your-domain.com:3001/ |
| MLflow | http://localhost:3004 | http://your-domain.com:3001/mlflow/ |
| Label Studio | http://localhost:3005 | http://your-domain.com:3001/annotation/ |
| CVAT | http://localhost:3006 | http://your-domain.com:3001/cvat/ |

### Frontend Pages

| Page | Path | What you do there |
|---|---|---|
| Dashboard | / | System overview, GPU status, recent jobs |
| Inference | /inference | Drop image, run detection / segmentation |
| Datasets | /datasets | Browse, import, validate datasets |
| Annotation | /annotation | Review VLM pre-labels, manage Label Studio |
| Label Studio | /labelstudio | Embedded Label Studio iframe |
| Training | /training | Start, monitor, compare training runs |
| Models | /models | Model registry, version management |
| Experiments | /experiments | MLflow experiment comparison |
| Morphology | /morphology | Morphology analysis results |
| Analytics | /analytics | Generate PDF / CSV / JSON reports |
| Video | /video | Video pipeline, frame extraction |
| Reports | /reports | Past report archive |
| Benchmarks | /benchmarks | Benchmark history and comparison |
| System | /system | Hardware stats, process monitor |
| Processes | /processes | Container manager, docker compose, live logs |
| Settings | /settings | Config, calibration, API keys |
| Wiki | /wiki | In-app documentation (EN/DE/ES, 14 pages) |

---

## 12. API Reference

Base path: `/api/v1/` — Full docs: http://localhost:8000/docs

### Detection

```bash
POST /detection/infer               # Single image inference
POST /detection/infer/tiled         # Tiled inference (4K images)
POST /detection/infer/batch         # Batch inference
POST /detection/calibrate           # Calibrate confidence scores
```

### Segmentation

```bash
POST /segmentation/segment          # SAM2 instance segmentation
POST /segmentation/refine           # Refine existing mask
GET  /segmentation/models           # Available SAM2 variants
```

### Maturity

```bash
POST /maturity/classify             # Classify maturity from image region
POST /maturity/classify/batch       # Batch classification
GET  /maturity/thresholds           # Current thresholds
PUT  /maturity/thresholds           # Update thresholds
```

### Training

```bash
POST /training/start                # Start training job
GET  /training/status               # Current training status
POST /training/stop                 # Stop training
GET  /training/runs                 # List all runs
GET  /training/runs/{run_id}        # Run details + metrics
POST /training/evaluate             # Evaluate on test set
```

### VLM Pre-labeling

```bash
POST /vlm/label                     # Run VLM pre-labeling (to review queue)
GET  /vlm/queue                     # Get pending review items
POST /vlm/queue/{id}/approve        # Approve pre-label
POST /vlm/queue/{id}/reject         # Reject pre-label
GET  /vlm/models                    # Available VLM models
```

### Active Learning

```bash
POST /active_learning/sample        # Get uncertain samples to label next
GET  /active_learning/trigger       # Check if retraining is triggered
POST /active_learning/priority      # Set labeling priority queue
```

### Annotation

```bash
POST /annotation/export             # Export from Label Studio (YOLO/COCO/CSV)
POST /annotation/stats              # Annotation quality statistics
POST /annotation/import             # Import annotation batch
GET  /annotation/projects           # List Label Studio projects
```

### Analytics & Reports

```bash
POST /analytics/report              # Generate report (PDF/CSV/JSON)
GET  /analytics/reports             # List past reports
GET  /analytics/reports/{id}        # Download specific report
POST /analytics/export/csv          # Export raw detection data
```

### System

```bash
GET  /system/health                 # Full health check + GPU stats
GET  /system/gpu                    # VRAM usage, temperature, utilization
GET  /system/version                # Component versions
GET  /models                        # Loaded model registry
```

### Container Management

```bash
GET  /containers                    # List all Docker containers (running + stopped)
POST /containers/{name}/start       # Start a stopped container
POST /containers/{name}/stop        # Stop a running container
POST /containers/{name}/restart     # Restart a container
POST /containers/{name}/pull        # Pull latest image + restart
DELETE /containers/{name}           # Stop + remove container

GET  /containers/{name}/logs        # Last N log lines (tail=200)
GET  /containers/{name}/logs/stream # SSE live log tail (docker logs -f)

GET  /containers/compose/config     # Compose services + .env key-value reader
POST /containers/compose/up         # docker compose up -d (blocking)
POST /containers/compose/down       # docker compose down (blocking)
GET  /containers/compose/up/stream  # SSE streaming compose up

# Background tasks (returns task_id immediately — poll for status)
POST /containers/compose/up/background        # Start annotation stack in background
POST /containers/compose/reinstall/background # Pull + force-recreate in background
GET  /containers/compose/task/{task_id}       # Poll: status/log/elapsed
GET  /containers/compose/tasks                # Last 20 background tasks
```

---

## 13. CLI Reference

```bash
# Detection
trichome detect --input image.jpg --tiled --tile-size 1280 --conf 0.25

# Segmentation
trichome segment --input image.jpg --model sam2-tiny

# Maturity
trichome maturity --input image.jpg

# Focus quality filter
trichome focus --input data/raw/ --output data/filtered/ --min-sharpness 80

# Dataset management
trichome dataset split --input data/annotated/ --output data/splits/ \
                       --train 0.75 --val 0.15 --test 0.10 --split-by session
trichome dataset verify --path data/splits/
trichome dataset stats  --path data/annotated/

# Training
trichome train detection --config configs/training/yolo11s_detection.yaml

# Evaluation
trichome benchmark detection --weights runs/.../best.pt --split test

# Engine building
trichome build-engine --onnx model.onnx --output model.engine --fp16 --imgsz 1280

# Report generation
trichome report --input results.json --format pdf --output report.pdf
```

---

## 14. Configuration

### `.env` Key Variables

```env
# Paths
TRICHOME_ROOT=/path/to/trichome-analysis
DATA_ROOT=/mnt/data/trichome
MODELS_ROOT=/mnt/models/trichome

# Hardware
CUDA_VISIBLE_DEVICES=0
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
VRAM_LIMIT_GB=8.0
GPU_INFERENCE_QUEUE_DEPTH=0        # 0 = fail-fast, no queue

# Database
DATABASE_URL=sqlite:///./trichome.db

# Annotation
LABEL_STUDIO_URL=http://localhost:3005
LABEL_STUDIO_API_KEY=your_key
CVAT_URL=http://localhost:3006

# Experiment Tracking
MLFLOW_TRACKING_URI=http://localhost:3004
EXPERIMENT_TRACKER=mlflow          # mlflow | wandb | both | none
WANDB_API_KEY=your_key

# VLM Models
FLORENCE2_MODEL_ID=microsoft/Florence-2-large
MOONDREAM_MODEL_ID=vikhyatk/moondream2
VLM_CACHE_DIR=/mnt/models/vlm_cache
```

---

## 15. Architecture

### Module Structure

Every scientific module follows Domain-Driven Design (DDD):

```
<module>/
  domain/          # Pure business logic, no framework deps
  application/     # Orchestrates domain objects (pipelines)
  infrastructure/  # Model backends, file I/O, external APIs
  api/             # FastAPI router for this module
  schemas/         # Pydantic request/response models
```

Modules: `detection/`, `segmentation/`, `maturity/`, `morphology/`, `measurement/`,
`focus/`, `vlm_labeling/`, `annotation/`, `active_learning/`, `training/`, `inference/`,
`video_pipeline/`, `analytics/`

### CV Pipeline

```
Image
  -> Focus scorer (reject blurry frames)
  -> Tiled inference (YOLO v11s, 1280px tiles, 20% overlap)
  -> Confidence calibration (temperature scaling)
  -> [Optional] RTMDet ensemble
  -> SAM2-tiny (prompted segmentation from YOLO boxes)
  -> Mask refinement (fill holes, smooth contours)
  -> Morphology classifier (stalked / sessile / bulbous)
  -> Maturity classifier (clear -> cloudy -> amber)
  -> Measurement (px -> µm via CalibrationScale)
  -> Analytics engine (statistics, report generation)
```

### Shared Domain Types (`shared/`)

All modules import from here — types are never defined locally:

- `shared/core/entities.py` — `Detection`, `Instance`, `MaturityLabel`, `TrichomeRegion`
- `shared/core/value_objects.py` — `BoundingBox`, `Confidence`, `Mask`, `Micrometer`, `CalibrationScale`
- `shared/core/enums.py` — `TrichomeType`, `MaturityStage`, `AnnotationSource`
- `shared/metrics/` — mAP, IoU, ECE/MCE calibration metrics

### Backend

- `backend/main.py` — FastAPI app factory, lifespan (DB init + GPU broadcast loop)
- `backend/config.py` — Settings via pydantic-settings, LRU-cached singleton
- `backend/middleware/gpu_guard.py` — VRAM budget enforcement, HTTP 429 when exceeded
- `asyncio.Semaphore(1)` — one GPU task at a time, globally enforced

---

## 16. Scientific Methodology

### Maturity Classification

Maturity is assessed purely from **optical characteristics** — no chemical claims:

| Feature Group | Features Used |
|---|---|
| Color (HSV) | Mean hue, saturation, value per trichome region |
| Color (LAB) | L* (lightness), a* (green-red), b* (blue-yellow) |
| Texture | LBP (Local Binary Patterns), GLCM (co-occurrence matrix), Gabor filters |
| Morphology | Head diameter, stalk length, circularity |

**Explicit limitations:**
- Amber coloration is not a direct proxy for cannabinoid degradation — it is optical observation
- Lighting conditions significantly affect color features — consistent imaging is critical
- This system does not predict THC, CBD, or any cannabinoid concentration

### Calibration

- Temperature scaling (preferred) — single scalar, preserves ranking
- Platt scaling — sigmoid fit on validation logits
- Reliability diagrams generated at every evaluation
- ECE (Expected Calibration Error) < 0.05 target

### Reproducibility

- `GLOBAL_SEED = 42` in all training, sampling, and augmentation pipelines
- Dataset splits are deterministic (seeded by session hash)
- All benchmark results stored in `docs/progress/benchmark_history.md`

---

## 17. Testing

```bash
# Full suite
pytest tests/ -v

# Fast (skip GPU and slow integration tests)
pytest tests/ -m "not gpu and not slow and not integration" -v

# Single module
pytest tests/unit/test_detection_metrics.py -v
pytest tests/unit/test_tensorrt_runner.py -v
pytest tests/unit/test_inference_tiling.py -v

# With coverage
pytest tests/ --cov=. --cov-report=html

# GPU tests (requires physical GPU + TRICHOME_ENGINE env var)
pytest tests/ -m gpu -v
```

**Current status: 960 passed, 4 skipped (GPU-only + reportlab guard)**

| Module | Tests |
|---|---|
| Detection metrics | 45 |
| Maturity classifier | 38 |
| Segmentation | 41 |
| VLM schema enforcer | 63 |
| Annotation statistics | 36 |
| Analytics export | 61 |
| TensorRT runner + builder | 35 |
| Tiled inference | 57 |
| All other modules | 584 |

---

---

# 🇩🇪 Deutsche Dokumentation

## Inhaltsverzeichnis

1. [Was ist das?](#de-1)
2. [Hardware-Anforderungen](#de-2)
3. [Installation](#de-3)
4. [Erste Schritte](#de-4)
5. [Datenerfassung — Bildtipps](#de-5)
6. [Labeling-Workflow](#de-6)
7. [Training-Workflow](#de-7)
8. [Verifizierung und Benchmarking](#de-8)
9. [Verbesserungs-Loop](#de-9)
10. [Docker-Deployment](#de-10)
11. [Alle URLs und Seiten](#de-11)
12. [API-Referenz](#de-12)
13. [CLI-Referenz](#de-13)
14. [Konfiguration](#de-14)
15. [Architektur](#de-15)
16. [Wissenschaftliche Methodik](#de-16)
17. [Tests](#de-17)

---

## DE 1. Was ist das? {#de-1}

CTIP ist eine **vollständige, produktionsreife Forschungsplattform** zur automatisierten Trichom-Analyse von *Cannabis sativa L.* unter digitaler Mikroskopie. Kein Demo, kein Spielzeug — ein vollständiges, laufendes System für echte wissenschaftliche Arbeit.

### Was es kann

| Fähigkeit | Methode | Zielwert |
|---|---|---|
| Trichom-Erkennung | YOLO v11s + RTMDet Ensemble | mAP50 > 0.88 |
| Instanz-Segmentierung | SAM2-tiny + Maskenverfeinerung | IoU > 0.82 |
| Reifegradklassifikation | HSV + LAB + Textur (LBP/GLCM/Gabor) | F1 > 0.85 |
| Morphologie-Typisierung | Geometrisch + CNN (gestielt/sitzend/kugelförmig) | Genauigkeit > 0.90 |
| Größenmessung | Kalibrierte px nach µm Umrechnung | ±5% Fehler |
| Fokusbeurteilung | Laplacian + Tenengrad + FFT | — |
| Videoanalyse | Frame-Qualitätsranking + temporale Deduplizierung | — |
| VLM-Vorlabeling | Moondream-2B / Florence-2 / Qwen2-VL (4-bit) | Mensch-in-Loop |
| Aktives Lernen | Unsicherheits- + Dissens-Sampling | — |
| TensorRT-Inferenz | FP16 Engine, async v3 API | RTX 4060 optimiert |

### Was es NICHT ist

- Keine THC/Cannabinoid-Konzentrationsvorhersagen (nur optische Reife)
- Keine Pseudowissenschaft
- VLM-Ausgaben gehen niemals direkt in Trainingsdaten (HITL-Gate ist Pflicht)

---

## DE 2. Hardware-Anforderungen {#de-2}

| Komponente | Minimum | Empfohlen |
|---|---|---|
| GPU | NVIDIA GTX 1080 (8 GB VRAM) | RTX 4060 / 3080 (8+ GB) |
| CPU | 6-Kern modern | i5-13400F oder besser |
| RAM | 16 GB | 32 GB |
| Speicher | 50 GB SSD | 500 GB NVMe |
| CUDA | 11.8+ | 12.6 |

### VRAM-Budget (RTX 4060, 8 GB)

| Komponente | VRAM |
|---|---|
| YOLO v11s Inferenz | ~0,9 GB |
| SAM2-tiny | ~1,8 GB |
| Florence-2 (4-bit) | ~2,1 GB |
| Moondream-2B (4-bit) | ~1,4 GB |
| Qwen2-VL-7B (4-bit) | ~4,8 GB |
| YOLO v11s Training (bs=8) | ~5,5 GB |

> Immer nur **ein GPU-Task gleichzeitig** — asyncio.Semaphore(1). Bewusst so designed für 8-GB-Karten.

---

## DE 3. Installation {#de-3}

### 3.1 Voraussetzungen

```bash
sudo apt update && sudo apt install -y \
    git curl wget build-essential \
    python3.12 python3.12-venv python3.12-dev \
    ffmpeg libgl1 libglib2.0-0 libsm6 libxext6

# uv installieren (schneller Python-Paketmanager)
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.bashrc

# Node.js 20 (für Frontend)
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs

nvcc --version && nvidia-smi
```

### 3.2 Klonen und Installieren

```bash
git clone https://github.com/deinuser/trichome-analysis.git
cd trichome-analysis

python3.12 -m venv .venv && source .venv/bin/activate

uv pip install -e ".[dev]"     # Kern + Dev
uv pip install -e ".[vlm]"     # + VLM-Modelle
uv pip install -e ".[sam]"     # + SAM2-Segmentierung
uv pip install -e ".[all]"     # alles
```

### 3.3 TensorRT (optional, für Produktions-Inferenz)

```bash
sudo apt install -y python3-libnvinfer python3-libnvinfer-dev tensorrt tensorrt-dev

export PATH=/usr/local/cuda-12.6/bin:$PATH && pip install pycuda

SITE=$(python -c "import site; print(site.getsitepackages()[0])")
printf "/usr/lib/python3/dist-packages\n/usr/lib/python3.12/dist-packages\n" > "$SITE/system_trt.pth"
echo 'export PATH=/usr/local/cuda-12.6/bin:$PATH' >> .venv/bin/activate

python -c "import tensorrt; print(tensorrt.__version__)"
```

### 3.4 Frontend

```bash
cd frontend && npm install && cd ..
```

### 3.5 Umgebungskonfiguration

```bash
cp .env.example .env
```

> **Tipp:** Nutze den integrierten **Einrichtungsassistenten** statt die `.env` manuell zu bearbeiten — er führt dich Schritt für Schritt durch alle Einstellungen (→ §4.1).

Für manuelle Konfiguration die wichtigsten Variablen:

```env
DATA_ROOT=/mnt/data/trichome          # oder ./data für lokale Entwicklung
MODELS_ROOT=/mnt/models/trichome
CUDA_VISIBLE_DEVICES=0
VRAM_LIMIT_GB=8.0
MLFLOW_TRACKING_URI=http://localhost:3004
EXPERIMENT_TRACKER=mlflow
LABEL_STUDIO_URL=http://localhost:3005
LABEL_STUDIO_API_KEY=dein_schluessel
```

---

## DE 4. Erste Schritte {#de-4}

### 4.1 Dev-Modus starten & Ersteinrichtung

```bash
# Terminal 1 — Backend-API
source .venv/bin/activate
uvicorn backend.main:app --reload --port 8000

# Terminal 2 — Frontend
cd frontend && npm run dev
```

**http://localhost:3000** öffnen — der **Einrichtungsassistent startet automatisch** beim ersten Start (solange keine `.env` konfiguriert ist).

Der Assistent führt durch 7 Schritte:

| Schritt | Konfiguriert |
|---|---|
| 🌐 Netzwerk | Öffentliche Domain vs. nur localhost, nginx-Port |
| ⚙️ Hardware | CUDA-Gerät, VRAM-Budget |
| 💾 Speicher | Datenwurzel, Modellverzeichnis, Ausgabepfad |
| 🔌 Dienste | Label Studio API-Key, MLflow URI, W&B (optional) |
| 🔒 Sicherheit | Secret Key (Auto-Generator), API-Token |
| ✅ Überprüfung | Zusammenfassung aller Einstellungen |
| 🎉 Fertig | Schreibt `.env`, zeigt Docker-Restart-Befehl |

Nach Abschluss wird die `.env` automatisch geschrieben — kein manuelles Bearbeiten nötig.
Erneut starten jederzeit über die Seitenleiste: **Ersteinrichtung**.

- API-Docs: http://localhost:8000/docs

### 4.2 System prüfen

```bash
pytest tests/ -v --tb=short
# Erwartet: 960 passed, 4 skipped

python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
curl http://localhost:8000/api/v1/system/health | python -m json.tool
```

### 4.3 Erste Erkennung ausführen

```bash
# Via CLI
trichome detect --input /pfad/zum/bild.jpg --tiled --tile-size 1280

# Via API
curl -X POST http://localhost:8000/api/v1/detection/infer \
  -F "file=@/pfad/zum/bild.jpg" -F "confidence_threshold=0.25" | python -m json.tool

# Via Frontend: http://localhost:3000/inference — Bild hineinziehen
```

---

## DE 5. Datenerfassung — Bildtipps {#de-5}

Gute Daten sind der wichtigste Faktor überhaupt. Was bei Trichom-Mikroskopie wirklich zählt:

### 5.1 Equipment

| Setup | Hinweise |
|---|---|
| Digitalmikroskop | 40x–200x Vergrößerung optimal. USB-Mikroskope (Andonstar, Celestron, Jiusion) reichen für den Einstieg. |
| Smartphone + Clip-Linse | Für größere Trichome geeignet, schlecht für kleine kugelförmige. |
| Stereomikroskop | Beste optische Qualität, schwieriger konsistent zu digitalisieren. |

### 5.2 Aufnahme-Protokoll (Kritisch)

```
MACHEN:
  - Konsistente Vergrößerung für jede Session (z.B. immer 100x)
  - RAW oder maximale JPEG-Qualität aufnehmen
  - Kalibrierungsbild mit bekannter Skala aufnehmen (Objektmikrometer, z.B. 1 mm)
    Ermöglicht px nach µm Kalibration
  - Mindestens 1920x1080, idealerweise 4K
  - Lichtquelle immer in derselben Position
  - Dateiname mit Metadaten: mikroskop01_100x_20260101_probe42_001.jpg
  - Alle Trichom-Typen in einer Session aufnehmen
  - Leere Hintergrundbereiche aufnehmen (keine Trichome) als Negativbeispiele

NICHT MACHEN:
  - Vergrößerungen mischen ohne Notiz (ruiniert Kalibration)
  - Auto-Belichtung verwenden (inkonsistente Helligkeit)
  - Nur perfekte Trichome fotografieren (teilweise sichtbare + überlappende einschließen)
  - Komprimierte Social-Media-Bilder verwenden
```

### 5.3 Reifestadien-Abdeckung

| Stadium | Optik | Ziel-% im Datensatz |
|---|---|---|
| Klar | Glasig, vollständig transparent | ~25% |
| Trüb | Weiß/milchig, opak | ~35% |
| Bernstein | Goldgelb-orange, degradiert | ~25% |
| Gemischt | Übergangspräparate | ~15% |

### 5.4 Datenorganisation

```
data/
├── raw/                    # Originale, unveränderte Bilder
│   ├── session_20260101/
│   └── session_20260115/
├── calibration/            # Objektmikrometer-Bilder pro Mikroskop+Vergrößerung
├── annotated/              # Nach dem Labeling (Label Studio exportiert hierher)
│   ├── images/
│   └── labels/             # YOLO-Format .txt-Dateien
└── splits/                 # train / val / test — niemals Sessions mischen!
    ├── train/
    ├── val/
    └── test/
```

> **Niemals** Bilder aus derselben Mikroskopie-Session in train UND val/test packen — das ist Data Leakage.
> Immer nach **Session** splitten, nicht nach Bild.

### 5.5 Fokusqualitäts-Filter

```bash
trichome focus --input data/raw/session_20260101/ \
               --output data/gefiltert/ \
               --min-sharpness 80.0 --copy-passing
```

### 5.6 Mindest-Datenmenge

| Phase | Bilder | Annotationen (Boxen) |
|---|---|---|
| Erstes funktionierendes Modell | 150–300 | 2.000–5.000 |
| Gute Generalisierung | 500–1.000 | 10.000–25.000 |
| Produktionsreif | 2.000+ | 50.000+ |

Klein anfangen, schnell trainieren, Schwachstellen identifizieren, gezielt sammeln. Das schlägt 1.000 zufällige Bilder jedes Mal.

---

## DE 6. Labeling-Workflow {#de-6}

### 6.1 Label Studio starten

```bash
# Docker (empfohlen)
cd docker && docker compose --profile annotation up -d label-studio
# Standalone
pip install label-studio && label-studio start --port 3005
```

Zugriff: http://localhost:3005

### 6.2 Projekt erstellen

1. **Create Project** — Namen vergeben
2. **Labeling Setup** → Object Detection with Bounding Boxes
3. Labels anlegen (exakte Schreibweise wichtig für YOLO-Export):

```
capitate-stalked    #FF4444 (Rot)
capitate-sessile    #44FF44 (Grün)
bulbous             #4444FF (Blau)
non-glandular       #FFAA00 (Orange)
```

4. **Bilder importieren**: Settings → Cloud Storage → Add Source Storage → Local Files → Pfad zur Session setzen

### 6.3 VLM-Vorlabeling (3–5x schneller annotieren)

```bash
curl -X POST http://localhost:8000/api/v1/vlm/label \
  -H "Content-Type: application/json" \
  -d '{
    "image_paths": ["data/raw/session_20260101/bild001.jpg"],
    "model": "florence2",
    "confidence_threshold": 0.3
  }'
```

Kandidaten-Boxen landen in der Review-Queue — nie direkt in Trainingsdaten. Im Label Studio sieht man vorgefüllte Boxen, die man korrigieren, ergänzen oder ablehnen kann.

| Modell | VRAM | Geschwindigkeit | Qualität |
|---|---|---|---|
| Moondream-2B (4-bit) | ~1,4 GB | Schnell | Gut für Erkennung |
| Florence-2-large (4-bit) | ~2,1 GB | Mittel | Beste Genauigkeit |
| Qwen2-VL-7B (4-bit) | ~4,8 GB | Langsam | Höchste Qualität |

### 6.4 Annotations-Standards

```
Box-Zeichenregeln:
  JA: Box eng um den Trichom-Kopf (nicht den Stiel)
  JA: Ganzen Kopf einschließen, auch wenn teilweise verdeckt
  JA: Trichome am Bildrand markieren
  JA: ALLE sichtbaren Trichome annotieren — kein selektives Überspringen
  JA: Unsicherheit gestielt/sitzend — sichtbaren Hals suchen

  NEIN: Boxen um nackte Stiele (kein Kopf)
  NEIN: Debris oder Artefakte annotieren
  NEIN: Erkennbare, unscharfe Trichome überspringen
```

### 6.5 Annotierungen exportieren

```bash
# Label Studio UI: Project → Export → YOLO format → Download

# Via API
curl -X POST http://localhost:8000/api/v1/annotation/export \
  -H "Content-Type: application/json" \
  -d '{"project_id": 1, "format": "yolo", "output_dir": "data/annotated/session_20260101"}'
```

### 6.6 Annotierungsqualität prüfen

```bash
curl -X POST http://localhost:8000/api/v1/annotation/stats \
  -H "Content-Type: application/json" \
  -d '{"annotation_dir": "data/annotated/session_20260101"}'
```

Ausgabe: Klassenverteilung, Cohen's κ, Box-Größenverteilung, verdächtige Annotierungen.

---

## DE 7. Training-Workflow {#de-7}

### 7.1 Dataset-Split vorbereiten

```bash
# Nach Session splitten (nicht nach Bild!) — verhindert Data Leakage
trichome dataset split \
  --input data/annotated/ --output data/splits/ \
  --train 0.75 --val 0.15 --test 0.10 --split-by session

trichome dataset verify --path data/splits/
```

### 7.2 Training konfigurieren

`configs/training/yolo11s_detection.yaml`:

```yaml
model: yolo11s.pt           # Startgewichte (automatischer Download)
task: detect
data: data/splits/dataset.yaml
imgsz: 1280                 # Für Tiled Inference nötig
batch: 8                    # Optimal für RTX 4060 8GB
workers: 4
epochs: 100
patience: 20
lr0: 0.01
cos_lr: true

# Augmentierung (mikroskopie-spezifisch)
degrees: 90.0               # Volle Rotation — Trichome haben keine Standardausrichtung
flipud: 0.5
fliplr: 0.5
hsv_h: 0.015                # Kleiner Farbtonversatz — Beleuchtung variiert
hsv_s: 0.7
mosaic: 0.3                 # Niedrigeres Mosaik

device: 0
amp: true                   # FP16 Mixed Precision
```

### 7.3 Training starten

```bash
# Via CLI
trichome train detection --config configs/training/yolo11s_detection.yaml

# Via API (nicht-blockierend, Progress via WebSocket)
curl -X POST http://localhost:8000/api/v1/training/start \
  -H "Content-Type: application/json" \
  -d '{"config_path": "configs/training/yolo11s_detection.yaml"}'

# Live-Dashboard:  http://localhost:3000/training
# WebSocket:       ws://localhost:8000/ws/training
# MLflow:          http://localhost:3004
```

### 7.4 Trainings-Ausgabe

```
runs/detect/trichome_yolo11s_20260101/
    weights/best.pt      <- dieses für Inferenz verwenden
    weights/last.pt
    results.csv
    confusion_matrix.png
    PR_curve.png
    val_batch0_pred.jpg
```

---

## DE 8. Verifizierung und Benchmarking {#de-8}

### 8.1 Auf Test-Set evaluieren

```bash
trichome benchmark detection \
  --weights runs/detect/trichome_yolo11s_20260101/weights/best.pt \
  --split test --data data/splits/dataset.yaml \
  --conf 0.25 --iou 0.5
```

Erwartete Ausgabe:

```
Klasse              P      R      mAP50  mAP50-95
all                 0.887  0.862  0.883  0.512
capitate-stalked    0.921  0.905  0.918  0.561
capitate-sessile    0.873  0.841  0.864  0.498
bulbous             0.841  0.812  0.832  0.445
non-glandular       0.913  0.890  0.918  0.543
```

### 8.2 Konfidenz-Kalibrierung

YOLO-Konfidenzscores sind unkalibriert. Vor dem Deployment korrigieren:

```bash
curl -X POST http://localhost:8000/api/v1/detection/calibrate \
  -H "Content-Type: application/json" \
  -d '{"weights_path": "runs/.../best.pt", "val_data": "data/splits/val/", "method": "temperature"}'
```

Ziel: ECE < 0,05. Reliability-Diagramme werden automatisch generiert.

### 8.3 TensorRT-Engine bauen (Produktions-Inferenz)

```bash
# YOLO nach ONNX exportieren
python -c "
from ultralytics import YOLO
YOLO('runs/.../best.pt').export(format='onnx', imgsz=1280, dynamic=True, half=True)
"

# TRT FP16 Engine bauen
trichome build-engine \
  --onnx runs/.../best.onnx \
  --output models/trichome_yolo11s_fp16.engine \
  --fp16 --imgsz 1280 --workspace-gb 4

# TRT vs. PyTorch benchmarken
trichome benchmark inference \
  --engine models/trichome_yolo11s_fp16.engine \
  --pytorch runs/.../best.pt \
  --image data/splits/test/images/ --n 100
```

### 8.4 Tiled Inference Benchmark

```bash
trichome benchmark tiled \
  --weights runs/.../best.pt \
  --image data/splits/test/images/hochaufloesung_001.jpg \
  --tile-sizes 640 1280 --overlaps 0.1 0.2 0.3
```

---

## DE 9. Verbesserungs-Loop {#de-9}

### 9.1 Aktives Lernen — Schwierige Fälle finden

```bash
curl -X POST http://localhost:8000/api/v1/active_learning/sample \
  -H "Content-Type: application/json" \
  -d '{"strategy": "uncertainty", "n_samples": 50, "unlabeled_dir": "data/raw/neue_session/"}'
```

Die zurückgegebenen Bilder zuerst labeln — sie verbessern das Modell am meisten pro Annotierungsstunde.

### 9.2 Häufige Schwachstellen

| Problem | Ursache | Lösung |
|---|---|---|
| Kugelförmige Trichome werden übersehen | Zu klein in Trainingsdaten | Gezielte Nahaufnahmen sammeln |
| Falsch-Positive auf Debris | Debris sieht aus wie Trichome | Debris als non-glandular annotieren |
| Gestielt/sitzend-Verwechslung | Kurzer Stiel aus schlechtem Winkel | Mehr Winkel-Varianten aufnehmen |
| Schlechte Erkennung an Bildrändern | Padding-Artefakte | Überlappung bei Tiled Inference erhöhen |
| Niedriges mAP bei hohem IoU | Zu lockere Box-Zeichnung | Engeres Annotations-Protokoll durchsetzen |

### 9.3 Checkliste nach jeder Trainingsrunde

```
  [ ] Konfusionsmatrix prüfen — welche Klasse wird verwechselt?
  [ ] val_batch*_pred.jpg ansehen — wo versagt das Modell?
  [ ] Aktives Lernen auf neue unlabeled Daten anwenden
  [ ] Klassenverteilung prüfen (ausgeglichen?)
  [ ] Gezielte Bilder für schwache Klassen hinzufügen
  [ ] Annotierungsqualität prüfen (Cohen kappa >= 0,80)
  [ ] Keine Session-Überschneidungen in Splits
  [ ] Kalibrierung nach jedem neuen Training neu ausführen
```

### 9.4 Retraining-Trigger

```bash
curl http://localhost:8000/api/v1/active_learning/trigger | python -m json.tool
```

Trigger feuern wenn:
- 100+ neue annotierte Bilder seit letztem Training
- Mittlere Unsicherheit > 0,45
- Neue Klassenverteilung > 15% Abweichung von der Trainingsverteilung

---

## DE 10. Docker-Deployment {#de-10}

### Core-Stack (nginx + backend + frontend + MLflow)

```bash
cd docker
docker compose build       # nur beim ersten Mal
docker compose up -d
docker compose logs -f
docker compose down
```

### Mit Annotations-Tools (Label Studio + CVAT + PostgreSQL)

```bash
docker compose --profile annotation up -d
```

### Mit GPU-Training-Stack

```bash
docker compose -f docker-compose.yml -f docker-compose.training.yml up -d
docker exec trichome-backend nvidia-smi   # GPU-Zugriff prüfen
```

### Nur Inferenz (leichtgewichtig, kein Frontend)

```bash
docker compose -f docker-compose.inference.yml up -d
```

### Umgebung für Docker einrichten

```bash
cp .env.example .env
# Container-interne Pfade verwenden:
# DATA_ROOT=/data
# MODELS_ROOT=/models
# MLFLOW_TRACKING_URI=http://mlflow:5000   <- internes Docker-DNS
```

### Daten-Volumes

```bash
docker volume ls | grep trichome
# trichome-models        Modell-Gewichte (geteilt)
# trichome-mlflow        Experiment-Daten
# trichome-db            SQLite-Datenbank
# trichome-label-studio  Label Studio-Daten
```

### Update / Rebuild

```bash
cd docker && git pull
docker compose build --no-cache && docker compose up -d
```

---

## DE 11. Alle URLs und Seiten {#de-11}

### Dev-Modus (kein Docker)

| Dienst | URL | Zweck |
|---|---|---|
| Frontend | http://localhost:3000 | Haupt-Web-UI |
| API Swagger | http://localhost:8000/docs | Interaktive API-Docs |
| API ReDoc | http://localhost:8000/redoc | API-Referenz |
| WS Training | ws://localhost:8000/ws/training | Live-Training-Stream |
| WS System | ws://localhost:8000/ws/system | System-/GPU-Stats |
| WS Jobs | ws://localhost:8000/ws/jobs | Hintergrund-Job-Status |
| WS Logs | ws://localhost:8000/ws/logs | Live-Log-Stream |

### Docker-Modus

| Dienst | Lokal | Öffentlich (via Nginx) |
|---|---|---|
| Nginx-Gateway | http://localhost:3001 | http://your-domain.com:3001 |
| Backend-API | http://localhost:3002/api/v1 | .../api/v1/ |
| API-Docs | http://localhost:3002/docs | .../docs |
| Frontend | http://localhost:3003 | .../ |
| MLflow | http://localhost:3004 | .../mlflow/ |
| Label Studio | http://localhost:3005 | .../annotation/ |
| CVAT | http://localhost:3006 | .../cvat/ |

### Frontend-Seiten

| Seite | Pfad | Funktion |
|---|---|---|
| Dashboard | / | Systemübersicht, GPU-Status, aktuelle Jobs |
| Inferenz | /inference | Bild hochladen, Erkennung/Segmentierung |
| Datensätze | /datasets | Datensätze verwalten |
| Annotierung | /annotation | VLM-Reviews, Label Studio verwalten |
| Label Studio | /labelstudio | Eingebettetes Label Studio |
| Training | /training | Training starten, überwachen, vergleichen |
| Modelle | /models | Modell-Registry, Versionen |
| Experimente | /experiments | MLflow-Vergleich |
| Morphologie | /morphology | Morphologie-Analyseergebnisse |
| Analytics | /analytics | PDF/CSV/JSON-Berichte generieren |
| Video | /video | Video-Pipeline, Frame-Extraktion |
| Berichte | /reports | Vergangene Berichte |
| Benchmarks | /benchmarks | Benchmark-Historie |
| System | /system | Hardware-Stats, Prozessmonitor |
| Prozesse | /processes | Container-Verwaltung, docker compose, Live-Logs |
| Einstellungen | /settings | Konfiguration, Kalibrierung, API-Schlüssel |
| Wiki | /wiki | In-App-Dokumentation (DE/EN/ES, 14 Seiten) |

---

## DE 12. API-Referenz {#de-12}

Basispfad: `/api/v1/` — Vollständige Docs: http://localhost:8000/docs

```bash
# Erkennung
POST /detection/infer               # Einzelbild-Inferenz
POST /detection/infer/tiled         # Tiled Inferenz (4K-Bilder)
POST /detection/infer/batch         # Batch-Inferenz
POST /detection/calibrate           # Konfidenz-Kalibrierung

# Segmentierung
POST /segmentation/segment          # SAM2-Instanz-Segmentierung
POST /segmentation/refine           # Maske verfeinern

# Reifegrad
POST /maturity/classify             # Reifegrad klassifizieren
POST /maturity/classify/batch       # Batch-Klassifikation

# Training
POST /training/start                # Training starten
GET  /training/status               # Trainingsstatus
POST /training/stop                 # Training stoppen
GET  /training/runs                 # Alle Runs auflisten
GET  /training/runs/{run_id}        # Run-Details + Metriken
POST /training/evaluate             # Auf Test-Set evaluieren

# VLM-Vorlabeling
POST /vlm/label                     # Vorlabeling starten (-> Review-Queue)
GET  /vlm/queue                     # Ausstehende Reviews
POST /vlm/queue/{id}/approve        # Vorlabel bestätigen
POST /vlm/queue/{id}/reject         # Vorlabel ablehnen

# Aktives Lernen
POST /active_learning/sample        # Unsichere Samples abrufen
GET  /active_learning/trigger       # Retraining-Trigger prüfen

# Annotierung
POST /annotation/export             # Export (YOLO/COCO/CSV)
POST /annotation/stats              # Qualitätsstatistiken
GET  /annotation/projects           # Label Studio-Projekte

# Berichte
POST /analytics/report              # PDF/CSV/JSON generieren

# System
GET  /system/health                 # Health-Check + GPU-Stats
GET  /system/gpu                    # VRAM, Temperatur, Auslastung
```

---

## DE 13. CLI-Referenz {#de-13}

```bash
# Erkennung
trichome detect   --input bild.jpg --tiled --tile-size 1280 --conf 0.25

# Segmentierung
trichome segment  --input bild.jpg --model sam2-tiny

# Reifegrad
trichome maturity --input bild.jpg

# Fokusfilter
trichome focus    --input data/raw/ --output data/gefiltert/ --min-sharpness 80

# Datensatz
trichome dataset split  --input data/annotated/ --output data/splits/ \
                        --train 0.75 --val 0.15 --test 0.10 --split-by session
trichome dataset verify --path data/splits/
trichome dataset stats  --path data/annotated/

# Training
trichome train detection --config configs/training/yolo11s_detection.yaml

# Evaluierung
trichome benchmark detection --weights runs/.../best.pt --split test

# Engine bauen
trichome build-engine --onnx model.onnx --output model.engine --fp16 --imgsz 1280

# Bericht generieren
trichome report   --input ergebnisse.json --format pdf --output bericht.pdf
```

---

## DE 14. Konfiguration {#de-14}

```env
# Pfade
TRICHOME_ROOT=/path/to/trichome-analysis
DATA_ROOT=/mnt/data/trichome
MODELS_ROOT=/mnt/models/trichome

# Hardware
CUDA_VISIBLE_DEVICES=0
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
VRAM_LIMIT_GB=8.0
GPU_INFERENCE_QUEUE_DEPTH=0

# Datenbank
DATABASE_URL=sqlite:///./trichome.db

# Annotierung
LABEL_STUDIO_URL=http://localhost:3005
LABEL_STUDIO_API_KEY=dein_schluessel
CVAT_URL=http://localhost:3006

# Experiment-Tracking
MLFLOW_TRACKING_URI=http://localhost:3004
EXPERIMENT_TRACKER=mlflow

# VLM-Modelle
FLORENCE2_MODEL_ID=microsoft/Florence-2-large
MOONDREAM_MODEL_ID=vikhyatk/moondream2
VLM_CACHE_DIR=/mnt/models/vlm_cache
```

---

## DE 15. Architektur {#de-15}

### Modul-Struktur (Domain-Driven Design)

```
<modul>/
  domain/          # Reine Geschäftslogik (keine Framework-Abhängigkeiten)
  application/     # Orchestriert Domain-Objekte (Pipelines)
  infrastructure/  # Modell-Backends, Datei-I/O, externe APIs
  api/             # FastAPI-Router
  schemas/         # Pydantic Request/Response-Modelle
```

### CV-Pipeline

```
Bild
  -> Fokus-Scorer (unscharfe Frames verwerfen)
  -> Tiled Inferenz (YOLO v11s, 1280px Kacheln, 20% Überlappung)
  -> Konfidenz-Kalibrierung (Temperature Scaling)
  -> [Optional] RTMDet Ensemble
  -> SAM2-tiny (Prompt-basierte Segmentierung mit YOLO-Boxen)
  -> Maskenverfeinerung (Löcher füllen, Konturen glätten)
  -> Morphologie-Klassifikator (gestielt / sitzend / kugelförmig)
  -> Reifegrad-Klassifikator (klar -> trüb -> bernstein)
  -> Messung (px -> µm via CalibrationScale)
  -> Analytics-Engine (Statistiken, Berichtgenerierung)
```

### Backend

- `backend/main.py` — FastAPI App Factory, Lifespan (DB-Init + GPU-Broadcast)
- `backend/config.py` — Settings via pydantic-settings, LRU-gecachter Singleton
- `backend/middleware/gpu_guard.py` — VRAM-Budget-Durchsetzung, HTTP 429 bei Überschreitung
- `asyncio.Semaphore(1)` — ein GPU-Task gleichzeitig, global durchgesetzt

---

## DE 16. Wissenschaftliche Methodik {#de-16}

### Reifegradklassifikation

Reifegrad wird **ausschließlich** aus optischen Eigenschaften beurteilt — keine chemischen Behauptungen:

| Merkmalgruppe | Verwendete Merkmale |
|---|---|
| Farbe (HSV) | Mittlerer Farbton, Sättigung, Hellwert pro Trichom-Region |
| Farbe (LAB) | L* (Helligkeit), a* (Grün-Rot), b* (Blau-Gelb) |
| Textur | LBP (Local Binary Patterns), GLCM (Co-Occurrence-Matrix), Gabor-Filter |
| Morphologie | Kopfdurchmesser, Stiellänge, Kreisförmigkeit |

**Explizite Einschränkungen:**
- Bernstein-Färbung ist kein direkter Proxy für Cannabinoid-Abbau — optische Beobachtung
- Beleuchtungsbedingungen beeinflussen Farbmerkmale erheblich
- Dieses System sagt keine THC-, CBD- oder sonstigen Cannabinoid-Konzentrationen vorher

### Kalibrierung

- Temperature Scaling (bevorzugt) — einzelner Skalar, bewahrt Ranking
- Platt Scaling — Sigmoid-Fit auf Validierungs-Logits
- Reliability-Diagramme bei jeder Evaluierung
- Ziel: ECE < 0,05

### Reproduzierbarkeit

- `GLOBAL_SEED = 42` in allen Trainings-, Sampling- und Augmentierungs-Pipelines
- Datensatz-Splits deterministisch (Session-Hash-basiert)
- Alle Benchmark-Ergebnisse in `docs/progress/benchmark_history.md`

---

## DE 17. Tests {#de-17}

```bash
# Vollständige Test-Suite
pytest tests/ -v

# Schnell (ohne GPU und langsame Integrationstests)
pytest tests/ -m "not gpu and not slow and not integration" -v

# Einzelnes Modul
pytest tests/unit/test_detection_metrics.py -v
pytest tests/unit/test_tensorrt_runner.py -v
pytest tests/unit/test_inference_tiling.py -v

# Mit Coverage
pytest tests/ --cov=. --cov-report=html

# GPU-Tests (erfordert physische GPU + TRICHOME_ENGINE env var)
pytest tests/ -m gpu -v
```

**Aktueller Stand: 960 passed, 4 skipped (GPU-only + reportlab-Guard)**

| Modul | Tests |
|---|---|
| Erkennungsmetriken | 45 |
| Reifegrad-Klassifikator | 38 |
| Segmentierung | 41 |
| VLM Schema Enforcer | 63 |
| Annotierungsstatistiken | 36 |
| Analytics Export | 61 |
| TensorRT Runner + Builder | 35 |
| Tiled Inference | 57 |
| Alle anderen Module | 584 |

---

---

# 🇪🇸 Documentación en Español

## Tabla de Contenidos

1. [¿Qué es esto?](#es-1)
2. [Requisitos de Hardware](#es-2)
3. [Instalación](#es-3)
4. [Primeros Pasos](#es-4)
5. [Recolección de Datos — Consejos para Imágenes](#es-5)
6. [Flujo de Trabajo de Etiquetado](#es-6)
7. [Flujo de Trabajo de Entrenamiento](#es-7)
8. [Verificación y Benchmarking](#es-8)
9. [Ciclo de Mejora](#es-9)
10. [Despliegue con Docker](#es-10)
11. [Todas las URLs y Páginas](#es-11)
12. [Referencia de API](#es-12)
13. [Referencia de CLI](#es-13)
14. [Configuración](#es-14)
15. [Arquitectura](#es-15)
16. [Metodología Científica](#es-16)
17. [Pruebas](#es-17)

---

## ES 1. ¿Qué es esto? {#es-1}

CTIP es una **plataforma de investigación completa y de calidad productiva** para análisis automatizado de tricomas de *Cannabis sativa L.* bajo microscopía digital. No es una demo — es un sistema completo y en funcionamiento para trabajo científico real.

### Capacidades

| Capacidad | Método | Objetivo |
|---|---|---|
| Detección de Tricomas | YOLO v11s + ensemble RTMDet | mAP50 > 0.88 |
| Segmentación de Instancias | SAM2-tiny + refinamiento de máscara | IoU > 0.82 |
| Clasificación de Madurez | HSV + LAB + Textura (LBP/GLCM/Gabor) | F1 > 0.85 |
| Tipificación Morfológica | Geométrico + CNN (pedunculado/sésil/bulboso) | Precisión > 0.90 |
| Medición de Tamaño | Conversión calibrada px a µm | ±5% error |
| Evaluación de Enfoque | Laplacian + Tenengrad + FFT | — |
| Análisis de Video | Ranking de calidad + deduplicación temporal | — |
| Pre-etiquetado VLM | Moondream-2B / Florence-2 / Qwen2-VL (4-bit) | Humano en el loop |
| Aprendizaje Activo | Muestreo por incertidumbre + desacuerdo | — |
| Inferencia TensorRT | Engine FP16, API async v3 | Optimizado para RTX 4060 |

### Lo que NO hace

- Sin predicciones de concentración de THC/cannabinoides (solo madurez óptica)
- Sin pseudociencia
- Las salidas VLM nunca van directamente a datos de entrenamiento (HITL obligatorio)

---

## ES 2. Requisitos de Hardware {#es-2}

| Componente | Mínimo | Recomendado |
|---|---|---|
| GPU | NVIDIA GTX 1080 (8 GB VRAM) | RTX 4060 / 3080 (8+ GB) |
| CPU | 6 núcleos modernos | i5-13400F o mejor |
| RAM | 16 GB | 32 GB |
| Almacenamiento | 50 GB SSD | 500 GB NVMe |
| CUDA | 11.8+ | 12.6 |

### Presupuesto VRAM (RTX 4060, 8 GB)

| Componente | VRAM |
|---|---|
| Inferencia YOLO v11s | ~0.9 GB |
| SAM2-tiny | ~1.8 GB |
| Florence-2 (4-bit) | ~2.1 GB |
| Moondream-2B (4-bit) | ~1.4 GB |
| Qwen2-VL-7B (4-bit) | ~4.8 GB |
| Entrenamiento YOLO v11s (bs=8) | ~5.5 GB |

> Solo **una tarea GPU a la vez** — aplicado mediante asyncio.Semaphore(1). Intencional para tarjetas de 8 GB.

---

## ES 3. Instalación {#es-3}

### 3.1 Prerequisitos

```bash
# Ubuntu 22.04 / 24.04
sudo apt update && sudo apt install -y \
    git curl wget build-essential \
    python3.12 python3.12-venv python3.12-dev \
    ffmpeg libgl1 libglib2.0-0 libsm6 libxext6

# Instalar uv (gestor de paquetes Python rápido)
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.bashrc

# Node.js 20 (para el frontend)
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs

# Verificar CUDA
nvcc --version && nvidia-smi
```

### 3.2 Clonar e Instalar

```bash
git clone https://github.com/tuusuario/trichome-analysis.git
cd trichome-analysis

python3.12 -m venv .venv && source .venv/bin/activate

uv pip install -e ".[dev]"       # núcleo + dev
uv pip install -e ".[vlm]"       # + modelos VLM
uv pip install -e ".[sam]"       # + segmentación SAM2
uv pip install -e ".[all]"       # todo
```

### 3.3 TensorRT (opcional, inferencia en producción)

```bash
sudo apt install -y python3-libnvinfer python3-libnvinfer-dev tensorrt tensorrt-dev

export PATH=/usr/local/cuda-12.6/bin:$PATH && pip install pycuda

SITE=$(python -c "import site; print(site.getsitepackages()[0])")
printf "/usr/lib/python3/dist-packages\n/usr/lib/python3.12/dist-packages\n" > "$SITE/system_trt.pth"
echo 'export PATH=/usr/local/cuda-12.6/bin:$PATH' >> .venv/bin/activate

python -c "import tensorrt; print(tensorrt.__version__)"
```

### 3.4 Frontend

```bash
cd frontend && npm install && cd ..
```

### 3.5 Configuración del Entorno

```bash
cp .env.example .env
```

> **Consejo:** Usa el **Asistente de Configuración** integrado en lugar de editar `.env` manualmente — te guía por cada ajuste de forma interactiva (→ §4.1).

Para configuración manual, las variables clave son:

```env
DATA_ROOT=/mnt/data/trichome          # o ./data para desarrollo local
MODELS_ROOT=/mnt/models/trichome
CUDA_VISIBLE_DEVICES=0
VRAM_LIMIT_GB=8.0
MLFLOW_TRACKING_URI=http://localhost:3004
EXPERIMENT_TRACKER=mlflow
LABEL_STUDIO_URL=http://localhost:3005
LABEL_STUDIO_API_KEY=tu_clave_aqui
```

---

## ES 4. Primeros Pasos {#es-4}

### 4.1 Iniciar en Modo Desarrollo y Configuración Inicial

```bash
# Terminal 1 — Backend API
source .venv/bin/activate
uvicorn backend.main:app --reload --port 8000

# Terminal 2 — Frontend
cd frontend && npm run dev
```

Abrir **http://localhost:3000** — el **Asistente de Configuración se inicia automáticamente** en el primer arranque (cuando no hay `.env` configurado).

El asistente recorre 7 pasos:

| Paso | Configura |
|---|---|
| 🌐 Red | Dominio público vs. solo localhost, puerto nginx |
| ⚙️ Hardware | Dispositivo CUDA, presupuesto VRAM |
| 💾 Almacenamiento | Directorio raíz, modelos, salidas |
| 🔌 Servicios | API Key de Label Studio, URI de MLflow, W&B (opcional) |
| 🔒 Seguridad | Clave secreta (generador automático), token API |
| ✅ Revisión | Resumen de todos los ajustes antes de guardar |
| 🎉 Listo | Escribe `.env`, muestra comando de reinicio Docker |

Al terminar, `.env` se escribe automáticamente — sin edición manual.
Volver a ejecutar en cualquier momento desde la barra lateral: **Configuración Inicial**.

- Docs API: http://localhost:8000/docs

### 4.2 Verificar el Sistema

```bash
pytest tests/ -v --tb=short
# Esperado: 960 passed, 4 skipped

python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
curl http://localhost:8000/api/v1/system/health | python -m json.tool
```

### 4.3 Primera Detección

```bash
# CLI
trichome detect --input /ruta/a/imagen.jpg --tiled --tile-size 1280

# API
curl -X POST http://localhost:8000/api/v1/detection/infer \
  -F "file=@imagen.jpg" -F "confidence_threshold=0.25" | python -m json.tool

# Frontend: http://localhost:3000/inference — arrastrar imagen
```

---

## ES 5. Recolección de Datos — Consejos para Imágenes {#es-5}

Los buenos datos son el factor más importante. Lo que realmente importa en microscopía de tricomas:

### 5.1 Equipamiento

| Configuración | Notas |
|---|---|
| Microscopio digital | 40x–200x de aumento. Los microscopios USB (Andonstar, Celestron, Jiusion) son válidos para empezar. |
| Teléfono + lente clip | Aceptable para tricomas grandes, malo para bulbosos/pequeños. |
| Microscopio estéreo | Mejor claridad óptica, más difícil de digitalizar consistentemente. |

### 5.2 Protocolo de Captura (Crítico)

```
HACER:
  - Magnificación consistente en cada sesión (ej. siempre 100x)
  - Capturar en RAW o máxima calidad JPEG
  - Imagen de calibración con micrómetro ocular (ej. 1 mm)
    Permite calibración px a µm
  - Mínimo 1920x1080, idealmente 4K
  - Misma posición de fuente de luz siempre
  - Nombre de archivo con metadatos:
    microscopio01_100x_20260101_muestra42_001.jpg
  - Todos los tipos de tricomas en una sesión
  - Parches de fondo vacíos como ejemplos negativos

NO HACER:
  - Mezclar magnificaciones sin registrarlas (arruina la calibración)
  - Usar exposición automática (brillo inconsistente)
  - Fotografiar solo tricomas perfectos (incluir parciales, solapados, en bordes)
  - Usar imágenes comprimidas de redes sociales
```

### 5.3 Cobertura de Etapas de Madurez

| Etapa | Visual | Objetivo % del dataset |
|---|---|---|
| Clara | Vítrea, completamente transparente | ~25% |
| Nublada | Blanca/lechosa, opaca | ~35% |
| Ámbar | Dorado-naranja, degradado | ~25% |
| Mixta | Especímenes en transición | ~15% |

### 5.4 Organización de Datos

```
data/
├── raw/                    # Imágenes originales sin modificar
│   ├── sesion_20260101/
│   └── sesion_20260115/
├── calibration/            # Imágenes de micrómetro por microscopio+magnificación
├── annotated/              # Tras etiquetar (Label Studio exporta aquí)
│   ├── images/
│   └── labels/             # Archivos .txt formato YOLO
└── splits/                 # train / val / test — NUNCA mezclar sesiones
    ├── train/
    ├── val/
    └── test/
```

> **Nunca** poner imágenes de la misma sesión en train Y val/test — eso es fuga de datos.
> Dividir siempre por **sesión**, no por imagen.

### 5.5 Filtro de Calidad de Enfoque

```bash
trichome focus --input data/raw/sesion_20260101/ \
               --output data/filtrado/ \
               --min-sharpness 80.0 --copy-passing

# O via API
curl -X POST http://localhost:8000/api/v1/focus/score -F "file=@imagen.jpg"
```

### 5.6 Tamaño Mínimo del Dataset

| Fase | Imágenes | Anotaciones (cajas) |
|---|---|---|
| Primer modelo funcional | 150–300 | 2.000–5.000 |
| Buena generalización | 500–1.000 | 10.000–25.000 |
| Listo para producción | 2.000+ | 50.000+ |

Empezar pequeño, entrenar rápido, identificar casos de fallo, recolectar imágenes específicas.

---

## ES 6. Flujo de Trabajo de Etiquetado {#es-6}

### 6.1 Iniciar Label Studio

```bash
# Docker (recomendado)
cd docker && docker compose --profile annotation up -d label-studio
# Independiente
pip install label-studio && label-studio start --port 3005
```

Acceso: http://localhost:3005

### 6.2 Crear Proyecto

1. **Create Project** — asignar nombre (ej. "Tricomas Sesión 20260101")
2. **Labeling Setup** → Object Detection with Bounding Boxes
3. Agregar etiquetas (ortografía exacta requerida para exportación YOLO):

```
capitate-stalked    #FF4444  (rojo)
capitate-sessile    #44FF44  (verde)
bulbous             #4444FF  (azul)
non-glandular       #FFAA00  (naranja)
```

4. **Importar imágenes**: Settings → Cloud Storage → Add Source Storage → Local Files → ruta a `data/raw/sesion_XXXXXXXX/`

### 6.3 Pre-etiquetado VLM (3–5x más rápido)

```bash
curl -X POST http://localhost:8000/api/v1/vlm/label \
  -H "Content-Type: application/json" \
  -d '{
    "image_paths": ["data/raw/sesion_20260101/img001.jpg"],
    "model": "florence2",
    "confidence_threshold": 0.3
  }'
```

Las anotaciones van a una **cola de revisión** — nunca directamente a datos de entrenamiento.

| Modelo | VRAM | Velocidad | Calidad |
|---|---|---|---|
| Moondream-2B (4-bit) | ~1.4 GB | Rápido | Bueno para detección |
| Florence-2-large (4-bit) | ~2.1 GB | Medio | Mejor para escenas complejas |
| Qwen2-VL-7B (4-bit) | ~4.8 GB | Lento | Mayor calidad |

### 6.4 Estándares de Anotación

```
Reglas para dibujar cajas:
  SI: Caja ajustada alrededor de la cabeza del tricoma (no el tallo)
  SI: Incluir cabeza completa aunque esté parcialmente ocluida
  SI: Marcar tricomas en los bordes de la imagen
  SI: Etiquetar TODOS los tricomas visibles — sin saltarse ninguno
  SI: Ante duda pedunculado/sésil: buscar cuello visible

  NO: Cajas alrededor de tallos desnudos (sin cabeza)
  NO: Etiquetar residuos o artefactos
  NO: Saltarse tricomas borrosos si son identificables
```

### 6.5 Exportar y Verificar Anotaciones

```bash
# Label Studio UI: Project → Export → YOLO format → Download

# Via API
curl -X POST http://localhost:8000/api/v1/annotation/export \
  -H "Content-Type: application/json" \
  -d '{"project_id": 1, "format": "yolo", "output_dir": "data/annotated/sesion_20260101"}'

# Verificar calidad
curl -X POST http://localhost:8000/api/v1/annotation/stats \
  -H "Content-Type: application/json" \
  -d '{"annotation_dir": "data/annotated/sesion_20260101"}'
```

---

## ES 7. Flujo de Trabajo de Entrenamiento {#es-7}

### 7.1 Preparar División del Dataset

```bash
# Dividir por sesión — previene fuga de datos
trichome dataset split \
  --input data/annotated/ --output data/splits/ \
  --train 0.75 --val 0.15 --test 0.10 --split-by session

trichome dataset verify --path data/splits/
```

### 7.2 Configurar Entrenamiento

`configs/training/yolo11s_detection.yaml`:

```yaml
model: yolo11s.pt           # Descarga automáticamente
task: detect
data: data/splits/dataset.yaml
imgsz: 1280                 # Requerido para inferencia tiled
batch: 8                    # Óptimo para RTX 4060 8GB
workers: 4
epochs: 100
patience: 20
lr0: 0.01
cos_lr: true

# Augmentación específica para microscopía
degrees: 90.0               # Rotación completa — tricomas sin orientación canónica
flipud: 0.5
fliplr: 0.5
hsv_h: 0.015                # Pequeño cambio de tono — iluminación variable
hsv_s: 0.7
mosaic: 0.3                 # Mosaico más bajo — contexto microscópico importa

device: 0
amp: true                   # Precisión mixta FP16
```

### 7.3 Iniciar Entrenamiento

```bash
# CLI
trichome train detection --config configs/training/yolo11s_detection.yaml

# API (no bloqueante, progreso via WebSocket)
curl -X POST http://localhost:8000/api/v1/training/start \
  -H "Content-Type: application/json" \
  -d '{"config_path": "configs/training/yolo11s_detection.yaml"}'

# Dashboard en vivo: http://localhost:3000/training
# Stream WebSocket:  ws://localhost:8000/ws/training
# MLflow UI:         http://localhost:3004
```

### 7.4 Salida del Entrenamiento

```
runs/detect/trichome_yolo11s_20260101/
    weights/best.pt      <- usar este para inferencia
    weights/last.pt
    results.csv
    confusion_matrix.png
    PR_curve.png
    val_batch0_pred.jpg
```

---

## ES 8. Verificación y Benchmarking {#es-8}

### 8.1 Evaluar en Conjunto de Prueba

```bash
trichome benchmark detection \
  --weights runs/detect/trichome_yolo11s_20260101/weights/best.pt \
  --split test --data data/splits/dataset.yaml \
  --conf 0.25 --iou 0.5
```

Salida esperada:

```
Clase               P      R      mAP50  mAP50-95
all                 0.887  0.862  0.883  0.512
capitate-stalked    0.921  0.905  0.918  0.561
capitate-sessile    0.873  0.841  0.864  0.498
bulbous             0.841  0.812  0.832  0.445
non-glandular       0.913  0.890  0.918  0.543
```

### 8.2 Calibración de Confianza

Las puntuaciones de confianza de YOLO no están calibradas. Corregir antes del despliegue:

```bash
curl -X POST http://localhost:8000/api/v1/detection/calibrate \
  -H "Content-Type: application/json" \
  -d '{
    "weights_path": "runs/.../best.pt",
    "val_data": "data/splits/val/",
    "method": "temperature"
  }'
```

Objetivo: ECE < 0.05. Diagramas de fiabilidad generados automáticamente.

### 8.3 Construir Engine TensorRT

```bash
# Exportar YOLO a ONNX
python -c "
from ultralytics import YOLO
YOLO('runs/.../best.pt').export(format='onnx', imgsz=1280, dynamic=True, half=True)
"

# Construir engine FP16
trichome build-engine \
  --onnx runs/.../best.onnx \
  --output models/trichome_yolo11s_fp16.engine \
  --fp16 --imgsz 1280 --workspace-gb 4

# Benchmark TRT vs PyTorch
trichome benchmark inference \
  --engine models/trichome_yolo11s_fp16.engine \
  --pytorch runs/.../best.pt \
  --image data/splits/test/images/ --n 100
```

### 8.4 Benchmark Inferencia Tiled

```bash
trichome benchmark tiled \
  --weights runs/.../best.pt \
  --image data/splits/test/images/alta_resolucion_001.jpg \
  --tile-sizes 640 1280 --overlaps 0.1 0.2 0.3
```

---

## ES 9. Ciclo de Mejora {#es-9}

### 9.1 Aprendizaje Activo — Encontrar Casos Difíciles

```bash
curl -X POST http://localhost:8000/api/v1/active_learning/sample \
  -H "Content-Type: application/json" \
  -d '{"strategy": "uncertainty", "n_samples": 50, "unlabeled_dir": "data/raw/nueva_sesion/"}'
```

Etiquetar primero las imágenes devueltas — enseñan más al modelo por hora de anotación.

### 9.2 Casos de Fallo Comunes

| Fallo | Causa | Solución |
|---|---|---|
| Tricomas bulbosos no detectados | Muy pequeños en training | Recolectar primeros planos específicos |
| Falsos positivos en residuos | Se parecen a tricomas | Etiquetar como non-glandular |
| Confusión pedunculado/sésil | Tallo corto en mal ángulo | Añadir más variaciones de ángulo |
| Detección pobre en bordes | Artefactos de relleno | Aumentar solapamiento en tiled inference |
| mAP bajo con IoU alto | Cajas dibujadas muy holgadas | Reforzar protocolo de anotación |

### 9.3 Lista de Verificación Post-entrenamiento

```
  [ ] Revisar matriz de confusión — ¿qué clase se confunde con cuál?
  [ ] Examinar val_batch*_pred.jpg — ¿dónde falla visualmente el modelo?
  [ ] Aplicar muestreo activo a nuevos datos sin etiquetar
  [ ] Verificar distribución de clases (¿equilibrada?)
  [ ] Añadir imágenes específicas para clases con bajo rendimiento
  [ ] Re-verificar calidad de anotación (Cohen kappa >= 0.80)
  [ ] Sin solapamiento de sesiones en splits
  [ ] Ejecutar calibración tras cada nuevo entrenamiento
```

### 9.4 Triggers de Reentrenamiento

```bash
curl http://localhost:8000/api/v1/active_learning/trigger | python -m json.tool
```

Se activan cuando:
- 100+ nuevas imágenes anotadas desde último entrenamiento
- Incertidumbre media de predicciones recientes > 0.45
- Nueva distribución de clases diverge > 15% de la distribución de entrenamiento

---

## ES 10. Despliegue con Docker {#es-10}

### Stack Principal (nginx + backend + frontend + MLflow)

```bash
cd docker
docker compose build       # solo la primera vez
docker compose up -d
docker compose logs -f
docker compose down
```

### Con Herramientas de Anotación (Label Studio + CVAT + PostgreSQL)

```bash
docker compose --profile annotation up -d
```

### Con Stack de Entrenamiento GPU

```bash
docker compose -f docker-compose.yml -f docker-compose.training.yml up -d
docker exec trichome-backend nvidia-smi   # verificar acceso GPU
```

### Solo Inferencia (ligero, sin frontend)

```bash
docker compose -f docker-compose.inference.yml up -d
```

### Configuración del Entorno para Docker

```bash
cp .env.example .env
# Usar rutas internas del contenedor:
# DATA_ROOT=/data
# MODELS_ROOT=/models
# MLFLOW_TRACKING_URI=http://mlflow:5000   <- DNS interno Docker
```

### Volúmenes de Datos

```bash
docker volume ls | grep trichome
# trichome-models        pesos de modelos (compartidos)
# trichome-mlflow        datos de experimentos
# trichome-db            base de datos SQLite
# trichome-label-studio  datos de Label Studio
```

### Actualizar / Reconstruir

```bash
cd docker && git pull
docker compose build --no-cache && docker compose up -d
```

---

## ES 11. Todas las URLs y Páginas {#es-11}

### Modo Desarrollo (sin Docker)

| Servicio | URL | Propósito |
|---|---|---|
| Frontend | http://localhost:3000 | Interfaz web principal |
| API Swagger | http://localhost:8000/docs | Docs API interactivos |
| API ReDoc | http://localhost:8000/redoc | Referencia API |
| WS Entrenamiento | ws://localhost:8000/ws/training | Stream entrenamiento en vivo |
| WS Sistema | ws://localhost:8000/ws/system | Stats sistema/GPU |
| WS Jobs | ws://localhost:8000/ws/jobs | Estado de trabajos en segundo plano |
| WS Logs | ws://localhost:8000/ws/logs | Stream de logs en vivo |

### Modo Docker

| Servicio | URL Local | URL Pública (via nginx) |
|---|---|---|
| Gateway Nginx | http://localhost:3001 | http://your-domain.com:3001 |
| Backend API | http://localhost:3002/api/v1 | .../api/v1/ |
| Docs API | http://localhost:3002/docs | .../docs |
| Frontend | http://localhost:3003 | .../ |
| MLflow | http://localhost:3004 | .../mlflow/ |
| Label Studio | http://localhost:3005 | .../annotation/ |
| CVAT | http://localhost:3006 | .../cvat/ |

### Páginas del Frontend

| Página | Ruta | Función |
|---|---|---|
| Dashboard | / | Resumen del sistema, estado GPU, trabajos recientes |
| Inferencia | /inference | Subir imagen, ejecutar detección/segmentación |
| Datasets | /datasets | Explorar, importar, validar datasets |
| Anotación | /annotation | Revisar pre-etiquetas VLM, gestionar Label Studio |
| Label Studio | /labelstudio | Label Studio integrado |
| Entrenamiento | /training | Iniciar, monitorizar, comparar entrenamientos |
| Modelos | /models | Registro de modelos, gestión de versiones |
| Experimentos | /experiments | Comparación de experimentos MLflow |
| Morfología | /morphology | Resultados de análisis morfológico |
| Analytics | /analytics | Generar informes PDF/CSV/JSON |
| Video | /video | Pipeline de video, extracción de frames |
| Informes | /reports | Archivo de informes pasados |
| Benchmarks | /benchmarks | Historial y comparación de benchmarks |
| Sistema | /system | Stats de hardware, monitor de procesos |
| Procesos | /processes | Gestión de contenedores Docker, compose, logs en vivo |
| Configuración | /settings | Config, calibración, claves API |
| Wiki | /wiki | Documentación integrada (ES/EN/DE, 14 páginas) |

---

## ES 12. Referencia de API {#es-12}

Ruta base: `/api/v1/` — Docs completos: http://localhost:8000/docs

```bash
# Detección
POST /detection/infer               # Inferencia imagen única
POST /detection/infer/tiled         # Inferencia tiled (imágenes 4K)
POST /detection/infer/batch         # Inferencia por lotes
POST /detection/calibrate           # Calibrar puntuaciones de confianza

# Segmentación
POST /segmentation/segment          # Segmentación instancias SAM2
POST /segmentation/refine           # Refinar máscara existente

# Madurez
POST /maturity/classify             # Clasificar madurez de región
POST /maturity/classify/batch       # Clasificación por lotes
GET  /maturity/thresholds           # Umbrales actuales
PUT  /maturity/thresholds           # Actualizar umbrales

# Entrenamiento
POST /training/start                # Iniciar trabajo de entrenamiento
GET  /training/status               # Estado actual del entrenamiento
POST /training/stop                 # Detener entrenamiento
GET  /training/runs                 # Listar todos los runs
GET  /training/runs/{run_id}        # Detalles del run + métricas
POST /training/evaluate             # Evaluar en conjunto de prueba

# Pre-etiquetado VLM
POST /vlm/label                     # Pre-etiquetado VLM (a cola de revisión)
GET  /vlm/queue                     # Elementos pendientes de revisión
POST /vlm/queue/{id}/approve        # Aprobar pre-etiqueta
POST /vlm/queue/{id}/reject         # Rechazar pre-etiqueta
GET  /vlm/models                    # Modelos VLM disponibles

# Aprendizaje Activo
POST /active_learning/sample        # Muestras inciertas para etiquetar
GET  /active_learning/trigger       # Verificar trigger de reentrenamiento
POST /active_learning/priority      # Definir cola de prioridad

# Anotación
POST /annotation/export             # Exportar desde Label Studio (YOLO/COCO/CSV)
POST /annotation/stats              # Estadísticas de calidad
POST /annotation/import             # Importar lote de anotaciones
GET  /annotation/projects           # Listar proyectos de Label Studio

# Analytics e Informes
POST /analytics/report              # Generar informe (PDF/CSV/JSON)
GET  /analytics/reports             # Listar informes pasados
GET  /analytics/reports/{id}        # Descargar informe específico

# Sistema
GET  /system/health                 # Health check completo + stats GPU
GET  /system/gpu                    # VRAM, temperatura, utilización
GET  /system/version                # Versiones de todos los componentes
GET  /models                        # Registro de modelos cargados
```

---

## ES 13. Referencia de CLI {#es-13}

```bash
# Detección
trichome detect   --input imagen.jpg --tiled --tile-size 1280 --conf 0.25

# Segmentación
trichome segment  --input imagen.jpg --model sam2-tiny

# Madurez
trichome maturity --input imagen.jpg

# Filtro de enfoque
trichome focus    --input data/raw/ --output data/filtrado/ --min-sharpness 80

# Gestión de datasets
trichome dataset split  --input data/annotated/ --output data/splits/ \
                        --train 0.75 --val 0.15 --test 0.10 --split-by session
trichome dataset verify --path data/splits/
trichome dataset stats  --path data/annotated/

# Entrenamiento
trichome train detection --config configs/training/yolo11s_detection.yaml

# Evaluación
trichome benchmark detection --weights runs/.../best.pt --split test

# Construir engine
trichome build-engine --onnx modelo.onnx --output modelo.engine --fp16 --imgsz 1280

# Generar informe
trichome report   --input resultados.json --format pdf --output informe.pdf
```

---

## ES 14. Configuración {#es-14}

```env
# Rutas
TRICHOME_ROOT=/path/to/trichome-analysis
DATA_ROOT=/mnt/data/trichome
MODELS_ROOT=/mnt/models/trichome

# Hardware
CUDA_VISIBLE_DEVICES=0
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
VRAM_LIMIT_GB=8.0
GPU_INFERENCE_QUEUE_DEPTH=0        # 0 = fail-fast, sin cola

# Base de datos
DATABASE_URL=sqlite:///./trichome.db

# Anotación
LABEL_STUDIO_URL=http://localhost:3005
LABEL_STUDIO_API_KEY=tu_clave
CVAT_URL=http://localhost:3006

# Seguimiento de experimentos
MLFLOW_TRACKING_URI=http://localhost:3004
EXPERIMENT_TRACKER=mlflow          # mlflow | wandb | both | none
WANDB_API_KEY=tu_clave

# Modelos VLM
FLORENCE2_MODEL_ID=microsoft/Florence-2-large
MOONDREAM_MODEL_ID=vikhyatk/moondream2
VLM_CACHE_DIR=/mnt/models/vlm_cache
```

---

## ES 15. Arquitectura {#es-15}

### Estructura de Módulos (Domain-Driven Design)

```
<modulo>/
  domain/          # Lógica de negocio pura (sin dependencias de framework)
  application/     # Orquesta objetos de dominio (pipelines)
  infrastructure/  # Backends de modelos, I/O de archivos, APIs externas
  api/             # Router FastAPI para este módulo
  schemas/         # Modelos Pydantic request/response
```

Módulos: `detection/`, `segmentation/`, `maturity/`, `morphology/`, `measurement/`,
`focus/`, `vlm_labeling/`, `annotation/`, `active_learning/`, `training/`, `inference/`,
`video_pipeline/`, `analytics/`

### Pipeline CV

```
Imagen
  -> Evaluador de enfoque (descartar frames borrosos)
  -> Inferencia tiled (YOLO v11s, tiles 1280px, 20% solapamiento)
  -> Calibración de confianza (temperature scaling)
  -> [Opcional] Ensemble RTMDet
  -> SAM2-tiny (segmentación por prompts con cajas YOLO)
  -> Refinamiento de máscaras (rellenar huecos, suavizar contornos)
  -> Clasificador morfológico (pedunculado / sésil / bulboso)
  -> Clasificador de madurez (clara -> nublada -> ámbar)
  -> Medición (px -> µm via CalibrationScale)
  -> Motor de analytics (estadísticas, generación de informes)
```

### Restricciones Arquitectónicas

- `asyncio.Semaphore(1)` — solo una tarea GPU a la vez
- Salidas VLM nunca escritas directamente en datos de entrenamiento
- `GLOBAL_SEED = 42` en todos los pipelines de entrenamiento y muestreo
- `backend/middleware/gpu_guard.py` — HTTP 429 cuando se supera el presupuesto VRAM

---

## ES 16. Metodología Científica {#es-16}

### Clasificación de Madurez

La madurez se evalúa únicamente a partir de **características ópticas** — sin afirmaciones químicas:

| Grupo de Características | Características Usadas |
|---|---|
| Color (HSV) | Tono medio, saturación, valor por región de tricoma |
| Color (LAB) | L* (luminosidad), a* (verde-rojo), b* (azul-amarillo) |
| Textura | LBP (Local Binary Patterns), GLCM (matriz de co-ocurrencia), filtros Gabor |
| Morfología | Diámetro de cabeza, longitud de tallo, circularidad |

**Limitaciones explícitas:**
- La coloración ámbar NO es un proxy directo de degradación de cannabinoides — es una observación óptica
- Las condiciones de iluminación afectan significativamente las características de color — la imagen consistente es crítica
- Este sistema **no** predice concentraciones de THC, CBD ni ningún otro cannabinoide

### Calibración

- Temperature scaling (preferido) — parámetro escalar único, preserva el ranking
- Platt scaling — ajuste sigmoide en logits de validación
- Diagramas de fiabilidad generados en cada evaluación
- ECE (Error de Calibración Esperado) < 0.05 como objetivo

### Reproducibilidad

- `GLOBAL_SEED = 42` en todos los pipelines
- Divisiones de dataset deterministas (hash de sesión)
- Todos los resultados de benchmark almacenados en `docs/progress/benchmark_history.md`

---

## ES 17. Pruebas {#es-17}

```bash
# Suite completa
pytest tests/ -v

# Rápido (sin GPU ni integración lenta)
pytest tests/ -m "not gpu and not slow and not integration" -v

# Módulo específico
pytest tests/unit/test_detection_metrics.py -v
pytest tests/unit/test_tensorrt_runner.py -v
pytest tests/unit/test_inference_tiling.py -v

# Con cobertura
pytest tests/ --cov=. --cov-report=html

# Pruebas GPU (requiere TRICHOME_ENGINE env var)
pytest tests/ -m gpu -v
```

**Estado actual: 960 passed, 4 skipped (GPU-only + guard de reportlab)**

| Módulo | Pruebas |
|---|---|
| Métricas de detección | 45 |
| Clasificador de madurez | 38 |
| Segmentación | 41 |
| VLM schema enforcer | 63 |
| Estadísticas de anotación | 36 |
| Exportación analytics | 61 |
| TensorRT runner + builder | 35 |
| Inferencia tiled | 57 |
| Resto de módulos | 584 |

---

---

# 🔧 Technology Stack & Rationale · Technologie-Stack & Begründung · Stack Tecnológico y Justificación

---

## 🇬🇧 Why These Technologies?

Every technology choice in CTIP was made deliberately. This section explains what we use and why we picked it over the alternatives.

---

### Python 3.12

**Why:** 3.12 brings significant performance improvements (5–15% faster than 3.10 in CPU-bound scientific workloads) and better error messages. The scientific ML ecosystem (PyTorch, NumPy, OpenCV) fully supports it. We use `uv` as the package manager — it is 10–100x faster than pip for dependency resolution and installs.

---

### PyTorch 2.x + CUDA 12.x

**Why PyTorch over TensorFlow:** PyTorch has dominated research and production CV since 2020. The ecosystem — Ultralytics, SAM2, Hugging Face Transformers — is all PyTorch-native. Dynamic computation graphs make debugging easier. TorchCompile (`torch.compile`) provides significant inference speedups without leaving Python.

**Why CUDA 12.x:** Required for TensorRT 10.x and the latest PyTorch CUDA kernels. CUDA 12.x introduces improved memory management (`expandable_segments:True` in `PYTORCH_CUDA_ALLOC_CONF`) which reduces fragmentation on 8 GB VRAM cards — critical for this hardware target.

---

### YOLO v11s — Detection Backbone

**Why YOLO over Faster-RCNN / DETR / RT-DETR:**
- YOLO v11s achieves comparable mAP to heavier two-stage detectors at 3–5x the inference speed
- Single-stage architecture fits in 8 GB VRAM alongside SAM2 and calibration
- Ultralytics provides production-grade training pipelines, augmentation, and ONNX/TensorRT export out of the box
- `s` (small) variant: best accuracy/VRAM tradeoff on RTX 4060 — `n` underfits on complex microscopy, `m` requires too much VRAM when SAM2 is co-loaded

**Why tiled inference (1280px tiles, 20% overlap):**
Trichomes in 4K microscopy images are tiny relative to image size. Training at 640px means small objects get lost. Tiled inference processes each 1280px crop independently and merges results with WBF (Weighted Boxes Fusion) — this recovers detection of small/edge trichomes that full-image inference misses.

---

### SAM2-tiny — Instance Segmentation

**Why SAM2 over Mask-RCNN / instance-trained YOLO:**
SAM2 (Segment Anything Model 2) is prompt-based — it takes the YOLO bounding boxes as spatial prompts and generates high-quality instance masks without needing mask annotations in the training dataset. This decouples detection labels (cheap to collect) from segmentation masks (expensive to annotate). The `tiny` variant fits in ~1.8 GB VRAM.

**Why not just YOLO segmentation (`yolo11s-seg`):**
YOLO seg produces coarser masks and requires segmentation annotations from the start. SAM2 produces significantly sharper masks for irregular biological shapes like trichome heads and requires zero mask annotation effort.

---

### TensorRT 10.x — Production Inference

**Why TensorRT over plain ONNX Runtime:**
TensorRT compiles the model into an optimized engine for the specific GPU it runs on. On RTX 4060, FP16 TRT engines are typically 2–4x faster than ONNX Runtime FP32 and 1.5–2x faster than ONNX FP16. The TRT 10.x API (`execute_async_v3`, `set_tensor_address`) is fully asynchronous and zero-copy where possible.

**Why FP16:**
FP16 (half precision) halves VRAM usage with negligible accuracy loss (< 0.5% mAP drop in our testing). On RTX 4060, Tensor Cores execute FP16 matmuls at 2x the throughput of FP32.

---

### FastAPI — Backend Framework

**Why FastAPI over Django / Flask / Express:**
- **Native async:** FastAPI is built on Starlette + asyncio. Background GPU tasks, WebSocket streams, and REST requests share one event loop without threading overhead.
- **Automatic OpenAPI docs:** All 128+ endpoints get Swagger UI and ReDoc for free — no manual documentation.
- **Pydantic validation:** Request/response types are validated, serialized, and documented from the same dataclass definitions.
- **Speed:** FastAPI benchmarks faster than Flask and Django in async I/O scenarios (the dominant pattern in this system).

**Why WebSockets for training/system monitoring:**
Training runs for minutes to hours. Polling REST every second is wasteful and laggy. WebSocket streams push metric updates at sub-second latency directly to the dashboard — no client polling needed.

---

### Next.js 14 (App Router) — Frontend

**Why Next.js over plain React / Vue / Svelte:**
- **App Router** (Next.js 14) enables React Server Components — static UI renders on the server, reducing JS bundle size and client load.
- **File-based routing** maps cleanly to the 16 distinct tool pages (inference, training, annotation, etc.)
- **TypeScript native:** The scientific tooling needs reliable type contracts between frontend and backend schemas.

**Why not a simple admin template (Grafana, Retool, etc.):**
This is a scientific instrument interface, not a dashboard. It needs custom annotation review workflows, overlaid bounding box / mask visualization, training loss curves with scientific context, and advanced configuration panels that generic dashboard tools cannot express.

---

### Label Studio + CVAT — Annotation

**Why two annotation tools:**
- **Label Studio:** Best for bounding box + classification annotation. Simple UI, fast keyboard shortcuts, built-in project management, Python SDK for programmatic import/export.
- **CVAT:** Better for polygon / instance mask annotation and multi-annotator workflows with task assignment. Useful when transitioning from box-only to mask annotation.

Both export YOLO-compatible formats. Using both gives flexibility — start with Label Studio for boxes, move to CVAT for masks when needed.

**Why not just use Roboflow / cloud annotation:**
Data sovereignty. Microscopy images may contain proprietary strain or cultivation information. All annotation stays on-premise. Label Studio and CVAT run fully air-gapped.

---

### Florence-2 / Moondream-2B / Qwen2-VL — VLM Pre-labeling

**Why Vision Language Models for pre-labeling:**
Manual annotation of microscopy images is slow and requires expertise. VLMs can generate 60–80% accurate bounding box candidates that a human then corrects in 20–30% of the time of manual annotation. This is the core speed multiplier of the labeling workflow.

**Why three VLMs:**
Different tradeoffs in quality, speed, and VRAM:

| Model | VRAM | Speed | Best for |
|---|---|---|---|
| Moondream-2B (4-bit) | 1.4 GB | Fast | High-volume first pass |
| Florence-2-large (4-bit) | 2.1 GB | Medium | Best detection accuracy |
| Qwen2-VL-7B (4-bit) | 4.8 GB | Slow | Ambiguous or complex images |

**Why 4-bit quantization:**
Full precision Florence-2 requires ~8 GB alone — the entire VRAM budget. 4-bit quantization via bitsandbytes reduces this to ~2 GB with less than 3% accuracy drop, enabling co-loading with other models.

**Why HITL (Human-In-The-Loop) is mandatory:**
VLMs hallucinate. They will confidently produce plausible but wrong bounding boxes. Sending VLM output directly to training data would introduce systematic errors that compound with each retraining cycle. All VLM outputs go to a review queue — a human approves each annotation before it becomes training data. This is a hard architectural constraint, not optional.

---

### Active Learning — Uncertainty Sampling

**Why active learning over random sampling:**
Not all images teach the model equally. Images the model is already confident on add little. Images near the decision boundary (high entropy predictions) improve calibration and generalization the most. Active learning prioritizes labeling effort toward where it matters — typically 30–50% fewer annotations needed to reach the same accuracy.

**Three strategies:**
- **Uncertainty:** High-entropy output distribution (model unsure between classes)
- **Disagreement:** Ensemble disagreement between YOLO and RTMDet predictions
- **Hybrid:** Weighted combination of both

---

### Confidence Calibration (Temperature / Platt Scaling)

**Why calibration matters:**
A YOLO model outputting confidence 0.85 does not mean 85% of such predictions are correct. Raw YOLO scores are often overconfident. Without calibration, downstream decisions (threshold tuning, active learning uncertainty estimates, report confidence intervals) are based on misleading numbers.

**Why temperature scaling:**
Temperature scaling is a single-parameter post-hoc calibration — it does not change the model, only rescales the output logits. It preserves ranking (mAP unchanged) while aligning confidence scores to actual accuracy. ECE (Expected Calibration Error) < 0.05 is the target.

---

### MLflow — Experiment Tracking

**Why MLflow over Weights & Biases:**
- **Self-hosted:** No data leaves the machine. Experiment data stays on-premise.
- **No account required:** W&B requires cloud sign-up even for local use in some modes.
- **ONNX/PyTorch model registry built in:** MLflow's model registry integrates directly with training output.
- W&B is supported too via `EXPERIMENT_TRACKER=wandb` — the choice is configurable.

---

### Docker + Nginx — Deployment

**Why Docker Compose over bare-metal / systemd:**
Docker Compose captures the entire dependency graph (nginx, backend, frontend, MLflow, Label Studio, CVAT, PostgreSQL) in version-controlled YAML. Any machine with Docker can reproduce the full stack in one command. No "works on my machine" problems.

**Why nginx as reverse proxy:**
Single entry point on port 3001. Nginx routes `/api/v1/` to backend, `/mlflow/` to MLflow, `/annotation/` to Label Studio, `/cvat/` to CVAT, and `/` to the Next.js frontend — all behind one port with TLS termination possible. This is production architecture, not a development shortcut.

**Default: localhost only — no public exposure out of the box.**
By default, nginx binds to `127.0.0.1:3001` — only accessible from the local machine. To expose publicly, set your domain in `.env` and the Setup Wizard (Settings → Setup in the web UI) handles nginx config automatically:

```env
# .env
PUBLIC_DOMAIN=your-domain.com     # e.g. mylab.ddns.net — leave empty for localhost only
PUBLIC_PORT=3001
```

To configure nginx manually:

```nginx
# docker/nginx/nginx.conf — find this block and change server_name:

server {
    listen 80;
    server_name localhost;       # ← change to your domain, or leave as localhost
    # server_name mylab.ddns.net; # ← uncomment and set your domain for public access
    ...
}
```

To **disable** public access completely (localhost only, no DDNS):
```env
PUBLIC_DOMAIN=                    # leave empty
```
Nginx will then only respond to `localhost` / `127.0.0.1` — safe for local-only research setups.

> 💡 The Setup Wizard in the web UI (`Settings → First-Time Setup`) configures all of this interactively without editing config files manually.

**Why separate docker-compose files for training:**
The GPU training container needs `runtime: nvidia`, elevated memory limits, and different restart policies. Keeping it separate means the core stack (inference + annotation) can run on machines without NVIDIA Docker runtime.

---

### SQLite → PostgreSQL

**Why SQLite in development:**
Zero configuration. No separate process. Works identically on any OS. For a single-node research platform with one user, SQLite is entirely sufficient — it handles thousands of queries per second.

**Why PostgreSQL for production:**
Multi-user annotation workflows, concurrent API requests, and production annotation databases (Label Studio + CVAT both use PostgreSQL internally) benefit from proper ACID transactions and concurrent write support. Migration is one config line: `DATABASE_URL=postgresql://...`

---

### Domain-Driven Design (DDD)

**Why DDD over flat module structure:**
Each scientific domain (detection, segmentation, maturity, morphology, measurement) has its own distinct rules, entities, and invariants. DDD separates these cleanly:

- `domain/` — pure scientific logic, no I/O, fully testable in isolation
- `application/` — orchestrates domain objects into pipelines
- `infrastructure/` — swappable backends (YOLO → RTMDet, SAM2 → MobileSAM, etc.)
- `api/` — thin FastAPI layer, no business logic

This means the maturity classifier can be tested without starting a server. The YOLO backend can be swapped for TRT without changing domain code. Scientific rules stay isolated from deployment concerns.

---

### asyncio.Semaphore(1) — GPU Guard

**Why a semaphore instead of a task queue:**
The RTX 4060 has 8 GB VRAM shared between inference, training, and VLM tasks. Two simultaneous GPU tasks will OOM-kill each other. A semaphore with depth 1 ensures exactly one GPU task runs at any moment. Requests that arrive during GPU use receive HTTP 429 immediately (fail-fast) rather than queuing indefinitely. This is intentional — the client (frontend) handles retry.

---

### AGPL-3.0 License

**Why AGPL over MIT or GPL:**
MIT and GPL have a "SaaS loophole" — a company can run the software as a cloud service without releasing modifications. AGPL closes this: anyone who deploys CTIP as a network service (API, web app) must release their modifications under the same license. This ensures the scientific community benefits from all improvements, even those made in commercial deployments.

---

---

## 🇩🇪 Warum diese Technologien?

Jede Technologieentscheidung in CTIP wurde bewusst getroffen. Dieser Abschnitt erklärt was wir verwenden und warum wir es gegenüber Alternativen bevorzugt haben.

---

### Python 3.12

**Warum:** Python 3.12 bringt signifikante Performanceverbesserungen (5–15% schneller als 3.10 bei CPU-lastigen wissenschaftlichen Workloads) und bessere Fehlermeldungen. Das gesamte wissenschaftliche ML-Ökosystem (PyTorch, NumPy, OpenCV) unterstützt es vollständig. Als Paketmanager verwenden wir `uv` — 10–100x schneller als pip bei der Abhängigkeitsauflösung.

---

### PyTorch 2.x + CUDA 12.x

**Warum PyTorch und nicht TensorFlow:** PyTorch dominiert seit 2020 Forschung und Produktion im Bereich CV. Das gesamte Ökosystem — Ultralytics, SAM2, Hugging Face Transformers — ist PyTorch-nativ. Dynamische Berechnungsgraphen erleichtern das Debugging erheblich.

**Warum CUDA 12.x:** Notwendig für TensorRT 10.x und die neuesten PyTorch CUDA-Kernel. `expandable_segments:True` in `PYTORCH_CUDA_ALLOC_CONF` reduziert Speicherfragmentierung auf 8-GB-Karten — entscheidend für unser Hardwareziel.

---

### YOLO v11s — Erkennungs-Backbone

**Warum YOLO und nicht Faster-RCNN / DETR:**
- YOLO v11s erreicht vergleichbares mAP wie schwerere Zweistufendetektoren bei 3–5x höherer Inferenzgeschwindigkeit
- Single-Stage-Architektur passt in 8 GB VRAM neben SAM2 und Kalibrierung
- Ultralytics bietet produktionsreife Trainingspipelines, Augmentierung und ONNX/TensorRT-Export
- Variante `s` (small): bestes Genauigkeits-/VRAM-Verhältnis auf RTX 4060

**Warum Tiled Inference (1280px Kacheln, 20% Überlappung):**
Trichome in 4K-Mikroskopiebildern sind winzig relativ zur Bildgröße. Tiled Inference verarbeitet jeden 1280px-Ausschnitt unabhängig und führt Ergebnisse mit WBF (Weighted Boxes Fusion) zusammen — das stellt Erkennung kleiner Trichome am Bildrand sicher.

---

### SAM2-tiny — Instanz-Segmentierung

**Warum SAM2 und nicht Mask-RCNN:**
SAM2 ist Prompt-basiert — es nimmt die YOLO-Bounding-Boxes als räumliche Prompts und erzeugt hochwertige Instanzmasken ohne Maskenannotierungen im Trainingsdatensatz. Dies entkoppelt Erkennungslabels (günstig zu sammeln) von Segmentierungsmasken (teuer zu annotieren). Die `tiny`-Variante benötigt nur ~1,8 GB VRAM.

---

### TensorRT 10.x — Produktions-Inferenz

**Warum TensorRT und nicht plain ONNX Runtime:**
TensorRT kompiliert das Modell in eine für die spezifische GPU optimierte Engine. Auf RTX 4060 sind FP16-TRT-Engines typischerweise 2–4x schneller als ONNX Runtime FP32.

**Warum FP16:**
FP16 halbiert den VRAM-Verbrauch bei vernachlässigbarem Genauigkeitsverlust (< 0,5% mAP-Rückgang). Auf RTX 4060 führen Tensor Cores FP16-Matmuls mit doppeltem Durchsatz gegenüber FP32 aus.

---

### FastAPI — Backend-Framework

**Warum FastAPI und nicht Django / Flask:**
- **Native async:** Gebaut auf Starlette + asyncio. GPU-Tasks, WebSocket-Streams und REST-Anfragen teilen sich einen Event-Loop.
- **Automatische OpenAPI-Docs:** Alle 128+ Endpunkte erhalten Swagger UI und ReDoc kostenlos.
- **Pydantic-Validierung:** Request/Response-Typen werden aus denselben Dataclass-Definitionen validiert, serialisiert und dokumentiert.

**Warum WebSockets für Training/System-Monitoring:**
Training läuft Minuten bis Stunden. REST-Polling jede Sekunde ist verschwenderisch und träge. WebSocket-Streams pushen Metrik-Updates mit Sub-Sekunden-Latenz direkt ins Dashboard.

---

### Next.js 14 (App Router) — Frontend

**Warum Next.js und nicht plain React / Vue:**
- App Router ermöglicht React Server Components — reduziert JS-Bundle-Größe
- Dateibasiertes Routing passt zu den 16 verschiedenen Tool-Seiten
- TypeScript-nativ: Wissenschaftliches Tooling braucht zuverlässige Typverträge

**Warum kein Admin-Template (Grafana, Retool):**
Das ist eine wissenschaftliche Instrumenten-Oberfläche. Sie braucht benutzerdefinierte Annotierungs-Review-Workflows, überlagerte Bounding-Box-Visualisierung, Trainingsverlust-Kurven mit wissenschaftlichem Kontext — das können generische Dashboard-Tools nicht ausdrücken.

---

### Label Studio + CVAT — Annotierung

**Warum zwei Annotierungstools:**
- **Label Studio:** Am besten für Bounding-Box-Annotation. Einfache UI, schnelle Tastenkürzel, Python SDK.
- **CVAT:** Besser für Polygon/Instanzmaske-Annotation und Multi-Annotator-Workflows mit Aufgabenzuweisung.

**Warum nicht Roboflow / Cloud-Annotierung:**
Datensouveränität. Mikroskopiebilder können proprietäre Zucht- oder Anbauinformationen enthalten. Alle Annotierungen bleiben On-Premise.

---

### Florence-2 / Moondream-2B / Qwen2-VL — VLM-Vorlabeling

**Warum VLMs für Vorlabeling:**
Manuelle Annotation von Mikroskopiebildern ist langsam und erfordert Expertise. VLMs können 60–80% genaue Bounding-Box-Kandidaten generieren, die ein Mensch dann in 20–30% der Zeit manueller Annotation korrigiert.

**Warum drei VLMs:**
Unterschiedliche Kompromisse bei Qualität, Geschwindigkeit und VRAM (siehe englische Tabelle oben).

**Warum HITL (Mensch-in-Loop) verpflichtend ist:**
VLMs halluzinieren. Sie produzieren selbstbewusst plausible aber falsche Bounding Boxes. VLM-Ausgaben direkt in Trainingsdaten zu schreiben würde systematische Fehler einführen, die sich mit jedem Retraining-Zyklus verstärken. Alle VLM-Ausgaben gehen in eine Review-Queue — ein Mensch bestätigt jede Annotierung bevor sie Trainingsdaten wird. Das ist eine harte Architekturinvariante.

---

### Aktives Lernen — Unsicherheits-Sampling

**Warum aktives Lernen statt zufälliger Auswahl:**
Nicht alle Bilder trainieren das Modell gleich stark. Bilder nahe der Entscheidungsgrenze (hohe Entropie-Vorhersagen) verbessern Kalibrierung und Generalisierung am meisten. Aktives Lernen priorisiert Annotierungsaufwand dort wo er am wichtigsten ist — typischerweise 30–50% weniger Annotierungen nötig.

---

### Konfidenz-Kalibrierung (Temperature / Platt Scaling)

**Warum Kalibrierung wichtig ist:**
Ein YOLO-Modell mit Konfidenz 0,85 bedeutet nicht, dass 85% solcher Vorhersagen korrekt sind. Roh-YOLO-Scores sind oft überkonfident. Ohne Kalibrierung sind Schwellenwert-Entscheidungen, Unsicherheitsschätzungen und Konfidenzintervalle in Berichten irreführend.

**Warum Temperature Scaling:**
Einzelner Post-hoc-Parameter — verändert das Modell nicht, skaliert nur die Ausgabe-Logits neu. Bewahrt das Ranking (mAP unverändert) bei verbesserter Konfidenzgenauigkeit. Ziel: ECE < 0,05.

---

### MLflow — Experiment-Tracking

**Warum MLflow und nicht Weights & Biases:**
- **Self-hosted:** Keine Daten verlassen die Maschine
- **Kein Account erforderlich:** W&B erfordert Cloud-Anmeldung
- **ONNX/PyTorch Model Registry integriert**
- W&B wird via `EXPERIMENT_TRACKER=wandb` ebenfalls unterstützt

---

### Docker + Nginx — Deployment

**Warum Docker Compose:**
Erfasst den gesamten Abhängigkeitsgraph in versioniertem YAML. Jede Maschine mit Docker kann den vollständigen Stack mit einem Befehl reproduzieren.

**Warum Nginx als Reverse Proxy:**
Einzelner Einstiegspunkt auf Port 3001. Nginx routet `/api/v1/` zum Backend, `/mlflow/` zu MLflow, `/annotation/` zu Label Studio — alles hinter einem Port mit möglicher TLS-Terminierung.

---

### Domain-Driven Design (DDD)

**Warum DDD statt flacher Modulstruktur:**
Jede wissenschaftliche Domäne hat eigene Regeln und Invarianten. DDD trennt diese sauber:
- `domain/` — reine wissenschaftliche Logik, kein I/O, vollständig isoliert testbar
- `infrastructure/` — austauschbare Backends (YOLO → RTMDet, SAM2 → MobileSAM)

Der Reifegrad-Klassifikator kann ohne gestarteten Server getestet werden. Das YOLO-Backend kann gegen TRT ausgetauscht werden ohne Domain-Code zu ändern.

---

### asyncio.Semaphore(1) — GPU-Guard

**Warum Semaphore statt Task-Queue:**
Die RTX 4060 hat 8 GB VRAM geteilt zwischen Inferenz, Training und VLM-Tasks. Zwei gleichzeitige GPU-Tasks würden sich gegenseitig mit OOM-Kill beenden. Semaphore mit Tiefe 1 stellt sicher, dass genau ein GPU-Task gleichzeitig läuft. Anfragen während GPU-Nutzung erhalten sofort HTTP 429 (Fail-Fast).

---

### AGPL-3.0-Lizenz

**Warum AGPL und nicht MIT oder GPL:**
MIT und GPL haben eine "SaaS-Lücke" — ein Unternehmen kann die Software als Cloud-Service betreiben ohne Änderungen zu veröffentlichen. AGPL schließt das: Jeder der CTIP als Netzwerkdienst betreibt muss seine Änderungen unter derselben Lizenz veröffentlichen.

---

---

## 🇪🇸 ¿Por qué estas tecnologías?

Cada decisión tecnológica en CTIP fue tomada deliberadamente. Esta sección explica qué usamos y por qué lo elegimos sobre las alternativas.

---

### Python 3.12

**Por qué:** Python 3.12 trae mejoras de rendimiento significativas (5–15% más rápido que 3.10 en cargas de trabajo científicas) y mejores mensajes de error. Todo el ecosistema ML científico (PyTorch, NumPy, OpenCV) lo soporta completamente. Usamos `uv` como gestor de paquetes — 10–100x más rápido que pip en resolución de dependencias.

---

### PyTorch 2.x + CUDA 12.x

**Por qué PyTorch y no TensorFlow:** PyTorch domina la investigación y producción en CV desde 2020. Todo el ecosistema — Ultralytics, SAM2, Hugging Face Transformers — es nativo de PyTorch. Los grafos de computación dinámicos facilitan la depuración.

**Por qué CUDA 12.x:** Necesario para TensorRT 10.x y los últimos kernels CUDA de PyTorch. `expandable_segments:True` reduce la fragmentación de memoria en tarjetas de 8 GB VRAM — crítico para nuestro hardware objetivo.

---

### YOLO v11s — Backbone de Detección

**Por qué YOLO y no Faster-RCNN / DETR:**
- YOLO v11s alcanza mAP comparable a detectores de dos etapas más pesados a 3–5x mayor velocidad de inferencia
- Arquitectura de una sola etapa cabe en 8 GB VRAM junto con SAM2 y calibración
- Ultralytics proporciona pipelines de entrenamiento de calidad productiva, augmentación y exportación ONNX/TensorRT

**Por qué inferencia tiled (tiles 1280px, 20% solapamiento):**
Los tricomas en imágenes de microscopía 4K son diminutos en relación al tamaño de la imagen. La inferencia tiled procesa cada recorte de 1280px independientemente y fusiona resultados con WBF — recupera la detección de tricomas pequeños y en bordes.

---

### SAM2-tiny — Segmentación de Instancias

**Por qué SAM2 y no Mask-RCNN:**
SAM2 es basado en prompts — toma las bounding boxes de YOLO como prompts espaciales y genera máscaras de instancia de alta calidad sin necesitar anotaciones de máscara en el dataset de entrenamiento. Esto desacopla las etiquetas de detección (baratas de recolectar) de las máscaras de segmentación (caras de anotar). La variante `tiny` usa solo ~1.8 GB VRAM.

---

### TensorRT 10.x — Inferencia en Producción

**Por qué TensorRT y no ONNX Runtime:**
TensorRT compila el modelo en un engine optimizado para la GPU específica. En RTX 4060, los engines TRT FP16 son típicamente 2–4x más rápidos que ONNX Runtime FP32.

**Por qué FP16:**
FP16 reduce a la mitad el uso de VRAM con pérdida de precisión despreciable (< 0.5% de caída en mAP). En RTX 4060, los Tensor Cores ejecutan matmuls FP16 al doble de rendimiento que FP32.

---

### FastAPI — Framework Backend

**Por qué FastAPI y no Django / Flask:**
- **Async nativo:** Construido sobre Starlette + asyncio. Tareas GPU, streams WebSocket y peticiones REST comparten un event loop.
- **Docs OpenAPI automáticas:** Todos los 128+ endpoints obtienen Swagger UI y ReDoc gratis.
- **Validación Pydantic:** Los tipos request/response se validan, serializan y documentan desde las mismas definiciones de dataclass.

**Por qué WebSockets para monitorización:**
El entrenamiento dura minutos a horas. El polling REST cada segundo es ineficiente. Los streams WebSocket empujan actualizaciones de métricas con latencia sub-segundo directamente al dashboard.

---

### Next.js 14 (App Router) — Frontend

**Por qué Next.js y no React puro / Vue:**
- App Router permite React Server Components — reduce el tamaño del bundle JS
- Enrutamiento basado en archivos se mapea limpiamente a las 16 páginas de herramientas
- TypeScript nativo: las herramientas científicas necesitan contratos de tipo fiables

**Por qué no una plantilla admin (Grafana, Retool):**
Esta es una interfaz de instrumento científico. Necesita flujos de trabajo de revisión de anotaciones personalizados, visualización de bounding boxes superpuestas, curvas de pérdida de entrenamiento con contexto científico — herramientas de dashboard genéricas no pueden expresar esto.

---

### Label Studio + CVAT — Anotación

**Por qué dos herramientas de anotación:**
- **Label Studio:** Mejor para anotación de bounding boxes. UI simple, atajos de teclado rápidos, SDK Python.
- **CVAT:** Mejor para anotación de polígonos/máscaras y flujos multi-anotador con asignación de tareas.

**Por qué no Roboflow / anotación en la nube:**
Soberanía de datos. Las imágenes de microscopía pueden contener información propietaria. Toda la anotación permanece on-premise.

---

### Florence-2 / Moondream-2B / Qwen2-VL — Pre-etiquetado VLM

**Por qué VLMs para pre-etiquetado:**
La anotación manual de imágenes de microscopía es lenta y requiere experiencia. Los VLMs pueden generar candidatos de bounding box con 60–80% de precisión que un humano corrige en 20–30% del tiempo de anotación manual.

**Por qué tres VLMs:**
Diferentes compromisos en calidad, velocidad y VRAM (ver tabla en sección inglesa).

**Por qué HITL (Humano en el Loop) es obligatorio:**
Los VLMs alucinan. Enviar salidas VLM directamente a datos de entrenamiento introduciría errores sistemáticos que se agravan con cada ciclo de reentrenamiento. Todas las salidas VLM van a una cola de revisión — un humano aprueba cada anotación antes de que se convierta en dato de entrenamiento. Esta es una invariante arquitectónica estricta.

---

### Aprendizaje Activo — Muestreo por Incertidumbre

**Por qué aprendizaje activo en lugar de muestreo aleatorio:**
No todas las imágenes entrenan al modelo igualmente. Las imágenes cerca del límite de decisión (predicciones de alta entropía) mejoran más la calibración y generalización. El aprendizaje activo prioriza el esfuerzo de etiquetado donde más importa — típicamente 30–50% menos anotaciones necesarias.

---

### Calibración de Confianza (Temperature / Platt Scaling)

**Por qué la calibración importa:**
Un modelo YOLO con confianza 0.85 no significa que el 85% de esas predicciones sean correctas. Las puntuaciones brutas de YOLO suelen ser demasiado confiadas. Sin calibración, las decisiones de umbral y los intervalos de confianza en informes son engañosos.

**Por qué temperature scaling:**
Un único parámetro post-hoc — no cambia el modelo, solo reescala los logits de salida. Preserva el ranking (mAP sin cambios). Objetivo: ECE < 0.05.

---

### MLflow — Seguimiento de Experimentos

**Por qué MLflow y no Weights & Biases:**
- **Auto-hospedado:** Ningún dato sale de la máquina
- **Sin cuenta requerida:** W&B requiere registro en la nube
- **Registro de modelos ONNX/PyTorch integrado**
- W&B también soportado via `EXPERIMENT_TRACKER=wandb`

---

### Docker + Nginx — Despliegue

**Por qué Docker Compose:**
Captura todo el grafo de dependencias en YAML versionado. Cualquier máquina con Docker puede reproducir el stack completo con un comando.

**Por qué Nginx como proxy inverso:**
Punto de entrada único en el puerto 3001. Nginx enruta `/api/v1/` al backend, `/mlflow/` a MLflow, `/annotation/` a Label Studio — todo detrás de un puerto con posible terminación TLS.

---

### Domain-Driven Design (DDD)

**Por qué DDD y no estructura plana de módulos:**
Cada dominio científico tiene sus propias reglas e invariantes. DDD las separa limpiamente:
- `domain/` — lógica científica pura, sin I/O, completamente testable en aislamiento
- `infrastructure/` — backends intercambiables (YOLO → RTMDet, SAM2 → MobileSAM)

El clasificador de madurez puede testearse sin iniciar un servidor. El backend YOLO puede cambiarse por TRT sin modificar código de dominio.

---

### asyncio.Semaphore(1) — GPU Guard

**Por qué un semáforo y no una cola de tareas:**
La RTX 4060 tiene 8 GB VRAM compartidos entre inferencia, entrenamiento y tareas VLM. Dos tareas GPU simultáneas se matarían mutuamente con OOM-kill. El semáforo con profundidad 1 garantiza exactamente una tarea GPU a la vez. Las peticiones durante el uso de GPU reciben HTTP 429 inmediatamente (fail-fast).

---

### Licencia AGPL-3.0

**Por qué AGPL y no MIT o GPL:**
MIT y GPL tienen un "agujero SaaS" — una empresa puede ejecutar el software como servicio en la nube sin publicar modificaciones. AGPL lo cierra: cualquiera que despliegue CTIP como servicio de red debe publicar sus modificaciones bajo la misma licencia.

---

## License

AGPL-3.0 — see [LICENSE](LICENSE)

---

*This platform is for scientific optical analysis only. No claims are made about cannabinoid concentrations, potency, or chemical composition. All maturity assessments are based purely on optical observation under digital microscopy.*
