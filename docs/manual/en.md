# CTIP Manual (English)

> [English](en.md) · [Deutsch](de.md) · [Español](es.md) · [Technology stack](tech-stack.md) · [Back to README](../../README.md)

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

### Supported GPU backends

| Backend | Hardware | `.env` setting | Notes |
|---------|----------|----------------|-------|
| **NVIDIA CUDA** | GTX 1080+ / RTX series | `CUDA_DEVICE="cuda:0"` | Recommended. Full feature set including TensorRT. |
| **AMD ROCm** | RX 6000 / RX 7000 (RDNA2/3) | `CUDA_DEVICE="cuda:0"` | ROCm 6.x required. TensorRT unavailable; use ONNX+MIGraphX. ~70% CUDA perf. |
| **Apple MPS** | Apple Silicon M1/M2/M3/M4 | `CUDA_DEVICE="mps"` | Metal Performance Shaders. No fp16 for all ops. Sequential inference only. |
| **CPU** | Any x86-64 / ARM64 | `CUDA_DEVICE="cpu"` | ~20× slower. Development, annotation, and dataset management only. |

### Resource requirements

| Component | Minimum | Recommended |
|---|---|---|
| GPU | NVIDIA GTX 1080 (8 GB VRAM) | RTX 4060 / 3080 (8+ GB) |
| CPU | 6-core modern | i5-13400F or better |
| RAM | 16 GB | 32 GB |
| Storage | 50 GB SSD | 500 GB NVMe |
| CUDA | 11.8+ (NVIDIA) / ROCm 6.x (AMD) | 12.6 / ROCm 6.1 |

### VRAM Budget (8 GB card — RTX 4060 / RX 7800 XT)

| Component | VRAM |
|---|---|
| YOLO v11s inference | ~0.9 GB |
| SAM2-tiny | ~1.8 GB |
| Florence-2 (4-bit) | ~2.1 GB |
| Moondream-2B (4-bit) | ~1.4 GB |
| Qwen2-VL-7B (4-bit) | ~4.8 GB |
| YOLO v11s training (bs=8) | ~5.5 GB |

> Only **one GPU task runs at a time** — enforced by `asyncio.Semaphore(1)`. Intentional for 8 GB VRAM cards.  
> Apple Silicon MPS shares system RAM — VRAM budget applies to unified memory.

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
git clone https://github.com/ottco-dev/ctip-oss.git
cd ctip-oss

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
# Expected: 1694 passed, 4 skipped (GPU-only + reportlab guard)

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
trichome video score image.jpg      # focus, exposure and noise score (also shown per image in the UI)
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
# Export a Label Studio project as a YOLO dataset. Whole imaging sessions go to one split each
# (task field data.session, else the image folder, else the file-name series), so near-identical
# frames never leak between train / val / test. The export log reports images and sessions per split.
curl -X POST http://localhost:8000/api/v1/training/prepare-ls-dataset \
  -H "Content-Type: application/json" \
  -d '{"project_id": 1, "train_ratio": 0.70, "val_ratio": 0.15, "seed": 42}'
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
trichome train start --config configs/training/yolo11s_detection.yaml

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
trichome convert tensorrt runs/.../best.pt \
  --imgsz 1280 --fp16 --workspace-gb 4

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
trichome status                                   # GPU, models, API
trichome serve                                    # start the FastAPI backend

trichome detect image.jpg                         # YOLO detection (tiled for large images)
trichome segment image.jpg                        # SAM2 instance segmentation
trichome maturity image.jpg                       # optical maturity (colour + texture)
trichome calibrate run                            # pixel -> µm from a stage micrometer
trichome video extract clip.mp4                   # best frames from a microscopy video

trichome train start --data dataset.yaml --model yolo11s --imgsz 1280 --batch 8
trichome train evaluate                           # metrics on a labelled split
trichome train list                               # recent runs

trichome benchmark all                            # focus, maturity, morphology, measurement, video, detection
trichome convert tensorrt best.pt --fp16          # .pt -> ONNX -> TensorRT engine
trichome export run <session> -f pdf,csv,json     # scientific reports
trichome annotate --help                          # VLM pre-labelling (always human-reviewed)
```

Every command has `--help` with all options.

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

**Current status: 1694 passed, 4 skipped (GPU-only + reportlab guard)**

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
