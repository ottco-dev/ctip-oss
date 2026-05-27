The Setup Wizard is a full-screen, OS-style first-run installer that configures every aspect of CTIP.
It launches automatically on first visit and writes all settings to `.env`.

## How it works

1. `SetupGuard` checks `GET /api/v1/setup/status` on every page load
2. If `completed: false` → redirect to `/setup`
3. After saving: sets `SETUP_COMPLETED="true"` in `.env`, sets `ctip-setup-checked` in `sessionStorage`
4. Re-open anytime from the sidebar: **First-Time Setup**

---

## Step 0 — Welcome

Overview of what the installer does. No user input required.

---

## Step 1 — System Check

Runs `GET /api/v1/setup/system-check`. Checks:

**Environment:**
| Check | What it verifies |
|-------|-----------------|
| Python | Version ≥ 3.11 |
| CUDA / nvidia-smi | GPU driver availability |
| Node.js | Version ≥ 18 |
| npm | Package manager |
| nginx | Web server |
| Git | Version control |

**Python packages (14 checked):**
`torch`, `ultralytics`, `supervision`, `sam2`, `fastapi`, `uvicorn`, `httpx`,
`pydantic`, `sqlmodel`, `mlflow`, `cv2`, `PIL`, `numpy`, `sklearn`

Red = missing required, yellow = missing optional. You can continue with warnings.

Fix missing Python packages:
```bash
source .venv/bin/activate
uv pip install -e ".[all]"
```

---

## Step 2 — Network

Configure how CTIP is exposed:

| Field | Default | Description |
|-------|---------|-------------|
| Enable public access | off | Toggle to enter a domain |
| Public Domain | (empty = localhost only) | Your DDNS hostname, e.g. `mylab.ddns.net` |
| Public Port | 3001 | nginx listens on this port |

**Localhost-only** (default): nginx only binds to `127.0.0.1:3001`. Not reachable from outside.

**Public**: Enter your domain → nginx listens on `0.0.0.0:3001` and sets `server_name`.
Update your router to port-forward to this machine.

---

## Step 3 — Hardware

| Field | Description |
|-------|-------------|
| CUDA Device | `cuda:0` / `mps` / `cpu` |
| CUDA_VISIBLE_DEVICES | GPU index (e.g. `0`) |
| Total VRAM (GB) | Your GPU's VRAM (RTX 4060 = 8.0) |
| Inference Reserve (GB) | Budget kept for inference tasks (default 2.0) |
| Environment | `development` or `production` |

CTIP enforces **one GPU task at a time** via an asyncio semaphore. The VRAM budget prevents OOM errors by refusing tasks that would exceed the limit.

---

## Step 4 — Storage

| Field | Description |
|-------|-------------|
| Data Root | Parent directory for all CTIP data |
| Models Directory | Trained weight files (`.pt`, `.engine`) |
| Outputs Directory | Detection results, PDFs, CSV exports |

All paths must be **absolute**. Directories are created automatically on startup.

Example for a standard Linux install:
```bash
DATA_ROOT="/home/youruser/ctip-oss/data"
MODELS_DIR="/home/youruser/ctip-oss/data/models"
OUTPUTS_DIR="/home/youruser/ctip-oss/data/outputs"
```

---

## Step 5 — Docker

Checks Docker availability and manages the annotation container stack.

**Status panel:**
- Docker Engine: reachable / not reachable
- docker group: yes / needs fix

**If not in docker group:**
```bash
sudo usermod -aG docker $USER && newgrp docker
```
The exact command is shown in the UI with a copy button.

**Container list:** shows all running/stopped annotation containers (Label Studio, CVAT, PostgreSQL).

**Start Annotation Stack** button runs:
```bash
docker compose --profile annotation up -d
```

---

## Step 6 — ML Models

Downloads YOLO11 and SAM2 model weights. Required models must be present before inference.

| Model | File | Size | Required | Purpose |
|-------|------|------|----------|---------|
| YOLO11n | yolo11n.pt | 5.4 MB | No | Fastest, lowest accuracy |
| YOLO11s | yolo11s.pt | 18.4 MB | **Yes** | Default — best balance for RTX 4060 |
| YOLO11m | yolo11m.pt | 43.0 MB | No | Higher accuracy, needs 12+ GB VRAM |
| SAM2-tiny | sam2_hiera_tiny.pt | 38.9 MB | **Yes** | Instance segmentation |
| SAM2-small | sam2_hiera_small.pt | 46.1 MB | No | Better masks, +20% VRAM |

**Download flow:**
1. Click **Download** → `POST /api/v1/setup/models/download` starts a background task
2. Progress bar polls `GET /api/v1/setup/models/download/{task_id}` every 500ms
3. On completion: catalog refreshes, model marked as present

**Download All Required** downloads yolo11s + sam2-tiny in parallel.

---

## Step 7 — Label Studio

Full Label Studio setup in 3 sub-sections:

### Account Creation
Enter email + password → `POST /api/v1/setup/label-studio/create-account`
- Creates the first admin account on a fresh LS instance
- If account already exists: retrieves the API token
- API key is auto-filled below

### Connection Test
Enter API key → `POST /api/v1/setup/label-studio/test`
- Checks reachability
- Authenticates via `GET /api/current-user/whoami`
- Shows username and existing project count

### Annotation Project
Enter project name → **Create Annotation Project** → `POST /api/v1/setup/label-studio/create-project`

Creates a project with this label config (pre-built):
```xml
<RectangleLabels name="label" toName="image">
  <Label value="stalked"        background="#22d3ee" hotkey="1"/>
  <Label value="sessile"        background="#34d399" hotkey="2"/>
  <Label value="bulbous"        background="#a78bfa" hotkey="3"/>
  <Label value="non-glandular"  background="#fb923c" hotkey="4"/>
</RectangleLabels>
<Choices name="quality">good / blurry / poor</Choices>
<TextArea name="notes" .../>
```

---

## Step 8 — Services (Experiment Tracking)

### MLflow
| Field | Description |
|-------|-------------|
| Tracking URI | `http://localhost:3004` — local MLflow server |
| Default Experiment | Name for training runs (e.g. `trichome-detection`) |

### Weights & Biases (optional)
Toggle W&B logging, enter API key + project name.

---

## Step 9 — Security

| Field | Description |
|-------|-------------|
| Secret Key | 64-char random key for session signing — generate with the button |
| API Token | Optional bearer token for API authentication (empty = no auth) |

If CTIP is internet-facing, always set both. The **Generate random** button creates a cryptographically random 64-character key.

---

## Step 10 — Review

Full summary of all configured values. Sensitive fields (keys, tokens) show as `••••••••`.
Click **Save & Finish** → writes to `.env` → advances to Verification.

---

## Step 11 — Verification

Live health check of all services. Terminal-style log:

```
✓ PASS  Backend API          HTTP 200  12ms
✓ PASS  Frontend             HTTP 200  8ms
✓ PASS  MLflow               HTTP 200  34ms
✗ FAIL  Label Studio         Connection refused
✓ PASS  nginx health check   HTTP 200  1ms
```

FAIL on optional services (Label Studio, CVAT) is expected if Docker isn't running.
Click **Go to Dashboard** when ready.
