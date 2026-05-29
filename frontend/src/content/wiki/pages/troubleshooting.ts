import type { WikiPage } from '../types';

const en = `
## Quick diagnosis checklist

Before diving into specific issues, run these first:

\`\`\`bash
# 1. Backend alive?
curl http://localhost:8000/api/v1/system/health

# 2. GPU visible?
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# 3. All services running?
docker ps --format "table {{.Names}}\\t{{.Status}}\\t{{.Ports}}"

# 4. Frontend compiling?
cd frontend && npm run type-check
\`\`\`

---

## Setup wizard issues

### "catalog is undefined" / TypeError on models step

**Cause**: Backend \`/setup/models/status\` returns a list directly, not \`{ models: [...] }\`.

**Fix**: Already patched in current version. If you see it:
\`\`\`bash
git pull origin main
cd frontend && npm run build
\`\`\`

If persists, check the API response manually:
\`\`\`bash
curl http://localhost:8000/api/v1/setup/models/status | python3 -m json.tool
# Should be a JSON array, not an object
\`\`\`

---

### Setup wizard keeps redirecting back to setup

The wizard marks itself complete by writing \`SETUP_COMPLETE=true\` to \`.env\`. If the flag is missing:

\`\`\`bash
grep SETUP_COMPLETE .env

# If missing, add it:
echo "SETUP_COMPLETE=true" >> .env

# Then restart backend:
pkill -f uvicorn && uvicorn backend.main:app --reload --port 8000 &
\`\`\`

---

### Can't create Label Studio account (step 8)

1. Confirm Label Studio is running:
   \`\`\`bash
   curl http://localhost:3005/api/health
   \`\`\`

2. Label Studio takes 10–30s to initialize after Docker start. Retry after waiting.

3. Check Docker logs:
   \`\`\`bash
   docker logs ctip-label-studio --tail 50
   \`\`\`

4. Port conflict: if something else is on 3005:
   \`\`\`bash
   lsof -i :3005
   \`\`\`

---

## Backend issues

### Backend won't start (ImportError / ModuleNotFoundError)

\`\`\`bash
# Ensure venv is active
source .venv/bin/activate
which python  # must point inside .venv/

# Reinstall
uv pip install -e ".[dev]"

# Try direct import
python -c "from backend.main import app; print('OK')"
\`\`\`

Common causes:
- Forgot to activate venv
- \`uv\` not installed (install: \`pip install uv\`)
- Missing system deps (libglib, libsm) — install via \`apt\`/\`dnf\`

---

### OOM (Out of Memory) / CUDA out of memory

**RTX 4060 (8 GB VRAM) hard limits:**

| Operation | Max batch | Typical VRAM |
|-----------|-----------|-------------|
| YOLO inference | 4 tiles | ~5.0 GB |
| YOLO training | batch=4 | ~7.1 GB |
| SAM2-tiny | N/A | ~3.2 GB |
| VLM (Moondream 4-bit) | 1 image | ~2.1 GB |

Fix OOM at inference:
\`\`\`bash
# Reduce tile batch size in .env:
YOLO_TILE_BATCH=1   # safest: 1 tile at a time (~1.8 GB)
\`\`\`

Fix OOM at training:
\`\`\`yaml
# In training config:
batch: 2    # down from 4
workers: 2  # fewer DataLoader workers
\`\`\`

Clear VRAM:
\`\`\`bash
# Kill all GPU processes
sudo fuser -v /dev/nvidia* 2>/dev/null
# or
pkill -f "python.*yolo"
\`\`\`

---

### GPU not detected by PyTorch

\`\`\`bash
nvidia-smi                          # NVIDIA driver check
python -c "import torch; print(torch.version.cuda)"   # CUDA version

# If cuda is None:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
\`\`\`

On WSL2: ensure \`nvidia-smi\` works inside WSL:
\`\`\`bash
# In WSL:
nvidia-smi
# If "no NVIDIA SMI", update WSL:
wsl --update
\`\`\`

---

### Port 8000 already in use

\`\`\`bash
lsof -i :8000
kill -9 <PID>

# Or run on alternate port:
uvicorn backend.main:app --reload --port 8001
# Update frontend NEXT_PUBLIC_API_URL in .env.local
\`\`\`

---

## Docker issues

### Docker compose fails: "Can't find docker-compose.yml"

Always run Docker commands from the \`docker/\` subdirectory, or use absolute paths:
\`\`\`bash
cd docker && docker compose up -d
# OR
docker compose -f /path/to/trichome-analysis/docker/docker-compose.yml up -d
\`\`\`

---

### "permission denied" when running Docker

\`\`\`bash
# Add yourself to the docker group
sudo usermod -aG docker $USER
newgrp docker   # apply without logout

# Verify
docker info | head -5
\`\`\`

---

### Label Studio or CVAT container crashes immediately

\`\`\`bash
docker logs ctip-label-studio --tail 100
docker logs ctip-cvat-server --tail 100
\`\`\`

Common causes:
- **PostgreSQL not ready yet**: wait 10s and retry \`docker compose up -d\`
- **Volume permissions**: \`sudo chown -R 1000:1000 data/label_studio/\`
- **Port conflict**: check \`lsof -i :3005\` and \`lsof -i :3006\`

---

## Frontend issues

### Wiki not loading / blank page

\`\`\`bash
cd frontend
npm run dev   # check terminal for errors

# TypeScript errors?
npm run type-check 2>&1 | head -40

# Rebuild node_modules
rm -rf node_modules .next && npm install && npm run dev
\`\`\`

---

### API requests fail with CORS or 404

The frontend proxies \`/api/v1/*\` to the backend via Next.js \`next.config.js\`:

\`\`\`bash
# Check backend is running:
curl http://localhost:8000/api/v1/system/health

# Check proxy config in frontend/next.config.js
# It should proxy to http://localhost:8000
\`\`\`

---

### WebSocket not connecting (live GPU stats broken)

\`\`\`bash
# Test WebSocket manually:
wscat -c ws://localhost:8000/ws/system

# If wscat not installed:
npm install -g wscat

# If connection refused: check backend logs for WebSocket errors
tail -f logs/backend.log | grep -i websocket
\`\`\`

---

## Model download issues

### Download task stuck at 0% / "downloading" forever

\`\`\`bash
# Check download task status:
curl http://localhost:8000/api/v1/setup/models/download/<task_id>

# Check backend logs for errors:
tail -f logs/backend.log | grep -i download

# Manually download model:
mkdir -p data/models
cd data/models
wget -c "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt"
\`\`\`

Common causes:
- GitHub rate limit: wait 10 min or use a VPN
- DNS issue: try \`ping github.com\`
- Disk full: \`df -h data/models/\`

---

### Model file exists but reports "missing"

The backend checks file presence in \`MODELS_DIR\` (default: \`data/models/\`):
\`\`\`bash
ls -lh data/models/
# Expected: yolo11s.pt (~18.4 MB), sam2_hiera_tiny.pt (~38.9 MB)

# If path differs, update .env:
echo "MODELS_DIR=/path/to/models" >> .env
\`\`\`

---

## Training issues

### Training job "queued" but never starts

\`\`\`bash
# Check if GPU is already busy:
curl http://localhost:8000/api/v1/system/gpu

# Check background task queue:
curl http://localhost:8000/api/v1/training/jobs

# Check logs:
tail -f logs/training.log
\`\`\`

If another inference or training job is holding the GPU semaphore, wait for it to finish or restart the backend.

---

### mAP@0.5 below 0.70 after training

| Issue | Likely cause | Fix |
|-------|-------------|-----|
| mAP < 0.50 | Too few annotations | Need at least 200 annotated trichomes |
| mAP 0.50–0.70 | Class imbalance | Use \`cls_weights\` or oversample |
| mAP stuck after epoch 30 | LR too high | Lower \`lr0\` to 0.005 |
| mAP drops after epoch 60 | Overfitting | Add \`dropout: 0.1\`, reduce epochs |
| High FP rate | conf threshold too low | Sweep conf: 0.1 to 0.5 |

\`\`\`bash
# Full benchmark with precision-recall analysis:
trichome benchmark detection \\
  --model data/models/yolo11s_custom.pt \\
  --split val \\
  --conf-sweep 0.1,0.2,0.3,0.4,0.5 \\
  --output reports/benchmark_v1.json
\`\`\`

---

### Training dies midway (NaN loss)

\`\`\`yaml
# Reduce LR if NaN appears early:
lr0: 0.001
warmup_epochs: 5    # longer warmup
clip_gradients: 10  # gradient clipping
\`\`\`

NaN loss most often caused by:
- LR too high
- Bad annotations (completely wrong labels)
- Corrupted images in dataset (check with \`trichome dataset verify\`)

---

## Annotation / Label Studio issues

### Can't export annotations in YOLO format

1. Go to Label Studio → Project → Export → YOLO format
2. If YOLO format is missing, update Label Studio image:
   \`\`\`bash
   docker pull heartexlabs/label-studio:latest
   docker compose --profile annotation up -d --force-recreate
   \`\`\`

3. Export via API:
   \`\`\`bash
   curl -H "Authorization: Token <your_token>" \\
     "http://localhost:3005/api/projects/1/export?exportType=YOLO" \\
     -o annotations.zip
   \`\`\`

---

### Annotations not showing up after import

1. Check task was successfully imported:
   \`\`\`bash
   curl -H "Authorization: Token <token>" \\
     "http://localhost:3005/api/projects/1/tasks/" | python3 -m json.tool | head -30
   \`\`\`

2. Check label config matches your class names (\`stalked\`, \`sessile\`, \`bulbous\`, \`non-glandular\`).

3. Label Studio uses 0-indexed class IDs in YOLO format — verify your \`data.yaml\`:
   \`\`\`yaml
   names: ['stalked', 'sessile', 'bulbous', 'non-glandular']
   \`\`\`

---

## Calibration / measurement issues

### Measurements show wildly wrong µm values

The µm/px calibration scale must be set per microscope + objective combination:

\`\`\`bash
# List available calibration profiles:
trichome calibrate list

# Set scale manually (measure a known-size object):
trichome calibrate set \\
  --microscope "DeltaOptical-TCA-1" \\
  --objective "40x" \\
  --um-per-px 0.27    # measure from calibration slide
\`\`\`

If you don't have a calibration slide, use a stage micrometer or a known reference structure (pollen grain = ~20 µm diameter).

---

### Confidence scores seem too high (> 0.95 for all detections)

Raw YOLO scores are overconfident. Apply calibration:

\`\`\`bash
trichome calibrate-confidence \\
  --model data/models/yolo11s.pt \\
  --val-data data/datasets/v1/val/ \\
  --method platt \\
  --plot    # shows reliability diagram
\`\`\`

Calibrated confidence is stored in \`data/models/yolo11s_calibrated.json\` and used automatically in inference.

---

## Log locations

| Service | Log location |
|---------|-------------|
| Backend | \`logs/backend.log\` |
| Training | \`logs/training.log\` |
| Frontend (dev) | terminal / \`frontend/.next/\` |
| Docker (all) | \`docker logs <container_name>\` |
| Label Studio | \`docker logs ctip-label-studio\` |
| nginx | \`docker logs ctip-nginx\` |

\`\`\`bash
# Follow all logs in real time:
tail -f logs/*.log

# Docker logs with timestamps:
docker logs ctip-backend -f --timestamps
\`\`\`

---

## Reset / clean state

\`\`\`bash
# Reset setup wizard (re-run all 11 steps):
sed -i '/SETUP_COMPLETE/d' .env

# Reset database (loses all records — NOT images or models):
rm trichome.db && uvicorn backend.main:app --reload --port 8000

# Stop all Docker containers:
cd docker && docker compose --profile annotation down

# Clean Docker volumes (DESTRUCTIVE — loses Label Studio data):
docker volume rm ctip_label_studio_data ctip_postgres_data

# Clear model download cache:
rm -f data/models/*.tmp
\`\`\`
`;

const de = `
## Schnell-Diagnose

\`\`\`bash
# 1. Backend läuft?
curl http://localhost:8000/api/v1/system/health

# 2. GPU erkannt?
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"

# 3. Alle Container laufen?
docker ps --format "table {{.Names}}\\t{{.Status}}"
\`\`\`

---

## Setup-Wizard Probleme

### Weiterleitung zurück zum Setup

Der Wizard speichert \`SETUP_COMPLETE=true\` in \`.env\`. Falls fehlend:

\`\`\`bash
grep SETUP_COMPLETE .env
echo "SETUP_COMPLETE=true" >> .env
# Backend neu starten
\`\`\`

### Label Studio Account kann nicht erstellt werden

1. Warte 10–30s nach Docker-Start (LS braucht Zeit zum Initialisieren)
2. Prüfe: \`curl http://localhost:3005/api/health\`
3. Logs: \`docker logs ctip-label-studio --tail 50\`

---

## Backend-Probleme

### OOM / CUDA out of memory

VRAM-Limits für RTX 4060 (8 GB):

| Operation | Max Batch | VRAM |
|-----------|-----------|------|
| YOLO Inferenz | 4 Kacheln | ~5,0 GB |
| YOLO Training | batch=4 | ~7,1 GB |
| SAM2-tiny | — | ~3,2 GB |

Lösung: \`YOLO_TILE_BATCH=1\` in \`.env\` setzen (sicherste Option).

### GPU nicht erkannt

\`\`\`bash
nvidia-smi
python -c "import torch; print(torch.version.cuda)"
# Falls None: PyTorch mit CUDA neu installieren
pip install torch --index-url https://download.pytorch.org/whl/cu121
\`\`\`

---

## Docker-Probleme

### docker-compose.yml nicht gefunden

\`\`\`bash
cd docker && docker compose up -d
\`\`\`

### "permission denied" bei Docker

\`\`\`bash
sudo usermod -aG docker $USER
newgrp docker
\`\`\`

---

## Training-Probleme

### mAP@0.5 unter 0,70

| Problem | Ursache | Lösung |
|---------|---------|--------|
| mAP < 0,50 | Zu wenig Annotationen | mind. 200 Trichome annotieren |
| mAP 0,50–0,70 | Klassenungleichgewicht | \`cls_weights\` nutzen |
| Hohe FP-Rate | conf zu niedrig | Sweep: \`--conf-sweep 0.1,0.2,0.3,0.4,0.5\` |

### Training endet mit NaN Loss

\`\`\`yaml
lr0: 0.001
warmup_epochs: 5
clip_gradients: 10
\`\`\`

---

## Log-Speicherorte

| Service | Pfad |
|---------|------|
| Backend | \`logs/backend.log\` |
| Training | \`logs/training.log\` |
| Docker | \`docker logs <container>\` |

---

## Reset

\`\`\`bash
# Setup-Wizard zurücksetzen:
sed -i '/SETUP_COMPLETE/d' .env

# Datenbank zurücksetzen (Bilder/Modelle bleiben):
rm trichome.db

# Docker-Container stoppen:
cd docker && docker compose --profile annotation down
\`\`\`
`;

const es = `
## Diagnóstico rápido

\`\`\`bash
curl http://localhost:8000/api/v1/system/health
nvidia-smi
docker ps --format "table {{.Names}}\\t{{.Status}}"
\`\`\`

---

## Problemas comunes

### OOM / CUDA sin memoria

Límites VRAM para RTX 4060 (8 GB):

| Operación | Batch máx | VRAM |
|-----------|-----------|------|
| Inferencia YOLO | 4 mosaicos | ~5.0 GB |
| Entrenamiento YOLO | batch=4 | ~7.1 GB |

Solución: \`YOLO_TILE_BATCH=1\` en \`.env\`

### GPU no detectada

\`\`\`bash
nvidia-smi
python -c "import torch; print(torch.version.cuda)"
pip install torch --index-url https://download.pytorch.org/whl/cu121
\`\`\`

### mAP@0.5 menor a 0.70

| Problema | Causa | Solución |
|---------|-------|----------|
| mAP < 0.50 | Pocas anotaciones | Anotar al menos 200 tricomas |
| Alta tasa FP | conf muy bajo | Usar \`--conf-sweep 0.1,...,0.5\` |

---

## Ubicación de logs

| Servicio | Ubicación |
|---------|-----------|
| Backend | \`logs/backend.log\` |
| Entrenamiento | \`logs/training.log\` |
| Docker | \`docker logs <contenedor>\` |

---

## Restablecer estado

\`\`\`bash
# Restablecer asistente de configuración:
sed -i '/SETUP_COMPLETE/d' .env

# Restablecer base de datos:
rm trichome.db

# Detener Docker:
cd docker && docker compose --profile annotation down
\`\`\`
`;

const page: WikiPage = {
  slug: 'troubleshooting',
  title: { en: 'Troubleshooting', de: 'Fehlerbehebung', es: 'Solución de problemas' },
  description: {
    en: 'Common errors, OOM fixes, Docker issues, training failures, calibration problems.',
    de: 'Häufige Fehler, OOM-Fixes, Docker-Probleme, Trainingsfehler, Kalibrierungsprobleme.',
    es: 'Errores comunes, correcciones OOM, problemas Docker, fallos de entrenamiento.',
  },
  content: { en, de, es },
  section: 'reference',
  icon: '🔧',
};

export default page;
