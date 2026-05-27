import type { WikiPage } from '../types';

const en = `
## Models used

| Model | Role | Default config |
|-------|------|----------------|
| YOLO11s | Trichome detection | yolo11s_detection.yaml |
| SAM2-tiny | Instance segmentation | sam2_hiera_tiny.yaml |
| Custom CNN | Maturity classification | maturity_classifier.yaml |

---

## Before training

\`\`\`bash
# 1. Export annotations from Label Studio
trichome export --project 1 --format yolo --output data/datasets/v1/

# 2. Verify dataset
trichome dataset verify --path data/datasets/v1/
# Checks: class distribution, image sizes, empty labels, train/val split

# 3. Check GPU
nvidia-smi
python -c "import torch; print(torch.cuda.get_device_properties(0).total_memory // 1024**2, 'MB VRAM')"
\`\`\`

---

## Detection training (YOLO11s)

### Config file

\`configs/training/yolo11s_detection.yaml\`:
\`\`\`yaml
model: yolo11s.pt          # pretrained weights (downloaded in setup)
data: data/datasets/v1/data.yaml
epochs: 100
imgsz: 1280                # tile size for training
batch: 4                   # reduce to 2 if OOM on 8 GB VRAM
workers: 4
lr0: 0.01
lrf: 0.01                  # final LR = lr0 * lrf
momentum: 0.937
weight_decay: 0.0005
warmup_epochs: 3
hsv_h: 0.015               # color augmentation (important for maturity detection)
hsv_s: 0.7
hsv_v: 0.4
flipud: 0.5
fliplr: 0.5
mosaic: 1.0
mixup: 0.1
device: cuda:0
seed: 42                   # reproducibility
project: mlruns/detection
name: yolo11s-v1
\`\`\`

### Start training

\`\`\`bash
# Via CLI
trichome train detection --config configs/training/yolo11s_detection.yaml

# Directly with ultralytics
source .venv/bin/activate
python -c "
from ultralytics import YOLO
model = YOLO('data/models/yolo11s.pt')
model.train(**yaml.safe_load(open('configs/training/yolo11s_detection.yaml')))
"

# Monitor in MLflow (open in browser)
http://localhost:3004

# Monitor in terminal
tail -f logs/training.log
\`\`\`

### VRAM usage by batch size (YOLO11s, imgsz=1280)

| Batch | VRAM | Recommendation |
|-------|------|----------------|
| 1 | ~3.5 GB | RTX 3060 6 GB |
| 2 | ~5.2 GB | RTX 3060 8 GB |
| **4** | **~7.1 GB** | **RTX 4060 8 GB ← default** |
| 8 | ~13.2 GB | RTX 3080 16 GB |
| 16 | ~25 GB | A100 40 GB |

---

## Segmentation training (SAM2)

SAM2 is used in **prompted mode** — it does not need bounding-box-free training.
The detection model (YOLO11s) provides prompts at inference time.

For custom segmentation masks:

\`\`\`bash
trichome train segmentation --config configs/training/sam2_finetune.yaml
# Note: SAM2 fine-tuning requires ~6 GB VRAM minimum
# On RTX 4060: only sam2-tiny is feasible
\`\`\`

---

## Maturity classifier training

\`\`\`bash
trichome train maturity --config configs/training/maturity_classifier.yaml
\`\`\`

The maturity classifier uses:
- **Color features**: HSV histogram (hue/saturation distribution)
- **Texture features**: LBP (Local Binary Pattern), GLCM, Gabor filters
- **Translucency**: gradient variance of trichome head region

Training input: cropped trichome head images (extracted from detection results).
Labels: \`clear\`, \`cloudy\`, \`amber\` — annotated separately in a classification task.

---

## Training via CTIP API

\`\`\`bash
# Start training job
curl -X POST http://localhost:8000/api/v1/training/start \\
  -H "Content-Type: application/json" \\
  -d '{"config": "yolo11s_detection.yaml", "dataset": "v1"}'

# Check status
curl http://localhost:8000/api/v1/training/jobs

# Live metrics via WebSocket
wscat -c ws://localhost:8000/ws/training
\`\`\`

---

## After training

\`\`\`bash
# Best model saved to:
mlruns/detection/yolo11s-v1/weights/best.pt

# Copy to models directory
cp mlruns/detection/yolo11s-v1/weights/best.pt data/models/yolo11s_custom.pt

# Benchmark on test set
trichome benchmark detection --model data/models/yolo11s_custom.pt --split test
\`\`\`

### Expected metrics (good dataset)

| Metric | Minimum | Target |
|--------|---------|--------|
| mAP@0.5 | 0.70 | ≥ 0.85 |
| mAP@0.5:0.95 | 0.45 | ≥ 0.60 |
| Precision | 0.75 | ≥ 0.88 |
| Recall | 0.72 | ≥ 0.85 |

---

## MLflow experiment tracking

All training runs automatically logged:

\`\`\`python
# Access experiments programmatically
import mlflow
mlflow.set_tracking_uri("http://localhost:3004")
runs = mlflow.search_runs(experiment_names=["trichome-detection"])
best = runs.sort_values("metrics.mAP50", ascending=False).iloc[0]
print(best["params.lr0"], best["metrics.mAP50"])
\`\`\`
`;

const de = `
## Verwendete Modelle

| Modell | Rolle | Standard-Config |
|--------|-------|-----------------|
| YOLO11s | Trichom-Erkennung | yolo11s_detection.yaml |
| SAM2-tiny | Instanz-Segmentierung | sam2_hiera_tiny.yaml |
| Custom CNN | Reifeklassifizierung | maturity_classifier.yaml |

---

## Vor dem Training

\`\`\`bash
# 1. Annotationen aus Label Studio exportieren
trichome export --project 1 --format yolo --output data/datasets/v1/

# 2. Datensatz prüfen
trichome dataset verify --path data/datasets/v1/

# 3. GPU prüfen
nvidia-smi
\`\`\`

---

## Erkennungs-Training (YOLO11s)

\`\`\`yaml
# configs/training/yolo11s_detection.yaml (Schlüsselparameter)
model: yolo11s.pt
data: data/datasets/v1/data.yaml
epochs: 100
imgsz: 1280
batch: 4          # Bei OOM auf 8 GB: auf 2 reduzieren
device: cuda:0
seed: 42
\`\`\`

\`\`\`bash
trichome train detection --config configs/training/yolo11s_detection.yaml
\`\`\`

### VRAM-Verbrauch (YOLO11s, imgsz=1280)

| Batch | VRAM | Empfehlung |
|-------|------|-----------|
| 2 | ~5,2 GB | RTX 3060 8 GB |
| **4** | **~7,1 GB** | **RTX 4060 8 GB ← Standard** |
| 8 | ~13,2 GB | RTX 3080 16 GB |

---

## Nach dem Training

\`\`\`bash
# Bestes Modell liegt in:
mlruns/detection/yolo11s-v1/weights/best.pt

# In models-Verzeichnis kopieren
cp mlruns/detection/yolo11s-v1/weights/best.pt data/models/yolo11s_custom.pt

# Benchmark
trichome benchmark detection --model data/models/yolo11s_custom.pt --split test
\`\`\`

### Erwartete Metriken (guter Datensatz)

| Metrik | Minimum | Ziel |
|--------|---------|------|
| mAP@0.5 | 0,70 | ≥ 0,85 |
| mAP@0.5:0.95 | 0,45 | ≥ 0,60 |
| Precision | 0,75 | ≥ 0,88 |
| Recall | 0,72 | ≥ 0,85 |

---

## MLflow-Experiment-Tracking

Alle Trainingsläufe werden automatisch geloggt:
- Hyperparameter
- mAP@0.5, mAP@0.5:0.95
- Loss-Kurven (box, cls, dfl)
- Modell-Artefakte

Zugriff: http://localhost:3004
`;

const es = `
## Modelos utilizados

| Modelo | Función | Config por defecto |
|--------|---------|-------------------|
| YOLO11s | Detección de tricomas | yolo11s_detection.yaml |
| SAM2-tiny | Segmentación de instancias | sam2_hiera_tiny.yaml |
| Custom CNN | Clasificación de madurez | maturity_classifier.yaml |

## Antes del entrenamiento

\`\`\`bash
trichome export --project 1 --format yolo --output data/datasets/v1/
trichome dataset verify --path data/datasets/v1/
\`\`\`

## Entrenamiento de detección (YOLO11s)

\`\`\`yaml
# Parámetros clave
model: yolo11s.pt
epochs: 100
imgsz: 1280
batch: 4      # Reducir a 2 si OOM en 8 GB VRAM
device: cuda:0
seed: 42
\`\`\`

\`\`\`bash
trichome train detection --config configs/training/yolo11s_detection.yaml
\`\`\`

## Métricas esperadas

| Métrica | Mínimo | Objetivo |
|---------|--------|----------|
| mAP@0.5 | 0.70 | ≥ 0.85 |
| Precision | 0.75 | ≥ 0.88 |
| Recall | 0.72 | ≥ 0.85 |

## Después del entrenamiento

\`\`\`bash
cp mlruns/detection/yolo11s-v1/weights/best.pt data/models/yolo11s_custom.pt
trichome benchmark detection --model data/models/yolo11s_custom.pt --split test
\`\`\`
`;

const page: WikiPage = {
  slug: 'training',
  title: { en: 'Training', de: 'Training', es: 'Entrenamiento' },
  description: {
    en: 'YOLO11s detection training, SAM2 segmentation, maturity classifier, VRAM budgets, MLflow.',
    de: 'YOLO11s Erkennungs-Training, SAM2 Segmentierung, Maturity Classifier, VRAM-Budget, MLflow.',
    es: 'Entrenamiento de detección YOLO11s, SAM2, clasificador de madurez, presupuesto VRAM, MLflow.',
  },
  content: { en, de, es },
  section: 'workflow',
  icon: '🧠',
};

export default page;
