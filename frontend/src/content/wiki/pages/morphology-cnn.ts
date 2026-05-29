import type { WikiPage } from '../types';

const en = `
## Morphology CNN

CTIP includes a fine-tuned **EfficientNet-B0** classifier that assigns each detected trichome instance to one of four morphological classes. The model is trained on cropped trichome regions produced by the SAM2 segmentation step.

---

## Classes

| Class | Description |
|---|---|
| \`CAPITATE_STALKED\` | Glandular trichome with a visible stalk and spherical head; primary focus for maturity assessment |
| \`CAPITATE_SESSILE\` | Glandular trichome with a flattened head directly on the epidermis; no stalk |
| \`BULBOUS\` | Smallest glandular type; single or few secretory cells |
| \`NON_GLANDULAR\` | Cystolithic or covering trichome; excluded from maturity analysis |

---

## Model architecture

\`\`\`
EfficientNet-B0 (pretrained on ImageNet-1k)
    │
    ▼
Global Average Pooling
    │
    ▼
Dropout(p=config.dropout)
    │
    ▼
Linear(1280 → 4)   # 4 morphology classes
    │
    ▼
Softmax → class probabilities
\`\`\`

Input: 224×224 RGB crops, normalized with ImageNet mean/std.

---

## Training augmentations

\`\`\`python
transforms.Compose([
    transforms.RandomRotation(180),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.RandomGrayscale(p=0.1),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
\`\`\`

\`RandomRotation(180)\` is critical because trichomes can be imaged at any orientation. \`RandomGrayscale\` improves robustness to illumination variation across microscope setups.

---

## MorphologyTrainingConfig

| Parameter | Default | Description |
|---|---|---|
| \`epochs\` | 50 | Training epochs |
| \`batch_size\` | 32 | Samples per batch |
| \`learning_rate\` | 1e-4 | AdamW learning rate |
| \`dropout\` | 0.3 | Dropout probability before final linear layer |
| \`fp16\` | \`true\` | Mixed-precision training (AMP) |
| \`augment\` | \`true\` | Enable the augmentation pipeline |

---

## API reference

### Start training

\`\`\`bash
POST /api/v1/morphology/training/start
Content-Type: application/json

{
  "dataset_root": "data/datasets/morphology/v1/",
  "epochs": 50,
  "batch_size": 32,
  "learning_rate": 1e-4,
  "dropout": 0.3,
  "fp16": true,
  "augment": true
}
\`\`\`

Response:
\`\`\`json
{ "job_id": "morph_t9x2", "status": "queued" }
\`\`\`

### Check training status

\`\`\`bash
GET /api/v1/morphology/training/status
\`\`\`

\`\`\`json
{
  "job_id": "morph_t9x2",
  "status": "running",
  "epoch": 18,
  "epochs_total": 50,
  "train_loss": 0.112,
  "val_accuracy": 0.934
}
\`\`\`

### Evaluate model

\`\`\`bash
POST /api/v1/morphology/training/evaluate
Content-Type: application/json

{
  "checkpoint": "data/models/morphology/best.pt",
  "test_root": "data/datasets/morphology/v1/test/"
}
\`\`\`

### Export to ONNX

\`\`\`bash
POST /api/v1/morphology/training/export
Content-Type: application/json

{
  "checkpoint": "data/models/morphology/best.pt",
  "output_path": "data/models/morphology/morphology_b0.onnx"
}
\`\`\`

---

## Training output

\`\`\`
data/models/morphology/
├── best.pt             # PyTorch checkpoint (best val accuracy)
├── last.pt             # Checkpoint at final epoch
├── morphology_b0.onnx  # ONNX export for runtime inference
└── training_run.json   # Config + per-epoch metrics
\`\`\`

---

## Frontend: Morphology page → Training tab

- **Config panel** — all \`MorphologyTrainingConfig\` fields.
- **Live metrics** — training loss and validation accuracy per epoch (Recharts line chart), streamed via WebSocket \`/ws/training\`.
- **Confusion matrix** — rendered after evaluation completes.
- **Export button** — triggers ONNX export; shows output path.

---

## VRAM usage

| Config | Approx. VRAM |
|---|---|
| \`batch_size=32\`, FP16 | ~1.4 GB |
| \`batch_size=64\`, FP16 | ~2.3 GB |

EfficientNet-B0 is deliberately lightweight; even \`batch_size=64\` fits comfortably on an 8 GB GPU.
`;

const de = `
## Morphologie-CNN

CTIP enthält einen feinabgestimmten **EfficientNet-B0**-Klassifikator, der jede erkannte Trichom-Instanz einer von vier morphologischen Klassen zuordnet. Das Modell wird auf zugeschnittenen Trichom-Regionen aus dem SAM2-Segmentierungsschritt trainiert.

---

## Klassen

| Klasse | Beschreibung |
|---|---|
| \`CAPITATE_STALKED\` | Drüsentrichom mit sichtbarem Stiel und kugelförmigem Kopf |
| \`CAPITATE_SESSILE\` | Drüsentrichom mit abgeflachtem Kopf direkt auf der Epidermis |
| \`BULBOUS\` | Kleinster Drüsentyp |
| \`NON_GLANDULAR\` | Deckend oder cystolithisch; von der Reifeanalyse ausgeschlossen |

---

## MorphologyTrainingConfig

| Parameter | Standard | Beschreibung |
|---|---|---|
| \`epochs\` | 50 | Trainings-Epochen |
| \`batch_size\` | 32 | Stichproben je Batch |
| \`learning_rate\` | 1e-4 | AdamW-Lernrate |
| \`dropout\` | 0.3 | Dropout-Wahrscheinlichkeit |
| \`fp16\` | \`true\` | Mixed-Precision-Training (AMP) |
| \`augment\` | \`true\` | Augmentierungs-Pipeline aktivieren |

---

## API-Referenz

\`\`\`bash
POST /api/v1/morphology/training/start
GET  /api/v1/morphology/training/status
POST /api/v1/morphology/training/evaluate
POST /api/v1/morphology/training/export
\`\`\`

---

## Trainingsausgabe

\`\`\`
data/models/morphology/
├── best.pt             # PyTorch-Checkpoint (beste Val-Genauigkeit)
├── last.pt             # Checkpoint der letzten Epoche
├── morphology_b0.onnx  # ONNX-Export für die Laufzeit-Inferenz
└── training_run.json   # Konfiguration + Metriken je Epoche
\`\`\`
`;

const es = `
## CNN de Morfología

CTIP incluye un clasificador **EfficientNet-B0** ajustado que asigna cada instancia de tricoma detectada a una de cuatro clases morfológicas.

---

## Clases

| Clase | Descripción |
|---|---|
| \`CAPITATE_STALKED\` | Tricoma glandular con tallo visible y cabeza esférica |
| \`CAPITATE_SESSILE\` | Tricoma glandular con cabeza aplanada directamente sobre la epidermis |
| \`BULBOUS\` | Tipo glandular más pequeño |
| \`NON_GLANDULAR\` | Cistolítico o cobertor; excluido del análisis de madurez |

---

## MorphologyTrainingConfig

| Parámetro | Defecto | Descripción |
|---|---|---|
| \`epochs\` | 50 | Épocas de entrenamiento |
| \`batch_size\` | 32 | Muestras por lote |
| \`learning_rate\` | 1e-4 | Tasa de aprendizaje AdamW |
| \`dropout\` | 0.3 | Probabilidad de dropout |
| \`fp16\` | \`true\` | Entrenamiento de precisión mixta (AMP) |
| \`augment\` | \`true\` | Activar pipeline de aumentación |

---

## Referencia de API

\`\`\`bash
POST /api/v1/morphology/training/start
GET  /api/v1/morphology/training/status
POST /api/v1/morphology/training/evaluate
POST /api/v1/morphology/training/export
\`\`\`

---

## Salida del entrenamiento

\`\`\`
data/models/morphology/
├── best.pt             # Checkpoint PyTorch (mejor accuracy de validación)
├── last.pt             # Checkpoint de la última época
├── morphology_b0.onnx  # Exportación ONNX para inferencia en tiempo de ejecución
└── training_run.json   # Configuración + métricas por época
\`\`\`
`;

const page: WikiPage = {
  slug: 'morphology-cnn',
  title: {
    en: 'Morphology CNN',
    de: 'Morphologie-CNN',
    es: 'CNN de Morfología',
  },
  description: {
    en: 'EfficientNet-B0 fine-tuned for 4-class trichome morphology classification.',
    de: 'EfficientNet-B0 feinabgestimmt für 4-Klassen-Trichom-Morphologieklassifizierung.',
    es: 'EfficientNet-B0 ajustado para clasificación morfológica de tricomas en 4 clases.',
  },
  content: { en, de, es },
  section: 'reference',
  icon: '🧬',
};

export default page;
