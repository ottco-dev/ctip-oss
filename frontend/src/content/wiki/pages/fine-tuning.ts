import type { WikiPage } from '../types';

const en = `
## When to fine-tune

Fine-tune when:
- mAP@0.5 drops below 0.75 on new strains
- You add a new microscope (different µm/px ratio, different image characteristics)
- You collect 50+ new annotated images
- False positive rate is high on a specific background type

Do NOT fine-tune when:
- You have fewer than 30 new annotated images
- Only 1-2 images per class in the new batch

---

## Data preparation for fine-tuning

\`\`\`bash
# Export new annotations
trichome export --project 1 --format yolo --output data/datasets/v2-incremental/

# Merge with existing training data
trichome dataset merge \\
  --base data/datasets/v1/ \\
  --new data/datasets/v2-incremental/ \\
  --output data/datasets/v2/ \\
  --val-split 0.15 \\
  --seed 42

# Verify merged dataset
trichome dataset verify --path data/datasets/v2/
\`\`\`

---

## Fine-tuning config

\`configs/training/yolo11s_finetune.yaml\`:
\`\`\`yaml
model: data/models/yolo11s_custom.pt   # your current best model
data: data/datasets/v2/data.yaml

# Shorter training — model already has good features
epochs: 30
imgsz: 1280
batch: 4

# Lower learning rate — don't destroy learned features
lr0: 0.001          # 10× lower than initial training
lrf: 0.01
warmup_epochs: 1

# Keep augmentation conservative
mosaic: 0.5         # less mosaic for fine-tuning
mixup: 0.0          # disable mixup

freeze: 10          # freeze first 10 backbone layers
device: cuda:0
seed: 42
\`\`\`

\`\`\`bash
trichome train detection --config configs/training/yolo11s_finetune.yaml
\`\`\`

---

## Hyperparameter tuning

### Learning rate

\`\`\`python
# Typical LR schedule (cosine decay)
# epoch 0–3:  warmup (0 → lr0)
# epoch 3–N:  cosine decay (lr0 → lr0 * lrf)

# RTX 4060 sweet spot for batch=4:
lr0: 0.005    # slightly below default 0.01 for stability
\`\`\`

### Confidence threshold

\`\`\`bash
# Find optimal confidence threshold on validation set
trichome benchmark detection \\
  --model data/models/yolo11s_custom.pt \\
  --split val \\
  --conf-sweep 0.1,0.2,0.3,0.4,0.5

# Output: precision-recall curve + F1 vs conf plot
\`\`\`

Typical optimal conf for trichome detection: **0.25 – 0.35**.
Lower conf → higher recall (fewer missed), higher FP.

### IoU threshold

\`\`\`yaml
iou: 0.45     # NMS IoU — lower = more aggressive deduplication
              # increase to 0.6 if trichomes are densely packed
\`\`\`

### Image augmentation for microscopy

\`\`\`yaml
# Recommended augmentation for microscopy images:
hsv_h: 0.02    # small hue shift (trichome color matters)
hsv_s: 0.5     # saturation variation
hsv_v: 0.4     # brightness variation (different lighting)
flipud: 0.5    # vertical flip (microscope images have no gravity orientation)
fliplr: 0.5    # horizontal flip
rotate: 15     # small rotation
blur: 0.1      # occasional slight blur (mimics focus variation)
noise: 0.05    # camera noise simulation
\`\`\`

**Do NOT use:**
- \`mosaic: 1.0\` for fine-tuning (too destructive to existing features)
- \`perspective: 0.5\` (microscope images have consistent perspective)

---

## Class imbalance handling

If bulbous or non-glandular trichomes are rare in your dataset:

\`\`\`yaml
# Option 1: class weights in loss function
cls_weights: [1.0, 1.0, 3.0, 2.0]   # stalked, sessile, bulbous, non-glandular

# Option 2: oversample rare classes during export
trichome export --project 1 --format yolo --oversample bulbous:3 --output data/datasets/v2/
\`\`\`

---

## Evaluating improvements

\`\`\`bash
# Compare two models side-by-side
trichome compare \\
  --model-a data/models/yolo11s_v1.pt \\
  --model-b mlruns/detection/yolo11s-v2/weights/best.pt \\
  --test-data data/datasets/v2/test/ \\
  --output reports/model_comparison.html

# Check calibration quality
trichome calibrate-confidence \\
  --model data/models/yolo11s_v2.pt \\
  --val-data data/datasets/v2/val/ \\
  --plot   # outputs ECE + reliability diagram
\`\`\`

---

## TensorRT export (production speed)

After training, export to TensorRT for ~3× faster inference:

\`\`\`bash
trichome export-model \\
  --model data/models/yolo11s_custom.pt \\
  --format engine \\
  --imgsz 1280 \\
  --device 0 \\
  --output data/models/yolo11s_custom.engine

# Use in inference
trichome detect --model data/models/yolo11s_custom.engine --input ...
\`\`\`

RTX 4060 inference speed (1280px, batch=1):
| Format | ms/image |
|--------|---------|
| PyTorch (.pt) | ~45ms |
| TensorRT FP16 (.engine) | ~15ms |
| TensorRT INT8 | ~10ms |

Note: INT8 requires calibration dataset and may reduce mAP by ~1-2%.
`;

const de = `
## Wann Fine-Tunen?

Fine-Tunen wenn:
- mAP@0.5 unter 0,75 für neue Sorten
- Neues Mikroskop hinzugefügt (anderes µm/px-Verhältnis)
- 50+ neue annotierte Bilder vorhanden
- Hohe False-Positive-Rate bei einem bestimmten Hintergrundtyp

**Nicht** fine-tunen bei:
- Weniger als 30 neuen annotierten Bildern
- Nur 1-2 Bilder je Klasse im neuen Batch

---

## Daten für Fine-Tuning vorbereiten

\`\`\`bash
trichome export --project 1 --format yolo --output data/datasets/v2-neu/

trichome dataset merge \\
  --base data/datasets/v1/ \\
  --new data/datasets/v2-neu/ \\
  --output data/datasets/v2/ \\
  --val-split 0.15 --seed 42

trichome dataset verify --path data/datasets/v2/
\`\`\`

---

## Fine-Tuning Konfiguration

\`\`\`yaml
model: data/models/yolo11s_custom.pt   # aktuell bestes Modell
data: data/datasets/v2/data.yaml
epochs: 30
lr0: 0.001          # 10× niedriger als initiales Training
warmup_epochs: 1
mosaic: 0.5         # weniger Mosaic beim Fine-Tuning
freeze: 10          # erste 10 Backbone-Schichten einfrieren
seed: 42
\`\`\`

---

## Hyperparameter

### Lernrate

Standard LR für RTX 4060 (batch=4): \`lr0: 0.005\`

### Konfidenz-Schwellenwert

\`\`\`bash
trichome benchmark detection \\
  --model data/models/yolo11s_custom.pt \\
  --split val \\
  --conf-sweep 0.1,0.2,0.3,0.4,0.5
\`\`\`

Typisch optimaler Wert für Trichom-Erkennung: **0,25 – 0,35**

### Augmentation für Mikroskopbilder

\`\`\`yaml
hsv_h: 0.02    # kleiner Farbtonverschiebung
hsv_v: 0.4     # Helligkeitsvariation
flipud: 0.5    # vertikal spiegeln (Mikroskop hat keine Schwerkraft-Orientierung)
fliplr: 0.5
blur: 0.1      # gelegentlich leicht verwischen
\`\`\`

---

## Klassenungleichgewicht behandeln

\`\`\`yaml
# Klassgewichte in der Loss-Funktion
cls_weights: [1.0, 1.0, 3.0, 2.0]   # stalked, sessile, bulbous, non-glandular

# Oder: seltene Klassen beim Export übergewichten
trichome export --oversample bulbous:3 ...
\`\`\`

---

## TensorRT-Export (Produktionsgeschwindigkeit)

\`\`\`bash
trichome export-model \\
  --model data/models/yolo11s_custom.pt \\
  --format engine --imgsz 1280 --device 0
\`\`\`

| Format | ms/Bild (RTX 4060) |
|--------|-------------------|
| PyTorch (.pt) | ~45ms |
| TensorRT FP16 | ~15ms |
| TensorRT INT8 | ~10ms |
`;

const es = `
## Cuándo hacer fine-tuning

- mAP@0.5 cae por debajo de 0.75 en nuevas cepas
- Nuevo microscopio añadido (diferente ratio µm/px)
- 50+ nuevas imágenes anotadas disponibles

**No** hacer fine-tuning con menos de 30 nuevas imágenes anotadas.

## Preparar datos

\`\`\`bash
trichome export --project 1 --format yolo --output data/datasets/v2-nuevo/
trichome dataset merge --base data/datasets/v1/ --new data/datasets/v2-nuevo/ --output data/datasets/v2/
\`\`\`

## Configuración de fine-tuning

\`\`\`yaml
model: data/models/yolo11s_custom.pt
epochs: 30
lr0: 0.001       # 10× menor que el entrenamiento inicial
freeze: 10       # congelar primeras 10 capas del backbone
\`\`\`

## Umbral de confianza óptimo

\`\`\`bash
trichome benchmark detection --model data/models/yolo11s_custom.pt --split val --conf-sweep 0.1,0.2,0.3,0.4,0.5
\`\`\`

Valor típico óptimo para detección de tricomas: **0.25 – 0.35**

## Exportar a TensorRT (velocidad de producción)

\`\`\`bash
trichome export-model --model data/models/yolo11s_custom.pt --format engine --imgsz 1280

# RTX 4060: PyTorch ~45ms → TensorRT FP16 ~15ms
\`\`\`
`;

const page: WikiPage = {
  slug: 'fine-tuning',
  title: { en: 'Fine-Tuning', de: 'Fine-Tuning', es: 'Fine-Tuning' },
  description: {
    en: 'When and how to fine-tune, hyperparameter tuning, augmentation, class imbalance, TensorRT export.',
    de: 'Wann und wie Fine-Tunen, Hyperparameter, Augmentation, Klassenungleichgewicht, TensorRT-Export.',
    es: 'Cuándo y cómo hacer fine-tuning, hiperparámetros, augmentation, desequilibrio de clases.',
  },
  content: { en, de, es },
  section: 'workflow',
  icon: '⚙️',
};

export default page;
