## Morphology CNN

CTIP includes a fine-tuned **EfficientNet-B0** classifier that assigns each detected trichome instance to one of four morphological classes. The model is trained on cropped trichome regions (produced by the SAM2 segmentation step) and runs on CPU or GPU after the detection pass.

---

## Classes

| Class | Description |
|---|---|
| `CAPITATE_STALKED` | Glandular trichome with a visible stalk and spherical head; primary focus for maturity assessment |
| `CAPITATE_SESSILE` | Glandular trichome with a flattened head directly on the epidermis; no stalk |
| `BULBOUS` | Smallest glandular type; single or few secretory cells |
| `NON_GLANDULAR` | Cystolithic or covering trichome; excluded from maturity analysis |

---

## Model architecture

```
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
```

Input: 224×224 RGB crops, normalized with ImageNet mean/std.

---

## Training augmentations

```python
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
```

`RandomRotation(180)` is critical because trichomes can be imaged at any orientation. `RandomGrayscale` improves robustness to illumination variation across microscope setups.

---

## MorphologyTrainingConfig

| Parameter | Default | Description |
|---|---|---|
| `epochs` | 50 | Training epochs |
| `batch_size` | 32 | Samples per batch |
| `learning_rate` | 1e-4 | AdamW learning rate |
| `dropout` | 0.3 | Dropout probability before the final linear layer |
| `fp16` | `true` | Mixed-precision training (AMP) |
| `augment` | `true` | Enable the augmentation pipeline above |

---

## API reference

### Start training

```bash
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
```

Response:
```json
{
  "job_id": "morph_t9x2",
  "status": "queued"
}
```

### Check training status

```bash
GET /api/v1/morphology/training/status
```

```json
{
  "job_id": "morph_t9x2",
  "status": "running",
  "epoch": 18,
  "epochs_total": 50,
  "train_loss": 0.112,
  "val_accuracy": 0.934
}
```

### Evaluate model

```bash
POST /api/v1/morphology/training/evaluate
Content-Type: application/json

{
  "checkpoint": "data/models/morphology/best.pt",
  "test_root": "data/datasets/morphology/v1/test/"
}
```

```json
{
  "accuracy": 0.941,
  "per_class": {
    "CAPITATE_STALKED": 0.963,
    "CAPITATE_SESSILE": 0.927,
    "BULBOUS": 0.891,
    "NON_GLANDULAR": 0.982
  },
  "confusion_matrix": [[...]]
}
```

### Export to ONNX

```bash
POST /api/v1/morphology/training/export
Content-Type: application/json

{
  "checkpoint": "data/models/morphology/best.pt",
  "output_path": "data/models/morphology/morphology_b0.onnx"
}
```

---

## Training output

```
data/models/morphology/
├── best.pt             # PyTorch checkpoint (best val accuracy)
├── last.pt             # Checkpoint at final epoch
├── morphology_b0.onnx  # ONNX export for runtime inference
└── training_run.json   # Config + per-epoch metrics
```

---

## Frontend: Morphology page → Training tab

- **Config panel** — all `MorphologyTrainingConfig` fields.
- **Live metrics** — training loss and validation accuracy per epoch (Recharts line chart), streamed via WebSocket `/ws/training`.
- **Confusion matrix** — rendered after evaluation completes.
- **Export button** — triggers ONNX export; shows output path.

---

## VRAM usage

| Config | Approx. VRAM |
|---|---|
| `batch_size=32`, FP16 | ~1.4 GB |
| `batch_size=64`, FP16 | ~2.3 GB |

EfficientNet-B0 is deliberately lightweight. Even `batch_size=64` fits comfortably alongside the OS on an 8 GB GPU.
