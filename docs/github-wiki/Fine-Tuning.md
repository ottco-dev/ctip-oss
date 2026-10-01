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

```bash
# Keep old and new annotations in one Label Studio project, then export it again —
# the session split keeps every imaging session in exactly one of train / val / test.
# Export the Label Studio project as a YOLO dataset (whole imaging sessions per split, seed 42)
curl -X POST http://localhost:8000/api/v1/training/prepare-ls-dataset \
  -H "Content-Type: application/json" \
  -d '{"project_id": 1, "train_ratio": 0.70, "val_ratio": 0.15, "seed": 42}'
# The export log lists images and sessions per split and warns about missing session information.
```

---

## Fine-tuning config

`configs/training/yolo11s_finetune.yaml`:
```yaml
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
```

```bash
trichome train detection --config configs/training/yolo11s_finetune.yaml
```

---

## Hyperparameter tuning

### Learning rate

```python
# Typical LR schedule (cosine decay)
# epoch 0–3:  warmup (0 → lr0)
# epoch 3–N:  cosine decay (lr0 → lr0 * lrf)

# RTX 4060 sweet spot for batch=4:
lr0: 0.005    # slightly below default 0.01 for stability
```

### Confidence threshold

```bash
# Find optimal confidence threshold on validation set
trichome benchmark detection \\
  --model data/models/yolo11s_custom.pt \\
  --split val \\
  --conf-sweep 0.1,0.2,0.3,0.4,0.5

# Output: precision-recall curve + F1 vs conf plot
```

Typical optimal conf for trichome detection: **0.25 – 0.35**.
Lower conf → higher recall (fewer missed), higher FP.

### IoU threshold

```yaml
iou: 0.45     # NMS IoU — lower = more aggressive deduplication
              # increase to 0.6 if trichomes are densely packed
```

### Image augmentation for microscopy

```yaml
# Recommended augmentation for microscopy images:
hsv_h: 0.02    # small hue shift (trichome color matters)
hsv_s: 0.5     # saturation variation
hsv_v: 0.4     # brightness variation (different lighting)
flipud: 0.5    # vertical flip (microscope images have no gravity orientation)
fliplr: 0.5    # horizontal flip
rotate: 15     # small rotation
blur: 0.1      # occasional slight blur (mimics focus variation)
noise: 0.05    # camera noise simulation
```

**Do NOT use:**
- `mosaic: 1.0` for fine-tuning (too destructive to existing features)
- `perspective: 0.5` (microscope images have consistent perspective)

---

## Class imbalance handling

If bulbous or non-glandular trichomes are rare in your dataset:

- **Label more of them.** Active learning (`/api/v1/al`) ranks unlabelled images by uncertainty —
  rare classes tend to be the uncertain ones. This is the most reliable fix.
- **Image-level oversampling.** Duplicate *training* images that contain rare classes after the export
  (never val/test). Ultralytics has no per-class loss weights; its `cls` gain scales all classes equally.
- **Report per-class metrics.** Overall mAP hides a weak rare class.

---

## Evaluating improvements

```bash
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
```

---

## TensorRT export (production speed)

After training, export to TensorRT for ~3× faster inference:

```bash
trichome export-model \\
  --model data/models/yolo11s_custom.pt \\
  --format engine \\
  --imgsz 1280 \\
  --device 0 \\
  --output data/models/yolo11s_custom.engine

# Use in inference
trichome detect --model data/models/yolo11s_custom.engine --input ...
```

RTX 4060 inference speed (1280px, batch=1):
| Format | ms/image |
|--------|---------|
| PyTorch (.pt) | ~45ms |
| TensorRT FP16 (.engine) | ~15ms |
| TensorRT INT8 | ~10ms |

Note: INT8 requires calibration dataset and may reduce mAP by ~1-2%.
