# End-to-End Workflow Guide

Complete walkthrough from raw microscopy images to a verified, benchmarked detection pipeline.  
**In-app version**: Open the **Guide** page inside CTIP for an interactive checklist version of this document.

---

## Overview

```
Images → Dataset → Labels → Training → Verification → Benchmarks → Pipeline
  1          2        3          4            5              6            7
```

Each step depends on quality from the previous one. A bad dataset cannot be fixed by better hyperparameters.

---

## Step 1 — Image Collection

### Equipment
- Digital microscope (Dino-Lite, AmScope, or similar) at **40× – 200×**
- Fixed lighting rig with diffuser — no auto-exposure between captures
- Calibration slide for µm/pixel measurement (step 4 uses this)

### Requirements

| Setting | Value |
|---------|-------|
| Resolution | ≥ 1280×1280 px |
| Format | PNG or TIFF (never JPEG for training data) |
| White balance | Manual, fixed |
| Focus | Sharp trichome head edges |
| Minimum per class | 50 images |
| Diversity | ≥ 3 strains, ≥ 2 operators |

### Metadata to capture per image
```json
{
  "strain": "Example OG",
  "harvest_date": "2025-05-15",
  "microscope": "Dino-Lite AM7915MZT",
  "magnification": 200,
  "lighting": "ring-diffuser-preset-A",
  "operator": "researcher_1"
}
```

### Tips
- Blurry images teach the model to detect blurry trichomes — discard them.
- Uneven lighting creates false amber in clear trichomes — use a diffuser.
- Capture at multiple focal planes per sample (shallow depth of field).

---

## Step 2 — Dataset Creation

### In CTIP
1. **Datasets** → New Dataset → descriptive name with version: `ctip-v3-multistrains`
2. Import images via folder upload or drag-and-drop
3. Run **Verify Dataset** — checks:
   - Image size distribution
   - Blur detection (Laplacian variance < 50 flagged)
   - Class distribution (requires labels from step 3)
   - Train/val/test leakage prevention
4. Set split: **70 / 20 / 10** (train / val / test)
5. Lock the test split — never add to it after this point

### Class targets

| Class | Minimum | Target |
|-------|---------|--------|
| Stalked (capitate-stalked) | 50 | 200+ |
| Sessile (capitate-sessile) | 40 | 150+ |
| Bulbous | 20 | 80+ |
| Non-glandular (hair) | 30 | 100+ |

### Warning
Do **not** combine datasets without checking for duplicate images. CTIP's dataset fingerprinting detects exact duplicates, but near-duplicates (same image, different crop) must be reviewed manually.

---

## Step 3 — Labeling

### Setup
1. **Annotation** → Label Studio tab → ensure Label Studio is running
2. Select your dataset in the dataset dropdown
3. Create or open a project with the CTIP bounding box template

### VLM-assisted labeling (recommended)
1. Enable auto-annotation in the Auto-Annotation panel
2. Select a VLM backend (Moondream-2B for speed, Florence-2 for accuracy)
3. Run on a batch of 50 images — generates first-pass bounding boxes
4. **Review 100% of auto-labels** — the system enforces this; no auto-label reaches training data unreviewed

### Class definitions

| Class | Description |
|-------|-------------|
| **Stalked** | Large glandular head on a visible stalk. Most prominent in mature flower. |
| **Sessile** | Smaller head, no visible stalk. Common on sugar leaves. |
| **Bulbous** | Tiny round head, no stalk. Often at base of larger trichomes. |
| **Non-glandular** | Hair structures (cystolithic trichomes). Long fibers, no head. |

### Quality gates
- **Inter-annotator agreement**: ≥ 95% on 50-image test set before full labeling
- **Box tightness**: boxes should touch the trichome head edge (not loose)
- **Exclusions**: partially-cropped trichomes at image edges — do NOT label
- **Ambiguous cases**: mark with "Needs review" tag, resolve with second annotator

### Export
```bash
# Via CTIP API
curl -X POST http://localhost:8000/api/v1/datasets/{id}/export \
  -d '{"format": "yolo", "split": "train"}'

# Result: data/datasets/ctip-v3/
#   ├── images/train/   images/val/   images/test/
#   ├── labels/train/   labels/val/   labels/test/
#   └── dataset.yaml
```

---

## Step 4 — Training

### In CTIP
1. **Training** → Runs tab → configure:

| Parameter | RTX 4060 8 GB | Notes |
|-----------|---------------|-------|
| Model variant | `yolo11s` | Best speed/accuracy tradeoff |
| Dataset | `ctip-v3-multistrains` | Bare name — CTIP resolves the path |
| Epochs | 150 | Early stopping patience=50 |
| Batch size | 4 | Reduce to 2 if OOM |
| imgsz | 1280 | Tile size for training |
| AMP | enabled | Saves ~1 GB VRAM |
| Seed | 42 | For reproducibility |

2. Click **Start Training**
3. Watch live metrics: `box_loss`, `cls_loss`, `mAP50`
4. After completion, model auto-registers in **Models** registry

### VRAM guide

| Model | Batch 4 | Batch 2 |
|-------|---------|---------|
| yolo11n | ~4.5 GB | ~3.0 GB |
| yolo11s | ~7.1 GB | ~5.2 GB |
| yolo11m | OOM | ~7.5 GB |

### Target metrics

| Metric | Minimum | Target |
|--------|---------|--------|
| mAP@0.5 | 0.70 | ≥ 0.85 |
| mAP@0.5:0.95 | 0.45 | ≥ 0.60 |
| Precision | 0.75 | ≥ 0.88 |
| Recall | 0.72 | ≥ 0.85 |

### MLflow tracking
All runs auto-logged. View at `http://localhost:3004`.

---

## Step 5 — Verification

### In CTIP
1. **Inference** → Workbench tab
2. Select trained model from registry dropdown
3. Upload **test set images only** (images the model has never seen)
4. Verify:
   - All 4 classes detected
   - Confidence scores in range 0.5–0.95 for clear cases
   - No systematic misclassifications (Sessile/Stalked confusion is common)

### Common failures

| Symptom | Cause | Fix |
|---------|-------|-----|
| Missing detections | Threshold too high or class underrepresented | Lower threshold or add labels |
| Too many false positives | Threshold too low or noise in training labels | Raise threshold or clean labels |
| Wrong class | Label inconsistency | Audit labels with inter-annotator test |
| Works on one strain, fails on another | Insufficient diversity | Add images from failing strain |

### Calibration check
1. **Evaluation** → Calibration tab
2. Paste confidence scores and correctness flags from a test run
3. ECE < 0.05 = well-calibrated, > 0.10 = significant miscalibration
4. If overconfident: apply temperature scaling post-training

---

## Step 6 — Benchmarking

### In CTIP
1. **Evaluation** → Benchmarks tab
2. Upload 10–20 test images (same lighting/resolution as production)
3. Run benchmark with `conf_threshold=0.35`
4. Record FPS, mean latency, VRAM

### Reference targets (RTX 4060, FP16)

| Model | imgsz | Tiled | Target FPS | Target ms/img |
|-------|-------|-------|-----------|---------------|
| yolo11n | 640 | No | ≥ 150 | ≤ 7 ms |
| yolo11s | 1280 | No | ≥ 75 | ≤ 13 ms |
| yolo11s | 4096 | Yes | ≥ 7 | ≤ 150 ms |

### Tips
- Run benchmark 3×, use average (first run has CUDA warmup overhead)
- Save results for regression tracking (new model version should not regress FPS by >20%)
- For production: test with `use_tiled=True` on 4K images if your microscope produces them

---

## Step 7 — Pipeline Builder

### In CTIP
1. **Inference** → Pipeline Builder tab
2. Build a test pipeline by dragging nodes from the left palette:

```
[Image Input] → [Model (yolo11s, conf=0.35)] → [Filter (minConf=0.4)] → [Detection Output]
                                                                        → [Stats Output]
```

3. Connect nodes by dragging from the output dot (right side) to the input dot (left side)
4. Upload a test image in the Image Input node
5. Click **Run**
6. Click **Save** → **Share** to copy a shareable URL

### Recommended pipelines to save

| Name | Config | Use case |
|------|--------|----------|
| high-recall | conf=0.20, no filter | Exploratory — find everything |
| default | conf=0.35, no filter | Standard research |
| high-precision | conf=0.60, stalked/sessile only | Final counts |
| 4k-tiled | conf=0.35, tiled=true | 4K microscopy images |

---

## Iteration cycle

After completing step 7, your model is operational. The improvement cycle is:

```
Collect failures in production
    ↓
Add failure cases to labeling queue (Annotation → Review tab)
    ↓
Re-label → export updated dataset
    ↓
Retrain (new version, same experiment group)
    ↓
Compare metrics with previous run in Training → Experiments tab
    ↓
If improved: update Models registry → run Benchmark regression check
```

Target: mAP50 ≥ 0.90 after 3 iterations with diverse training data.

---

## Scientific constraints

> CTIP never predicts cannabinoid concentrations.
> Maturity stage (Clear / Cloudy / Amber) is an observable optical property only.
> No inference about THC, CBD, or other cannabinoid content can be made from visual trichome appearance.

This constraint is enforced in all API responses (`scientific_caveat` field in maturity endpoints).

---

*See also: [[Training]], [[Inference & Analysis|Inference-and-Analysis]], [[Architecture]]*
