# Trichome Detection — Evaluation Methodology

**Version**: 1.0  
**Date**: 2026-05-26  
**Target hardware**: RTX 4060 8 GB / i5-13400F / 16 GB RAM  

This document defines the complete evaluation protocol for all models and pipelines
in the Trichome Analysis Platform. Reproducibility is a hard requirement: every
benchmark must be runnable from scratch with `GLOBAL_SEED=42`.

---

## 1. Detection Evaluation

### 1.1 Primary Metrics

| Metric | Threshold | Rationale |
|---|---|---|
| **mAP@0.5** | IoU ≥ 0.50 | Standard; matches annotation precision |
| **mAP@0.5:0.95** | IoU 0.50–0.95, step 0.05 | Localization quality |
| **Precision** | @ conf=0.35 | Calibrated operating point |
| **Recall** | @ conf=0.35 | Miss rate for trichome counting |
| **F1** | harmonic mean P/R | Single-number summary |
| **AR@100** | max 100 dets | Detection capacity |

**Operating confidence threshold**: 0.35 (default). Chosen based on Platt-calibrated
confidence curve such that calibrated precision ≥ 0.80 on the held-out validation set.

### 1.2 Per-Class Evaluation

Trichome classes and their relative frequency in the reference dataset:

| Class | Approximate Frequency | Detection Challenge |
|---|---|---|
| `stalked_glandular` | ~55% | Baseline; stalks obscured by focal plane |
| `bulbous_glandular` | ~20% | Small (< 30px); high miss rate |
| `sessile_glandular` | ~18% | Overlapping with stalked at low magnification |
| `non_glandular` | ~7% | Rare; background-like in texture |

**Required per-class reporting**: P, R, AP@0.5, AP@0.5:0.95 for each class.  
Class imbalance ≥ 3× between most and least frequent: use **weighted mAP** as
the primary summary metric, not macro-averaged.

### 1.3 Size-Stratified Evaluation

Trichome size varies with magnification and specimen preparation. Evaluate separately:

| Size Bin | Criterion | Notes |
|---|---|---|
| **Micro** | bbox area < 20×20 px | Bulbous at 4× objective |
| **Small** | 20×20 – 60×60 px | Most trichomes at 10× |
| **Medium** | 60×60 – 150×150 px | Stalked at 40× |
| **Large** | > 150×150 px | Close-up crops |

Do NOT use COCO's absolute 32²/96² thresholds — they are calibrated for natural images
and are meaningless at 1280px microscopy resolution.

### 1.4 IoU Matching Protocol

Follow COCO protocol for IoU matching:
1. Sort predictions descending by confidence
2. Greedy match to ground truth boxes (each GT matched at most once)
3. IoU threshold: 0.5 (AP@0.5) or stepped 0.05–0.95 (AP@0.5:0.95)
4. Unmatched predictions = False Positives; unmatched GT = False Negatives

Implementation: `shared/metrics/detection.py` → `match_predictions_to_gt()`

### 1.5 Tiled Inference Evaluation

For full FOV images (≥ 1280px), tiled inference with 20% overlap + WBF merging is used.
Evaluate separately:
- **Tile-level mAP**: detection quality within each tile
- **FOV-level mAP**: after WBF merging, against full-image annotations
- Report both; FOV-level is the production metric

---

## 2. Confidence Calibration Evaluation

### 2.1 Expected Calibration Error (ECE)

```
ECE = Σ_{m=1}^{M} (|B_m| / n) × |acc(B_m) - conf(B_m)|
```

- M = 15 bins (equal-width in [0, 1])
- n = total predictions
- acc(B_m) = fraction of predictions in bin m that are correct (IoU ≥ 0.5)
- conf(B_m) = mean confidence of predictions in bin m

**Quality thresholds**:

| ECE Range | Assessment |
|---|---|
| < 0.05 | Well-calibrated ✅ |
| 0.05 – 0.10 | Acceptable ⚠️ |
| > 0.10 | Poorly calibrated ❌ — apply temperature scaling |

### 2.2 Maximum Calibration Error (MCE)

```
MCE = max_{m} |acc(B_m) - conf(B_m)|
```

MCE identifies the worst-case bin. ECE can be low while MCE is high if the model
is severely miscalibrated at extreme confidence values.

**Threshold**: MCE < 0.15 required for production deployment.

### 2.3 Calibration Curve (Reliability Diagram)

Report a reliability diagram with:
- Bar per bin (overconfident = orange if conf > acc, underconfident = blue if conf < acc)
- Diagonal reference (perfect calibration)
- Confidence histogram (right panel)

Implementation: `analytics/visualization/plotter.py` → `plot_reliability_diagram_from_bins()`

### 2.4 Calibration Training Protocol

1. Split dataset: 60% train / 20% val / 20% test (stratified by class)
2. Train detection model on train split
3. Run inference on val split; collect (confidence, IoU) pairs
4. Fit temperature T via NLL minimization on val pairs: T = argmin_T NLL(val | T)
5. Evaluate ECE/MCE on test split with calibrated confidences
6. Log calibration artifacts to MLflow: `eval/ece`, `eval/mce`, `confidence_scores.npy`, `is_correct.npy`

---

## 3. Maturity Stage Evaluation

### 3.1 Classification Metrics

Maturity stages: `clear`, `cloudy`, `amber` (3-class problem).

| Metric | Calculation |
|---|---|
| **Accuracy** | Overall correct / total |
| **Weighted F1** | F1 per class weighted by support |
| **Macro F1** | Unweighted mean F1 across classes |
| **Confusion Matrix** | 3×3 matrix — critical for amber/cloudy boundary |

Report macro and weighted F1. Macro is more informative when class distributions
differ between train/test (which is expected: amber is rare at harvest time).

### 3.2 Maturity Stage Distribution Validation

**Scientific constraint**: Results must reflect observable optical properties only.
No claim about THC/CBD content is permissible. Reports must include the caveat:

> *"Maturity stage is an observable optical property of trichome head colour and
> translucency. No inference about cannabinoid concentration can be made from
> visual appearance alone."*

### 3.3 Population-Level Analysis

For whole-plant or whole-session analysis, report:
- Clear fraction (%)
- Cloudy fraction (%)
- Amber fraction (%)
- 95% confidence interval on each fraction (binomial CI, Wilson score)
- Number of trichomes scored (N)
- Maturity index: MI = cloudy_fraction + 2 × amber_fraction (0=all clear, 2=all amber)

---

## 4. Segmentation Evaluation

### 4.1 Mask Quality Metrics

| Metric | Formula | Notes |
|---|---|---|
| **mIoU** | mean intersection over union | Over all classes |
| **Dice** | 2|P∩G| / (|P|+|G|) | Shape similarity |
| **Boundary IoU** | IoU computed on ±2px boundary | Localization precision |
| **AP_mask** | Mask-level AP at IoU 0.5:0.95 | COCO instance segmentation standard |

### 4.2 SAM2 Prompt Evaluation

SAM2 is prompted with YOLO bounding boxes. Evaluate:
- Box-prompted IoU vs. point-prompted IoU
- Mask quality as a function of box localization error
- Failure rate: fraction of masks with IoU < 0.5

---

## 5. Measurement Evaluation

### 5.1 Calibration Accuracy

| Metric | Target |
|---|---|
| Scale bar detection accuracy | ±2% of known scale |
| µm/pixel uncertainty (u_rel) | < 5% relative |
| Diameter measurement repeatability (σ) | < 3 µm within session |
| Diameter measurement reproducibility | < 10 µm across sessions |

### 5.2 Ground Truth Reference

Stage micrometer with known pitch (e.g., 10 µm divisions) photographed under
identical conditions. All calibration measurements traced to this reference.

### 5.3 Uncertainty Reporting

Per GUM (ISO/IEC Guide 98-3), all measurement results must include:
- Measured value ± expanded uncertainty U (k=2, 95% CI)
- Unit (µm)
- Source of uncertainty (calibration, detection, pixel quantization)

---

## 6. Dataset Splits and Leakage Prevention

### 6.1 Split Strategy

```
Dataset
├── train/        60%   — model training only
├── val/          20%   — hyperparameter tuning, early stopping, calibration fitting
└── test/         20%   — held-out final evaluation (single use)
```

**Hard rule**: Test set is evaluated exactly ONCE per model version. No tuning
on test set under any circumstances.

### 6.2 Leakage Sources to Prevent

| Risk | Prevention |
|---|---|
| Same plant across splits | Split by plant ID, not image ID |
| Same session across splits | Split by acquisition session |
| VLM pseudo-labels in training | HITL review required before train inclusion |
| Calibration data in test | Calibration fit on val only |
| Near-duplicate frames | pHash deduplication before splitting |

### 6.3 Metadata Required Per Sample

Every training sample must have:
- `plant_id` — prevents plant-level leakage
- `session_id` — prevents session-level leakage
- `magnification` — enables stratified evaluation
- `annotation_source` — `human` | `vlm_approved` (never `vlm_raw`)
- `split` — `train` | `val` | `test`

---

## 7. Reproducibility Requirements

### 7.1 Seeds

```python
GLOBAL_SEED = 42
torch.manual_seed(GLOBAL_SEED)
numpy.random.seed(GLOBAL_SEED)
random.seed(GLOBAL_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

`benchmark = False` trades ~10% throughput for determinism. Use `benchmark = True`
only for latency benchmarks, not evaluation runs.

### 7.2 Model Checkpointing

Every evaluation must log:
- Model weights hash (SHA-256 of `.pt` file)
- Ultralytics version
- Python version
- CUDA version
- `torch.__version__`
- Dataset version (DVC commit or MD5 of dataset manifest)

### 7.3 Evaluation Run Artifacts (MLflow)

Required MLflow artifacts per evaluation run:
```
run/
├── metrics/        mAP50, mAP50-95, precision, recall, ECE, MCE
├── artifacts/
│   ├── confidence_scores.npy
│   ├── is_correct.npy
│   ├── calibration.json
│   ├── confusion_matrix.png
│   └── reliability_diagram.png
└── tags/
    ├── model_sha256
    ├── dataset_version
    └── eval_date
```

---

## 8. Benchmark Hardware Baseline (RTX 4060)

All published benchmarks in this repository are measured on:

| Component | Spec |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 8 GB GDDR6 |
| CPU | Intel Core i5-13400F (10 cores, 16 threads) |
| RAM | 16 GB DDR4 |
| OS | Ubuntu 22.04 LTS (WSL2) |
| CUDA | 12.x |
| PyTorch | 2.x with CUDA 12.x |

**Inference mode**: FP16 (half precision), single GPU, batch_size=1 (interactive),
batch_size=4 (throughput benchmark). Tiled inference at 1280px tile size, 20% overlap.

---

## 9. Reporting Checklist

Before publishing any evaluation result, verify:

- [ ] Train/val/test split sizes documented
- [ ] Leakage checks passed (plant ID, session ID)
- [ ] Seed fixed to 42 and logged
- [ ] Model weights hash logged to MLflow
- [ ] ECE ≤ 0.10 (or calibration reason documented)
- [ ] Per-class AP reported for all 4 trichome classes
- [ ] Size-stratified evaluation (micro/small/medium/large)
- [ ] Confidence operating threshold documented (default 0.35)
- [ ] Scientific caveat present in all maturity reports
- [ ] Uncertainty reported for all measurement values (GUM)
- [ ] Benchmark hardware documented
