# Benchmark Design for Trichome Detection Models

## Overview

This document describes the evaluation methodology for trichome detection and
classification models. A rigorous benchmark must:
1. Be reproducible (fixed seeds, versioned data)
2. Report uncertainty (bootstrap confidence intervals, not just point estimates)
3. Use domain-appropriate metrics
4. Be honest about limitations

---

## Dataset Splits

### Standard Split
```
70% train / 20% val / 10% test
```

The test split must:
- Never be seen during training OR hyperparameter tuning
- Be held out until final evaluation only
- Represent the deployment distribution (multiple strains, microscopes, conditions)

### Cross-Validation (for small datasets < 500 images)
5-fold stratified cross-validation with class-balanced folds.

```
Fold 1: train=[2,3,4,5], val=[1]
Fold 2: train=[1,3,4,5], val=[2]
...
```

Report mean ± std across folds.

---

## Detection Metrics

### Primary: mAP50 (COCO-style)
- IoU threshold: 0.50
- Precision-recall curve area
- Standard for object detection benchmarks

### Secondary: mAP50-95
- IoU averaged over thresholds [0.50, 0.55, ..., 0.95]
- More strict, better correlates with localization quality
- Trichomes are small — 0.75+ IoU is high bar

### Per-Class AP
Report separately for: capitate-stalked, capitate-sessile, bulbous, non-glandular

Reason: class imbalance means aggregate mAP can mask poor per-class performance.

### Small Object Metrics
Trichomes < 32px²: report AP_small separately (COCO definition).

---

## Calibration Metrics

Model confidence must be meaningful for uncertainty estimation in active learning.

### Expected Calibration Error (ECE)
```
ECE = Σ_b |B_b|/n × |acc(b) - conf(b)|
```

Target: ECE < 0.05 (5% calibration error)

### Maximum Calibration Error (MCE)
Worst-case bin error. Important for safety-critical use.

### Reliability Diagram
Plot observed accuracy vs. predicted confidence across 15 bins.
Well-calibrated model: diagonal line.

Reference: Guo et al. (2017). On Calibration of Modern Neural Networks. ICML.

---

## Evaluation Protocol

### Step 1: Data Preparation
```bash
python -m detection.benchmarks.benchmark_detection \
    --dataset data/trichome_dataset/data.yaml \
    --split test \
    --seed 42
```

### Step 2: Model Inference
```
- Confidence threshold: 0.001 (use full PR curve)
- IoU threshold: 0.45 (NMS)
- Image size: 1280px (full resolution)
- Tiled inference: enabled for 4K images (tile=1280, overlap=0.2)
```

### Step 3: Metric Computation
```python
from shared.metrics.detection_metrics import compute_detection_metrics

result = compute_detection_metrics(
    predictions=predictions,
    ground_truth=ground_truth,
    iou_threshold=0.50,
    bootstrap_iters=1000,
)

print(f"mAP50: {result.map50:.4f} ± {result.map50_ci[1] - result.map50:.4f}")
```

### Step 4: Bootstrap CI
- 1000 bootstrap iterations over test set
- Report 95% confidence interval
- Example: mAP50 = 0.842 ± 0.018 [95% CI: 0.806, 0.878]

---

## Maturity Analysis Evaluation

### Classification Task
4 classes: clear, cloudy, amber, mixed

### Metrics
1. **Accuracy**: fraction correctly classified
2. **Cohen's κ**: agreement corrected for chance
3. **Macro F1**: average F1 across all classes (handles imbalance)
4. **Confusion Matrix**: which classes are confused (cloudy↔amber is common)

### Important Limitations
Maturity classification ground truth is inherently subjective.
Inter-annotator agreement (Cohen's κ) among human annotators on the same
images typically ranges 0.55–0.75 (moderate to substantial).

A model achieving κ ≈ 0.70 matches human-level agreement.
Report model vs. ground truth AND human vs. ground truth for context.

---

## Segmentation Evaluation

### Primary: Mask IoU (instance-level)
Average IoU between predicted and ground truth masks.

### Secondary: Boundary IoU
Penalizes boundary errors more than interior fill errors.
More relevant for trichome head diameter measurement accuracy.

Reference: Cheng et al. (2021). Boundary IoU. CVPR 2021.

### Size Measurement Accuracy
When calibration is available (px→µm):
- Report mean absolute error (MAE) in µm for head diameter
- Compare against ground truth from scanning electron microscopy (SEM)

---

## Hardware-Specific Benchmarks (RTX 4060)

Report inference throughput for production planning:

| Model | Image Size | Tiled | ms/img | FPS | VRAM |
|-------|-----------|-------|--------|-----|------|
| YOLO11n | 640 | No | ~6ms | ~150 | 0.6 GB |
| YOLO11n | 1280 | No | ~11ms | ~90 | 0.8 GB |
| YOLO11s | 1280 | No | ~13ms | ~75 | 1.2 GB |
| YOLO11s | 4K | Yes (6 tiles) | ~150ms | ~7 | 1.2 GB |
| YOLO11s+SAM2 | 1280 | No | ~80ms | ~12 | 5.0 GB |

*Measured on RTX 4060 8GB, FP16, batch=1*

---

## Reporting Template

```
Model: [name]
Dataset: [N] images, [M] instances
Evaluation date: YYYY-MM-DD
Seed: 42

Detection Results (test set):
  mAP50: X.XXX [95% CI: X.XXX, X.XXX]  n_bootstrap=1000
  mAP50-95: X.XXX
  Per-class AP50:
    capitate-stalked: X.XXX
    capitate-sessile: X.XXX
    bulbous: X.XXX
    non-glandular: X.XXX

Calibration:
  ECE: X.XXX (target < 0.05)
  MCE: X.XXX

Inference (RTX 4060):
  Throughput: XX ms/image (FP16, batch=1)
  VRAM: X.X GB peak
```

---

## Anti-Patterns to Avoid

1. **Reporting val metrics as test metrics**: Always use the held-out test split for final numbers.
2. **No confidence intervals**: Single numbers without CI are uninformative.
3. **Choosing threshold by accuracy**: Calibrate threshold on val, report on test.
4. **Ignoring small-object performance**: Bulbous trichomes are rare AND small — report separately.
5. **Claiming THC correlation**: Visual maturity analysis does NOT allow THC quantification.

---

## References

1. **Lin et al. (2014)**. Microsoft COCO: Common Objects in Context.
   ECCV 2014. arXiv:1405.0312.

2. **Guo et al. (2017)**. On Calibration of Modern Neural Networks.
   ICML 2017. arXiv:1706.04599.

3. **Cheng et al. (2021)**. Boundary IoU: Improving Object-Centric Image Segmentation.
   CVPR 2021. DOI: 10.1109/CVPR46437.2021.00500.

4. **Efron & Tibshirani (1993)**. An Introduction to the Bootstrap.
   Chapman & Hall.

5. **Jocher et al. (2023)**. Ultralytics YOLO.
   https://github.com/ultralytics/ultralytics
