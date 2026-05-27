# Small Object Detection in Trichome Microscopy

## Problem Statement

Trichome detection presents unique challenges for standard object detection pipelines:

| Challenge | Trichome-Specific Problem | Solution Used |
|---|---|---|
| Small object size | Bulbous trichomes: 10-15µm (~30-50px at typical magnification) | Tiled inference, high imgsz |
| High density | 50-200+ trichomes per field of view | WBF merging, careful NMS |
| Class overlap | Sessile vs. stalked hard to distinguish at low resolution | Morphology classifier post-processing |
| Background clutter | Plant tissue, debris | Training data diversity |
| Depth-of-field | Only part of image in focus | Focus quality scoring |

---

## Tiled Inference

Standard YOLO input: 640px or 1280px.
Microscopy image: 4K (3840×2160) or higher.
Naive resize: small trichomes become too small for detection.

**Solution: Sliding window tiled inference**
- Tile size: 1280px × 1280px
- Overlap: 20% (256px)
- Border replication padding: eliminates edge artifacts
- WBF merging: handles duplicate detections across tile boundaries

Performance (RTX 4060):
- 4K image, tile=1280, overlap=0.2 → ~6 tiles → ~150ms total
- 1080p image, tile=1280 → 1-2 tiles → ~30ms

**WBF vs. NMS for tile merging:**
WBF (Solovyev et al. 2021) is preferred over standard NMS because:
- NMS discards all but one box per cluster → loses localization information
- WBF merges overlapping boxes → more accurate final coordinates
- WBF handles confidence scores better (weighted average)

Reference: Solovyev, R. et al. (2021). Weighted boxes fusion: Ensembling boxes
from different object detection models. *Image and Vision Computing* 107:104117.

---

## Ensemble Detection

Running 2-3 models and fusing predictions:
- YOLO11n (fast, high recall)
- YOLO11s (balanced)
- RTMDet-m (different architecture, complementary errors)

WBF fusion across models provides:
1. Better recall (catches detections any model finds)
2. Better precision (consensus boxes only)
3. Epistemic uncertainty estimate (inter-model disagreement)

VRAM constraint on RTX 4060: Cannot run all models simultaneously.
Sequential inference + CPU-side WBF merge.

---

## Anchor-Free Detection

YOLO11 uses anchor-free detection (Task-Aligned Learning).
This is beneficial for trichomes because:
1. Bulbous trichomes are nearly circular — no aspect ratio bias
2. Capitate-stalked can be tall and thin — no anchor mismatch
3. Eliminates anchor tuning step (which requires dataset analysis)

---

## Data Requirements

For reliable detection (mAP50 > 0.80):
- Minimum: ~500 annotated images with diverse conditions
- Recommended: 2,000-5,000 images
- Split: 70% train, 20% val, 10% test
- Diversity: multiple strains, microscopes, magnifications, focus levels

Class imbalance typical:
- Capitate-stalked: ~60% (most common on calyxes)
- Capitate-sessile: ~25%
- Bulbous: ~10%
- Non-glandular: ~5%

Handling imbalance:
- Weighted sampling (training/samplers/weighted_sampler.py)
- Focal loss (training/losses/focal_loss.py)
- YOLO's built-in class-weighted BCE

---

## Key References

1. **Jocher, G. et al. (2023)**. Ultralytics YOLO. GitHub.
   https://github.com/ultralytics/ultralytics

2. **Solovyev, R. et al. (2021)**. Weighted boxes fusion.
   *Image and Vision Computing* 107:104117.
   DOI: 10.1016/j.imavis.2021.104117

3. **Ren, S. et al. (2015)**. Faster R-CNN: Towards Real-Time Object Detection.
   *NeurIPS* 2015. arXiv:1506.01497

4. **Li, C. et al. (2022)**. YOLOv6: A Single-Stage Object Detection Framework.
   arXiv:2209.02976

5. **Cheng, B. et al. (2021)**. Boundary IoU: Improving Object-Centric Image Segmentation.
   *CVPR 2021*. DOI: 10.1109/CVPR46437.2021.00500
