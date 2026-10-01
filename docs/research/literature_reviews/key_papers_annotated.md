# Annotated Bibliography — Trichome Analysis System

Curated references organized by topic. Each entry includes:
- Full citation with DOI
- Relevance to this system
- Key findings used in implementation
- Limitations or caveats

---

## 1. Object Detection

### 1.1 YOLO Family

**Jocher, G. et al. (2023)**. Ultralytics YOLO.
GitHub: https://github.com/ultralytics/ultralytics
*Citation: Jocher, G. et al. (2023). YOLO by Ultralytics.*

- **Relevance**: Primary detection backbone. YOLO11 used for trichome detection.
- **Key findings**:
  - Anchor-free detection eliminates aspect ratio bias (good for bulbous trichomes)
  - Task-Aligned Learning (TAL) improves small object detection
  - 1280px input significantly improves detection of structures 30-50px
- **Implementation**: `detection/infrastructure/yolo_backend.py`

---

### 1.2 Weighted Boxes Fusion

**Solovyev, R., Wang, W. & Gabruseva, T. (2021)**.
Weighted boxes fusion: Ensembling boxes from different object detection models.
*Image and Vision Computing* 107:104117.
DOI: 10.1016/j.imavis.2021.104117

- **Relevance**: WBF merging for tiled inference + model ensemble.
- **Key findings**:
  - WBF outperforms NMS and soft-NMS for ensemble fusion
  - Preserves localization information lost by NMS
  - Handles confidence scores properly via weighted averaging
- **Implementation**: `detection/domain/ensemble.py` (WBF merge step)

---

### 1.3 Focal Loss

**Lin, T.Y., Goyal, P., Girshick, R., He, K. & Dollár, P. (2017)**.
Focal loss for dense object detection.
*IEEE International Conference on Computer Vision (ICCV)* 2017. pp. 2980-2988.
DOI: 10.1109/ICCV.2017.324. arXiv:1708.02002.

- **Relevance**: Class imbalance in trichome dataset (stalked 60%, bulbous 10%).
- **Key findings**:
  - γ=2.0 reduces loss weight of easy negatives by 100×
  - α=0.25 compensates for foreground/background imbalance
  - Critical for rare class (non-glandular trichomes ~5%)
- **Implementation**: `training/losses/focal_loss.py`

---

## 2. Instance Segmentation

### 2.1 Segment Anything Model 2

**Ravi, N. et al. (2024)**.
SAM 2: Segment Anything in Images and Videos.
arXiv:2408.00714.

- **Relevance**: Instance segmentation backend (SAM2-tiny).
- **Key findings**:
  - Memory-efficient streaming architecture
  - SAM2-tiny: 38.9M params, ~3.8 GB VRAM on RTX 4060
  - Multimask output for ambiguous cases (merged heads)
  - Point and box prompts enable detect-then-segment pipeline
- **Implementation**: `segmentation/infrastructure/sam2_backend.py`

---

### 2.2 MobileSAM

**Zhang, C., Han, D., Qiao, Y., Kim, J.U., Bae, S.H., Lee, S. & Hong, C.S. (2023)**.
Faster Segment Anything: Towards Lightweight SAM for Mobile Applications.
arXiv:2306.14289.

- **Relevance**: CPU-capable fallback when RTX 4060 is occupied.
- **Key findings**:
  - TinyViT encoder: 5M params (vs SAM's ViT-H 636M)
  - Identical SAM decoder: same output format
  - ~12ms on RTX 4060 vs SAM2-tiny ~40ms
  - Accuracy loss acceptable for quick preview / CPU machines
- **Implementation**: `segmentation/infrastructure/mobile_sam.py`

---

### 2.3 Tversky Loss

**Salehi, S.S.M., Erdogmus, D. & Gholipour, A. (2017)**.
Tversky Loss Function for Image Segmentation Using 3D Fully Convolutional Deep Networks.
*MLMI 2017.* arXiv:1706.05721.

**Abraham, N. & Khan, N.M. (2019)**.
A Novel Focal Tversky Loss Function With Improved Attention U-Net for Lesion Segmentation.
*ISBI 2019.* arXiv:1810.07842.

- **Relevance**: Trichome boundary segmentation with class imbalance.
- **Key findings**:
  - β > α penalizes missed segments more than false positives
  - Recommended α=0.3, β=0.7 for biomedical segmentation
  - Focal variant focuses on hard examples (partially occluded trichomes)
- **Implementation**: `training/losses/tversky_loss.py`

---

## 3. Model Calibration

### 3.1 Temperature Scaling

**Guo, C., Pleiss, G., Sun, Y. & Weinberger, K.Q. (2017)**.
On Calibration of Modern Neural Networks.
*Proceedings of the 34th ICML*. pp. 1321-1330.
arXiv:1706.04599.

- **Relevance**: Calibration for uncertainty estimation in active learning.
- **Key findings**:
  - Modern deep networks are systematically overconfident
  - Temperature scaling: single learned parameter T divides logits
  - Post-hoc calibration: doesn't change predictions, only confidence
  - ECE < 0.05 achievable from ECE > 0.15 pre-calibration
- **Implementation**: `detection/domain/confidence_calibrator.py`

---

### 3.2 Calibration Metrics

**Naeini, M.P., Cooper, G. & Hauskrecht, M. (2015)**.
Obtaining Well Calibrated Probabilities Using Bayesian Binning.
*AAAI 2015*. pp. 2901-2907.

- **Relevance**: ECE and reliability diagram implementation.
- **Implementation**: `shared/metrics/calibration_metrics.py`

---

## 4. Active Learning

### 4.1 Uncertainty Sampling

**Lewis, D.D. & Gale, W.A. (1994)**.
A Sequential Algorithm for Training Text Classifiers.
*SIGIR 1994*. pp. 3-12.
DOI: 10.1007/978-1-4471-2099-5_1.

- **Relevance**: Foundation for uncertainty-based sampling strategies.
- **Key findings**:
  - Least confidence, margin sampling, entropy sampling are equivalent in many settings
  - Entropy-based sampling is most generally applicable for multi-class

---

### 4.2 Dataset Drift

**Rabanser, S., Günnemann, S. & Lipton, Z. (2019)**.
Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift.
*NeurIPS 2019*. arXiv:1810.11953.

- **Relevance**: Drift detection before adding new data to training set.
- **Key findings**:
  - MMD and KS tests most reliable for feature-space drift detection
  - Prediction-based drift detection complements feature-based methods
  - Univariate tests (KS per feature) often outperform multivariate MMD
- **Implementation**: `active_learning/analysis/drift.py`

---

## 5. VLM-Assisted Labeling

### 5.1 Moondream-2

**Moondream (2024)**. Moondream: Small Vision Language Model.
HuggingFace: https://huggingface.co/vikhyatk/moondream2

- **Relevance**: Primary VLM for auto-labeling (~2.1 GB VRAM 4-bit).
- **Key findings**:
  - Optimized for embedded/edge deployment
  - Good performance on structured visual Q&A
  - Answer question + encode image API is efficient

---

### 5.2 Florence-2

**Xiao, B., Wu, H., Xu, W. et al. (2023)**.
Florence-2: Advancing a Unified Representation for a Variety of Vision Tasks.
arXiv:2311.06242.

- **Relevance**: Structured detection + captioning tasks (~3.5 GB VRAM fp16).
- **Key findings**:
  - Native task tokens: <CAPTION>, <OVD>, <REGION_TO_DESCRIPTION>
  - Multi-task training on FLD-5B dataset
  - Strong open-vocabulary detection capability

---

## 6. Trichome Biology

### 6.1 Cannabinoid Correlation

**ElSohly, M.A. & Slade, D. (2005)**.
Chemical constituents of marijuana: The complex mixture of natural cannabinoids.
*Life Sciences* 78(5):539-548.
DOI: 10.1016/j.lsc.2005.09.011.

- **Relevance**: THC→CBN oxidation pathway (amber coloration mechanism).
- **Key findings**:
  - THC degrades to CBN via photo-oxidative dehydrogenation
  - UV exposure + heat → 2-step oxidation mechanism
  - Amber coloration correlates with oxidative degradation (not THC content)
- **Scientific caveat**: Amber = degradation signal, NOT THC% predictor.

**Elzinga, S., Fischedick, J., Bassetti, R. & Raber, J.C. (2015)**.
Cannabinoids and terpenes as chemotaxonomic markers in cannabis.
*Natural Products Chemistry & Research* 3:181.
DOI: 10.4172/2329-6836.1000181.

- **Key findings**:
  - Identical visual appearance → 10-20% THC difference between strains
  - Confirms: visual analysis alone CANNOT determine cannabinoid content

---

### 6.2 Maturity Markers

**Fischedick, J.T. et al. (2010)**.
Metabolic fingerprinting of Cannabis sativa L., cannabinoids and terpenoids
for chemotaxonomic and drug standardization purposes.
*Phytochemistry* 71(17-18):2058-2073.
DOI: 10.1016/j.phytochem.2010.09.015.

- **Relevance**: Understanding what the "cloudy" visual stage actually reflects.
- **Key findings**:
  - Opaque appearance from Mie scattering of dense resin droplets
  - THCA crystallization increases refractive index contrast
  - Does NOT directly measure concentration — only optical density

---

## 7. Measurement and Calibration

### 7.1 Pixel-to-Micron Calibration

Standard methodology for microscopy calibration:

- Stage micrometer: ruled glass slide, 100µm divisions
- Objective-specific calibration: must be repeated for each magnification
- Reference: ISO 10012 (Measurement management systems)

Practical implementations in this system:
- `measurement/calibration/stage_micrometer.py`
- Pre-set profiles for common objectives (4×, 10×, 20×, 40×, 100×)

---

## 8. Data Augmentation

**Buslaev, A., Iglovikov, V.I., Khvedchenya, E., Parinov, A., Druzhinin, M. & Kalinin, A.A. (2020)**.
Albumentations: Fast and Flexible Image Augmentations.
*Information* 11(2):125.
DOI: 10.3390/info11020125.

- **Relevance**: Domain-specific augmentations for trichome images.
- **Key findings**:
  - Conservative augmentation required: color semantics (hue) encode maturity information
  - HSV hue shift must be ≤ 8° to avoid changing "cloudy" → "amber" artificially
  - Rotation (180°), flip, mild brightness OK for trichome training
- **Implementation**: `training/augmentation/microscopy_aug.py`

---

## 9. Weighted Sampling

**Cui, Y., Jia, M., Lin, T.Y., Song, Y. & Belongie, S. (2019)**.
Class-Balanced Loss Based on Effective Number of Samples.
*CVPR 2019*. arXiv:1901.05555.
DOI: 10.1109/CVPR.2019.00949.

- **Relevance**: Handling trichome class imbalance (stalked 60%, non-glandular 5%).
- **Key findings**:
  - Effective sample number (1-β^n)/(1-β) is more stable than pure inverse frequency
  - β=0.9999 approximates inverse frequency for large datasets
  - Significantly outperforms plain re-weighting on long-tail distributions
- **Implementation**: `training/samplers/weighted_sampler.py`

---

## 10. Focus Quality Metrics

**Pertuz, S., Puig, D. & Garcia, M.A. (2013)**.
Analysis of focus measure operators for shape-from-focus.
*Pattern Recognition* 46(5):1415-1432.
DOI: 10.1016/j.patcog.2012.11.011.

- **Relevance**: Comprehensive survey of focus metrics; basis for our multi-metric composite.
- **Key findings**:
  - Laplacian variance (LVAR) is simple and robust for general microscopy
  - Tenengrad outperforms LVAR for phase-contrast and bright-field at low magnification
  - Modified Laplacian (MLAP) was proposed specifically for electron microscopy
  - No single metric dominates across all conditions — composite is preferred
  - Brenner is fastest computationally (sum of squared differences)
- **Implementation**: `focus/metrics/laplacian.py`, `focus/metrics/tenengrad.py`, `focus/metrics/composite.py`

---

**Vollath, D. (1987)**.
Automatic focusing by correlative methods.
*Journal of Microscopy* 147(3):279-288.
DOI: 10.1111/j.1365-2818.1987.tb02839.x.

- **Relevance**: Vollath F4 focus metric — self-correlation based.
- **Key findings**:
  - F4 metric: F4 = Σ[I(r) × I(r+δ)] - Σ[I(r) × I(r+2δ)]
  - Less sensitive to shot noise than gradient methods
  - Good for low-contrast brightfield microscopy
- **Implementation**: `focus/metrics/fft_metrics.py` (vollath_f4)

---

## 11. Perceptual Hashing for Video Deduplication

**Zauner, C. (2010)**.
Implementation and Benchmarking of Perceptual Image Hash Functions.
Bachelor Thesis, Upper Austria University of Applied Sciences.

- **Relevance**: DCT-based pHash used in video frame deduplication.
- **Key findings**:
  - DCT-based pHash outperforms color histogram and gradient hashes
  - 64-bit hash (8×8 DCT block) robust to JPEG compression, minor brightness changes
  - Hamming threshold 10/64 (~15%) captures near-duplicates without false matches
- **Implementation**: `video_pipeline/domain/hasher.py` (perceptual_hash, deduplicate_frames)

---

## 12. Uncertainty Propagation (GUM)

**Joint Committee for Guides in Metrology (2008)**.
JCGM 100:2008 — Evaluation of measurement data: Guide to the Expression
of Uncertainty in Measurement (GUM).
BIPM, Sèvres, France.
https://www.bipm.org/utils/common/documents/jcgm/JCGM_100_2008_E.pdf

- **Relevance**: Foundation for all uncertainty propagation in the measurement pipeline.
- **Key findings**:
  - Combined standard uncertainty: u_c = √(Σ (∂f/∂xi)² × u(xi)²)
  - Expanded uncertainty: U = k × u_c (k=2 for 95% CI, assuming normal distribution)
  - Type A: statistical evaluation (repeated measurements)
  - Type B: other means (manufacturer spec, calibration certificate)
- **Implementation**: `measurement/domain/propagation.py`

---

## 13. Optical Flow for Motion Estimation

**Lucas, B.D. & Kanade, T. (1981)**.
An iterative image registration technique with an application to stereo vision.
*IJCAI 1981* 81(1):674-679.

- **Relevance**: Lucas-Kanade sparse optical flow used in video pipeline motion estimation.
- **Key findings**:
  - Sparse tracking via iterative Gauss-Newton minimization
  - Assumes constant intensity, small displacements, spatial coherence (aperture problem)
  - Works well for stage motion estimation (large coherent translation, minimal deformation)
  - ~200 feature points sufficient for robust RANSAC affine estimation
- **Implementation**: `video_pipeline/domain/motion.py` (estimate_motion)

---

## 14. Noise Estimation

**Immerkaer, J. (1996)**.
Fast noise variance estimation.
*Computer Vision and Image Understanding* 64(2):300-302.
DOI: 10.1006/cviu.1996.0060.

- **Relevance**: Laplacian-based noise estimator used in video frame quality scoring.
- **Key findings**:
  - σ_noise = (1/36) × Σ |ΔI| with specific convolution kernel
  - Robust to image content (structures subtracted by Laplacian)
  - Fast: single convolution pass
- **Implementation**: `video_pipeline/domain/scorer.py` (_score_noise)

---

## 15. Active Learning

**Gal, Y. & Ghahramani, Z. (2016)**.
Dropout as a Bayesian approximation: Representing model uncertainty in deep learning.
*International Conference on Machine Learning (ICML)* 2016. pp. 1050-1059.
arXiv:1506.02142.

- **Relevance**: MC Dropout as a tractable epistemic uncertainty estimator without ensembles.
- **Key findings**:
  - Running T stochastic forward passes with dropout active approximates Bayesian inference
  - Variance across passes estimates epistemic (model) uncertainty
  - Predictive entropy H[y|x,D] decomposes into aleatoric + epistemic terms
  - Works at T=20 passes with acceptable accuracy; T=50 used for high-stakes decisions
  - Drawback: only valid when model was trained with dropout; YOLO11 lacks explicit dropout
- **Implementation**: `active_learning/sampling/uncertainty.py` (mc_dropout_uncertainty)
- **Caveat**: For YOLO detectors, we fall back to prediction entropy as uncertainty proxy

---

**Kirsch, A., van Amersfoort, J. & Gal, Y. (2019)**.
BatchBALD: Efficient and diverse batch acquisition for deep Bayesian active learning.
*Advances in Neural Information Processing Systems (NeurIPS)* 32.
arXiv:1906.08158.

- **Relevance**: BALD (Bayesian Active Learning by Disagreement) — information gain criterion.
- **Key findings**:
  - BALD score: I(y; θ | x, D) = H[y|x,D] - E_{θ~q}[H[y|x,θ]]
  - Selects samples that maximise mutual information between prediction and model parameters
  - BatchBALD extends to correlated batch selection (avoids redundant samples)
  - For trichome data: BALD identifies morphologically ambiguous samples (cloudy–amber boundary)
- **Implementation**: `active_learning/sampling/disagreement.py` (bald_score in DisagreementScore)

---

**Lewis, D.D. & Gale, W.A. (1994)**.
A sequential algorithm for training text classifiers.
*ACM SIGIR Conference on Research and Development in Information Retrieval*. pp. 3-12.
DOI: 10.1007/978-1-4471-2099-5_1.

- **Relevance**: Least-confidence sampling — the simplest active learning baseline.
- **Key findings**:
  - Selects samples where max class probability is lowest: LC = 1 - max_k P(y_k|x)
  - Computationally trivial; no repeated inference required
  - Outperforms random sampling by 2-5× on benchmark tasks
  - For trichome data: effective for selecting ambiguous maturity stage samples
- **Implementation**: `active_learning/sampling/uncertainty.py` (compute_least_confidence)

---

## 16. Confidence Calibration

**Platt, J.C. (1999)**.
Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods.
*Advances in Large Margin Classifiers* 10(3):61-74.

- **Relevance**: Platt scaling (sigmoid calibration) for converting SVM scores to probabilities.
- **Key findings**:
  - Fits sigmoid P(y=1|f) = 1/(1 + exp(Af + B)) on held-out data
  - Requires only 2 parameters; fast to fit even with small calibration sets
  - Extended to multi-class via one-vs-rest or softmax post-hoc
  - Works well when base model is approximately monotone in probability
- **Implementation**: `detection/domain/confidence_calibrator.py` (PlattScalingCalibrator)

---

**Guo, C., Pleiss, G., Sun, Y. & Weinberger, K.Q. (2017)**.
On calibration of modern neural networks.
*International Conference on Machine Learning (ICML)* 2017. pp. 1321-1330.
arXiv:1706.04599.

- **Relevance**: Temperature scaling — primary calibration method for YOLO detection confidence.
- **Key findings**:
  - Modern neural networks are overconfident: high confidence ≠ high accuracy
  - Temperature scaling: p̂ = softmax(z/T), single scalar T fit on validation set
  - Minimises NLL on validation set; does not change argmax predictions
  - ECE (Expected Calibration Error) as primary calibration metric
  - ECE = Σ_m |acc(B_m) - conf(B_m)| × |B_m|/n
  - 15 bins with equal spacing in [0,1] used throughout this system
- **Implementation**: `detection/domain/confidence_calibrator.py`, `shared/metrics/calibration.py`
- **Note**: `Confidence` value object carries calibration state; raw scores never reported as probabilities

---

## 17. Dataset Drift Detection

**Gretton, A., Borgwardt, K.M., Rasch, M.J., Schölkopf, B. & Smola, A. (2012)**.
A kernel two-sample test.
*Journal of Machine Learning Research* 13(25):723-773.
https://jmlr.org/papers/v13/gretton12a.html.

- **Relevance**: Maximum Mean Discrepancy (MMD) for detecting distribution shift in trichome datasets.
- **Key findings**:
  - MMD²(X,Y) = E[k(x,x')] - 2E[k(x,y)] + E[k(y,y')] with RBF kernel k
  - Unbiased estimator: O(n²) but practical for n ≤ 1000 feature vectors
  - Consistent test: power → 1 as n → ∞ for any fixed alternative
  - Bandwidth h = median pairwise distance (median heuristic) works in practice
- **Implementation**: `active_learning/analysis/drift.py` (compute_mmd)
- **Threshold**: MMD > 0.05 triggers drift warning in this system; 0.10 = severe

---

## 18. Semi-Automatic Annotation

**Bearman, A., Russakovsky, O., Ferrari, V. & Fei-Fei, L. (2016)**.
What's the point: Semantic segmentation with point supervision.
*European Conference on Computer Vision (ECCV)* 2016. pp. 549-565.
arXiv:1506.02106.

- **Relevance**: Point-annotation workflow for fast dataset building under annotation budget.
- **Key findings**:
  - Point clicks are 12× faster than full bounding boxes, 79× faster than pixel masks
  - Semi-automated: point annotation + model-guided mask completion
  - Active learning reduces annotation cost by 3-5× on object detection tasks
- **Implementation context**: VLM pre-labeling + HITL review is our equivalent strategy
- **Caveat**: Trichome boundaries require precise masks for morphology analysis; point annotation insufficient for production data

---

## 19. Evaluation of Small Object Detection

**Zhu, C., He, Y. & Savvides, M. (2019)**.
Feature selective anchor-free module for single-shot object detection.
*IEEE Conference on Computer Vision and Pattern Recognition (CVPR)* 2019. pp. 840-849.
DOI: 10.1109/CVPR.2019.00093. arXiv:1903.00621.

- **Relevance**: Benchmark methodology for small-object detectors (trichomes: 10-100px diameter).
- **Key findings**:
  - COCO "small" definition (area < 32²) inadequate for microscopy: use relative size (< 2% of FOV)
  - Separate mAP evaluation by object size critical for trichome analysis
  - Recall@IoU=0.5 vs @IoU=0.75 reveals localization vs classification errors
  - AR (Average Recall) at maxDets=1/10/100 reveals detection capacity limits
- **Implementation**: `shared/metrics/detection.py` (compute_map, size-stratified evaluation)

---

*Bibliography total: 19 entries (15 original + 4 new: active learning, calibration, drift, annotation)*
*Last updated: 2026-05-26*
