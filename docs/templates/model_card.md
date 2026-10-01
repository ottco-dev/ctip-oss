# Model card — <name>

> Fill in every section before sharing weights. Numbers without the evaluation protocol are not meaningful.

## Model
- **Task:** detection / segmentation / morphology / maturity
- **Architecture and size:** e.g. YOLO11s, input 1280 px
- **Weights:** file, SHA-256, licence (YOLO11 derivatives are AGPL-3.0)
- **Training code:** CTIP commit hash, config file, seed (default 42)

## Training data
- Dataset card(s) and version, sessions and images used, split method (CTIP session split)
- Augmentations

## Evaluation
- **Test data:** which sessions / setups; never seen during training
- **Detection:** mAP50, mAP50-95, precision, recall — per class
- **Calibration:** ECE and reliability diagram (before / after temperature scaling)
- **Measurement:** µm error against a stage micrometer, if applicable
- **Uncertainty:** how confidence should be read; ensemble / disagreement if used
- **Runtime:** GPU, precision, image size, tiled or not, latency p50/p95, peak VRAM
  (`benchmarks/detection/gpu_detection_benchmark.py`)

## Failure cases
- Where it fails (blur, glare, small sessile heads, unusual lighting, other microscopes) — with example images

## Intended use and limits
- Optical trichome analysis on images similar to the training data
- **Not** a cannabinoid (THC/CBD) measurement, not a lab analysis, not a harvest guarantee
- Expect lower accuracy on other microscopes, lighting or magnifications until re-evaluated

## Changelog
- <date>: initial release
