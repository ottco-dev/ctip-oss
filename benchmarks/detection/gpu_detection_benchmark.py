"""
benchmarks.detection.gpu_detection_benchmark — end-to-end detection runtime on the GPU.

Measures the real CTIP detection pipeline (preprocessing, YOLO11, tiling, filtering) on the GPU:
  - latency p50 / p95 and throughput
  - peak VRAM (torch allocator) and process RAM
  - FP16 vs FP32
  - full image vs tiled inference (1280 px tiles, 20 % overlap) on a 4K camera frame

RUNTIME ONLY. The weights are the generic COCO ``yolo11s.pt`` (no trichome model is released yet), so detection
counts and accuracy are meaningless here; the network cost is the same as for a fine-tuned YOLO11s.

Images are synthetic microscopy-like frames (seed 42) so the run is reproducible without a dataset.

Usage:
    python benchmarks/detection/gpu_detection_benchmark.py [--runs 30] [--weights yolo11s.pt]
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

GLOBAL_SEED = 42


def synthetic_frame(h: int, w: int, seed: int = GLOBAL_SEED) -> np.ndarray:
    """Bright-field-like background with round glandular-head blobs and sensor noise (RGB uint8)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.empty((h, w, 3), np.float32)
    img[...] = (200 + 20 * yy[..., None] / h - 10 * xx[..., None] / w)
    for _ in range(max(30, h * w // 40_000)):
        cy, cx, r = rng.integers(0, h), rng.integers(0, w), rng.integers(8, 40)
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
        img[mask] = rng.uniform(120, 250, 3)
    img += rng.normal(0, 4, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def measure(fn, runs: int, warmup: int = 3) -> dict:
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return {
        "p50_ms": round(statistics.median(times), 2),
        "p95_ms": round(times[min(len(times) - 1, int(0.95 * len(times)))], 2),
        "mean_ms": round(statistics.fmean(times), 2),
        "fps": round(1000 / statistics.fmean(times), 2),
        "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1),
        "runs": runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--weights", default="yolo11s.pt", help="YOLO weights (downloaded by Ultralytics if missing)")
    parser.add_argument("--output-dir", default="benchmarks/detection")
    args = parser.parse_args()

    import psutil
    import torch

    if not torch.cuda.is_available():
        sys.exit("CUDA GPU required for this benchmark")

    from detection.application.detect_pipeline import DetectionPipeline, PipelineConfig
    from detection.domain.detector import DetectionConfig
    from detection.infrastructure.yolo_backend import YOLODetector

    weights = Path(args.weights)
    if not weights.exists():                       # generic Ultralytics asset, e.g. yolo11s.pt
        from ultralytics.utils.downloads import attempt_download_asset

        weights = Path(attempt_download_asset(str(weights)))
    detector = YOLODetector(model_id="yolo11s-coco", weights_path=weights, device="cuda:0")
    detector.load()

    frames = {"1280x1280": synthetic_frame(1280, 1280), "3840x2160": synthetic_frame(2160, 3840)}
    scenarios = [
        ("1280x1280", False, True), ("1280x1280", False, False),
        ("3840x2160", False, True), ("3840x2160", True, True), ("3840x2160", True, False),
    ]
    results = []
    for size, tiled, fp16 in scenarios:
        cfg = PipelineConfig(detection=DetectionConfig(use_fp16=fp16), use_tiling=tiled, tile_size=1280, tile_overlap=0.2)
        pipe = DetectionPipeline(detector=detector, config=cfg)
        frame = frames[size]
        r = measure(lambda pipe=pipe, frame=frame: pipe.run(frame, image_id="bench"), args.runs)
        r.update(image=size, tiled=tiled, precision="fp16" if fp16 else "fp32")
        results.append(r)
        print(f"{size:>10} {'tiled' if tiled else 'full ':>5} {r['precision']}: p50 {r['p50_ms']:7.1f} ms  "
              f"p95 {r['p95_ms']:7.1f} ms  {r['fps']:6.1f} FPS  peak VRAM {r['peak_vram_mb']:7.1f} MB")

    report = {
        "benchmark": "gpu_detection",
        "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "note": "runtime only - generic COCO yolo11s weights, synthetic frames (seed 42)",
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cpu": platform.processor() or platform.machine(),
        "process_rss_mb": round(psutil.Process().memory_info().rss / 2**20, 1),
        "results": results,
    }
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"results_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(report, indent=2))
    print(f"\nRAM (process RSS): {report['process_rss_mb']} MB\nSaved → {path}")


if __name__ == "__main__":
    main()
