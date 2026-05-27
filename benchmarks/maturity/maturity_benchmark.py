"""
benchmarks/maturity/maturity_benchmark.py — Maturity pipeline throughput benchmark.

Measures per-function latency and FPS for all maturity analysis components.
Designed for RTX 4060 / i5-13400F reference hardware.

Functions benchmarked:
  --- Color features ---
  - extract_color_features
  - rule_based_maturity_estimate
  --- Texture features ---
  - compute_lbp
  - compute_glcm_features
  - compute_gabor_features
  - compute_shannon_entropy
  - extract_texture_features
  --- Translucency ---
  - estimate_translucency
  --- Degradation ---
  - detect_color_degradation
  - detect_structural_collapse
  - detect_texture_irregularity
  - assess_degradation
  --- Pipeline ---
  - MaturityPipeline.analyze_crop (rule-based path)
  - MaturityPipeline.analyze_batch (N=10 instances)

Output:
  - Terminal table
  - benchmarks/maturity/results_YYYYMMDD_HHMMSS.json

Usage:
    python benchmarks/maturity/maturity_benchmark.py
    python benchmarks/maturity/maturity_benchmark.py --n 200 --size 64
    python benchmarks/maturity/maturity_benchmark.py --warmup 10

Reproducibility:
    GLOBAL_SEED=42, fixed numpy RNG for all synthetic images.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ── Fixtures ──────────────────────────────────────────────────────────────────

_STAGES = ["clear", "cloudy", "amber", "degraded"]


def _make_trichome_crops(
    n: int,
    h: int = 64,
    w: int = 64,
    seed: int = 42,
) -> list[np.ndarray]:
    """
    Generate synthetic trichome-like crops for benchmarking.

    Cycle through four characteristic color patterns matching real
    optical states:
      - Clear: high-saturation gray-white (bright, low color)
      - Cloudy: desaturated white-gray (milky, uniform)
      - Amber: warm amber/golden hue (HSV: 25-40°)
      - Degraded: dark brown (HSV: 10-20°, low value)
    All images are BGR uint8 (as returned by OpenCV).
    """
    rng = np.random.default_rng(seed)
    crops: list[np.ndarray] = []

    for i in range(n):
        stage = i % 4

        if stage == 0:  # clear — bright, nearly white
            hsv = np.full((h, w, 3), [120, 30, 240], dtype=np.uint8)
            noise = rng.integers(-15, 15, (h, w, 3), dtype=np.int16)
            img_hsv = np.clip(hsv.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

        elif stage == 1:  # cloudy — milky white-gray
            hsv = np.full((h, w, 3), [0, 25, 200], dtype=np.uint8)
            noise = rng.integers(-20, 20, (h, w, 3), dtype=np.int16)
            img_hsv = np.clip(hsv.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

        elif stage == 2:  # amber — warm golden
            hsv = np.full((h, w, 3), [20, 180, 200], dtype=np.uint8)
            noise = rng.integers(-15, 15, (h, w, 3), dtype=np.int16)
            img_hsv = np.clip(hsv.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

        else:  # degraded — dark brown
            hsv = np.full((h, w, 3), [12, 150, 80], dtype=np.uint8)
            noise = rng.integers(-10, 10, (h, w, 3), dtype=np.int16)
            img_hsv = np.clip(hsv.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

        crops.append(img)

    return crops


def _make_gray_crops(
    n: int, h: int = 64, w: int = 64, seed: int = 42
) -> list[np.ndarray]:
    """Grayscale versions for texture/translucency functions."""
    bgr = _make_trichome_crops(n, h, w, seed)
    return [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in bgr]


def _make_binary_masks(
    n: int, h: int = 64, w: int = 64, seed: int = 42
) -> list[np.ndarray]:
    """Elliptical binary masks (uint8, 0/255) simulating segmented heads."""
    rng = np.random.default_rng(seed)
    masks: list[np.ndarray] = []
    for _ in range(n):
        m = np.zeros((h, w), dtype=np.uint8)
        cx = w // 2 + rng.integers(-4, 4)
        cy = h // 2 + rng.integers(-4, 4)
        ax = w // 3 + rng.integers(-4, 4)
        ay = h // 3 + rng.integers(-4, 4)
        cv2.ellipse(m, (cx, cy), (max(ax, 5), max(ay, 5)), 0, 0, 360, 255, -1)
        masks.append(m)
    return masks


# ── Benchmark runner ──────────────────────────────────────────────────────────

def _run(fn, items: list, warmup: int = 5) -> dict:
    """Benchmark a single function over a list of inputs. Returns latency stats."""
    # Warmup
    for item in items[:warmup]:
        try:
            fn(item)
        except Exception:
            return {"error": "warmup_failed", "avg_ms": None, "fps": None}

    latencies_ms: list[float] = []
    for item in items:
        t0 = time.perf_counter()
        try:
            fn(item)
        except Exception:
            return {"error": "benchmark_failed", "avg_ms": None, "fps": None}
        latencies_ms.append((time.perf_counter() - t0) * 1000)

    n = len(latencies_ms)
    avg = statistics.mean(latencies_ms)
    med = statistics.median(latencies_ms)
    p95 = sorted(latencies_ms)[int(0.95 * n)]
    fps = 1000.0 / avg if avg > 0 else 0.0

    return {
        "n": n,
        "avg_ms": round(avg, 3),
        "median_ms": round(med, 3),
        "p95_ms": round(p95, 3),
        "fps": round(fps, 1),
    }


# ── Main benchmark ────────────────────────────────────────────────────────────

def run_maturity_benchmark(
    n: int = 200,
    crop_size: int = 64,
    warmup: int = 5,
) -> dict:
    """Run full maturity benchmark suite."""
    print(f"\n{'='*62}")
    print(f"  Maturity Analysis Benchmark")
    print(f"  N={n} crops | {crop_size}×{crop_size}px | warmup={warmup}")
    print(f"{'='*62}")

    bgr_crops = _make_trichome_crops(n, crop_size, crop_size)
    # Convert BGR→RGB for functions that expect RGB input (most maturity functions)
    rgb_crops = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in bgr_crops]
    gray_crops = _make_gray_crops(n, crop_size, crop_size)
    masks = _make_binary_masks(n, crop_size, crop_size)
    print(f"  Generated {n} synthetic trichome crops ({crop_size}×{crop_size}px)")

    metrics: dict[str, dict] = {}

    # ── Imports ───────────────────────────────────────────────────────────────
    from maturity.domain.color_features import (
        extract_color_features,
        rule_based_maturity_estimate,
    )
    from maturity.domain.texture_features import (
        compute_lbp,
        compute_glcm_features,
        compute_gabor_features,
        compute_shannon_entropy,
        extract_texture_features,
    )
    from maturity.domain.translucency import estimate_translucency
    from maturity.domain.degradation import (
        detect_color_degradation,
        detect_structural_collapse,
        detect_texture_irregularity,
        assess_degradation,
    )
    from maturity.application.maturity_pipeline import MaturityPipeline, MaturityPipelineConfig
    from shared.core.entities import Instance

    # Warm up imports / JIT
    _ = extract_color_features(bgr_crops[0])
    _ = extract_texture_features(gray_crops[0])

    # ── Color features ────────────────────────────────────────────────────────
    benchmark_fns: dict[str, tuple] = {
        "extract_color_features":        (extract_color_features, rgb_crops),
        "rule_based_maturity_estimate":  (
            lambda img: rule_based_maturity_estimate(extract_color_features(img)),
            rgb_crops,
        ),
        # ── Texture features ─────────────────────────────────────────────────
        "compute_lbp":                   (compute_lbp, gray_crops),
        "compute_glcm_features":         (compute_glcm_features, gray_crops),
        "compute_gabor_features":        (compute_gabor_features, gray_crops),
        "compute_shannon_entropy":       (compute_shannon_entropy, gray_crops),
        "extract_texture_features":      (extract_texture_features, gray_crops),
        # ── Translucency ─────────────────────────────────────────────────────
        "estimate_translucency":         (estimate_translucency, rgb_crops),
        # ── Degradation ──────────────────────────────────────────────────────
        "detect_color_degradation":      (detect_color_degradation, rgb_crops),
        "detect_structural_collapse":    (
            detect_structural_collapse,
            rgb_crops,
        ),
        "detect_texture_irregularity":   (
            detect_texture_irregularity,
            rgb_crops,
        ),
        "assess_degradation":            (assess_degradation, rgb_crops),
    }

    # ── Pipeline (crop level) ─────────────────────────────────────────────────
    cfg = MaturityPipelineConfig(use_analyzer=False)  # rule-based path (no CNN)
    pipeline = MaturityPipeline(cfg)

    benchmark_fns["pipeline_analyze_crop"] = (
        lambda img: pipeline.analyze_crop(img),
        rgb_crops,
    )

    # ── Pipeline (batch of 10 via analyze()) ──────────────────────────────────
    batch_instances = [Instance(crop=rgb_crops[i % len(rgb_crops)]) for i in range(10)]

    def _batch_fn(_ignored):
        return pipeline.analyze(batch_instances[:])

    # Use rgb_crops to drive timing; the fn does a full 10-item batch each call
    benchmark_fns["pipeline_analyze_batch_10"] = (_batch_fn, rgb_crops)

    # ── Run benchmarks ────────────────────────────────────────────────────────
    print(f"\n  {'Function':<38}  {'Avg(ms)':>8}  {'Med(ms)':>8}  {'p95(ms)':>8}  {'FPS':>8}")
    print(f"  {'-'*38}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")

    for name, (fn, items) in benchmark_fns.items():
        stats = _run(fn, items, warmup=warmup)
        metrics[name] = stats
        if stats.get("error"):
            print(f"  {name:<38}  ERROR: {stats['error']}")
        else:
            print(
                f"  {name:<38}  {stats['avg_ms']:>8.3f}  "
                f"{stats['median_ms']:>8.3f}  {stats['p95_ms']:>8.3f}  "
                f"{stats['fps']:>8.1f}"
            )

    print()
    return metrics


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Maturity analysis benchmark")
    parser.add_argument("--n",       type=int, default=200, help="Number of crops")
    parser.add_argument("--size",    type=int, default=64,  help="Crop size (square px)")
    parser.add_argument("--warmup",  type=int, default=5,   help="Warmup iterations")
    parser.add_argument("--output",  type=str, default="",  help="Output JSON path")
    args = parser.parse_args()

    # Hardware info
    try:
        import torch
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"GPU: {gpu} ({vram:.1f} GB VRAM)")
        else:
            print("GPU: not available (CPU only)")
    except ImportError:
        print("GPU: PyTorch not installed")

    import platform
    print(f"Platform: {platform.processor()} | Python {platform.python_version()}")

    metrics = run_maturity_benchmark(n=args.n, crop_size=args.size, warmup=args.warmup)

    # Summary
    valid = {k: v for k, v in metrics.items() if not v.get("error")}
    if valid:
        fastest = min(valid, key=lambda k: valid[k]["avg_ms"])
        slowest = max(valid, key=lambda k: valid[k]["avg_ms"])
        print(f"  Fastest: {fastest} ({valid[fastest]['avg_ms']:.3f} ms)")
        print(f"  Slowest: {slowest} ({valid[slowest]['avg_ms']:.3f} ms)")
        crop_stat = valid.get("pipeline_analyze_crop")
        if crop_stat:
            print(f"  Pipeline crop FPS: {crop_stat['fps']:.1f}")

    # Save
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = args.output or f"benchmarks/maturity/results_{timestamp}.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    result = {
        "benchmark": "maturity_analysis",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {"n": args.n, "crop_size": args.size, "warmup": args.warmup},
        "metrics": metrics,
    }
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Results saved: {output_path}")


if __name__ == "__main__":
    main()
