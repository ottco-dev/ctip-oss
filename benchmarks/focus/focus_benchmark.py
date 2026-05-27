"""
benchmarks/focus/focus_benchmark.py — Focus metric throughput benchmark.

Measures per-metric latency and FPS for all focus quality metrics.
Designed for RTX 4060 / i5-13400F reference hardware.

Metrics benchmarked:
  - Laplacian Variance (LVAR)
  - Modified Laplacian (MLAP)
  - Squared Laplacian Gradient (SLG)
  - Tenengrad (TENG)
  - Tenengrad Variance (TENGV)
  - FFT High-Frequency Ratio
  - DCT High-Frequency Score
  - Brenner Focus
  - Composite Score (all metrics combined)
  - Regional Composite (4×4 grid)
  - Focus Heatmap generation

Output:
  - Terminal table
  - benchmarks/focus/results_YYYYMMDD_HHMMSS.json

Usage:
    python benchmarks/focus/focus_benchmark.py
    python benchmarks/focus/focus_benchmark.py --n 500 --size 640
    python benchmarks/focus/focus_benchmark.py --warmup 10

Reproducibility:
    GLOBAL_SEED=42, fixed numpy RNG for image generation.
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

def _make_test_images(n: int, h: int, w: int, seed: int = 42) -> list[np.ndarray]:
    """Generate a mix of sharp and blurred grayscale images."""
    rng = np.random.default_rng(seed)
    images = []
    for i in range(n):
        # Alternate between sharp (noise) and blurred patterns
        if i % 3 == 0:
            # High-frequency: checkerboard
            img = np.zeros((h, w), dtype=np.uint8)
            block = max(h // 16, 1)
            for r in range(h):
                for c in range(w):
                    if (r // block + c // block) % 2 == 0:
                        img[r, c] = 255
        elif i % 3 == 1:
            # Medium: Gaussian noise
            img = rng.integers(0, 256, (h, w), dtype=np.uint8)
        else:
            # Low: Blurred uniform
            base = np.full((h, w), 180, dtype=np.uint8)
            img = cv2.GaussianBlur(base, (31, 31), 0)
        images.append(img)
    return images


# ── Benchmark runner ──────────────────────────────────────────────────────────

def _run(fn, images: list[np.ndarray], warmup: int = 5) -> dict:
    """Run benchmark for a single metric function. Returns stats."""
    # Warmup
    for img in images[:warmup]:
        try:
            fn(img)
        except Exception:
            return {"error": "warmup_failed", "avg_ms": None, "fps": None}

    # Timed runs
    latencies_ms = []
    for img in images:
        t0 = time.perf_counter()
        try:
            fn(img)
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


def run_focus_benchmark(n: int = 200, h: int = 512, w: int = 512, warmup: int = 5) -> dict:
    """Run full focus benchmark suite."""
    print(f"\n{'='*60}")
    print(f"  Focus Metrics Benchmark")
    print(f"  N={n} images | {w}×{h}px | warmup={warmup}")
    print(f"{'='*60}")

    images = _make_test_images(n, h, w)
    print(f"  Generated {n} test images ({w}×{h})")

    metrics = {}

    # ── Laplacian metrics ───────────────────────────────────────────────────
    from focus.metrics.laplacian import (
        laplacian_variance,
        modified_laplacian,
        squared_laplacian_gradient,
        laplacian_energy_of_gradient,
        regional_laplacian_variance,
    )
    from focus.metrics.tenengrad import (
        tenengrad,
        tenengrad_variance,
        absolute_gradient_sum,
    )
    from focus.metrics.fft_metrics import (
        fft_high_frequency_ratio,
        dct_high_frequency_score,
        brenner_focus,
        vollath_f4,
    )
    from focus.metrics.composite import compute_focus_score, generate_focus_heatmap
    from focus.guidance.heatmap import generate_focus_heatmap as hm_generate

    benchmark_fns = {
        "laplacian_variance": laplacian_variance,
        "modified_laplacian": modified_laplacian,
        "squared_laplacian_gradient": squared_laplacian_gradient,
        "laplacian_energy_of_gradient": laplacian_energy_of_gradient,
        "regional_laplacian_variance": regional_laplacian_variance,
        "tenengrad": tenengrad,
        "tenengrad_variance": tenengrad_variance,
        "absolute_gradient_sum": absolute_gradient_sum,
        "fft_high_frequency_ratio": fft_high_frequency_ratio,
        "dct_high_frequency_score": dct_high_frequency_score,
        "brenner_focus": brenner_focus,
        "vollath_f4": vollath_f4,
        "composite_score": lambda img: compute_focus_score(img),
        "composite_regional_4x4": lambda img: compute_focus_score(img, compute_regional=True, region_grid=(4, 4)),
        "generate_heatmap_composite": lambda img: generate_focus_heatmap(img, grid=(4, 4)),
        "generate_heatmap_guidance": lambda img: hm_generate(np.stack([img, img, img], axis=-1), grid=(8, 8)),
    }

    # Header
    print(f"\n  {'Metric':<36}  {'Avg(ms)':>8}  {'Med(ms)':>8}  {'p95(ms)':>8}  {'FPS':>8}")
    print(f"  {'-'*36}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")

    for name, fn in benchmark_fns.items():
        stats = _run(fn, images, warmup=warmup)
        metrics[name] = stats
        if stats.get("error"):
            print(f"  {name:<36}  ERROR: {stats['error']}")
        else:
            print(
                f"  {name:<36}  {stats['avg_ms']:>8.3f}  "
                f"{stats['median_ms']:>8.3f}  {stats['p95_ms']:>8.3f}  "
                f"{stats['fps']:>8.1f}"
            )

    print()
    return metrics


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Focus metrics benchmark")
    parser.add_argument("--n", type=int, default=200, help="Number of test images")
    parser.add_argument("--size", type=int, default=512, help="Image size (square)")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--output", type=str, default="", help="Output JSON path")
    args = parser.parse_args()

    # Check GPU info
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"GPU: {gpu_name} ({vram_gb:.1f} GB VRAM)")
        else:
            print("GPU: not available (CPU only)")
    except ImportError:
        print("GPU: PyTorch not installed")

    import platform
    print(f"Platform: {platform.processor()} | Python {platform.python_version()}")

    metrics = run_focus_benchmark(n=args.n, h=args.size, w=args.size, warmup=args.warmup)

    # Summary
    valid = {k: v for k, v in metrics.items() if not v.get("error")}
    if valid:
        fastest = min(valid, key=lambda k: valid[k]["avg_ms"])
        slowest = max(valid, key=lambda k: valid[k]["avg_ms"])
        print(f"  Fastest: {fastest} ({valid[fastest]['avg_ms']:.3f} ms)")
        print(f"  Slowest: {slowest} ({valid[slowest]['avg_ms']:.3f} ms)")
        composite = valid.get("composite_score", {})
        if composite:
            print(f"  Composite FPS: {composite['fps']:.1f}")

    # Save results
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = args.output or f"benchmarks/focus/results_{timestamp}.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    result = {
        "benchmark": "focus_metrics",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {"n": args.n, "image_size": args.size, "warmup": args.warmup},
        "metrics": metrics,
    }
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Results saved: {output_path}")


if __name__ == "__main__":
    main()
