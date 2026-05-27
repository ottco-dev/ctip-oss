"""
benchmarks/morphology/morphology_benchmark.py — Morphology pipeline throughput benchmark.

Measures per-function latency and FPS for all morphology analysis components.
Designed for RTX 4060 / i5-13400F reference hardware.

Functions benchmarked:
  --- Geometric descriptors ---
  - extract_geometric_descriptors
  - contour_from_mask
  --- Stalk/head detection ---
  - detect_stalk_and_head
  --- Density map ---
  - compute_density_map (grid, KDE)
  --- Classification ---
  - extract_geometric_features
  - classify_morphology_geometric
  - MorphologyClassifier.predict_geometric
  --- Pipeline ---
  - MorphologyPipeline.process_instance
  - MorphologyPipeline.process_batch (N=10)

Output:
  - Terminal table
  - benchmarks/morphology/results_YYYYMMDD_HHMMSS.json

Usage:
    python benchmarks/morphology/morphology_benchmark.py
    python benchmarks/morphology/morphology_benchmark.py --n 200 --size 128
    python benchmarks/morphology/morphology_benchmark.py --warmup 10

Reproducibility:
    GLOBAL_SEED=42, fixed numpy RNG.
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
from numpy.typing import NDArray

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_masks(n: int, h: int = 128, w: int = 128, seed: int = 42) -> list[NDArray]:
    """
    Generate synthetic trichome-like binary masks (uint8, 0/255).

    Cycles through three morphology types:
      - Bulbous:  large circular/elliptical head
      - Sessile:  squat ellipse with slight stalk
      - Stalked:  elongated with constriction at stalk/head junction
    """
    rng = np.random.default_rng(seed)
    masks: list[NDArray] = []

    for i in range(n):
        m = np.zeros((h, w), dtype=np.uint8)
        morph = i % 3

        if morph == 0:  # Bulbous: large round head
            cx, cy = w // 2, h // 2
            r = int(min(h, w) * 0.35 + rng.integers(-5, 5))
            cv2.circle(m, (cx, cy), max(r, 10), 255, -1)

        elif morph == 1:  # Sessile: squat ellipse
            cx, cy = w // 2, h // 2
            ax = int(w * 0.28 + rng.integers(-4, 4))
            ay = int(h * 0.22 + rng.integers(-4, 4))
            cv2.ellipse(m, (cx, cy), (max(ax, 8), max(ay, 8)), 0, 0, 360, 255, -1)

        else:  # Stalked: elongated vertical
            # Head at top
            hx, hy = w // 2, h // 4
            head_r = int(min(h, w) * 0.18 + rng.integers(-3, 3))
            cv2.circle(m, (hx, hy), max(head_r, 6), 255, -1)
            # Stalk below
            stalk_w = max(int(w * 0.08 + rng.integers(-2, 2)), 4)
            stalk_top = hy + head_r
            stalk_bot = int(h * 0.85)
            cv2.rectangle(
                m,
                (w // 2 - stalk_w, stalk_top),
                (w // 2 + stalk_w, stalk_bot),
                255, -1
            )

        masks.append(m)

    return masks


def _make_crops(
    n: int, h: int = 128, w: int = 128, seed: int = 42
) -> list[NDArray]:
    """Generate RGB crops (3-channel uint8 BGR) for density map benchmark."""
    rng = np.random.default_rng(seed)
    crops: list[NDArray] = []
    for _ in range(n):
        img = rng.integers(100, 200, (h, w, 3), dtype=np.uint8)
        crops.append(img)
    return crops


def _make_centroids(
    n: int, h: int = 512, w: int = 512, seed: int = 42
) -> list[tuple[float, float]]:
    """Random trichome centroid positions for density map testing."""
    rng = np.random.default_rng(seed)
    xs = rng.uniform(0, w, n)
    ys = rng.uniform(0, h, n)
    return list(zip(xs.tolist(), ys.tolist()))


# ── Benchmark runner ──────────────────────────────────────────────────────────

def _run(fn, items: list, warmup: int = 5) -> dict:
    """Benchmark a single function. Returns latency stats."""
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

def run_morphology_benchmark(
    n: int = 200,
    mask_size: int = 128,
    warmup: int = 5,
) -> dict:
    """Run full morphology benchmark suite."""
    print(f"\n{'='*62}")
    print(f"  Morphology Analysis Benchmark")
    print(f"  N={n} masks | {mask_size}×{mask_size}px | warmup={warmup}")
    print(f"{'='*62}")

    masks = _make_masks(n, mask_size, mask_size)
    crops = _make_crops(n, mask_size, mask_size)
    print(f"  Generated {n} synthetic trichome masks ({mask_size}×{mask_size}px)")

    metrics: dict[str, dict] = {}

    # ── Imports ───────────────────────────────────────────────────────────────
    from morphology.domain.geometric import (
        extract_geometric_descriptors,
        contour_from_mask,
    )
    from morphology.domain.stalk_detector import detect_stalk_and_head
    from morphology.domain.density_map import compute_density_map, TrichomeCentroid
    from morphology.classification.classifier import (
        extract_geometric_features,
        classify_morphology_geometric,
        MorphologyClassifier,
    )
    from morphology.application.morphology_pipeline import (
        MorphologyPipeline,
        MorphologyPipelineConfig,
    )
    from shared.core.entities import Instance

    # Build centroid list for density map benchmarks
    centroids = [
        TrichomeCentroid(x=float(i % mask_size), y=float((i * 7) % mask_size))
        for i in range(n)
    ]

    # Pre-extract geometric features and descriptors for downstream benchmarks
    geo_features_list = [extract_geometric_features(m.astype(bool)) for m in masks]
    geo_desc_list = [extract_geometric_descriptors(m) for m in masks]

    classifier = MorphologyClassifier()

    benchmark_fns: dict[str, tuple] = {
        # ── Geometric ────────────────────────────────────────────────────────
        "contour_from_mask":              (contour_from_mask,               masks),
        "extract_geometric_descriptors":  (extract_geometric_descriptors,   masks),
        # ── Stalk/head ────────────────────────────────────────────────────────
        "detect_stalk_and_head":          (detect_stalk_and_head,           masks),
        # ── Classification (takes GeometricFeatures, not mask) ────────────────
        "extract_geometric_features":     (
            lambda m: extract_geometric_features(m.astype(bool)),
            masks,
        ),
        "classify_morphology_geometric":  (classify_morphology_geometric,  geo_features_list),
        "MorphologyClassifier.predict":   (
            lambda feats: classifier.predict_geometric(features=feats),
            geo_features_list,
        ),
        # ── Density map (single call over 50 centroids) ──────────────────────
        "compute_density_map_50pts": (
            lambda _: compute_density_map(
                centroids[:50],
                image_height=512,
                image_width=512,
                grid_rows=8,
                grid_cols=8,
            ),
            masks,  # driven by mask list length
        ),
        "compute_density_map_50pts_kde": (
            lambda _: compute_density_map(
                centroids[:50],
                image_height=512,
                image_width=512,
                grid_rows=8,
                grid_cols=8,
                kde_bandwidth=30.0,
            ),
            masks,
        ),
    }

    # ── Pipeline (single instance) ────────────────────────────────────────────
    # MorphologyPipeline.analyze() requires Instance.mask to be set
    cfg = MorphologyPipelineConfig(classifier_model_path=None)
    pipeline = MorphologyPipeline(cfg)

    from shared.core.value_objects import Mask as MaskVO

    def _make_inst(mask_arr):
        inst = Instance(crop=crops[0])
        inst.mask = MaskVO.from_uint8(mask_arr)
        return inst

    single_inst = _make_inst(masks[0])
    benchmark_fns["pipeline_analyze_single"] = (
        lambda _: pipeline.analyze([single_inst]),
        masks,
    )

    # ── Pipeline (batch of 10) ────────────────────────────────────────────────
    batch_instances = [_make_inst(masks[i % len(masks)]) for i in range(10)]

    benchmark_fns["pipeline_analyze_batch_10"] = (
        lambda _: pipeline.analyze(batch_instances[:]),
        masks,
    )

    # ── Run all ───────────────────────────────────────────────────────────────
    print(f"\n  {'Function':<40}  {'Avg(ms)':>8}  {'Med(ms)':>8}  {'p95(ms)':>8}  {'FPS':>8}")
    print(f"  {'-'*40}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")

    for name, (fn, items) in benchmark_fns.items():
        stats = _run(fn, items, warmup=warmup)
        metrics[name] = stats
        if stats.get("error"):
            print(f"  {name:<40}  ERROR: {stats['error']}")
        else:
            print(
                f"  {name:<40}  {stats['avg_ms']:>8.3f}  "
                f"{stats['median_ms']:>8.3f}  {stats['p95_ms']:>8.3f}  "
                f"{stats['fps']:>8.1f}"
            )

    print()
    return metrics


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Morphology analysis benchmark")
    parser.add_argument("--n",      type=int, default=200, help="Number of masks")
    parser.add_argument("--size",   type=int, default=128, help="Mask size (square px)")
    parser.add_argument("--warmup", type=int, default=5,   help="Warmup iterations")
    parser.add_argument("--output", type=str, default="",  help="Output JSON path")
    args = parser.parse_args()

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

    metrics = run_morphology_benchmark(n=args.n, mask_size=args.size, warmup=args.warmup)

    valid = {k: v for k, v in metrics.items() if not v.get("error")}
    if valid:
        fastest = min(valid, key=lambda k: valid[k]["avg_ms"])
        slowest = max(valid, key=lambda k: valid[k]["avg_ms"])
        print(f"  Fastest: {fastest} ({valid[fastest]['avg_ms']:.3f} ms)")
        print(f"  Slowest: {slowest} ({valid[slowest]['avg_ms']:.3f} ms)")
        pipe_stat = valid.get("pipeline_analyze_single")
        if pipe_stat:
            print(f"  Pipeline instance FPS: {pipe_stat['fps']:.1f}")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = args.output or f"benchmarks/morphology/results_{timestamp}.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    result = {
        "benchmark": "morphology_analysis",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {"n": args.n, "mask_size": args.size, "warmup": args.warmup},
        "metrics": metrics,
    }
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Results saved: {output_path}")


if __name__ == "__main__":
    main()
