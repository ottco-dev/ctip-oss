"""
benchmarks/measurement/measurement_benchmark.py — Measurement pipeline throughput benchmark.

Measures per-function latency and FPS for all measurement pipeline components.
Designed for RTX 4060 / i5-13400F reference hardware.

Functions benchmarked:
  --- Uncertainty propagation ---
  - combine_uncertainties
  - propagate_linear
  - propagate_area
  - propagate_ratio
  - focus_induced_uncertainty
  --- Measurer ---
  - Measurer.measure (single trichome)
  --- Calibration ---
  - estimate_scale_from_objective
  --- Pipeline ---
  - MeasurementPipeline.measure_instances (single)
  - MeasurementPipeline.measure_instances (batch=10)

Output:
  - Terminal table
  - benchmarks/measurement/results_YYYYMMDD_HHMMSS.json

Usage:
    python benchmarks/measurement/measurement_benchmark.py
    python benchmarks/measurement/measurement_benchmark.py --n 500
    python benchmarks/measurement/measurement_benchmark.py --warmup 10

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

_SEED = 42


def _make_pixel_measurements(n: int, seed: int = _SEED) -> list[dict]:
    """
    Generate synthetic pixel-space measurements for Measurer.measure().

    Values are drawn from realistic trichome size distributions
    (bulbous head: 40–80µm diameter at 40×; stalked: 100–200µm height).
    At 40× generic profile (0.1625 µm/px), those correspond to:
      head_diameter_px ≈ 246–492 px (at very low magnification)
      → We use realistic 64×64 crops instead:
      head_diameter_px ≈ 20–55 px
      stalk_length_px  ≈ 30–90 px
    """
    rng = np.random.default_rng(seed)
    return [
        {
            "head_diameter_px": float(rng.uniform(20, 55)),
            "head_area_px":     float(rng.uniform(300, 2400)),
            "head_circularity": float(rng.uniform(0.65, 0.95)),
            "stalk_length_px":  float(rng.uniform(30, 90)),
            "stalk_width_px":   float(rng.uniform(4, 12)),
            "total_height_px":  float(rng.uniform(50, 150)),
            "total_area_px":    float(rng.uniform(400, 4000)),
        }
        for _ in range(n)
    ]


def _make_ellipse_masks(n: int, h: int = 128, w: int = 128, seed: int = _SEED) -> list[NDArray]:
    """Generate elliptical binary masks (uint8 0/255)."""
    rng = np.random.default_rng(seed)
    masks = []
    for _ in range(n):
        m = np.zeros((h, w), dtype=np.uint8)
        cx, cy = w // 2, h // 2
        ax = int(w * 0.3 + rng.integers(-5, 5))
        ay = int(h * 0.25 + rng.integers(-5, 5))
        cv2.ellipse(m, (cx, cy), (max(ax, 6), max(ay, 6)), 0, 0, 360, 255, -1)
        masks.append(m)
    return masks


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

def run_measurement_benchmark(n: int = 500, warmup: int = 5) -> dict:
    """Run full measurement benchmark suite."""
    print(f"\n{'='*62}")
    print(f"  Measurement Pipeline Benchmark")
    print(f"  N={n} measurements | warmup={warmup}")
    print(f"{'='*62}")

    px_measurements = _make_pixel_measurements(n)
    masks = _make_ellipse_masks(n)
    print(f"  Generated {n} synthetic measurement records")

    metrics: dict[str, dict] = {}

    # ── Imports ───────────────────────────────────────────────────────────────
    from measurement.domain.propagation import (
        combine_uncertainties,
        propagate_linear,
        propagate_area,
        propagate_ratio,
        focus_induced_uncertainty,
    )
    from measurement.domain.measurer import Measurer
    from measurement.domain.profile_manager import MicroscopeProfile
    from measurement.calibration.stage_micrometer import estimate_scale_from_objective
    from measurement.application.measurement_pipeline import MeasurementPipeline
    from shared.core.entities import Instance
    from morphology.domain.geometric import extract_geometric_descriptors

    # Use a realistic 40× profile
    profile_40x = MicroscopeProfile(
        profile_id="bench_40x",
        name="Benchmark 40×",
        um_per_pixel=0.1625,
        objective="40x",
        uncertainty_um=0.005,
    )
    measurer = Measurer(profile_40x)

    # Pre-extract geometric descriptors for instances
    geo_descs = [extract_geometric_descriptors(m) for m in masks[:n]]

    # ── Propagation functions ─────────────────────────────────────────────────

    def _bench_combine(px_dict):
        return combine_uncertainties(0.005, 0.001 * px_dict["head_diameter_px"])

    def _bench_propagate_linear(px_dict):
        return propagate_linear(
            px_dict["head_diameter_px"],
            0.1625,
            calibration_uncertainty_um=0.005,
            edge_uncertainty_px=1.0,
        )

    def _bench_propagate_area(px_dict):
        return propagate_area(
            px_dict["head_area_px"],
            0.1625,
            calibration_uncertainty_um=0.005,
            edge_uncertainty_px=1.0,
        )

    def _bench_propagate_ratio(px_dict):
        # Build two MeasurementWithUncertainty objects for the ratio
        head_meas = propagate_linear(px_dict["head_diameter_px"], 0.1625)
        stalk_meas = propagate_linear(px_dict["stalk_length_px"], 0.1625)
        return propagate_ratio(head_meas, stalk_meas)

    def _bench_focus_uncertainty(px_dict):
        return focus_induced_uncertainty(
            focus_score=0.75,
            pixel_size_um=0.1625,
        )

    def _bench_measurer(px_dict):
        return measurer.measure(
            head_diameter_px=px_dict["head_diameter_px"],
            head_area_px=px_dict["head_area_px"],
            head_circularity=px_dict["head_circularity"],
            stalk_length_px=px_dict["stalk_length_px"],
            stalk_width_px=px_dict["stalk_width_px"],
            total_height_px=px_dict["total_height_px"],
            total_area_px=px_dict["total_area_px"],
        )

    def _bench_estimate_scale(_):
        # estimate_scale_from_objective(objective_magnification, digital_zoom,
        #                               sensor_pixel_size_um, camera_binning)
        return estimate_scale_from_objective(
            objective_magnification=40,
            sensor_pixel_size_um=2.4,
        )

    benchmark_fns: dict[str, tuple] = {
        "combine_uncertainties":         (_bench_combine,            px_measurements),
        "propagate_linear":              (_bench_propagate_linear,   px_measurements),
        "propagate_area":                (_bench_propagate_area,     px_measurements),
        "propagate_ratio":               (_bench_propagate_ratio,    px_measurements),
        "focus_induced_uncertainty":     (_bench_focus_uncertainty,  px_measurements),
        "measurer_measure":              (_bench_measurer,           px_measurements),
        "estimate_scale_from_objective": (_bench_estimate_scale,     px_measurements),
    }

    # ── Pipeline ──────────────────────────────────────────────────────────────
    pipeline = MeasurementPipeline(profile=profile_40x)

    # Single instance
    single_inst = [Instance(crop=np.zeros((64, 64, 3), dtype=np.uint8))]

    benchmark_fns["pipeline_measure_single"] = (
        lambda _: pipeline.measure_instances(single_inst),
        px_measurements,
    )

    # Batch of 10
    batch_insts = [Instance(crop=np.zeros((64, 64, 3), dtype=np.uint8)) for _ in range(10)]
    benchmark_fns["pipeline_measure_batch_10"] = (
        lambda _: pipeline.measure_instances(batch_insts),
        px_measurements,
    )

    # ── Print & collect ───────────────────────────────────────────────────────
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
    parser = argparse.ArgumentParser(description="Measurement pipeline benchmark")
    parser.add_argument("--n",      type=int, default=500, help="Number of measurements")
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

    metrics = run_measurement_benchmark(n=args.n, warmup=args.warmup)

    valid = {k: v for k, v in metrics.items() if not v.get("error")}
    if valid:
        fastest = min(valid, key=lambda k: valid[k]["avg_ms"])
        slowest = max(valid, key=lambda k: valid[k]["avg_ms"])
        print(f"  Fastest: {fastest} ({valid[fastest]['avg_ms']:.3f} ms)")
        print(f"  Slowest: {slowest} ({valid[slowest]['avg_ms']:.3f} ms)")
        pipe_stat = valid.get("pipeline_measure_single", {})
        if pipe_stat:
            print(f"  Pipeline single FPS: {pipe_stat['fps']:.1f}")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = args.output or f"benchmarks/measurement/results_{timestamp}.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    result = {
        "benchmark": "measurement_pipeline",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {"n": args.n, "warmup": args.warmup},
        "metrics": metrics,
    }
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Results saved: {output_path}")


if __name__ == "__main__":
    main()
