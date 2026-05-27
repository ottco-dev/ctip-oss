"""
benchmarks/video/video_benchmark.py — Video pipeline throughput benchmark.

Measures per-function latency and FPS for all video pipeline components.
Designed for RTX 4060 / i5-13400F reference hardware.

Functions benchmarked:
  --- Frame quality scoring ---
  - score_frame (fast path, use_focus_composite=False)
  - score_frame (composite path, use_focus_composite=True)
  --- Perceptual hashing ---
  - perceptual_hash
  - hamming_distance
  - deduplicate_frames (N=50 frame pool)
  --- Motion estimation ---
  - estimate_motion (frame pair)
  - classify_motion_sequence (N=20 estimates)
  --- Frame ranking ---
  - rank_top_n (N=50 frames, select 10)
  - rank_diverse_n (N=50 frames, select 10)
  - rank_adaptive (N=50 frames, select 10)

Output:
  - Terminal table
  - benchmarks/video/results_YYYYMMDD_HHMMSS.json

Usage:
    python benchmarks/video/video_benchmark.py
    python benchmarks/video/video_benchmark.py --n 200 --size 512
    python benchmarks/video/video_benchmark.py --warmup 10

Reproducibility:
    GLOBAL_SEED=42, fixed numpy RNG.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from numpy.typing import NDArray

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ── Fixtures ──────────────────────────────────────────────────────────────────

_SEED = 42


def _make_frames(
    n: int,
    h: int = 512,
    w: int = 512,
    seed: int = _SEED,
) -> list[NDArray]:
    """
    Generate synthetic microscopy-like frames (RGB uint8).

    Alternates between:
      - Sharp checkerboard (high frequency — good focus)
      - Gaussian blurred random noise (medium quality)
      - Nearly uniform with small gradient (low frequency — poor focus)
    """
    rng = np.random.default_rng(seed)
    frames = []

    for i in range(n):
        style = i % 3

        if style == 0:
            # Sharp: fine checkerboard in gray + slight color cast
            block = max(h // 32, 1)
            gray = np.zeros((h, w), dtype=np.uint8)
            for r in range(h):
                for c in range(w):
                    if (r // block + c // block) % 2 == 0:
                        gray[r, c] = 220
                    else:
                        gray[r, c] = 40
            frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

        elif style == 1:
            # Medium: Gaussian-blurred noise
            noise = rng.integers(0, 256, (h, w), dtype=np.uint8)
            blurred = cv2.GaussianBlur(noise, (5, 5), 0)
            frame = cv2.cvtColor(blurred, cv2.COLOR_GRAY2RGB)

        else:
            # Blurry: uniform soft gradient
            base = np.full((h, w), 160, dtype=np.uint8)
            blurred = cv2.GaussianBlur(base, (31, 31), 5)
            # Add subtle color gradient for realism
            frame = np.stack([
                blurred,
                np.clip(blurred.astype(int) + 20, 0, 255).astype(np.uint8),
                np.clip(blurred.astype(int) - 10, 0, 255).astype(np.uint8),
            ], axis=-1)

        frames.append(frame)

    return frames


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

def run_video_benchmark(
    n: int = 200,
    frame_size: int = 512,
    warmup: int = 5,
) -> dict:
    """Run full video pipeline benchmark suite."""
    print(f"\n{'='*62}")
    print(f"  Video Pipeline Benchmark")
    print(f"  N={n} frames | {frame_size}×{frame_size}px | warmup={warmup}")
    print(f"{'='*62}")

    frames = _make_frames(n, frame_size, frame_size)
    print(f"  Generated {n} synthetic frames ({frame_size}×{frame_size}px RGB)")

    metrics: dict[str, dict] = {}

    # ── Imports ───────────────────────────────────────────────────────────────
    from video_pipeline.domain.scorer import score_frame, FrameQualityScore
    from video_pipeline.domain.hasher import (
        perceptual_hash,
        hamming_distance,
        deduplicate_frames,
    )
    from video_pipeline.domain.motion import estimate_motion, classify_motion_sequence
    from video_pipeline.domain.ranker import (
        RankedFrame,
        rank_top_n,
        rank_diverse_n,
        rank_adaptive,
    )

    # ── Pre-build supporting data ─────────────────────────────────────────────

    # Build a pool of pre-scored RankedFrame objects for ranking benchmarks
    print(f"  Pre-scoring {min(50, n)} frames for ranking benchmarks...")
    pool_frames = frames[:min(50, n)]
    ranked_pool: list[RankedFrame] = []
    for idx, frame in enumerate(pool_frames):
        score = score_frame(frame, use_focus_composite=False)
        phash = perceptual_hash(frame)
        ranked_pool.append(
            RankedFrame(
                frame_index=idx,
                timestamp_s=float(idx) / 30.0,  # simulate 30 fps
                quality=score,
                phash=phash,
            )
        )

    # Pre-compute hashes for deduplication benchmark
    pool_hashes = [perceptual_hash(f) for f in pool_frames]

    # Pre-compute motion estimates for classify_motion_sequence
    print(f"  Pre-computing motion estimates for sequence benchmark...")
    motion_estimates = []
    for i in range(min(20, n - 1)):
        m = estimate_motion(frames[i], frames[i + 1])
        motion_estimates.append(m)

    # ── Define benchmark functions ────────────────────────────────────────────
    benchmark_fns: dict[str, tuple] = {
        # Scoring
        "score_frame_fast":              (
            lambda f: score_frame(f, use_focus_composite=False),
            frames,
        ),
        "score_frame_composite":         (
            lambda f: score_frame(f, use_focus_composite=True),
            frames,
        ),
        # Hashing
        "perceptual_hash":               (perceptual_hash,     frames),
        "hamming_distance":              (
            lambda f: hamming_distance(perceptual_hash(f), perceptual_hash(f)),
            frames,
        ),
        # Deduplication: operates on a 50-frame pool (same pool each call)
        "deduplicate_frames_50":         (
            lambda _: deduplicate_frames(pool_hashes[:50], threshold=10),
            frames,
        ),
        # Motion estimation: consecutive pair
        "estimate_motion":               (
            lambda f: estimate_motion(frames[0], f),
            frames,
        ),
        # Motion sequence classification: 20 pre-computed estimates
        "classify_motion_sequence_20":   (
            lambda _: classify_motion_sequence(motion_estimates[:20]),
            frames,
        ),
        # Ranking (pool of 50 RankedFrames, select 10)
        "rank_top_n_50->10":             (
            lambda _: rank_top_n(ranked_pool, n=10),
            frames,
        ),
        "rank_diverse_n_50->10":         (
            lambda _: rank_diverse_n(ranked_pool, n=10),
            frames,
        ),
        "rank_adaptive_50->10":          (
            lambda _: rank_adaptive(ranked_pool, n=10),
            frames,
        ),
    }

    # ── Run ───────────────────────────────────────────────────────────────────
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
    parser = argparse.ArgumentParser(description="Video pipeline benchmark")
    parser.add_argument("--n",      type=int, default=200, help="Number of frames")
    parser.add_argument("--size",   type=int, default=512, help="Frame size (square px)")
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

    metrics = run_video_benchmark(n=args.n, frame_size=args.size, warmup=args.warmup)

    valid = {k: v for k, v in metrics.items() if not v.get("error")}
    if valid:
        fastest = min(valid, key=lambda k: valid[k]["avg_ms"])
        slowest = max(valid, key=lambda k: valid[k]["avg_ms"])
        print(f"  Fastest: {fastest} ({valid[fastest]['avg_ms']:.3f} ms)")
        print(f"  Slowest: {slowest} ({valid[slowest]['avg_ms']:.3f} ms)")
        fast_score = valid.get("score_frame_fast", {})
        comp_score = valid.get("score_frame_composite", {})
        if fast_score and comp_score:
            speedup = comp_score["avg_ms"] / fast_score["avg_ms"] if fast_score["avg_ms"] > 0 else 1.0
            print(f"  score_frame speedup (fast vs composite): {speedup:.1f}×")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = args.output or f"benchmarks/video/results_{timestamp}.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    result = {
        "benchmark": "video_pipeline",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {"n": args.n, "frame_size": args.size, "warmup": args.warmup},
        "metrics": metrics,
    }
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Results saved: {output_path}")


if __name__ == "__main__":
    main()
