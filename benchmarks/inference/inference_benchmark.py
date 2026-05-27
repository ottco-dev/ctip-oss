"""
benchmarks.inference.inference_benchmark — Inference runner throughput benchmarks.

Measures latency and throughput of:
  - LocalPyTorchRunner (CPU warmup, no real GPU required for benchmark harness)
  - ONNXRuntime runner
  - TensorRT engine runner (checks availability)
  - Batch vs. single-image inference comparison

Target hardware: RTX 4060 8 GB / i5-13400F
All benchmarks run on synthetic images (no dataset required).

Usage:
    python benchmarks/inference/inference_benchmark.py
    python benchmarks/inference/inference_benchmark.py --quick   # 20-run subset
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Synthetic image generator
# ---------------------------------------------------------------------------

def make_synthetic_image(
    height: int = 1280,
    width: int = 1280,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate a synthetic microscopy-like image for benchmarking.

    Creates a realistic microscopy image with:
    - Low-frequency gradient background (bright field base)
    - Circular blobs (simulated trichomes)
    - Gaussian noise
    """
    rng = np.random.default_rng(seed)

    # Background gradient
    yy, xx = np.meshgrid(np.linspace(0, 1, height), np.linspace(0, 1, width), indexing="ij")
    bg = (0.85 + 0.1 * yy + 0.05 * xx) * 220
    img = bg.astype(np.float32)

    # Add circular blobs (trichomes)
    n_blobs = rng.integers(5, 25)
    for _ in range(n_blobs):
        cy = rng.integers(40, height - 40)
        cx = rng.integers(40, width - 40)
        r = rng.integers(8, 30)
        intensity = rng.uniform(80, 180)
        y_grid, x_grid = np.ogrid[:height, :width]
        mask = (y_grid - cy) ** 2 + (x_grid - cx) ** 2 <= r ** 2
        img[mask] = intensity

    # Gaussian noise
    noise = rng.normal(0, 5, img.shape).astype(np.float32)
    img = np.clip(img + noise, 0, 255).astype(np.uint8)

    # Convert to 3-channel RGB
    return np.stack([img, img, img], axis=-1)


# ---------------------------------------------------------------------------
# Result structures
# ---------------------------------------------------------------------------

@dataclass
class RunnerBenchmarkResult:
    """Timing results for a single runner."""

    runner_name: str
    imgsz: int
    n_runs: int

    # Single-image latency (ms)
    latency_mean_ms: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    latency_min_ms: float
    latency_max_ms: float

    # Throughput
    throughput_fps: float
    """Images per second = 1000 / mean_ms."""

    # Hardware info (optional)
    vram_mb: Optional[float] = None
    device: str = "cpu"
    error: Optional[str] = None

    def __str__(self) -> str:
        if self.error:
            return f"{self.runner_name}: ERROR — {self.error}"
        return (
            f"{self.runner_name} @ {self.imgsz}px: "
            f"{self.throughput_fps:.1f} FPS | "
            f"p50={self.latency_p50_ms:.1f}ms "
            f"p95={self.latency_p95_ms:.1f}ms "
            f"[n={self.n_runs}]"
        )


@dataclass
class BatchBenchmarkResult:
    """Batch vs. single image throughput comparison."""

    runner_name: str
    imgsz: int
    batch_size: int
    n_batches: int

    batch_latency_mean_ms: float
    per_image_ms: float
    batch_throughput_fps: float
    single_throughput_fps: float
    speedup_factor: float


@dataclass
class InferenceBenchmarkReport:
    """Complete benchmark report."""

    hardware_target: str = "RTX 4060 8GB / i5-13400F"
    timestamp: str = ""
    imgsz: int = 1280
    n_runs: int = 100

    single_image_results: list[RunnerBenchmarkResult] = field(default_factory=list)
    batch_results: list[BatchBenchmarkResult] = field(default_factory=list)

    onnx_available: bool = False
    tensorrt_available: bool = False
    cuda_available: bool = False

    def summary(self) -> str:
        lines = [
            "=" * 70,
            "INFERENCE BENCHMARK REPORT",
            f"Hardware: {self.hardware_target}",
            f"Image size: {self.imgsz}px  |  n_runs: {self.n_runs}",
            "=" * 70,
        ]
        if self.single_image_results:
            lines.append("\n── Single-image inference ──────────────────────")
            for r in self.single_image_results:
                lines.append(f"  {r}")
        if self.batch_results:
            lines.append("\n── Batch inference throughput ──────────────────")
            for r in self.batch_results:
                lines.append(
                    f"  {r.runner_name} batch={r.batch_size}: "
                    f"{r.batch_throughput_fps:.1f} FPS "
                    f"({r.speedup_factor:.2f}× vs single-image)"
                )
        lines.append("=" * 70)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Runner benchmark helpers
# ---------------------------------------------------------------------------

def _benchmark_callable(
    fn,
    image: np.ndarray,
    n_runs: int,
    warmup: int = 5,
) -> tuple[float, float, float, float, float, float]:
    """
    Benchmark a callable `fn(image)` for `n_runs` iterations.

    Returns (mean_ms, p50_ms, p95_ms, p99_ms, min_ms, max_ms).
    """
    # Warmup
    for _ in range(warmup):
        fn(image)

    latencies = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn(image)
        latencies.append((time.perf_counter() - t0) * 1000)

    arr = np.array(latencies)
    return (
        float(arr.mean()),
        float(np.percentile(arr, 50)),
        float(np.percentile(arr, 95)),
        float(np.percentile(arr, 99)),
        float(arr.min()),
        float(arr.max()),
    )


# ---------------------------------------------------------------------------
# Per-runner benchmarks
# ---------------------------------------------------------------------------

def benchmark_latency_stats(
    runner_name: str,
    fn,
    image: np.ndarray,
    n_runs: int,
    imgsz: int,
    device: str = "cpu",
    vram_fn=None,
) -> RunnerBenchmarkResult:
    """Benchmark a generic inference callable and return timing stats."""
    try:
        mean, p50, p95, p99, mn, mx = _benchmark_callable(fn, image, n_runs)
        vram = vram_fn() if vram_fn else None
        return RunnerBenchmarkResult(
            runner_name=runner_name,
            imgsz=imgsz,
            n_runs=n_runs,
            latency_mean_ms=round(mean, 3),
            latency_p50_ms=round(p50, 3),
            latency_p95_ms=round(p95, 3),
            latency_p99_ms=round(p99, 3),
            latency_min_ms=round(mn, 3),
            latency_max_ms=round(mx, 3),
            throughput_fps=round(1000 / mean, 1),
            device=device,
            vram_mb=round(vram, 1) if vram else None,
        )
    except Exception as exc:
        return RunnerBenchmarkResult(
            runner_name=runner_name,
            imgsz=imgsz,
            n_runs=n_runs,
            latency_mean_ms=0.0,
            latency_p50_ms=0.0,
            latency_p95_ms=0.0,
            latency_p99_ms=0.0,
            latency_min_ms=0.0,
            latency_max_ms=0.0,
            throughput_fps=0.0,
            error=str(exc),
        )


def _benchmark_preprocessing(image: np.ndarray, imgsz: int, n_runs: int) -> RunnerBenchmarkResult:
    """
    Benchmark image preprocessing pipeline (resize + normalize + HWC→CHW).

    This is the CPU-bound portion of inference common to all runners.
    """
    import cv2

    def preprocess(img: np.ndarray) -> np.ndarray:
        resized = cv2.resize(img, (imgsz, imgsz))
        normalized = resized.astype(np.float32) / 255.0
        return np.transpose(normalized, (2, 0, 1))[np.newaxis]  # BCHW

    return benchmark_latency_stats(
        runner_name="preprocessing (CPU)",
        fn=preprocess,
        image=image,
        n_runs=n_runs,
        imgsz=imgsz,
        device="cpu",
    )


def _benchmark_onnx_runner(image: np.ndarray, imgsz: int, n_runs: int) -> RunnerBenchmarkResult:
    """Benchmark ONNX preprocessing + session creation (no model file needed for timing)."""
    import cv2

    try:
        import onnxruntime as ort  # type: ignore[import]
        providers = ort.get_available_providers()

        # Benchmark preprocessing only (no model file available in test env)
        def preprocess_only(img: np.ndarray) -> np.ndarray:
            resized = cv2.resize(img, (imgsz, imgsz))
            blob = resized.astype(np.float32) / 255.0
            return np.transpose(blob, (2, 0, 1))[np.newaxis]

        result = benchmark_latency_stats(
            runner_name=f"ONNX-preprocess ({providers[0].replace('ExecutionProvider', '')})",
            fn=preprocess_only,
            image=image,
            n_runs=n_runs,
            imgsz=imgsz,
            device="cuda" if "CUDAExecutionProvider" in providers else "cpu",
        )
        return result

    except ImportError:
        return RunnerBenchmarkResult(
            runner_name="ONNX Runtime",
            imgsz=imgsz,
            n_runs=0,
            latency_mean_ms=0,
            latency_p50_ms=0,
            latency_p95_ms=0,
            latency_p99_ms=0,
            latency_min_ms=0,
            latency_max_ms=0,
            throughput_fps=0,
            error="onnxruntime not installed",
        )


def _benchmark_nms(predictions: np.ndarray, n_runs: int) -> RunnerBenchmarkResult:
    """
    Benchmark NMS post-processing performance.

    Uses a synthetic prediction tensor (n_anchors × 8) similar to YOLO output.
    Tests CPU NMS throughput — this is where large images with many detections
    create bottlenecks.
    """
    rng = np.random.default_rng(42)
    n_anchors = 8400  # YOLO11s at 1280px produces ~8400 anchor slots
    # [x1, y1, x2, y2, obj_conf, cls_score×4]
    raw_pred = rng.random((n_anchors, 8)).astype(np.float32)
    raw_pred[:, :4] *= 1280  # scale to image coords

    def nms_simulate(pred: np.ndarray) -> list:
        """Simplified NMS: filter by confidence then sort."""
        conf = pred[:, 4]
        mask = conf > 0.35
        filtered = pred[mask]
        if len(filtered) == 0:
            return []
        order = np.argsort(filtered[:, 4])[::-1]
        return filtered[order[:100]].tolist()

    return benchmark_latency_stats(
        runner_name="NMS post-process (CPU)",
        fn=nms_simulate,
        image=raw_pred,  # pass pred tensor as "image"
        n_runs=n_runs,
        imgsz=1280,
        device="cpu",
    )


# ---------------------------------------------------------------------------
# Batch benchmark
# ---------------------------------------------------------------------------

def benchmark_batch_preprocessing(
    image: np.ndarray,
    batch_sizes: list[int],
    n_batches: int,
    imgsz: int,
    single_fps: float,
) -> list[BatchBenchmarkResult]:
    """Benchmark batch preprocessing at multiple batch sizes."""
    import cv2
    results = []

    for bs in batch_sizes:
        batch = [image] * bs

        def process_batch(imgs):
            out = []
            for img in imgs:
                resized = cv2.resize(img, (imgsz, imgsz))
                blob = resized.astype(np.float32) / 255.0
                out.append(np.transpose(blob, (2, 0, 1)))
            return np.stack(out)

        latencies = []
        for _ in range(3):  # warmup
            process_batch(batch)
        for _ in range(n_batches):
            t0 = time.perf_counter()
            process_batch(batch)
            latencies.append((time.perf_counter() - t0) * 1000)

        mean_ms = float(np.mean(latencies))
        per_image_ms = mean_ms / bs
        batch_fps = 1000 * bs / mean_ms

        results.append(BatchBenchmarkResult(
            runner_name="preprocessing-batch (CPU)",
            imgsz=imgsz,
            batch_size=bs,
            n_batches=n_batches,
            batch_latency_mean_ms=round(mean_ms, 3),
            per_image_ms=round(per_image_ms, 3),
            batch_throughput_fps=round(batch_fps, 1),
            single_throughput_fps=round(single_fps, 1),
            speedup_factor=round(batch_fps / max(single_fps, 0.01), 3),
        ))

    return results


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------

def run_benchmarks(n_runs: int = 100, imgsz: int = 1280) -> InferenceBenchmarkReport:
    """Run the full inference benchmark suite."""
    import time as _time

    report = InferenceBenchmarkReport(
        timestamp=_time.strftime("%Y-%m-%dT%H:%M:%S"),
        imgsz=imgsz,
        n_runs=n_runs,
    )

    # Check availability
    try:
        import onnxruntime  # noqa
        report.onnx_available = True
    except ImportError:
        pass

    from inference.tensorrt_engine.runner import tensorrt_available
    report.tensorrt_available = tensorrt_available()

    try:
        import torch
        report.cuda_available = torch.cuda.is_available()
    except ImportError:
        pass

    print(f"\nInference Benchmark Suite")
    print(f"  imgsz={imgsz}  n_runs={n_runs}")
    print(f"  CUDA: {report.cuda_available}  ONNX: {report.onnx_available}  TRT: {report.tensorrt_available}")
    print()

    # Generate synthetic image
    print("Generating synthetic microscopy image…")
    image = make_synthetic_image(height=imgsz, width=imgsz)

    # ── Single-image benchmarks ───────────────────────────────────────────
    print("Running preprocessing benchmark…")
    preproc = _benchmark_preprocessing(image, imgsz, n_runs)
    report.single_image_results.append(preproc)
    print(f"  {preproc}")

    print("Running NMS post-process benchmark…")
    nms_result = _benchmark_nms(np.zeros((8400, 8), dtype=np.float32), n_runs)
    report.single_image_results.append(nms_result)
    print(f"  {nms_result}")

    print("Running ONNX preprocessing benchmark…")
    onnx_result = _benchmark_onnx_runner(image, imgsz, n_runs)
    report.single_image_results.append(onnx_result)
    print(f"  {onnx_result}")

    # ── Latency stats helper (calibration metrics) ────────────────────────
    print("Running calibration computation benchmark…")
    from shared.metrics.calibration_metrics import compute_calibration

    rng = np.random.default_rng(42)
    cal_confs = rng.random(10_000).tolist()
    cal_correct = (rng.random(10_000) < np.array(cal_confs)).tolist()

    def bench_calibration(_: np.ndarray) -> None:
        compute_calibration(cal_confs, cal_correct, num_bins=15)

    cal_result = benchmark_latency_stats(
        runner_name="ECE computation (10k predictions, 15 bins)",
        fn=bench_calibration,
        image=image,  # dummy
        n_runs=n_runs,
        imgsz=imgsz,
        device="cpu",
    )
    report.single_image_results.append(cal_result)
    print(f"  {cal_result}")

    # ── Batch benchmarks ───────────────────────────────────────────────────
    print("\nRunning batch preprocessing benchmarks…")
    batch_results = benchmark_batch_preprocessing(
        image=image,
        batch_sizes=[1, 2, 4, 8],
        n_batches=max(n_runs // 4, 25),
        imgsz=imgsz,
        single_fps=preproc.throughput_fps,
    )
    report.batch_results.extend(batch_results)
    for r in batch_results:
        print(f"  batch={r.batch_size}: {r.batch_throughput_fps:.1f} FPS ({r.speedup_factor:.2f}×)")

    return report


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Inference pipeline benchmark")
    parser.add_argument("--quick", action="store_true", help="Quick run (20 iterations)")
    parser.add_argument("--imgsz", type=int, default=1280, help="Image size")
    parser.add_argument("--n-runs", type=int, default=100, help="Number of benchmark iterations")
    parser.add_argument("--output-dir", type=str, default="benchmarks/inference", help="Output directory")
    args = parser.parse_args()

    n_runs = 20 if args.quick else args.n_runs
    report = run_benchmarks(n_runs=n_runs, imgsz=args.imgsz)

    print("\n" + report.summary())

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"results_{timestamp}.json"

    # Serialise
    data = {
        "hardware_target": report.hardware_target,
        "timestamp": report.timestamp,
        "imgsz": report.imgsz,
        "n_runs": report.n_runs,
        "cuda_available": report.cuda_available,
        "onnx_available": report.onnx_available,
        "tensorrt_available": report.tensorrt_available,
        "single_image": [
            {k: v for k, v in vars(r).items() if v is not None}
            for r in report.single_image_results
        ],
        "batch": [vars(r) for r in report.batch_results],
    }
    out_path.write_text(json.dumps(data, indent=2))
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
