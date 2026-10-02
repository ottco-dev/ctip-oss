"""
Run one job in its own process: `python -m apps.worker.run_job <workdir>`.

<workdir>/job.json    {"job_id", "spec", "inputs": {...}, "device", "batch", "threads"}
<workdir>/progress.json   written atomically after every epoch / step
<workdir>/result.json     final metrics and result files
Only the job kinds of shared.compute.schemas exist; the spec is validated again here.
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
from pathlib import Path

from pydantic import TypeAdapter

from shared.compute.schemas import BenchmarkSpec, JobSpec, YoloEvalSpec, YoloTrainSpec


def write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data))
    os.replace(tmp, path)


class Run:
    def __init__(self, workdir: Path) -> None:
        self.dir = workdir
        self.job = json.loads((workdir / "job.json").read_text())
        self.spec = TypeAdapter(JobSpec).validate_python(self.job["spec"])
        self.inputs = self.job.get("inputs", {})
        self.device = self.job.get("device", "cpu")

    def progress(self, progress: float, epoch: int | None = None, metrics: dict | None = None, message: str = "") -> None:
        write_json(self.dir / "progress.json", {"progress": round(min(max(progress, 0), 1), 4), "epoch": epoch,
                                                "metrics": {k: round(float(v), 5) for k, v in (metrics or {}).items()},
                                                "message": message[:300], "t": time.time()})

    def result(self, metrics: dict, files: dict[str, str], message: str = "") -> None:
        write_json(self.dir / "result.json", {"metrics": {k: round(float(v), 5) for k, v in metrics.items()},
                                              "files": files, "message": message})


def _metrics(d: dict) -> dict:
    keys = {"metrics/mAP50(B)": "map50", "metrics/mAP50-95(B)": "map50_95", "metrics/precision(B)": "precision",
            "metrics/recall(B)": "recall", "train/box_loss": "box_loss", "val/box_loss": "val_box_loss"}
    return {short: float(d[k]) for k, short in keys.items() if k in d and d[k] == d[k]}


def yolo_train(run: Run, spec: YoloTrainSpec) -> None:
    from ultralytics import YOLO

    from training.pipelines.yolo_trainer import TrainingConfig

    project, name = run.dir / "runs", "train"
    ckpt = run.inputs.get("checkpoint")
    if ckpt:                                           # resume elsewhere: point the stored training args at this machine
        import torch

        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        args = state.get("train_args", {})
        args.update(data=run.inputs["data_yaml"], project=str(project), name=name, device=run.device, exist_ok=True,
                    workers=run.job.get("loader_workers", 2))
        state["train_args"] = args
        torch.save(state, ckpt)
        model = YOLO(ckpt)
    else:
        model = YOLO(f"{spec.model}.pt")

    def on_epoch(trainer) -> None:
        ep = trainer.epoch + 1
        m = _metrics({**trainer.metrics, **trainer.label_loss_items(trainer.tloss, prefix="train")})
        run.progress(ep / trainer.epochs, ep, m, f"epoch {ep}/{trainer.epochs}")

    model.add_callback("on_fit_epoch_end", on_epoch)
    run.progress(0.0, 0, message="starting")
    if ckpt:
        model.train(resume=True)
    else:
        batch = run.job.get("batch", 4)
        if isinstance(batch, float):                   # share of free GPU memory -> a whole batch size (AutoBatch)
            from ultralytics.utils.autobatch import check_train_batch_size

            batch = max(1, int(check_train_batch_size(model.model.to(f"cuda:{run.device.split(',')[0]}"), spec.imgsz, True, batch)))
            run.progress(0.0, 0, message=f"batch {batch} from free VRAM")
        base = TrainingConfig(model_variant=spec.model, num_classes=1, epochs=spec.epochs, patience=spec.patience,
                              imgsz=spec.imgsz, seed=spec.seed, degrees=spec.degrees, mosaic=spec.mosaic,
                              batch_size=batch, gradient_accumulation_steps=1, use_mlflow=False)
        kw = base.to_ultralytics_kwargs()
        kw.update(data=run.inputs["data_yaml"], project=str(project), name=name, exist_ok=True, device=run.device,
                  workers=run.job.get("loader_workers", 2), save_period=-1, plots=False, cache=False, verbose=False)
        model.train(**{k: v for k, v in kw.items() if v is not None})
    w = project / name / "weights"
    rows = []
    csv = project / name / "results.csv"
    if csv.exists():
        import csv as _csv

        rows = list(_csv.DictReader(csv.open()))
    best = max(rows, key=lambda r: float(r.get("metrics/mAP50(B)", 0) or 0)) if rows else {}
    run.result(_metrics(best) | {"epochs_run": float(len(rows))},
               {k: str(p) for k, p in {"best": w / "best.pt", "last": w / "last.pt", "results": csv}.items() if p.exists()})


def yolo_eval(run: Run, spec: YoloEvalSpec) -> None:
    from ultralytics import YOLO

    run.progress(0.1, message="evaluating")
    v = YOLO(run.inputs["model"]).val(data=run.inputs["data_yaml"], split=spec.split, imgsz=spec.imgsz, device=run.device,
                                      batch=1, plots=False, verbose=False, project=str(run.dir / "runs"), name="eval")
    run.result({"map50": v.box.map50, "map50_95": v.box.map, "precision": float(v.box.mp), "recall": float(v.box.mr)}, {})


def benchmark(run: Run, spec: BenchmarkSpec) -> None:
    import numpy as np
    import torch
    from ultralytics import YOLO

    from benchmarks.detection.gpu_detection_benchmark import synthetic_frame

    model = YOLO(f"{spec.model}.pt")
    frame = synthetic_frame(spec.imgsz, spec.imgsz)
    cuda = run.device not in ("cpu", "mps")
    half = cuda
    for _ in range(3):
        model.predict(frame, imgsz=spec.imgsz, device=run.device, half=half, verbose=False)
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    times = []
    for i in range(spec.runs):
        t0 = time.perf_counter()
        model.predict(frame, imgsz=spec.imgsz, device=run.device, half=half, verbose=False)
        if cuda:
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
        run.progress((i + 1) / spec.runs, message=f"run {i + 1}/{spec.runs}")
    times.sort()
    m = {"p50_ms": statistics.median(times), "p95_ms": times[min(len(times) - 1, int(0.95 * len(times)))],
         "fps": 1000 / float(np.mean(times))}
    if cuda:
        m["peak_vram_mb"] = torch.cuda.max_memory_allocated() / 2**20
    run.result(m, {})


def main() -> int:
    workdir = Path(sys.argv[1]).resolve()
    run = Run(workdir)
    weights = Path(run.job.get("weights_dir") or workdir / "weights")
    weights.mkdir(parents=True, exist_ok=True)
    os.chdir(weights)                      # Ultralytics downloads base weights (yolo11s.pt …) into the cwd: cache them here
    threads = int(run.job.get("threads", 2))
    try:
        import torch

        torch.set_num_threads(threads)
    except Exception:
        pass
    dispatch = {"yolo_train": yolo_train, "yolo_eval": yolo_eval, "detection_benchmark": benchmark}
    dispatch[run.spec.kind](run, run.spec)
    return 0


if __name__ == "__main__":
    sys.exit(main())
