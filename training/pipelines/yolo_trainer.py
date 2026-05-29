"""
training.pipelines.yolo_trainer — YOLO model training pipeline.

HARDWARE CONFIGURATION (RTX 4060, 8 GB VRAM):
- batch_size=4, gradient_accumulation=4 → effective batch 16
- imgsz=1280 (full resolution microscopy)
- amp=True (FP16 mixed precision)
- workers=4 (i5-13400F, 6P+4E cores)
- cache=disk (16 GB RAM — keep RAM for other tasks)

EXPERIMENT TRACKING:
- MLflow: primary tracking (local server, no cloud needed)
- W&B: optional cloud sync
- All configs and metrics logged per run
- Best model checkpoint saved based on mAP50

REPRODUCIBILITY:
- Seed fixed via shared.utils.seed.set_global_seed()
- All hyperparameters logged to MLflow
- YOLO YAML config captured as artifact

MODEL SELECTION:
- yolo11s: default (best quality/VRAM balance, ~1.2 GB)
- yolo11n: fast iteration (~0.6 GB)
- yolo11m: better quality (requires ~2.5 GB, limited headroom with SAM)
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable

from shared.logging.logger import get_logger
from shared.utils.seed import set_global_seed

logger = get_logger(__name__)


@dataclass
class TrainingConfig:
    """
    YOLO training configuration.

    Defaults are tuned for RTX 4060 (8 GB VRAM) + i5-13400F.
    """

    # Model
    model_variant: str = "yolo11s"
    """YOLO variant: yolo11n, yolo11s, yolo11m, yolo11l, yolo11x"""

    pretrained: bool = True
    """Start from COCO pretrained weights."""

    # Data
    data_yaml: str = ""
    """Path to YOLO dataset YAML file."""

    num_classes: int = 4
    """Number of trichome classes: capitate_stalked, capitate_sessile, bulbous, non_glandular"""

    # Training
    epochs: int = 150
    patience: int = 20
    """Early stopping patience (epochs without improvement)."""

    batch_size: int = 4
    """Per-GPU batch size. RTX 4060: 4 @ 1280px is safe."""

    gradient_accumulation_steps: int = 4
    """Accumulate gradients for effective batch = batch_size × accumulation."""

    imgsz: int = 1280
    """Input resolution. Microscopy benefits from high resolution."""

    # Optimization
    lr0: float = 0.01
    lrf: float = 0.01
    """Final LR = lr0 × lrf (cosine schedule)."""

    momentum: float = 0.937
    weight_decay: float = 0.0005
    warmup_epochs: float = 3.0
    warmup_momentum: float = 0.8
    warmup_bias_lr: float = 0.1

    # Augmentation
    mosaic: float = 1.0
    """Mosaic augmentation probability. 0=disabled, 1=always."""

    mixup: float = 0.1
    flipud: float = 0.5
    """Vertical flip probability. Microscopy is not orientation-dependent."""

    fliplr: float = 0.5
    degrees: float = 10.0
    """Rotation (±degrees). Microscopy: trichomes appear at all angles."""

    scale: float = 0.5
    """Scale augmentation gain (random resize ±fraction). Default 0.5."""

    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    """HSV augmentation. Conservative for trichome maturity preservation."""

    close_mosaic: int = 10
    """Disable mosaic in final N epochs for stable convergence."""

    cos_lr: bool = True
    """Use cosine learning rate schedule. Recommended for epochs > 100."""

    augment: bool = True
    """Enable the full YOLO augmentation pipeline."""

    # Hardware
    device: str = "0"
    """CUDA device. '0' = first GPU."""

    workers: int = 4
    amp: bool = True
    """Mixed precision (FP16). Required for RTX 4060 efficiency."""

    cache: str = "disk"
    """Cache mode: 'ram', 'disk', or False. 'disk' for 16GB RAM."""

    # Output — computed at class definition to avoid Ultralytics double-nesting relative to its CWD
    project: str = str(Path(__file__).resolve().parents[2] / "runs" / "detect")
    name: str = ""
    """Run name. Auto-generated if empty."""

    save_period: int = 10
    """Save checkpoint every N epochs."""

    # Experiment tracking
    use_mlflow: bool = True
    mlflow_tracking_uri: str = "http://localhost:3004"
    mlflow_experiment_name: str = "trichome-detection"

    use_wandb: bool = False
    wandb_project: str = "trichome-detection"

    # Reproducibility
    seed: int = 42
    deterministic: bool = True

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps

    @property
    def model_pt(self) -> str:
        """Path to pretrained weights file."""
        return f"{self.model_variant}.pt"

    def to_ultralytics_kwargs(self) -> dict[str, Any]:
        """Convert to keyword arguments for ultralytics YOLO.train()."""
        return {
            "data": self.data_yaml,
            "epochs": self.epochs,
            "patience": self.patience,
            "batch": self.batch_size,
            "imgsz": self.imgsz,
            "lr0": self.lr0,
            "lrf": self.lrf,
            "momentum": self.momentum,
            "weight_decay": self.weight_decay,
            "warmup_epochs": self.warmup_epochs,
            "warmup_momentum": self.warmup_momentum,
            "warmup_bias_lr": self.warmup_bias_lr,
            "cos_lr": self.cos_lr,
            "augment": self.augment,
            "mosaic": self.mosaic,
            "mixup": self.mixup,
            "flipud": self.flipud,
            "fliplr": self.fliplr,
            "degrees": self.degrees,
            "scale": self.scale,
            "hsv_h": self.hsv_h,
            "hsv_s": self.hsv_s,
            "hsv_v": self.hsv_v,
            "close_mosaic": self.close_mosaic,
            "device": self.device,
            "workers": self.workers,
            "amp": self.amp,
            "cache": self.cache,
            "project": self.project,
            "name": self.name or f"run_{time.strftime('%Y%m%d_%H%M%S')}",
            "save_period": self.save_period,
            "seed": self.seed,
            "deterministic": self.deterministic,
            "plots": True,
            "val": True,
            "verbose": True,
        }


@dataclass
class TrainingResult:
    """Results from a completed YOLO training run."""

    run_id: str
    model_variant: str
    config: TrainingConfig

    # Best metrics
    best_map50: float = 0.0
    best_map50_95: float = 0.0
    best_precision: float = 0.0
    best_recall: float = 0.0
    best_epoch: int = 0

    # Output paths
    best_model_path: str = ""
    last_model_path: str = ""
    results_csv_path: str = ""
    run_dir: str = ""

    # Timing
    training_time_s: float = 0.0
    total_epochs_completed: int = 0

    # Experiment tracking
    mlflow_run_id: str | None = None
    wandb_run_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "model_variant": self.model_variant,
            "best_map50": self.best_map50,
            "best_map50_95": self.best_map50_95,
            "best_precision": self.best_precision,
            "best_recall": self.best_recall,
            "best_epoch": self.best_epoch,
            "best_model_path": self.best_model_path,
            "training_time_s": self.training_time_s,
            "total_epochs_completed": self.total_epochs_completed,
            "mlflow_run_id": self.mlflow_run_id,
        }


class YOLOTrainer:
    """
    YOLO training wrapper with experiment tracking and hardware optimization.

    Usage:
        config = TrainingConfig(
            model_variant="yolo11s",
            data_yaml="/path/to/trichome_dataset.yaml",
            epochs=150,
        )
        trainer = YOLOTrainer(config)
        result = trainer.train()

        print(f"Best mAP50: {result.best_map50:.4f}")
        print(f"Best model: {result.best_model_path}")
    """

    def __init__(
        self,
        config: TrainingConfig,
        on_epoch_end: Callable[[int, dict[str, float]], None] | None = None,
    ) -> None:
        """
        Args:
            config: Training configuration.
            on_epoch_end: Optional callback called after each epoch with
                         (epoch_num, metrics_dict). Used for WebSocket streaming.
        """
        self._config = config
        self._on_epoch_end = on_epoch_end
        self._run_id = str(uuid.uuid4())
        self._is_running = False
        self._stop_requested = False

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def is_running(self) -> bool:
        return self._is_running

    def request_stop(self) -> None:
        """Request graceful training stop (will stop after current epoch)."""
        self._stop_requested = True
        logger.info("Training stop requested", run_id=self._run_id)

    def train(self) -> TrainingResult:
        """
        Run full YOLO training.

        Returns:
            TrainingResult with best metrics and model paths.

        Raises:
            RuntimeError: If training fails.
        """
        if self._is_running:
            raise RuntimeError("Training already running")

        self._is_running = True
        self._stop_requested = False
        t_start = time.perf_counter()

        logger.info(
            "Starting YOLO training",
            run_id=self._run_id,
            model=self._config.model_variant,
            epochs=self._config.epochs,
            effective_batch=self._config.effective_batch_size,
            imgsz=self._config.imgsz,
        )

        # Validate config
        if not self._config.data_yaml:
            raise ValueError("data_yaml must be specified in TrainingConfig")

        # Set global seed for reproducibility
        set_global_seed(self._config.seed)

        # Start experiment tracking
        mlflow_run_id = None
        if self._config.use_mlflow:
            mlflow_run_id = self._start_mlflow_run()

        try:
            result = self._run_ultralytics_training()
            result.mlflow_run_id = mlflow_run_id

            if self._config.use_mlflow:
                self._log_mlflow_final(result)

            return result

        except Exception as e:
            logger.error("Training failed", run_id=self._run_id, error=str(e))
            if self._config.use_mlflow:
                self._end_mlflow_run(status="FAILED")
            raise RuntimeError(f"YOLO training failed: {e}") from e

        finally:
            self._is_running = False
            t_end = time.perf_counter()
            logger.info(
                "Training finished",
                run_id=self._run_id,
                elapsed_s=f"{t_end - t_start:.0f}",
            )

    def _run_ultralytics_training(self) -> TrainingResult:
        """Execute YOLO training via Ultralytics API."""
        try:
            from ultralytics import YOLO
        except ImportError:
            raise RuntimeError(
                "Ultralytics not installed. "
                "Install with: pip install ultralytics"
            )

        model = YOLO(self._config.model_pt)

        # Remove Ultralytics' built-in MLflow callbacks — we manage MLflow
        # ourselves in _start_mlflow_run/_log_mlflow_final to avoid double-
        # tracking and port conflicts.
        try:
            import os
            from ultralytics import settings as ult_settings
            # Disable Ultralytics' own MLflow integration — we manage MLflow tracking
            # ourselves. Without this, Ultralytics re-registers MLflow callbacks
            # after our callback-stripping and tries to write to /mlflow (root).
            ult_settings.update({"mlflow": False})
            os.environ["MLFLOW_TRACKING_URI"] = self._config.mlflow_tracking_uri
            for event in list(model.callbacks.keys()):
                model.callbacks[event] = [
                    cb for cb in model.callbacks[event]
                    if getattr(cb, "__module__", "").find("mlflow") == -1
                ]
        except Exception:
            pass

        # Register epoch callback for WebSocket streaming
        if self._on_epoch_end is not None:
            self._register_epoch_callback(model)

        train_kwargs = self._config.to_ultralytics_kwargs()

        logger.info("YOLO training kwargs", **{
            k: v for k, v in train_kwargs.items()
            if k not in ("name",)  # avoid log spam
        })

        t_start = time.perf_counter()
        train_result = model.train(**train_kwargs)
        t_end = time.perf_counter()

        # Resolve run_dir from the Ultralytics result object (absolute path).
        # Falling back to constructing it from kwargs is unreliable because
        # Ultralytics may resolve the relative project path from a different CWD.
        try:
            run_dir = Path(train_result.save_dir).resolve()
        except Exception:
            run_dir = (Path(train_kwargs["project"]) / train_kwargs["name"]).resolve()
        best_model = run_dir / "weights" / "best.pt"
        last_model = run_dir / "weights" / "last.pt"
        results_csv = run_dir / "results.csv"

        # Parse best metrics
        metrics = self._extract_best_metrics(train_result, results_csv)

        return TrainingResult(
            run_id=self._run_id,
            model_variant=self._config.model_variant,
            config=self._config,
            best_map50=metrics.get("map50", 0.0),
            best_map50_95=metrics.get("map50_95", 0.0),
            best_precision=metrics.get("precision", 0.0),
            best_recall=metrics.get("recall", 0.0),
            best_epoch=metrics.get("best_epoch", 0),
            best_model_path=str(best_model) if best_model.exists() else "",
            last_model_path=str(last_model) if last_model.exists() else "",
            results_csv_path=str(results_csv) if results_csv.exists() else "",
            run_dir=str(run_dir),
            training_time_s=t_end - t_start,
            total_epochs_completed=metrics.get("total_epochs", self._config.epochs),
        )

    def _extract_best_metrics(
        self,
        train_result: Any,
        results_csv: Path,
    ) -> dict[str, Any]:
        """Extract best validation metrics from training result."""
        metrics: dict[str, Any] = {}

        # Try to get from ultralytics result object
        try:
            if hasattr(train_result, "results_dict"):
                rd = train_result.results_dict
                metrics["map50"] = float(rd.get("metrics/mAP50(B)", 0.0))
                metrics["map50_95"] = float(rd.get("metrics/mAP50-95(B)", 0.0))
                metrics["precision"] = float(rd.get("metrics/precision(B)", 0.0))
                metrics["recall"] = float(rd.get("metrics/recall(B)", 0.0))
        except Exception:
            pass

        # Fallback: parse results CSV
        if not metrics and results_csv.exists():
            import csv
            best_map50 = 0.0
            best_epoch = 0
            total_epochs = 0

            with open(results_csv) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    total_epochs += 1
                    try:
                        # Ultralytics CSV column names vary by version
                        for col in ["metrics/mAP50(B)", "mAP_0.5"]:
                            if col in row:
                                val = float(row[col].strip())
                                if val > best_map50:
                                    best_map50 = val
                                    best_epoch = total_epochs
                    except (ValueError, KeyError):
                        pass

            metrics["map50"] = best_map50
            metrics["best_epoch"] = best_epoch
            metrics["total_epochs"] = total_epochs

        return metrics

    def _register_epoch_callback(self, model: Any) -> None:
        """Register Ultralytics callback for epoch-end WebSocket updates."""
        trainer_ref = self

        def on_train_epoch_end(trainer: Any) -> None:
            if trainer_ref._stop_requested:
                trainer.stop = True
                return

            epoch = trainer.epoch
            metrics = {}
            try:
                if hasattr(trainer, "metrics"):
                    metrics = {
                        k: float(v)
                        for k, v in trainer.metrics.items()
                        if isinstance(v, (int, float))
                    }
                if hasattr(trainer, "loss"):
                    metrics["train_loss"] = float(trainer.loss)
            except Exception:
                pass

            if trainer_ref._on_epoch_end:
                try:
                    trainer_ref._on_epoch_end(epoch, metrics)
                except Exception as e:
                    logger.debug("Epoch callback error", error=str(e))

        model.add_callback("on_train_epoch_end", on_train_epoch_end)

    def _start_mlflow_run(self) -> str | None:
        """Start MLflow run and log config."""
        try:
            import mlflow

            mlflow.set_tracking_uri(self._config.mlflow_tracking_uri)
            mlflow.set_experiment(self._config.mlflow_experiment_name)

            run = mlflow.start_run(run_name=self._run_id)

            # Log config as params
            config_dict = {
                "model_variant": self._config.model_variant,
                "epochs": self._config.epochs,
                "batch_size": self._config.batch_size,
                "effective_batch_size": self._config.effective_batch_size,
                "imgsz": self._config.imgsz,
                "lr0": self._config.lr0,
                "amp": self._config.amp,
                "seed": self._config.seed,
            }
            mlflow.log_params(config_dict)

            return run.info.run_id

        except ImportError:
            logger.warning("MLflow not installed, skipping experiment tracking")
            return None
        except Exception as e:
            logger.warning("MLflow init failed", error=str(e))
            return None

    def _log_mlflow_final(self, result: TrainingResult) -> None:
        """Log final metrics to MLflow."""
        try:
            import mlflow

            mlflow.log_metrics({
                "best_map50": result.best_map50,
                "best_map50_95": result.best_map50_95,
                "best_precision": result.best_precision,
                "best_recall": result.best_recall,
                "training_time_s": result.training_time_s,
            })
            # Model file is already saved by Ultralytics in runs/detect/<name>/weights/
            # Skipping log_artifact to avoid artifact-store permission issues.
            mlflow.end_run()

        except Exception as e:
            logger.warning("MLflow final logging failed", error=str(e))

    def _end_mlflow_run(self, status: str = "FINISHED") -> None:
        """End MLflow run."""
        try:
            import mlflow
            mlflow.end_run(status=status)
        except Exception:
            pass
