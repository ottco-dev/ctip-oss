"""
training.callbacks.checkpoint_callback — Model checkpointing during YOLO training.

Responsibilities:
1. Save best model checkpoint (highest mAP50)
2. Save periodic checkpoints (every N epochs)
3. Register saved checkpoints in the model registry (SQLite via SQLModel)
4. Trigger WebSocket notification when a new best is saved
5. Apply early stopping (optional, YOLO also has patience built-in)

YOLO already saves runs/detect/<name>/weights/best.pt and last.pt.
This callback adds:
  - Custom checkpoint directory with human-readable names
  - MLflow artifact logging
  - Registry integration for the web UI model browser
"""

from __future__ import annotations

import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CheckpointCallbackConfig:
    """Configuration for the checkpoint callback."""

    save_dir: str = "checkpoints"
    """Directory to save checkpoints (relative to working directory)."""

    run_name: str = "run"
    """Human-readable name prefix for checkpoint files."""

    save_best: bool = True
    """Save checkpoint when mAP50 improves."""

    save_every_n_epochs: int = 0
    """Save checkpoint every N epochs (0 = disable periodic saving)."""

    min_map50_to_save: float = 0.0
    """Only save best checkpoint if mAP50 exceeds this threshold."""

    # MLflow artifact logging
    log_to_mlflow: bool = True
    mlflow_run_id: str | None = None

    # Model registry
    register_in_db: bool = True
    model_type: str = "detection"
    framework: str = "pytorch"
    num_classes: int = 4


class CheckpointCallback:
    """
    Checkpoint management callback for YOLO training.

    Usage::

        callback = CheckpointCallback(config)
        model.add_callback("on_train_epoch_end", callback.on_epoch_end)
        model.add_callback("on_train_end", callback.on_train_end)
    """

    def __init__(self, config: CheckpointCallbackConfig | None = None) -> None:
        self.config = config or CheckpointCallbackConfig()
        self._best_map50: float = 0.0
        self._best_checkpoint_path: Path | None = None
        self._checkpoints: list[dict] = []
        self._save_dir = Path(self.config.save_dir) / self.config.run_name
        self._save_dir.mkdir(parents=True, exist_ok=True)
        logger.info("CheckpointCallback: saving to %s", self._save_dir)

    def on_epoch_end(self, trainer: Any) -> None:
        """Called after each training epoch by YOLO trainer."""
        try:
            epoch = int(getattr(trainer, "epoch", 0))
            metrics = dict(getattr(trainer, "metrics", {}) or {})

            # Extract mAP50
            map50 = 0.0
            for key in ["metrics/mAP50(B)", "mAP50", "val/mAP50"]:
                if key in metrics:
                    try:
                        map50 = float(
                            metrics[key].item()
                            if hasattr(metrics[key], "item")
                            else metrics[key]
                        )
                        break
                    except (TypeError, ValueError):
                        pass

            # Best checkpoint
            if (
                self.config.save_best
                and map50 > self._best_map50
                and map50 >= self.config.min_map50_to_save
            ):
                self._save_best(trainer, epoch, map50, metrics)

            # Periodic checkpoint
            if (
                self.config.save_every_n_epochs > 0
                and epoch > 0
                and epoch % self.config.save_every_n_epochs == 0
            ):
                self._save_periodic(trainer, epoch, map50)

        except Exception as e:
            logger.warning("CheckpointCallback.on_epoch_end failed: %s", e)

    def on_train_end(self, trainer: Any) -> None:
        """Called at end of training — log final checkpoint to MLflow."""
        try:
            if self._best_checkpoint_path and self.config.log_to_mlflow:
                self._log_artifact_mlflow(self._best_checkpoint_path)
        except Exception as e:
            logger.warning("CheckpointCallback.on_train_end failed: %s", e)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _save_best(
        self,
        trainer: Any,
        epoch: int,
        map50: float,
        metrics: dict,
    ) -> None:
        """Save best checkpoint and update registry."""
        prev_best = self._best_map50
        self._best_map50 = map50

        # Get YOLO's best weights path
        yolo_best = self._get_yolo_weights_path(trainer, "best")
        if yolo_best is None or not yolo_best.exists():
            logger.debug("YOLO best weights not yet available at epoch %d", epoch)
            return

        # Copy to our checkpoint directory
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        ckpt_name = f"{self.config.run_name}_best_ep{epoch:03d}_map{map50:.4f}_{timestamp}.pt"
        dest = self._save_dir / ckpt_name
        shutil.copy2(str(yolo_best), str(dest))

        logger.info(
            "New best checkpoint: mAP50=%.4f (prev=%.4f) → %s",
            map50, prev_best, dest,
        )
        self._best_checkpoint_path = dest
        self._checkpoints.append({"epoch": epoch, "map50": map50, "path": str(dest), "type": "best"})

        # Register in model registry
        if self.config.register_in_db:
            self._register_checkpoint(dest, epoch, map50)

        # Notify WebSocket
        self._notify_websocket(epoch, map50, str(dest))

    def _save_periodic(self, trainer: Any, epoch: int, map50: float) -> None:
        """Save periodic checkpoint."""
        yolo_last = self._get_yolo_weights_path(trainer, "last")
        if yolo_last is None or not yolo_last.exists():
            return

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        ckpt_name = f"{self.config.run_name}_ep{epoch:03d}_{timestamp}.pt"
        dest = self._save_dir / ckpt_name
        shutil.copy2(str(yolo_last), str(dest))

        logger.debug("Periodic checkpoint saved: %s", dest)
        self._checkpoints.append({"epoch": epoch, "map50": map50, "path": str(dest), "type": "periodic"})

    @staticmethod
    def _get_yolo_weights_path(trainer: Any, which: str) -> Path | None:
        """Extract YOLO best/last weights path from trainer."""
        try:
            # Ultralytics trainer stores save_dir
            save_dir = getattr(trainer, "save_dir", None)
            if save_dir:
                p = Path(save_dir) / "weights" / f"{which}.pt"
                if p.exists():
                    return p
            # Fallback: check last attribute
            weights = getattr(trainer, f"{which}", None)
            if weights and Path(str(weights)).exists():
                return Path(str(weights))
        except Exception:
            pass
        return None

    def _register_checkpoint(self, path: Path, epoch: int, map50: float) -> None:
        """Register checkpoint in SQLite model registry."""
        try:
            import json
            from backend.database import get_engine
            from backend.models.model_registry import RegisteredModel
            from sqlmodel import Session

            engine = get_engine()
            with Session(engine) as session:
                record = RegisteredModel(
                    name=f"{self.config.run_name}_ep{epoch}",
                    model_type=self.config.model_type,
                    framework=self.config.framework,
                    variant=f"epoch_{epoch}",
                    file_path=str(path),
                    vram_required_gb=1.2,  # YOLO11s estimate
                    metrics_json=json.dumps({"mAP50": map50, "epoch": epoch}),
                )
                session.add(record)
                session.commit()
                logger.debug("Registered checkpoint in model registry: %s", path.name)

        except Exception as e:
            logger.debug("Failed to register checkpoint in DB: %s", e)

    def _notify_websocket(self, epoch: int, map50: float, path: str) -> None:
        """Broadcast new best checkpoint to WebSocket clients."""
        try:
            import asyncio
            from backend.websocket.manager import ws_manager

            payload = {
                "type": "new_best_checkpoint",
                "epoch": epoch,
                "map50": map50,
                "path": path,
            }
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(ws_manager.broadcast_to_topic("training", payload))
        except Exception as e:
            logger.debug("WebSocket notify failed: %s", e)

    def _log_artifact_mlflow(self, path: Path) -> None:
        """Log checkpoint as MLflow artifact.

        Uses MlflowClient.log_artifact() to avoid creating nested runs when called
        inside an already-active run context (MLflow 3.x behaviour).
        """
        try:
            import mlflow
            if self.config.mlflow_run_id:
                client = mlflow.tracking.MlflowClient()
                client.log_artifact(self.config.mlflow_run_id, str(path), artifact_path="checkpoints")
                logger.debug("Logged checkpoint artifact to MLflow: %s", path.name)
        except Exception as e:
            logger.debug("MLflow artifact logging failed: %s", e)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def best_map50(self) -> float:
        return self._best_map50

    @property
    def best_checkpoint_path(self) -> Path | None:
        return self._best_checkpoint_path

    def get_checkpoints(self) -> list[dict]:
        """Return all saved checkpoint records."""
        return list(self._checkpoints)
