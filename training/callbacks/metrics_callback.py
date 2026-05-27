"""
training.callbacks.metrics_callback — MLflow + WebSocket metrics callback for YOLO training.

Integrates with:
1. MLflow: logs metrics per epoch (loss, mAP50, precision, recall)
2. WebSocket: broadcasts live metrics to frontend via backend.websocket.manager
3. Optionally W&B: if WANDB_API_KEY env var is set

Design: callback is called by YOLOTrainer after each epoch via
model.add_callback("on_train_epoch_end", callback_fn).

All I/O is best-effort — training continues even if logging fails.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metric key normalization
# ---------------------------------------------------------------------------

# YOLO epoch result dict keys → human-readable names
YOLO_METRIC_KEYS: dict[str, str] = {
    "train/box_loss": "box_loss",
    "train/cls_loss": "cls_loss",
    "train/dfl_loss": "dfl_loss",
    "metrics/precision(B)": "precision",
    "metrics/recall(B)": "recall",
    "metrics/mAP50(B)": "mAP50",
    "metrics/mAP50-95(B)": "mAP50_95",
    "val/box_loss": "val_box_loss",
    "val/cls_loss": "val_cls_loss",
    "val/dfl_loss": "val_dfl_loss",
    "lr/pg0": "lr",
}


def normalize_metrics(raw: dict[str, Any]) -> dict[str, float]:
    """
    Normalize YOLO metric dict to clean float values.

    Handles:
    - torch.Tensor values → float
    - numpy scalar values → float
    - Key renaming from YOLO internal names to readable names
    """
    result: dict[str, float] = {}

    for raw_key, clean_key in YOLO_METRIC_KEYS.items():
        value = raw.get(raw_key)
        if value is None:
            continue
        try:
            if hasattr(value, "item"):
                value = value.item()
            result[clean_key] = float(value)
        except (TypeError, ValueError):
            pass

    # Also pass through any keys not in the map
    for key, value in raw.items():
        if key not in YOLO_METRIC_KEYS:
            try:
                if hasattr(value, "item"):
                    value = value.item()
                result[key] = float(value)
            except (TypeError, ValueError):
                pass

    return result


# ---------------------------------------------------------------------------
# Callback class
# ---------------------------------------------------------------------------

@dataclass
class MetricsCallbackConfig:
    """Configuration for the metrics callback."""

    mlflow_run_id: str | None = None
    """Active MLflow run ID. Set by YOLOTrainer before training starts."""

    run_uuid: str = ""
    """Backend BackgroundJob UUID for WebSocket broadcasting."""

    log_to_mlflow: bool = True
    log_to_websocket: bool = True
    log_to_wandb: bool = False

    # WebSocket broadcast interval (skip if epoch < interval)
    ws_broadcast_every_n_epochs: int = 1

    # Extra custom metrics computed per epoch
    custom_metric_fns: list[Callable[[int, dict[str, float]], dict[str, float]]] = field(
        default_factory=list
    )


class MetricsCallback:
    """
    Epoch-end metrics callback for YOLO training.

    Logs to MLflow and broadcasts to WebSocket simultaneously.
    All operations are best-effort (failures are logged, not raised).

    Usage::

        callback = MetricsCallback(config)

        # Register with YOLO model
        model.add_callback("on_train_epoch_end", callback.on_epoch_end)

        # After training
        callback.get_history()  # full metrics history
    """

    def __init__(self, config: MetricsCallbackConfig | None = None) -> None:
        self.config = config or MetricsCallbackConfig()
        self._history: list[dict[str, Any]] = []
        self._best_map50: float = 0.0
        self._start_time = time.monotonic()
        self._mlflow_client: Any | None = None

    def on_epoch_end(self, trainer: Any) -> None:
        """
        Called by YOLO after each training epoch.

        Args:
            trainer: Ultralytics trainer object with .metrics and .epoch attributes.
        """
        try:
            epoch = int(getattr(trainer, "epoch", 0))
            raw_metrics = dict(getattr(trainer, "metrics", {}) or {})

            # Add epoch-level training losses if available
            loss_items = getattr(trainer, "loss_items", None)
            if loss_items is not None:
                try:
                    raw_metrics["train/box_loss"] = float(loss_items[0])
                    raw_metrics["train/cls_loss"] = float(loss_items[1])
                    raw_metrics["train/dfl_loss"] = float(loss_items[2])
                except (IndexError, TypeError):
                    pass

            # Learning rate
            if hasattr(trainer, "optimizer") and trainer.optimizer is not None:
                try:
                    raw_metrics["lr/pg0"] = trainer.optimizer.param_groups[0]["lr"]
                except (KeyError, IndexError):
                    pass

            metrics = normalize_metrics(raw_metrics)

            # Custom metrics
            for fn in self.config.custom_metric_fns:
                try:
                    extra = fn(epoch, metrics)
                    metrics.update(extra)
                except Exception as e:
                    logger.debug("Custom metric fn failed: %s", e)

            # Track best mAP50
            if "mAP50" in metrics:
                if metrics["mAP50"] > self._best_map50:
                    self._best_map50 = metrics["mAP50"]

            # Store history
            record = {"epoch": epoch, "timestamp": time.monotonic() - self._start_time, **metrics}
            self._history.append(record)

            # Log to MLflow
            if self.config.log_to_mlflow:
                self._log_mlflow(epoch, metrics)

            # Broadcast via WebSocket
            if (
                self.config.log_to_websocket
                and epoch % self.config.ws_broadcast_every_n_epochs == 0
            ):
                self._broadcast_websocket(epoch, metrics)

            # Log to W&B
            if self.config.log_to_wandb:
                self._log_wandb(epoch, metrics)

            logger.debug(
                "Epoch %d: mAP50=%.4f, box_loss=%.4f",
                epoch,
                metrics.get("mAP50", 0),
                metrics.get("box_loss", 0),
            )

        except Exception as e:
            logger.warning("MetricsCallback.on_epoch_end failed: %s", e)

    def _log_mlflow(self, epoch: int, metrics: dict[str, float]) -> None:
        """Log metrics directly to an MLflow run by ID.

        Uses MlflowClient.log_metric() to avoid creating nested runs when called
        inside an already-active run context (MLflow 3.x behaviour).
        """
        try:
            import mlflow

            if self.config.mlflow_run_id:
                client = mlflow.tracking.MlflowClient()
                for key, value in metrics.items():
                    client.log_metric(self.config.mlflow_run_id, key, value, step=epoch)
        except Exception as e:
            logger.debug("MLflow logging failed: %s", e)

    def _broadcast_websocket(self, epoch: int, metrics: dict[str, float]) -> None:
        """Broadcast training metrics to frontend via WebSocket."""
        try:
            import asyncio

            from backend.websocket.manager import ws_manager

            payload = {
                "type": "training_metrics",
                "epoch": epoch,
                "run_uuid": self.config.run_uuid,
                "metrics": metrics,
                "best_map50": self._best_map50,
                "elapsed_s": self._history[-1]["timestamp"] if self._history else 0,
            }

            # Run async broadcast in sync context
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(ws_manager.broadcast_to_topic("training", payload))
            else:
                loop.run_until_complete(ws_manager.broadcast_to_topic("training", payload))

        except Exception as e:
            logger.debug("WebSocket broadcast failed: %s", e)

    def _log_wandb(self, epoch: int, metrics: dict[str, float]) -> None:
        """Log to Weights & Biases if available."""
        try:
            import wandb
            if wandb.run is not None:
                wandb.log({**metrics, "epoch": epoch}, step=epoch)
        except Exception as e:
            logger.debug("W&B logging failed: %s", e)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_history(self) -> list[dict[str, Any]]:
        """Return full training history."""
        return list(self._history)

    @property
    def best_map50(self) -> float:
        """Best mAP50 seen during training."""
        return self._best_map50

    def get_final_metrics(self) -> dict[str, float]:
        """Return the last epoch's metrics."""
        if not self._history:
            return {}
        last = self._history[-1]
        return {k: v for k, v in last.items() if isinstance(v, float)}

    def reset(self) -> None:
        """Reset state for a new training run."""
        self._history.clear()
        self._best_map50 = 0.0
        self._start_time = time.monotonic()
