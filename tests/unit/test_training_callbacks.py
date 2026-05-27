"""
tests.unit.test_training_callbacks — Unit tests for training metrics callback.

Tests:
  - normalize_metrics: YOLO key remapping, tensor handling, float coercion
  - MetricsCallback: history accumulation, best mAP50 tracking
  - MetricsCallback: graceful handling of missing/malformed trainer attributes
  - MetricsCallbackConfig: defaults and validation
"""

from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from training.callbacks.metrics_callback import (
    MetricsCallback,
    MetricsCallbackConfig,
    normalize_metrics,
)


# ---------------------------------------------------------------------------
# normalize_metrics
# ---------------------------------------------------------------------------

class TestNormalizeMetrics:
    """Tests for the YOLO metric key normalization function."""

    def test_empty_dict_returns_empty(self):
        assert normalize_metrics({}) == {}

    def test_renames_box_loss(self):
        result = normalize_metrics({"train/box_loss": 0.42})
        assert "box_loss" in result
        assert abs(result["box_loss"] - 0.42) < 1e-9

    def test_renames_map50(self):
        result = normalize_metrics({"metrics/mAP50(B)": 0.85})
        assert "mAP50" in result
        assert abs(result["mAP50"] - 0.85) < 1e-9

    def test_renames_precision_recall(self):
        raw = {
            "metrics/precision(B)": 0.91,
            "metrics/recall(B)": 0.87,
        }
        result = normalize_metrics(raw)
        assert "precision" in result
        assert "recall" in result

    def test_renames_val_box_loss(self):
        result = normalize_metrics({"val/box_loss": 0.33})
        assert "val_box_loss" in result

    def test_handles_tensor_like_values(self):
        """Handles objects with .item() method (torch.Tensor-like)."""
        mock_tensor = MagicMock()
        mock_tensor.item.return_value = 0.77
        result = normalize_metrics({"train/box_loss": mock_tensor})
        assert abs(result["box_loss"] - 0.77) < 1e-9

    def test_skips_none_values(self):
        """Keys not present in raw dict are skipped."""
        result = normalize_metrics({"train/cls_loss": 0.1})
        assert "box_loss" not in result
        assert "cls_loss" in result

    def test_passes_through_unknown_keys(self):
        """Keys not in the rename map are preserved with their original name."""
        result = normalize_metrics({"custom_metric": 42.0})
        assert "custom_metric" in result
        assert result["custom_metric"] == 42.0

    def test_coerces_to_float(self):
        result = normalize_metrics({"train/box_loss": 1})
        assert isinstance(result["box_loss"], float)

    def test_handles_all_standard_yolo_keys(self):
        raw = {
            "train/box_loss": 0.1,
            "train/cls_loss": 0.2,
            "train/dfl_loss": 0.3,
            "metrics/precision(B)": 0.91,
            "metrics/recall(B)": 0.87,
            "metrics/mAP50(B)": 0.85,
            "metrics/mAP50-95(B)": 0.61,
            "val/box_loss": 0.4,
            "val/cls_loss": 0.5,
            "val/dfl_loss": 0.6,
            "lr/pg0": 0.001,
        }
        result = normalize_metrics(raw)
        assert len(result) == 11
        assert "mAP50" in result
        assert "lr" in result


# ---------------------------------------------------------------------------
# MetricsCallbackConfig
# ---------------------------------------------------------------------------

class TestMetricsCallbackConfig:
    def test_defaults(self):
        cfg = MetricsCallbackConfig()
        assert cfg.mlflow_run_id is None
        assert cfg.log_to_mlflow is True
        assert cfg.log_to_websocket is True
        assert cfg.log_to_wandb is False
        assert cfg.ws_broadcast_every_n_epochs == 1
        assert cfg.custom_metric_fns == []

    def test_custom_init(self):
        cfg = MetricsCallbackConfig(
            mlflow_run_id="abc123",
            log_to_mlflow=False,
            log_to_wandb=True,
        )
        assert cfg.mlflow_run_id == "abc123"
        assert cfg.log_to_mlflow is False
        assert cfg.log_to_wandb is True


# ---------------------------------------------------------------------------
# MetricsCallback — history accumulation
# ---------------------------------------------------------------------------

class TestMetricsCallbackHistory:
    """Tests for MetricsCallback metric recording without external I/O."""

    def _make_trainer(self, epoch: int, metrics: dict, loss_items=None) -> SimpleNamespace:
        """Build a mock Ultralytics trainer object."""
        return SimpleNamespace(
            epoch=epoch,
            metrics=metrics,
            loss_items=loss_items,
            optimizer=None,
        )

    def _make_callback(self, **kwargs) -> MetricsCallback:
        cfg = MetricsCallbackConfig(
            log_to_mlflow=False,
            log_to_websocket=False,
            log_to_wandb=False,
            **kwargs,
        )
        return MetricsCallback(cfg)

    def test_history_accumulates_across_epochs(self):
        cb = self._make_callback()
        for epoch in range(5):
            trainer = self._make_trainer(
                epoch=epoch,
                metrics={"metrics/mAP50(B)": 0.5 + epoch * 0.05},
            )
            cb.on_epoch_end(trainer)
        assert len(cb.get_history()) == 5

    def test_best_map50_tracked_correctly(self):
        cb = self._make_callback()
        map_values = [0.50, 0.72, 0.68, 0.81, 0.79]
        for epoch, val in enumerate(map_values):
            trainer = self._make_trainer(
                epoch=epoch,
                metrics={"metrics/mAP50(B)": val},
            )
            cb.on_epoch_end(trainer)
        assert abs(cb.best_map50 - 0.81) < 1e-9

    def test_history_includes_epoch_number(self):
        cb = self._make_callback()
        trainer = self._make_trainer(epoch=7, metrics={"metrics/mAP50(B)": 0.65})
        cb.on_epoch_end(trainer)
        history = cb.get_history()
        assert history[0]["epoch"] == 7

    def test_loss_items_added_to_metrics(self):
        cb = self._make_callback()
        trainer = self._make_trainer(
            epoch=0,
            metrics={},
            loss_items=[0.45, 0.22, 0.11],
        )
        cb.on_epoch_end(trainer)
        history = cb.get_history()
        assert "box_loss" in history[0]
        assert abs(history[0]["box_loss"] - 0.45) < 1e-6

    def test_handles_missing_metrics_gracefully(self):
        """Empty metrics dict → no crash, epoch recorded."""
        cb = self._make_callback()
        trainer = self._make_trainer(epoch=0, metrics={})
        cb.on_epoch_end(trainer)
        assert len(cb.get_history()) == 1

    def test_handles_invalid_trainer_gracefully(self):
        """Completely invalid trainer → on_epoch_end must not raise."""
        cb = self._make_callback()
        cb.on_epoch_end(None)  # Should not raise
        cb.on_epoch_end("not_a_trainer")  # Should not raise

    def test_custom_metric_fn_applied(self):
        """Custom metric functions are called and results merged."""
        def extra_fn(epoch: int, metrics: dict) -> dict:
            return {"custom_score": float(epoch) * 0.1}

        cfg = MetricsCallbackConfig(
            log_to_mlflow=False,
            log_to_websocket=False,
            log_to_wandb=False,
            custom_metric_fns=[extra_fn],
        )
        cb = MetricsCallback(cfg)
        trainer = self._make_trainer(epoch=3, metrics={"metrics/mAP50(B)": 0.7})
        cb.on_epoch_end(trainer)
        history = cb.get_history()
        assert "custom_score" in history[0]
        assert abs(history[0]["custom_score"] - 0.3) < 1e-9

    def test_custom_metric_fn_exception_does_not_crash(self):
        """A crashing custom metric fn must not abort the epoch."""
        def broken_fn(epoch, metrics):
            raise RuntimeError("simulated failure")

        cfg = MetricsCallbackConfig(
            log_to_mlflow=False,
            log_to_websocket=False,
            log_to_wandb=False,
            custom_metric_fns=[broken_fn],
        )
        cb = MetricsCallback(cfg)
        trainer = self._make_trainer(epoch=0, metrics={})
        cb.on_epoch_end(trainer)  # Must not raise
        assert len(cb.get_history()) == 1

    def test_ws_broadcast_respects_interval(self):
        """WebSocket broadcast skipped when epoch % interval != 0."""
        cfg = MetricsCallbackConfig(
            log_to_mlflow=False,
            log_to_websocket=True,
            log_to_wandb=False,
            ws_broadcast_every_n_epochs=5,
        )
        cb = MetricsCallback(cfg)
        with patch.object(cb, "_broadcast_websocket") as mock_ws:
            for epoch in range(10):
                trainer = self._make_trainer(epoch=epoch, metrics={"metrics/mAP50(B)": 0.5})
                cb.on_epoch_end(trainer)
            # Should broadcast at epochs 0, 5 → 2 calls
            assert mock_ws.call_count == 2

    def test_history_timestamp_increases_monotonically(self):
        cb = self._make_callback()
        for epoch in range(3):
            trainer = self._make_trainer(epoch=epoch, metrics={})
            cb.on_epoch_end(trainer)
            time.sleep(0.001)  # Ensure monotonic clock advances
        timestamps = [h["timestamp"] for h in cb.get_history()]
        assert all(timestamps[i] <= timestamps[i + 1] for i in range(len(timestamps) - 1))
