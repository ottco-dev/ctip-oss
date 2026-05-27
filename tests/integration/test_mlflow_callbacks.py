"""
tests.integration.test_mlflow_callbacks — MLflow + W&B integration tests for
training callbacks.

Tests verify that metrics, checkpoints, and artifacts are correctly logged to
a real MLflow tracking server (SQLite backend, ephemeral per test).

Design:
  - Each test creates a fresh MLite-backed MLflow tracking URI in a temp dir.
  - MetricsCallback and CheckpointCallback are instantiated directly;
    no YOLO model is required.
  - A minimal fake ``trainer`` namespace mimics the Ultralytics trainer API.
  - W&B integration is tested via unittest.mock (actual W&B not required).

Sections:
  TestMLflowMetricsLogging  — per-epoch metric logging via MetricsCallback
  TestMLflowRunScope        — run_id scope, nested runs, missing run_id
  TestMLflowBestMetrics     — best_map50 tracking across epochs
  TestMLflowCheckpointArtifacts — CheckpointCallback artifact upload
  TestWandBIntegration      — W&B wandb.log mock verification
  TestCallbackBestEffort    — failures do not crash training loop
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mlflow_uri(tmpdir: str) -> str:
    """Create a per-test SQLite MLflow tracking URI."""
    return f"sqlite:///{tmpdir}/mlflow.db"


def _make_trainer(
    epoch: int = 0,
    metrics: dict | None = None,
    loss_items: tuple | None = None,
    lr: float = 0.001,
) -> SimpleNamespace:
    """Minimal fake Ultralytics trainer object."""
    trainer = SimpleNamespace()
    trainer.epoch = epoch
    trainer.metrics = metrics or {
        "metrics/mAP50(B)": 0.75,
        "metrics/precision(B)": 0.80,
        "metrics/recall(B)": 0.70,
        "val/box_loss": 0.35,
    }
    trainer.loss_items = loss_items or (0.42, 0.23, 0.18)

    class FakeOptimizer:
        param_groups = [{"lr": lr}]

    trainer.optimizer = FakeOptimizer()
    return trainer


def _make_config(
    tmpdir: str,
    tracking_uri: str,
    run_id: str | None = None,
    log_wandb: bool = False,
) -> "MetricsCallbackConfig":
    from training.callbacks.metrics_callback import MetricsCallbackConfig

    return MetricsCallbackConfig(
        mlflow_run_id=run_id,
        run_uuid="test-backend-job-uuid",
        log_to_mlflow=True,
        log_to_websocket=False,   # no WebSocket in tests
        log_to_wandb=log_wandb,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mlflow_tmp():
    """Provide a temp dir + MLflow tracking URI for each test."""
    import mlflow as _mlflow
    with tempfile.TemporaryDirectory() as tmpdir:
        uri = _make_mlflow_uri(tmpdir)
        _mlflow.set_tracking_uri(uri)
        _mlflow.set_experiment("trichome_test")
        yield tmpdir, uri
    # Reset to default after test
    _mlflow.set_tracking_uri("")


# ---------------------------------------------------------------------------
# TestMLflowMetricsLogging
# ---------------------------------------------------------------------------

class TestMLflowMetricsLogging:
    """Per-epoch metric logging via MetricsCallback._log_mlflow()."""

    def test_metrics_logged_to_mlflow_run(self, mlflow_tmp):
        import mlflow
        from training.callbacks.metrics_callback import MetricsCallback, MetricsCallbackConfig

        tmpdir, uri = mlflow_tmp

        with mlflow.start_run() as run:
            run_id = run.info.run_id
            cfg = MetricsCallbackConfig(
                mlflow_run_id=run_id,
                log_to_mlflow=True,
                log_to_websocket=False,
            )
            cb = MetricsCallback(cfg)
            trainer = _make_trainer(epoch=1)
            cb.on_epoch_end(trainer)

        client = mlflow.tracking.MlflowClient(uri)
        history = client.get_metric_history(run_id, "mAP50")
        assert len(history) == 1
        assert abs(history[0].value - 0.75) < 1e-6
        assert history[0].step == 1

    def test_box_loss_logged(self, mlflow_tmp):
        import mlflow
        from training.callbacks.metrics_callback import MetricsCallback, MetricsCallbackConfig

        tmpdir, uri = mlflow_tmp

        with mlflow.start_run() as run:
            run_id = run.info.run_id
            cfg = MetricsCallbackConfig(
                mlflow_run_id=run_id,
                log_to_mlflow=True,
                log_to_websocket=False,
            )
            cb = MetricsCallback(cfg)
            trainer = _make_trainer(epoch=3, loss_items=(0.42, 0.23, 0.18))
            cb.on_epoch_end(trainer)

        client = mlflow.tracking.MlflowClient(uri)
        history = client.get_metric_history(run_id, "box_loss")
        assert len(history) == 1
        assert abs(history[0].value - 0.42) < 1e-6
        assert history[0].step == 3

    def test_lr_logged(self, mlflow_tmp):
        import mlflow
        from training.callbacks.metrics_callback import MetricsCallback, MetricsCallbackConfig

        tmpdir, uri = mlflow_tmp

        with mlflow.start_run() as run:
            run_id = run.info.run_id
            cfg = MetricsCallbackConfig(
                mlflow_run_id=run_id,
                log_to_mlflow=True,
                log_to_websocket=False,
            )
            cb = MetricsCallback(cfg)
            trainer = _make_trainer(epoch=5, lr=0.0025)
            cb.on_epoch_end(trainer)

        client = mlflow.tracking.MlflowClient(uri)
        history = client.get_metric_history(run_id, "lr")
        assert len(history) == 1
        assert abs(history[0].value - 0.0025) < 1e-9

    def test_multiple_epochs_all_logged(self, mlflow_tmp):
        import mlflow
        from training.callbacks.metrics_callback import MetricsCallback, MetricsCallbackConfig

        tmpdir, uri = mlflow_tmp
        n_epochs = 5

        with mlflow.start_run() as run:
            run_id = run.info.run_id
            cfg = MetricsCallbackConfig(
                mlflow_run_id=run_id,
                log_to_mlflow=True,
                log_to_websocket=False,
            )
            cb = MetricsCallback(cfg)
            for e in range(n_epochs):
                trainer = _make_trainer(
                    epoch=e,
                    metrics={"metrics/mAP50(B)": 0.5 + e * 0.05},
                )
                cb.on_epoch_end(trainer)

        client = mlflow.tracking.MlflowClient(uri)
        history = client.get_metric_history(run_id, "mAP50")
        assert len(history) == n_epochs
        values = [h.value for h in sorted(history, key=lambda h: h.step)]
        for i, v in enumerate(values):
            assert abs(v - (0.5 + i * 0.05)) < 1e-6

    def test_precision_and_recall_logged(self, mlflow_tmp):
        import mlflow
        from training.callbacks.metrics_callback import MetricsCallback, MetricsCallbackConfig

        tmpdir, uri = mlflow_tmp

        with mlflow.start_run() as run:
            run_id = run.info.run_id
            cfg = MetricsCallbackConfig(
                mlflow_run_id=run_id,
                log_to_mlflow=True,
                log_to_websocket=False,
            )
            cb = MetricsCallback(cfg)
            trainer = _make_trainer(
                epoch=1,
                metrics={
                    "metrics/mAP50(B)": 0.80,
                    "metrics/precision(B)": 0.87,
                    "metrics/recall(B)": 0.74,
                },
            )
            cb.on_epoch_end(trainer)

        client = mlflow.tracking.MlflowClient(uri)
        assert len(client.get_metric_history(run_id, "precision")) == 1
        assert len(client.get_metric_history(run_id, "recall")) == 1
        assert abs(client.get_metric_history(run_id, "precision")[0].value - 0.87) < 1e-6


# ---------------------------------------------------------------------------
# TestMLflowRunScope
# ---------------------------------------------------------------------------

class TestMLflowRunScope:

    def test_no_run_id_skips_mlflow(self, mlflow_tmp):
        """When mlflow_run_id is None, no metrics should be logged."""
        import mlflow
        from training.callbacks.metrics_callback import MetricsCallback, MetricsCallbackConfig

        tmpdir, uri = mlflow_tmp

        # No run_id → MLflow logging should be skipped (best-effort)
        cfg = MetricsCallbackConfig(
            mlflow_run_id=None,
            log_to_mlflow=True,
            log_to_websocket=False,
        )
        cb = MetricsCallback(cfg)
        # Should not raise even with no active run
        cb.on_epoch_end(_make_trainer(epoch=0))

    def test_log_to_mlflow_false_skips_logging(self, mlflow_tmp):
        """log_to_mlflow=False: _log_mlflow should not be called."""
        from training.callbacks.metrics_callback import MetricsCallback, MetricsCallbackConfig

        tmpdir, uri = mlflow_tmp

        import mlflow
        with mlflow.start_run() as run:
            run_id = run.info.run_id
            cfg = MetricsCallbackConfig(
                mlflow_run_id=run_id,
                log_to_mlflow=False,    # disabled
                log_to_websocket=False,
            )
            cb = MetricsCallback(cfg)
            cb.on_epoch_end(_make_trainer(epoch=1))

        client = mlflow.tracking.MlflowClient(uri)
        history = client.get_metric_history(run_id, "mAP50")
        assert len(history) == 0   # nothing was logged


# ---------------------------------------------------------------------------
# TestMLflowBestMetrics
# ---------------------------------------------------------------------------

class TestMLflowBestMetrics:

    def test_best_map50_updated_across_epochs(self, mlflow_tmp):
        import mlflow
        from training.callbacks.metrics_callback import MetricsCallback, MetricsCallbackConfig

        tmpdir, uri = mlflow_tmp

        with mlflow.start_run() as run:
            run_id = run.info.run_id
            cfg = MetricsCallbackConfig(
                mlflow_run_id=run_id,
                log_to_mlflow=True,
                log_to_websocket=False,
            )
            cb = MetricsCallback(cfg)

            map_values = [0.60, 0.72, 0.68, 0.80, 0.78]
            for i, m in enumerate(map_values):
                trainer = _make_trainer(
                    epoch=i,
                    metrics={"metrics/mAP50(B)": m},
                )
                cb.on_epoch_end(trainer)

        assert abs(cb.best_map50 - 0.80) < 1e-9

    def test_best_map50_never_decreases(self, mlflow_tmp):
        import mlflow
        from training.callbacks.metrics_callback import MetricsCallback, MetricsCallbackConfig

        tmpdir, uri = mlflow_tmp

        with mlflow.start_run() as run:
            run_id = run.info.run_id
            cfg = MetricsCallbackConfig(
                mlflow_run_id=run_id,
                log_to_mlflow=True,
                log_to_websocket=False,
            )
            cb = MetricsCallback(cfg)

            best = 0.0
            for i, m in enumerate([0.5, 0.3, 0.7, 0.2, 0.6]):
                trainer = _make_trainer(
                    epoch=i,
                    metrics={"metrics/mAP50(B)": m},
                )
                cb.on_epoch_end(trainer)
                best = max(best, m)
                assert cb.best_map50 >= best - 1e-9, (
                    f"Epoch {i}: best_map50 dropped below peak {best}"
                )

    def test_get_history_length(self, mlflow_tmp):
        import mlflow
        from training.callbacks.metrics_callback import MetricsCallback, MetricsCallbackConfig

        tmpdir, uri = mlflow_tmp
        n = 10

        with mlflow.start_run() as run:
            run_id = run.info.run_id
            cfg = MetricsCallbackConfig(
                mlflow_run_id=run_id,
                log_to_mlflow=True,
                log_to_websocket=False,
            )
            cb = MetricsCallback(cfg)
            for i in range(n):
                cb.on_epoch_end(_make_trainer(epoch=i))

        assert len(cb.get_history()) == n

    def test_get_final_metrics_returns_last_epoch(self, mlflow_tmp):
        import mlflow
        from training.callbacks.metrics_callback import MetricsCallback, MetricsCallbackConfig

        tmpdir, uri = mlflow_tmp

        with mlflow.start_run() as run:
            run_id = run.info.run_id
            cfg = MetricsCallbackConfig(
                mlflow_run_id=run_id,
                log_to_mlflow=True,
                log_to_websocket=False,
            )
            cb = MetricsCallback(cfg)
            for i, m in enumerate([0.5, 0.6, 0.7]):
                trainer = _make_trainer(
                    epoch=i,
                    metrics={"metrics/mAP50(B)": m},
                )
                cb.on_epoch_end(trainer)

        final = cb.get_final_metrics()
        assert abs(final.get("mAP50", 0) - 0.7) < 1e-9


# ---------------------------------------------------------------------------
# TestMLflowCheckpointArtifacts
# ---------------------------------------------------------------------------

class TestMLflowCheckpointArtifacts:

    def test_checkpoint_artifact_logged_on_train_end(self, mlflow_tmp):
        """CheckpointCallback logs checkpoint as MLflow artifact on train end."""
        import mlflow
        from training.callbacks.checkpoint_callback import (
            CheckpointCallback,
            CheckpointCallbackConfig,
        )

        tmpdir, uri = mlflow_tmp

        with mlflow.start_run() as run:
            run_id = run.info.run_id

            cfg = CheckpointCallbackConfig(
                save_dir=tmpdir,
                run_name="test_run",
                log_to_mlflow=True,
                mlflow_run_id=run_id,
                save_best=True,
                min_map50_to_save=0.0,
            )
            cb = CheckpointCallback(cfg)

            # Simulate epoch with improving mAP
            trainer = _make_trainer(
                epoch=0,
                metrics={"metrics/mAP50(B)": 0.75},
            )

            # Directly write a fake checkpoint file (normally YOLO saves best.pt)
            save_dir = Path(tmpdir) / "test_run"
            save_dir.mkdir(parents=True, exist_ok=True)
            fake_ckpt = save_dir / "best.pt"
            fake_ckpt.write_bytes(b"FAKE_PYTORCH_WEIGHTS")

            # Patch _save_best to set the internal path directly
            cb._best_checkpoint_path = fake_ckpt
            cb._best_map50 = 0.75

            # Trigger on_train_end
            cb.on_train_end(trainer)

        # Verify artifact uploaded
        client = mlflow.tracking.MlflowClient(uri)
        artifacts = client.list_artifacts(run_id, "checkpoints")
        assert len(artifacts) >= 1
        names = [a.path for a in artifacts]
        assert any("best.pt" in n for n in names), f"Artifacts: {names}"

    def test_no_artifact_when_no_checkpoint_path(self, mlflow_tmp):
        """on_train_end with no best checkpoint → no artifact logged."""
        import mlflow
        from training.callbacks.checkpoint_callback import (
            CheckpointCallback,
            CheckpointCallbackConfig,
        )

        tmpdir, uri = mlflow_tmp

        with mlflow.start_run() as run:
            run_id = run.info.run_id
            cfg = CheckpointCallbackConfig(
                save_dir=tmpdir,
                run_name="test_no_ckpt",
                log_to_mlflow=True,
                mlflow_run_id=run_id,
            )
            cb = CheckpointCallback(cfg)
            # _best_checkpoint_path is None → nothing to log
            cb.on_train_end(_make_trainer(epoch=0))

        client = mlflow.tracking.MlflowClient(uri)
        artifacts = client.list_artifacts(run_id, "checkpoints")
        assert len(artifacts) == 0

    def test_artifact_not_logged_when_disabled(self, mlflow_tmp):
        """log_to_mlflow=False → no artifact even if checkpoint exists."""
        import mlflow
        from training.callbacks.checkpoint_callback import (
            CheckpointCallback,
            CheckpointCallbackConfig,
        )

        tmpdir, uri = mlflow_tmp

        with mlflow.start_run() as run:
            run_id = run.info.run_id
            cfg = CheckpointCallbackConfig(
                save_dir=tmpdir,
                run_name="test_disabled",
                log_to_mlflow=False,    # disabled
                mlflow_run_id=run_id,
            )
            cb = CheckpointCallback(cfg)

            save_dir = Path(tmpdir) / "test_disabled"
            save_dir.mkdir(parents=True, exist_ok=True)
            fake_ckpt = save_dir / "best.pt"
            fake_ckpt.write_bytes(b"FAKE")
            cb._best_checkpoint_path = fake_ckpt

            cb.on_train_end(_make_trainer(epoch=0))

        client = mlflow.tracking.MlflowClient(uri)
        artifacts = client.list_artifacts(run_id, "checkpoints")
        assert len(artifacts) == 0


# ---------------------------------------------------------------------------
# TestWandBIntegration
# ---------------------------------------------------------------------------

class TestWandBIntegration:
    """Verify W&B logging calls via mock — does not require actual W&B account."""

    def test_wandb_log_called_when_enabled(self):
        """With log_to_wandb=True and wandb.run active, wandb.log is called."""
        from training.callbacks.metrics_callback import MetricsCallback, MetricsCallbackConfig

        cfg = MetricsCallbackConfig(
            mlflow_run_id=None,
            log_to_mlflow=False,
            log_to_websocket=False,
            log_to_wandb=True,
        )
        cb = MetricsCallback(cfg)

        mock_run = MagicMock()
        with patch("wandb.run", mock_run), \
             patch("wandb.log") as mock_log:
            cb.on_epoch_end(_make_trainer(epoch=2))

        mock_log.assert_called_once()
        call_kwargs = mock_log.call_args[1]  # kwargs
        # step should match epoch
        assert call_kwargs.get("step") == 2

    def test_wandb_log_not_called_when_disabled(self):
        """With log_to_wandb=False, wandb.log is never called."""
        from training.callbacks.metrics_callback import MetricsCallback, MetricsCallbackConfig

        cfg = MetricsCallbackConfig(
            mlflow_run_id=None,
            log_to_mlflow=False,
            log_to_websocket=False,
            log_to_wandb=False,
        )
        cb = MetricsCallback(cfg)

        with patch("wandb.log") as mock_log:
            cb.on_epoch_end(_make_trainer(epoch=1))

        mock_log.assert_not_called()

    def test_wandb_log_not_called_when_run_is_none(self):
        """With wandb.run=None (not initialised), wandb.log should be skipped."""
        from training.callbacks.metrics_callback import MetricsCallback, MetricsCallbackConfig

        cfg = MetricsCallbackConfig(
            mlflow_run_id=None,
            log_to_mlflow=False,
            log_to_websocket=False,
            log_to_wandb=True,
        )
        cb = MetricsCallback(cfg)

        with patch("wandb.run", None), \
             patch("wandb.log") as mock_log:
            cb.on_epoch_end(_make_trainer(epoch=1))

        mock_log.assert_not_called()

    def test_wandb_failure_does_not_crash_training(self):
        """If wandb.log raises, training loop continues (best-effort)."""
        from training.callbacks.metrics_callback import MetricsCallback, MetricsCallbackConfig

        cfg = MetricsCallbackConfig(
            mlflow_run_id=None,
            log_to_mlflow=False,
            log_to_websocket=False,
            log_to_wandb=True,
        )
        cb = MetricsCallback(cfg)

        mock_run = MagicMock()
        with patch("wandb.run", mock_run), \
             patch("wandb.log", side_effect=RuntimeError("W&B exploded")):
            # Must not raise
            cb.on_epoch_end(_make_trainer(epoch=1))

        # History should still be recorded locally
        assert len(cb.get_history()) == 1


# ---------------------------------------------------------------------------
# TestCallbackBestEffort
# ---------------------------------------------------------------------------

class TestCallbackBestEffort:
    """Failures in logging backends must not propagate to training loop."""

    def test_mlflow_error_does_not_crash_epoch(self, mlflow_tmp):
        """If MLflow log_metric raises, on_epoch_end completes normally."""
        from training.callbacks.metrics_callback import MetricsCallback, MetricsCallbackConfig

        tmpdir, uri = mlflow_tmp

        import mlflow
        with mlflow.start_run() as run:
            run_id = run.info.run_id
            cfg = MetricsCallbackConfig(
                mlflow_run_id=run_id,
                log_to_mlflow=True,
                log_to_websocket=False,
            )
            cb = MetricsCallback(cfg)

            with patch("mlflow.log_metric", side_effect=ConnectionError("MLflow down")):
                cb.on_epoch_end(_make_trainer(epoch=0))

        # History still accumulated locally despite MLflow failure
        assert len(cb.get_history()) == 1

    def test_malformed_trainer_metrics_handled(self, mlflow_tmp):
        """Trainer with non-numeric metrics must not crash on_epoch_end."""
        from training.callbacks.metrics_callback import MetricsCallback, MetricsCallbackConfig

        tmpdir, uri = mlflow_tmp

        cfg = MetricsCallbackConfig(
            mlflow_run_id=None,
            log_to_mlflow=False,
            log_to_websocket=False,
        )
        cb = MetricsCallback(cfg)

        # Trainer with bad metric values
        trainer = SimpleNamespace()
        trainer.epoch = 0
        trainer.metrics = {"metrics/mAP50(B)": "NOT_A_FLOAT", "garbage": None}
        trainer.loss_items = None
        trainer.optimizer = None

        cb.on_epoch_end(trainer)  # must not raise

    def test_none_trainer_attributes_handled(self):
        """Trainer with missing attributes handled gracefully."""
        from training.callbacks.metrics_callback import MetricsCallback, MetricsCallbackConfig

        cfg = MetricsCallbackConfig(
            mlflow_run_id=None,
            log_to_mlflow=False,
            log_to_websocket=False,
        )
        cb = MetricsCallback(cfg)

        # Trainer with no attributes at all
        cb.on_epoch_end(SimpleNamespace())   # must not raise

    def test_checkpoint_callback_survives_missing_trainer_model(self, mlflow_tmp):
        """CheckpointCallback.on_epoch_end handles trainer with no model."""
        from training.callbacks.checkpoint_callback import (
            CheckpointCallback,
            CheckpointCallbackConfig,
        )
        tmpdir, uri = mlflow_tmp

        cfg = CheckpointCallbackConfig(
            save_dir=tmpdir,
            run_name="robustness_test",
            log_to_mlflow=False,
        )
        cb = CheckpointCallback(cfg)
        # Trainer with no model attribute
        cb.on_epoch_end(SimpleNamespace())  # must not raise
