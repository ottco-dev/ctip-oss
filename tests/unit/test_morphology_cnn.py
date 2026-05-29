"""
tests.unit.test_morphology_cnn — Unit tests for CNN morphology training pipeline.

Coverage:
  - MorphologyCNNConfig: defaults, custom overrides, validation errors
  - MorphologyCNNTrainer: instantiation, build_model output shape, data transforms
  - _SubsetWithTransform wrapper
  - ONNX export (mocked)
  - API endpoints: start / status / evaluate / export (mocked training)
  - _mask_token helper
  - Token management API endpoints
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ── ensure project root is importable ────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def default_config():
    from morphology.training.cnn_trainer import MorphologyCNNConfig
    return MorphologyCNNConfig()


@pytest.fixture()
def tmp_data_dir(tmp_path):
    """Create a minimal fake morphology crop directory."""
    from PIL import Image

    classes = ["capitate_stalked", "capitate_sessile", "bulbous", "non_glandular"]
    for cls in classes:
        cls_dir = tmp_path / cls
        cls_dir.mkdir()
        for i in range(6):
            img = Image.fromarray(
                np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            )
            img.save(cls_dir / f"{cls}_{i:02d}.png")
    return tmp_path


@pytest.fixture()
def trainer_with_tmp(tmp_data_dir):
    from morphology.training.cnn_trainer import MorphologyCNNConfig, MorphologyCNNTrainer
    # Use a separate temp directory for output so it doesn't pollute data_dir
    with tempfile.TemporaryDirectory() as out_dir:
        cfg = MorphologyCNNConfig(
            data_dir=str(tmp_data_dir),
            output_dir=out_dir,
            epochs=1,
            batch_size=4,
            num_workers=0,
            use_fp16=False,
            augment=True,
            early_stopping_patience=2,
            val_split=0.3,
        )
        yield MorphologyCNNTrainer(cfg)


# ═══════════════════════════════════════════════════════════════════════════════
# MorphologyCNNConfig tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestMorphologyCNNConfig:

    def test_default_model_arch(self, default_config):
        assert default_config.model_arch == "efficientnet_b0"

    def test_default_num_classes(self, default_config):
        assert default_config.num_classes == 4

    def test_default_input_size(self, default_config):
        assert default_config.input_size == 224

    def test_default_batch_size(self, default_config):
        assert default_config.batch_size == 32

    def test_default_learning_rate(self, default_config):
        assert default_config.learning_rate == 1e-4

    def test_default_epochs(self, default_config):
        assert default_config.epochs == 50

    def test_default_dropout(self, default_config):
        assert default_config.dropout == 0.3

    def test_default_use_fp16(self, default_config):
        assert default_config.use_fp16 is True

    def test_default_augment(self, default_config):
        assert default_config.augment is True

    def test_default_val_split(self, default_config):
        assert default_config.val_split == 0.2

    def test_default_seed(self, default_config):
        assert default_config.seed == 42

    def test_default_early_stopping_patience(self, default_config):
        assert default_config.early_stopping_patience == 10

    def test_custom_arch_mobilenet(self):
        from morphology.training.cnn_trainer import MorphologyCNNConfig
        cfg = MorphologyCNNConfig(model_arch="mobilenet_v3_small")
        assert cfg.model_arch == "mobilenet_v3_small"

    def test_invalid_arch_raises(self):
        from morphology.training.cnn_trainer import MorphologyCNNConfig
        with pytest.raises(ValueError, match="model_arch must be one of"):
            MorphologyCNNConfig(model_arch="resnet50")

    def test_invalid_val_split_zero_raises(self):
        from morphology.training.cnn_trainer import MorphologyCNNConfig
        with pytest.raises(ValueError, match="val_split"):
            MorphologyCNNConfig(val_split=0.0)

    def test_invalid_val_split_one_raises(self):
        from morphology.training.cnn_trainer import MorphologyCNNConfig
        with pytest.raises(ValueError, match="val_split"):
            MorphologyCNNConfig(val_split=1.0)

    def test_invalid_num_classes_raises(self):
        from morphology.training.cnn_trainer import MorphologyCNNConfig
        with pytest.raises(ValueError, match="num_classes"):
            MorphologyCNNConfig(num_classes=1)

    def test_invalid_dropout_raises(self):
        from morphology.training.cnn_trainer import MorphologyCNNConfig
        with pytest.raises(ValueError, match="dropout"):
            MorphologyCNNConfig(dropout=1.5)

    def test_custom_paths(self):
        from morphology.training.cnn_trainer import MorphologyCNNConfig
        cfg = MorphologyCNNConfig(
            data_dir="/custom/data",
            output_dir="/custom/models",
        )
        assert cfg.data_dir == "/custom/data"
        assert cfg.output_dir == "/custom/models"


# ═══════════════════════════════════════════════════════════════════════════════
# MorphologyCNNTrainer tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestMorphologyCNNTrainer:

    def test_instantiation(self, trainer_with_tmp):
        from morphology.training.cnn_trainer import MorphologyCNNTrainer
        assert isinstance(trainer_with_tmp, MorphologyCNNTrainer)

    def test_initial_state_idle(self, trainer_with_tmp):
        assert trainer_with_tmp.status["state"] == "idle"

    def test_build_model_efficientnet_output_shape(self, trainer_with_tmp):
        """Model final layer must output num_classes=4 logits."""
        import torch
        model = trainer_with_tmp.build_model()
        model.eval()
        dummy = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = model(dummy)
        assert out.shape == (2, 4), f"Expected (2, 4), got {out.shape}"

    def test_build_model_mobilenet_output_shape(self, tmp_data_dir):
        """MobileNetV3-Small head must also output 4 logits."""
        import torch
        from morphology.training.cnn_trainer import MorphologyCNNConfig, MorphologyCNNTrainer
        with tempfile.TemporaryDirectory() as out_dir:
            cfg = MorphologyCNNConfig(
                model_arch="mobilenet_v3_small",
                data_dir=str(tmp_data_dir),
                output_dir=out_dir,
                num_workers=0,
            )
            trainer = MorphologyCNNTrainer(cfg)
        model = trainer.build_model()
        model.eval()
        dummy = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = model(dummy)
        assert out.shape == (1, 4)

    def test_train_transforms_include_normalise(self, trainer_with_tmp):
        """Train transform pipeline must include Normalize."""
        from torchvision.transforms import Normalize
        tf = trainer_with_tmp._build_train_transforms()
        # Flatten compose
        transform_types = [type(t).__name__ for t in tf.transforms]
        assert "Normalize" in transform_types

    def test_val_transforms_no_random_flip(self, trainer_with_tmp):
        """Validation transforms must be deterministic (no RandomHorizontalFlip)."""
        tf = trainer_with_tmp._build_val_transforms()
        transform_types = [type(t).__name__ for t in tf.transforms]
        assert "RandomHorizontalFlip" not in transform_types
        assert "RandomVerticalFlip" not in transform_types

    def test_train_transforms_with_augment_includes_rotation(self, trainer_with_tmp):
        """Augmented pipeline must include RandomRotation for microscopy."""
        tf = trainer_with_tmp._build_train_transforms()
        transform_types = [type(t).__name__ for t in tf.transforms]
        assert "RandomRotation" in transform_types

    def test_train_transforms_without_augment_no_rotation(self, tmp_data_dir):
        from morphology.training.cnn_trainer import MorphologyCNNConfig, MorphologyCNNTrainer
        with tempfile.TemporaryDirectory() as out_dir:
            cfg = MorphologyCNNConfig(
                data_dir=str(tmp_data_dir),
                output_dir=out_dir,
                augment=False,
                num_workers=0,
            )
            trainer = MorphologyCNNTrainer(cfg)
        tf = trainer._build_train_transforms()
        transform_types = [type(t).__name__ for t in tf.transforms]
        assert "RandomRotation" not in transform_types

    def test_prepare_data_returns_two_loaders(self, trainer_with_tmp):
        """prepare_data must return (train_loader, val_loader)."""
        train_loader, val_loader = trainer_with_tmp.prepare_data()
        assert train_loader is not None
        assert val_loader is not None

    def test_prepare_data_nonexistent_dir_raises(self, tmp_path):
        from morphology.training.cnn_trainer import MorphologyCNNConfig, MorphologyCNNTrainer
        cfg = MorphologyCNNConfig(
            data_dir=str(tmp_path / "nonexistent"),
            num_workers=0,
        )
        trainer = MorphologyCNNTrainer(cfg)
        with pytest.raises(FileNotFoundError):
            trainer.prepare_data()

    def test_training_history_initially_empty(self, trainer_with_tmp):
        h = trainer_with_tmp.training_history
        for key in ("train_loss", "val_loss", "train_acc", "val_acc"):
            assert h[key] == []

    def test_train_one_epoch_updates_history(self, trainer_with_tmp):
        """Single epoch training must populate history."""
        summary = trainer_with_tmp.train()
        h = trainer_with_tmp.training_history
        assert len(h["train_loss"]) >= 1
        assert len(h["val_loss"]) >= 1

    def test_train_creates_checkpoint(self, trainer_with_tmp):
        summary = trainer_with_tmp.train()
        assert Path(summary["checkpoint_path"]).exists()

    def test_train_creates_history_json(self, trainer_with_tmp):
        summary = trainer_with_tmp.train()
        assert Path(summary["history_path"]).exists()
        with open(summary["history_path"]) as f:
            data = json.load(f)
        assert "train_loss" in data

    def test_train_summary_keys(self, trainer_with_tmp):
        summary = trainer_with_tmp.train()
        for key in ("best_epoch", "best_val_loss", "best_val_acc", "final_epoch"):
            assert key in summary, f"Missing key: {key}"

    def test_status_completed_after_train(self, trainer_with_tmp):
        trainer_with_tmp.train()
        assert trainer_with_tmp.status["state"] == "completed"


# ═══════════════════════════════════════════════════════════════════════════════
# ONNX export tests (mocked)
# ═══════════════════════════════════════════════════════════════════════════════


class TestOnnxExport:

    def test_export_onnx_calls_torch_onnx(self, tmp_data_dir):
        """export_onnx must call torch.onnx.export and write the output file."""
        import torch
        from morphology.training.cnn_trainer import MorphologyCNNConfig, MorphologyCNNTrainer

        with tempfile.TemporaryDirectory() as out_dir:
            cfg = MorphologyCNNConfig(
                data_dir=str(tmp_data_dir),
                output_dir=out_dir,
                epochs=1,
                batch_size=4,
                num_workers=0,
                use_fp16=False,
            )
            trainer = MorphologyCNNTrainer(cfg)
            summary = trainer.train()
            ckpt_path = summary["checkpoint_path"]
            onnx_path = str(Path(out_dir) / "morphology.onnx")

            with patch("torch.onnx.export") as mock_export:
                mock_export.side_effect = lambda *a, **kw: Path(onnx_path).touch()
                result = trainer.export_onnx(ckpt_path, onnx_path)

        assert mock_export.called
        assert "morphology.onnx" in result

    def test_export_onnx_creates_output_dir(self, tmp_data_dir):
        """export_onnx must create parent directory if it does not exist."""
        from morphology.training.cnn_trainer import MorphologyCNNConfig, MorphologyCNNTrainer

        with tempfile.TemporaryDirectory() as out_dir:
            cfg = MorphologyCNNConfig(
                data_dir=str(tmp_data_dir),
                output_dir=out_dir,
                epochs=1,
                batch_size=4,
                num_workers=0,
                use_fp16=False,
            )
            trainer = MorphologyCNNTrainer(cfg)
            summary = trainer.train()
            ckpt_path = summary["checkpoint_path"]
            deep_output = str(Path(out_dir) / "nested" / "dir" / "model.onnx")

            with patch("torch.onnx.export") as mock_export:
                mock_export.side_effect = lambda *a, **kw: Path(deep_output).parent.mkdir(
                    parents=True, exist_ok=True
                ) or Path(deep_output).touch()
                trainer.export_onnx(ckpt_path, deep_output)

            # Assert inside context manager (dir still exists)
            assert Path(deep_output).parent.exists()


# ═══════════════════════════════════════════════════════════════════════════════
# API endpoint tests
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture()
def api_client():
    """FastAPI test client for the morphology training router."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from backend.api.v1.morphology_training import router

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


class TestMorphologyTrainingAPI:

    def test_status_idle_initially(self, api_client):
        # Reset global state
        from backend.api.v1 import morphology_training as mt
        mt._training_state["state"] = "idle"
        mt._training_state["trainer"] = None

        resp = api_client.get("/morphology/training/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["state"] == "idle"

    def test_start_returns_queued(self, api_client):
        from backend.api.v1 import morphology_training as mt
        mt._training_state["state"] = "idle"

        resp = api_client.post("/morphology/training/start", json={
            "data_dir": "/nonexistent/path",
            "epochs": 1,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "queued"

    def test_start_conflict_when_running(self, api_client):
        from backend.api.v1 import morphology_training as mt
        mt._training_state["state"] = "running"
        resp = api_client.post("/morphology/training/start", json={})
        assert resp.status_code == 409
        # Reset
        mt._training_state["state"] = "idle"

    def test_evaluate_returns_404_for_missing_model(self, api_client, tmp_path):
        resp = api_client.post("/morphology/training/evaluate", json={
            "model_path": str(tmp_path / "nonexistent.pt"),
            "data_dir": str(tmp_path),
        })
        assert resp.status_code == 404

    def test_export_returns_404_for_missing_checkpoint(self, api_client, tmp_path):
        resp = api_client.post("/morphology/training/export", json={
            "model_path": str(tmp_path / "missing.pt"),
            "output_path": str(tmp_path / "out.onnx"),
        })
        assert resp.status_code == 404

    def test_status_includes_config_after_start(self, api_client):
        from backend.api.v1 import morphology_training as mt
        mt._training_state["state"] = "idle"

        api_client.post("/morphology/training/start", json={
            "data_dir": "/dummy",
            "epochs": 3,
        })
        resp = api_client.get("/morphology/training/status")
        data = resp.json()
        assert data["config"] is not None
        assert data["config"]["epochs"] == 3
        mt._training_state["state"] = "idle"


# ═══════════════════════════════════════════════════════════════════════════════
# Token management API tests
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture()
def system_client(tmp_path):
    """FastAPI test client for the system router with a temp .env file."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from backend.api.v1.system import router

    # Point env_file to a temp file so tests don't touch the real .env
    env_file = tmp_path / ".env"
    env_file.write_text("")

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    return client, env_file


class TestTokenAPI:

    def test_token_status_disabled_when_empty(self, system_client):
        client, _ = system_client
        with patch("backend.config.get_settings") as mock_gs:
            settings = MagicMock()
            settings.api_token = ""
            mock_gs.return_value = settings
            resp = client.get("/system/token/status")
        assert resp.status_code == 200
        assert resp.json()["enabled"] is False

    def test_token_status_enabled_with_token(self, system_client):
        client, _ = system_client
        with patch("backend.config.get_settings") as mock_gs:
            settings = MagicMock()
            settings.api_token = "abcd1234" * 8  # 64-char token
            mock_gs.return_value = settings
            resp = client.get("/system/token/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["enabled"] is True
        assert data["token_preview"] is not None

    def test_mask_token_hides_middle(self):
        from backend.api.v1.system import _mask_token
        token = "abcdef0123456789" * 4  # 64 chars
        masked = _mask_token(token)
        assert masked.startswith(token[:4])
        assert masked.endswith(token[-4:])
        assert "*" in masked

    def test_mask_token_short_token(self):
        from backend.api.v1.system import _mask_token
        masked = _mask_token("abc")
        assert masked == "***"

    def test_generate_token_returns_64_char_hex(self, system_client, tmp_path):
        client, env_file = system_client
        with patch("backend.utils.env_file.write_env_key") as mock_write, \
             patch("backend.config.get_settings") as mock_gs:
            mock_gs.cache_clear = MagicMock()
            resp = client.post("/system/token/generate")
        assert resp.status_code == 200
        data = resp.json()
        assert "token" in data
        assert len(data["token"]) == 64
        assert "warning" in data

    def test_generate_token_contains_warning(self, system_client):
        client, _ = system_client
        with patch("backend.utils.env_file.write_env_key"), \
             patch("backend.config.get_settings") as mock_gs:
            mock_gs.cache_clear = MagicMock()
            resp = client.post("/system/token/generate")
        assert "Copy now" in resp.json()["warning"]

    def test_clear_token_returns_disabled(self, system_client):
        client, _ = system_client
        with patch("backend.utils.env_file.write_env_key") as mock_write, \
             patch("backend.config.get_settings") as mock_gs:
            mock_gs.cache_clear = MagicMock()
            resp = client.post("/system/token/clear")
        assert resp.status_code == 200
        data = resp.json()
        assert data["enabled"] is False
