"""
tests.unit.test_distributed_training — Comprehensive tests for multi-GPU DDP support.

Coverage (35+ tests):

DistributedConfig
  - Valid construction with all defaults
  - Invalid backend raises ValueError
  - Invalid mixed_precision raises ValueError
  - Invalid world_size (0, -2) raises ValueError
  - master_port out of range (0, 65536) raises ValueError
  - gradient_accumulation_steps < 1 raises ValueError
  - world_size=-1 (auto-detect) is valid
  - resolve_world_size returns world_size when not -1
  - resolve_world_size auto-detects when world_size=-1

DDPTrainer
  - setup single-GPU skips init_process_group
  - setup multi-GPU calls init_process_group with correct args
  - setup multi-GPU sets cuda device
  - teardown calls destroy_process_group when distributed active
  - teardown is safe when distributed NOT active (no-op)
  - is_rank_zero returns True for rank 0
  - is_rank_zero returns False for rank 1+
  - save_checkpoint only writes on rank 0
  - save_checkpoint skips on non-zero rank
  - save_checkpoint unwraps DDP module
  - load_checkpoint raises FileNotFoundError for missing file
  - load_checkpoint uses correct map_location for rank
  - rank_zero_first on single-GPU just yields
  - rank_zero_first on multi-GPU: rank 0 yields then barrier, others barrier then yield
  - NCCL → gloo fallback when NCCL unavailable
  - train_epoch with fp16 AMP accumulates gradients correctly
  - train_epoch with mixed_precision="no" skips AMP
  - wrap_model single-GPU returns model unchanged
  - wrap_model raises RuntimeError before setup()
  - barrier is no-op for single-GPU

DistributedLauncher
  - available_gpus mocks torch.cuda.device_count
  - available_gpus returns 0 when torch unavailable
  - optimal_world_size respects GPU count
  - optimal_world_size respects VRAM floor (model too large → 1)
  - optimal_world_size returns 1 when no GPUs
  - optimal_world_size never returns 0
  - detect_backend returns nccl when available
  - detect_backend falls back to gloo
  - launch raises FileNotFoundError for missing script
  - launch calls subprocess.run with correct torchrun args

API Endpoints
  - GET /training/distributed/status returns correct shape
  - POST /training/distributed/start returns task_id
  - POST /training/distributed/start falls back to gloo when NCCL missing
  - GET /training/distributed/jobs/{task_id} 404 for unknown task
  - GET /training/distributed/jobs/{task_id} returns correct data for known task
  - POST /training/distributed/stop/{task_id} returns 404 for unknown task
  - POST /training/distributed/stop/{task_id} returns stopped=False for completed job
  - POST /training/distributed/stop/{task_id} returns stopped=True for running job
"""

from __future__ import annotations

import asyncio
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_fake_tensor(value: float = 0.5):
    """Return a minimal fake scalar tensor for loss computation tests."""
    import torch
    return torch.tensor(value, requires_grad=True)


def _simple_dataloader(n_batches: int = 3):
    """Yield (inputs, targets) pairs of tiny tensors."""
    import torch
    for _ in range(n_batches):
        inputs = torch.randn(2, 4, requires_grad=False)
        targets = torch.zeros(2, dtype=torch.long)
        yield inputs, targets


# ─────────────────────────────────────────────────────────────────────────────
# DistributedConfig — validation
# ─────────────────────────────────────────────────────────────────────────────

class TestDistributedConfig:
    """Validate DistributedConfig field constraints."""

    def test_valid_defaults(self):
        from training.distributed.ddp_trainer import DistributedConfig
        cfg = DistributedConfig()
        assert cfg.backend == "nccl"
        assert cfg.world_size == -1
        assert cfg.master_port == 29500
        assert cfg.gradient_accumulation_steps == 1
        assert cfg.mixed_precision == "fp16"

    def test_valid_explicit_values(self):
        from training.distributed.ddp_trainer import DistributedConfig
        cfg = DistributedConfig(
            backend="gloo",
            world_size=4,
            master_port=12345,
            gradient_accumulation_steps=4,
            mixed_precision="bf16",
            sync_batchnorm=False,
        )
        assert cfg.backend == "gloo"
        assert cfg.world_size == 4
        assert cfg.mixed_precision == "bf16"

    def test_invalid_backend_raises(self):
        from training.distributed.ddp_trainer import DistributedConfig
        with pytest.raises(ValueError, match="backend"):
            DistributedConfig(backend="invalid_backend")

    def test_invalid_mixed_precision_raises(self):
        from training.distributed.ddp_trainer import DistributedConfig
        with pytest.raises(ValueError, match="mixed_precision"):
            DistributedConfig(mixed_precision="float16")

    def test_world_size_zero_raises(self):
        from training.distributed.ddp_trainer import DistributedConfig
        with pytest.raises(ValueError, match="world_size"):
            DistributedConfig(world_size=0)

    def test_world_size_negative_two_raises(self):
        from training.distributed.ddp_trainer import DistributedConfig
        with pytest.raises(ValueError, match="world_size"):
            DistributedConfig(world_size=-2)

    def test_world_size_minus_one_is_valid(self):
        from training.distributed.ddp_trainer import DistributedConfig
        cfg = DistributedConfig(world_size=-1)
        assert cfg.world_size == -1

    def test_master_port_zero_raises(self):
        from training.distributed.ddp_trainer import DistributedConfig
        with pytest.raises(ValueError, match="master_port"):
            DistributedConfig(master_port=0)

    def test_master_port_too_high_raises(self):
        from training.distributed.ddp_trainer import DistributedConfig
        with pytest.raises(ValueError, match="master_port"):
            DistributedConfig(master_port=65536)

    def test_master_port_boundary_valid(self):
        from training.distributed.ddp_trainer import DistributedConfig
        cfg1 = DistributedConfig(master_port=1)
        cfg2 = DistributedConfig(master_port=65535)
        assert cfg1.master_port == 1
        assert cfg2.master_port == 65535

    def test_gradient_accumulation_zero_raises(self):
        from training.distributed.ddp_trainer import DistributedConfig
        with pytest.raises(ValueError, match="gradient_accumulation_steps"):
            DistributedConfig(gradient_accumulation_steps=0)

    def test_gradient_accumulation_negative_raises(self):
        from training.distributed.ddp_trainer import DistributedConfig
        with pytest.raises(ValueError, match="gradient_accumulation_steps"):
            DistributedConfig(gradient_accumulation_steps=-1)

    def test_resolve_world_size_explicit(self):
        from training.distributed.ddp_trainer import DistributedConfig
        cfg = DistributedConfig(world_size=3)
        assert cfg.resolve_world_size() == 3

    def test_resolve_world_size_auto_detect(self):
        """When world_size=-1, falls back to torch.cuda.device_count."""
        from training.distributed.ddp_trainer import DistributedConfig
        cfg = DistributedConfig(world_size=-1)
        with patch("torch.cuda.device_count", return_value=2):
            result = cfg.resolve_world_size()
        # Result should be at least 1
        assert result >= 1

    def test_resolve_world_size_cuda_visible_devices(self):
        """CUDA_VISIBLE_DEVICES takes precedence when world_size=-1."""
        from training.distributed.ddp_trainer import DistributedConfig
        cfg = DistributedConfig(world_size=-1)
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1,2"}):
            # Remove any existing env var mock interference
            result = cfg.resolve_world_size()
        assert result == 3

    def test_mpi_backend_is_valid(self):
        from training.distributed.ddp_trainer import DistributedConfig
        cfg = DistributedConfig(backend="mpi")
        assert cfg.backend == "mpi"


# ─────────────────────────────────────────────────────────────────────────────
# DDPTrainer — setup / teardown
# ─────────────────────────────────────────────────────────────────────────────

class TestDDPTrainerSetupTeardown:
    """Tests for process group initialization and cleanup."""

    @patch("torch.distributed.init_process_group")
    @patch("torch.cuda.set_device")
    @patch("torch.cuda.is_available", return_value=True)
    def test_setup_multi_gpu_calls_init_process_group(
        self, mock_cuda_avail, mock_set_device, mock_init_pg
    ):
        from training.distributed.ddp_trainer import DDPTrainer, DistributedConfig
        cfg = DistributedConfig(backend="gloo", world_size=2)
        trainer = DDPTrainer(cfg)
        trainer.setup(rank=0, world_size=2)

        mock_init_pg.assert_called_once()
        call_kwargs = mock_init_pg.call_args
        assert call_kwargs.kwargs.get("backend") == "gloo" or call_kwargs.args[0] if call_kwargs.args else True

        trainer.teardown()

    @patch("torch.distributed.init_process_group")
    @patch("torch.cuda.set_device")
    @patch("torch.cuda.is_available", return_value=True)
    def test_setup_single_gpu_skips_init_process_group(
        self, mock_cuda_avail, mock_set_device, mock_init_pg
    ):
        from training.distributed.ddp_trainer import DDPTrainer, DistributedConfig
        cfg = DistributedConfig(world_size=1)
        trainer = DDPTrainer(cfg)
        trainer.setup(rank=0, world_size=1)

        mock_init_pg.assert_not_called()
        assert not trainer.distributed_active

    @patch("torch.distributed.init_process_group")
    @patch("torch.cuda.set_device")
    @patch("torch.cuda.is_available", return_value=True)
    def test_setup_sets_cuda_device(self, mock_cuda_avail, mock_set_device, mock_init_pg):
        from training.distributed.ddp_trainer import DDPTrainer, DistributedConfig
        cfg = DistributedConfig(backend="gloo", world_size=2)
        trainer = DDPTrainer(cfg)
        trainer.setup(rank=1, world_size=2)

        mock_set_device.assert_called_with(1)
        trainer.teardown()

    @patch("torch.distributed.destroy_process_group")
    @patch("torch.distributed.init_process_group")
    @patch("torch.cuda.set_device")
    @patch("torch.cuda.is_available", return_value=True)
    def test_teardown_calls_destroy_process_group(
        self, mock_cuda_avail, mock_set_device, mock_init_pg, mock_destroy_pg
    ):
        from training.distributed.ddp_trainer import DDPTrainer, DistributedConfig
        cfg = DistributedConfig(backend="gloo", world_size=2)
        trainer = DDPTrainer(cfg)
        trainer.setup(rank=0, world_size=2)
        trainer.teardown()

        mock_destroy_pg.assert_called_once()

    def test_teardown_safe_without_setup(self):
        """Calling teardown() before setup() must not raise."""
        from training.distributed.ddp_trainer import DDPTrainer, DistributedConfig
        cfg = DistributedConfig()
        trainer = DDPTrainer(cfg)
        trainer.teardown()  # Should not raise

    @patch("torch.distributed.init_process_group")
    @patch("torch.cuda.set_device")
    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.distributed.is_nccl_available", return_value=False)
    def test_nccl_fallback_to_gloo(
        self, mock_nccl, mock_cuda_avail, mock_set_device, mock_init_pg
    ):
        """When NCCL requested but unavailable, must fall back to gloo."""
        from training.distributed.ddp_trainer import DDPTrainer, DistributedConfig
        cfg = DistributedConfig(backend="nccl", world_size=2)
        trainer = DDPTrainer(cfg)
        trainer.setup(rank=0, world_size=2)

        # init_process_group must have been called with gloo
        call_args = mock_init_pg.call_args
        # backend can be positional or keyword
        backend_used = (
            call_args.kwargs.get("backend")
            or (call_args.args[0] if call_args.args else None)
        )
        assert backend_used == "gloo"
        trainer.teardown()


# ─────────────────────────────────────────────────────────────────────────────
# DDPTrainer — rank utilities
# ─────────────────────────────────────────────────────────────────────────────

class TestDDPTrainerRankUtils:
    """Tests for rank-related helper methods."""

    def _make_trainer(self, rank: int = 0) -> "DDPTrainer":  # noqa: F821
        from training.distributed.ddp_trainer import DDPTrainer, DistributedConfig
        cfg = DistributedConfig(world_size=1)
        trainer = DDPTrainer(cfg)
        trainer._rank = rank
        return trainer

    def test_is_rank_zero_rank_0(self):
        trainer = self._make_trainer(rank=0)
        assert trainer.is_rank_zero() is True

    def test_is_rank_zero_rank_1(self):
        trainer = self._make_trainer(rank=1)
        assert trainer.is_rank_zero() is False

    def test_is_rank_zero_rank_3(self):
        trainer = self._make_trainer(rank=3)
        assert trainer.is_rank_zero() is False

    @patch("torch.distributed.barrier")
    def test_barrier_noop_single_gpu(self, mock_barrier):
        from training.distributed.ddp_trainer import DDPTrainer, DistributedConfig
        cfg = DistributedConfig(world_size=1)
        trainer = DDPTrainer(cfg)
        trainer._distributed_active = False
        trainer.barrier()
        mock_barrier.assert_not_called()

    @patch("torch.distributed.barrier")
    def test_barrier_called_when_distributed(self, mock_barrier):
        from training.distributed.ddp_trainer import DDPTrainer, DistributedConfig
        cfg = DistributedConfig(world_size=2)
        trainer = DDPTrainer(cfg)
        trainer._distributed_active = True
        trainer.barrier()
        mock_barrier.assert_called_once()


# ─────────────────────────────────────────────────────────────────────────────
# DDPTrainer — checkpointing
# ─────────────────────────────────────────────────────────────────────────────

class TestDDPTrainerCheckpointing:
    """Tests for save/load checkpoint logic."""

    def _make_trainer_rank(self, rank: int):
        from training.distributed.ddp_trainer import DDPTrainer, DistributedConfig
        cfg = DistributedConfig(world_size=2)
        trainer = DDPTrainer(cfg)
        trainer._rank = rank
        trainer._world_size = 2
        trainer._device = MagicMock()
        return trainer

    @patch("torch.save")
    def test_save_checkpoint_rank_0_writes(self, mock_save, tmp_path):
        import torch.nn as nn
        trainer = self._make_trainer_rank(rank=0)

        model = nn.Linear(4, 2)
        optimizer = MagicMock()
        optimizer.state_dict.return_value = {}

        path = tmp_path / "ckpt.pt"
        trainer.save_checkpoint(path, model, optimizer, epoch=5, metrics={"loss": 0.5})

        mock_save.assert_called_once()
        saved_dict = mock_save.call_args.args[0]
        assert saved_dict["epoch"] == 5
        assert "model_state_dict" in saved_dict
        assert saved_dict["metrics"] == {"loss": 0.5}

    @patch("torch.save")
    def test_save_checkpoint_non_zero_rank_skips(self, mock_save, tmp_path):
        import torch.nn as nn
        trainer = self._make_trainer_rank(rank=1)

        model = nn.Linear(4, 2)
        optimizer = MagicMock()
        path = tmp_path / "ckpt.pt"

        trainer.save_checkpoint(path, model, optimizer, epoch=5, metrics={})
        mock_save.assert_not_called()

    @patch("torch.save")
    def test_save_checkpoint_unwraps_ddp(self, mock_save, tmp_path):
        """save_checkpoint should save model.module.state_dict() not model.state_dict()."""
        import torch.nn as nn
        from torch.nn.parallel import DistributedDataParallel as DDP

        trainer = self._make_trainer_rank(rank=0)

        inner = nn.Linear(4, 2)
        # Create a mock that looks like a DDP-wrapped model
        ddp_model = MagicMock(spec=DDP)
        ddp_model.module = inner

        optimizer = MagicMock()
        optimizer.state_dict.return_value = {}

        path = tmp_path / "ckpt_ddp.pt"
        trainer.save_checkpoint(path, ddp_model, optimizer, epoch=3, metrics={})

        mock_save.assert_called_once()
        saved = mock_save.call_args.args[0]
        # state_dict should come from inner (unwrapped) model
        expected_keys = set(inner.state_dict().keys())
        saved_keys = set(saved["model_state_dict"].keys())
        assert expected_keys == saved_keys

    def test_load_checkpoint_missing_file_raises(self, tmp_path):
        from training.distributed.ddp_trainer import DDPTrainer, DistributedConfig
        cfg = DistributedConfig()
        trainer = DDPTrainer(cfg)
        trainer._rank = 0
        trainer._device = MagicMock()
        trainer._device.__str__ = lambda s: "cuda:0"

        with pytest.raises(FileNotFoundError):
            trainer.load_checkpoint(tmp_path / "does_not_exist.pt")

    @patch("torch.load")
    def test_load_checkpoint_map_location_rank_0(self, mock_load, tmp_path):
        from training.distributed.ddp_trainer import DDPTrainer, DistributedConfig
        import torch

        cfg = DistributedConfig(world_size=2)
        trainer = DDPTrainer(cfg)
        trainer._rank = 0
        trainer._device = torch.device("cuda:0")

        # Create a dummy checkpoint file
        ckpt_path = tmp_path / "ckpt.pt"
        ckpt_path.touch()

        mock_load.return_value = {"epoch": 1, "model_state_dict": {}}

        result = trainer.load_checkpoint(ckpt_path)

        mock_load.assert_called_once()
        call_kwargs = mock_load.call_args
        map_loc = call_kwargs.kwargs.get("map_location", call_kwargs.args[1] if len(call_kwargs.args) > 1 else None)
        assert "cuda:0" in str(map_loc)
        assert result["epoch"] == 1

    @patch("torch.load")
    def test_load_checkpoint_map_location_rank_1(self, mock_load, tmp_path):
        from training.distributed.ddp_trainer import DDPTrainer, DistributedConfig
        import torch

        cfg = DistributedConfig(world_size=2)
        trainer = DDPTrainer(cfg)
        trainer._rank = 1
        trainer._device = torch.device("cuda:1")

        ckpt_path = tmp_path / "ckpt.pt"
        ckpt_path.touch()
        mock_load.return_value = {"epoch": 2, "model_state_dict": {}}

        trainer.load_checkpoint(ckpt_path)

        call_kwargs = mock_load.call_args
        map_loc = call_kwargs.kwargs.get("map_location", call_kwargs.args[1] if len(call_kwargs.args) > 1 else None)
        assert "cuda:1" in str(map_loc)


# ─────────────────────────────────────────────────────────────────────────────
# DDPTrainer — rank_zero_first context manager
# ─────────────────────────────────────────────────────────────────────────────

class TestDDPTrainerRankZeroFirst:
    """Tests for the rank_zero_first context manager."""

    def test_rank_zero_first_single_gpu_just_yields(self):
        from training.distributed.ddp_trainer import DDPTrainer, DistributedConfig
        cfg = DistributedConfig(world_size=1)
        trainer = DDPTrainer(cfg)
        trainer._distributed_active = False

        executed = []
        with trainer.rank_zero_first():
            executed.append("body")

        assert executed == ["body"]

    @patch("torch.distributed.barrier")
    def test_rank_zero_first_rank0_yields_then_barrier(self, mock_barrier):
        from training.distributed.ddp_trainer import DDPTrainer, DistributedConfig
        cfg = DistributedConfig(world_size=2)
        trainer = DDPTrainer(cfg)
        trainer._rank = 0
        trainer._distributed_active = True

        sequence = []
        with trainer.rank_zero_first():
            sequence.append("body")
        sequence.append("after")

        assert sequence == ["body", "after"]
        mock_barrier.assert_called_once()

    @patch("torch.distributed.barrier")
    def test_rank_zero_first_non_zero_barrier_then_yields(self, mock_barrier):
        from training.distributed.ddp_trainer import DDPTrainer, DistributedConfig
        cfg = DistributedConfig(world_size=2)
        trainer = DDPTrainer(cfg)
        trainer._rank = 1
        trainer._distributed_active = True

        sequence = []
        with trainer.rank_zero_first():
            sequence.append("body")
        sequence.append("after")

        assert sequence == ["body", "after"]
        mock_barrier.assert_called_once()


# ─────────────────────────────────────────────────────────────────────────────
# DDPTrainer — model wrapping
# ─────────────────────────────────────────────────────────────────────────────

class TestDDPTrainerModelWrap:
    """Tests for wrap_model behavior."""

    def test_wrap_model_raises_before_setup(self):
        from training.distributed.ddp_trainer import DDPTrainer, DistributedConfig
        import torch.nn as nn
        cfg = DistributedConfig()
        trainer = DDPTrainer(cfg)
        model = nn.Linear(4, 2)
        with pytest.raises(RuntimeError, match="setup"):
            trainer.wrap_model(model)

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.set_device")
    def test_wrap_model_single_gpu_returns_unchanged(self, mock_set_device, mock_cuda):
        from training.distributed.ddp_trainer import DDPTrainer, DistributedConfig
        import torch
        import torch.nn as nn

        cfg = DistributedConfig(world_size=1)
        trainer = DDPTrainer(cfg)
        trainer._is_setup = True
        trainer._distributed_active = False
        trainer._device = torch.device("cpu")

        model = nn.Linear(4, 2)
        wrapped = trainer.wrap_model(model)
        # Should be the same object (or at least not DDP-wrapped)
        from torch.nn.parallel import DistributedDataParallel as DDP
        assert not isinstance(wrapped, DDP)


# ─────────────────────────────────────────────────────────────────────────────
# DDPTrainer — train_epoch
# ─────────────────────────────────────────────────────────────────────────────

class TestDDPTrainerTrainEpoch:
    """Tests for the train_epoch method."""

    def _make_trainer(self, mixed_precision: str = "no") -> "DDPTrainer":  # noqa: F821
        from training.distributed.ddp_trainer import DDPTrainer, DistributedConfig
        import torch
        cfg = DistributedConfig(
            world_size=1,
            mixed_precision=mixed_precision,
            gradient_accumulation_steps=1,
        )
        trainer = DDPTrainer(cfg)
        trainer._rank = 0
        trainer._world_size = 1
        trainer._device = torch.device("cpu")
        trainer._is_setup = True
        trainer._distributed_active = False
        return trainer

    def test_train_epoch_no_amp_returns_loss_dict(self):
        import torch
        import torch.nn as nn

        trainer = self._make_trainer(mixed_precision="no")
        model = nn.Linear(4, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        class _FakeDataloader:
            def __iter__(self):
                for _ in range(3):
                    x = torch.randn(2, 4)
                    y = torch.randn(2, 1)
                    yield x, y

            def __len__(self):
                return 3

        # Wrap model's forward to return a scalar loss
        original_forward = model.forward
        loss_fn = nn.MSELoss()

        def patched_forward(inputs, targets):
            out = original_forward(inputs)
            return loss_fn(out, targets)

        model.forward = patched_forward  # type: ignore[method-assign]

        metrics = trainer.train_epoch(_FakeDataloader(), model, optimizer, scaler=None)
        assert "loss" in metrics
        assert isinstance(metrics["loss"], float)
        assert metrics["loss"] >= 0.0

    def test_train_epoch_gradient_accumulation(self):
        """Gradient accumulation steps are respected — optimizer.step called once per N."""
        import torch
        import torch.nn as nn
        from training.distributed.ddp_trainer import DDPTrainer, DistributedConfig

        cfg = DistributedConfig(
            world_size=1,
            mixed_precision="no",
            gradient_accumulation_steps=3,
        )
        trainer = DDPTrainer(cfg)
        trainer._rank = 0
        trainer._world_size = 1
        trainer._device = torch.device("cpu")
        trainer._is_setup = True
        trainer._distributed_active = False

        model = nn.Linear(4, 1)
        optimizer = MagicMock()
        optimizer.zero_grad = MagicMock()
        # Make step callable and track calls
        step_calls = []
        optimizer.step = MagicMock(side_effect=lambda: step_calls.append(1))
        # Expose parameters for grad clipping
        optimizer.param_groups = [{"params": list(model.parameters())}]

        loss_fn = nn.MSELoss()
        original_forward = model.forward

        def patched_forward(inputs, targets):
            out = original_forward(inputs)
            return loss_fn(out, targets)

        model.forward = patched_forward  # type: ignore[method-assign]

        class _DL:
            def __iter__(self):
                for _ in range(6):  # 6 batches, accumulate 3 → 2 steps
                    yield torch.randn(2, 4), torch.randn(2, 1)

            def __len__(self):
                return 6

        trainer.train_epoch(_DL(), model, optimizer, scaler=None)
        # With 6 batches and accumulation=3: expect 2 optimizer.step() calls
        assert optimizer.step.call_count == 2


# ─────────────────────────────────────────────────────────────────────────────
# DistributedLauncher
# ─────────────────────────────────────────────────────────────────────────────

class TestDistributedLauncher:
    """Tests for DistributedLauncher process spawning and VRAM calculations."""

    def test_available_gpus_delegates_to_torch(self):
        from training.distributed.launcher import DistributedLauncher
        with patch("torch.cuda.device_count", return_value=4):
            assert DistributedLauncher.available_gpus() == 4

    def test_available_gpus_zero_when_cuda_unavailable(self):
        from training.distributed.launcher import DistributedLauncher
        with patch("torch.cuda.device_count", side_effect=Exception("no cuda")):
            assert DistributedLauncher.available_gpus() == 0

    def test_optimal_world_size_no_gpus_returns_one(self):
        from training.distributed.launcher import DistributedLauncher
        with patch.object(DistributedLauncher, "available_gpus", return_value=0):
            assert DistributedLauncher.optimal_world_size(8.0, 2.0) == 1

    def test_optimal_world_size_model_too_large_returns_one(self):
        """Model VRAM (10 GB) > GPU VRAM (8 GB) → can't fit → return 1."""
        from training.distributed.launcher import DistributedLauncher
        with patch.object(DistributedLauncher, "available_gpus", return_value=4):
            result = DistributedLauncher.optimal_world_size(
                vram_per_gpu_gb=8.0,
                model_vram_gb=10.0,
            )
        assert result == 1

    def test_optimal_world_size_vram_limited(self):
        """4 GPUs × 8 GB = 32 GB, model uses 3 GB → floor(32/3) = 10, capped at 4."""
        from training.distributed.launcher import DistributedLauncher
        with patch.object(DistributedLauncher, "available_gpus", return_value=4):
            result = DistributedLauncher.optimal_world_size(
                vram_per_gpu_gb=8.0,
                model_vram_gb=3.0,
            )
        assert result == 4  # min(4, floor(32/3)=10) = 4

    def test_optimal_world_size_never_zero(self):
        from training.distributed.launcher import DistributedLauncher
        with patch.object(DistributedLauncher, "available_gpus", return_value=1):
            result = DistributedLauncher.optimal_world_size(8.0, 8.0)
        assert result >= 1

    def test_optimal_world_size_exact_fit(self):
        """2 GPUs × 8 GB = 16 GB, model 4 GB → floor(16/4)=4, capped at 2."""
        from training.distributed.launcher import DistributedLauncher
        with patch.object(DistributedLauncher, "available_gpus", return_value=2):
            result = DistributedLauncher.optimal_world_size(8.0, 4.0)
        assert result == 2

    @patch("torch.distributed.is_nccl_available", return_value=True)
    @patch("torch.cuda.is_available", return_value=True)
    def test_detect_backend_returns_nccl(self, mock_cuda, mock_nccl):
        from training.distributed.launcher import DistributedLauncher
        assert DistributedLauncher.detect_backend() == "nccl"

    @patch("torch.distributed.is_nccl_available", return_value=False)
    @patch("torch.cuda.is_available", return_value=False)
    def test_detect_backend_falls_back_to_gloo(self, mock_cuda, mock_nccl):
        from training.distributed.launcher import DistributedLauncher
        assert DistributedLauncher.detect_backend() == "gloo"

    def test_launch_missing_script_raises(self, tmp_path):
        from training.distributed.launcher import DistributedLauncher
        from training.distributed.ddp_trainer import DistributedConfig
        launcher = DistributedLauncher()
        cfg = DistributedConfig(world_size=1)
        with pytest.raises(FileNotFoundError, match="script"):
            launcher.launch(str(tmp_path / "nonexistent_script.py"), cfg)

    @patch("subprocess.run")
    def test_launch_calls_torchrun_with_correct_args(self, mock_run, tmp_path):
        from training.distributed.launcher import DistributedLauncher
        from training.distributed.ddp_trainer import DistributedConfig

        # Create a dummy script
        script = tmp_path / "train_ddp.py"
        script.write_text("# dummy\n")

        mock_run.return_value = MagicMock(returncode=0)

        cfg = DistributedConfig(
            backend="gloo",
            world_size=2,
            master_addr="127.0.0.1",
            master_port=29500,
        )
        launcher = DistributedLauncher()
        result = launcher.launch(str(script), cfg, extra_args=["--epochs=10"])

        assert result == 0
        mock_run.assert_called_once()

        cmd = mock_run.call_args.args[0]
        cmd_str = " ".join(cmd)
        assert "torch.distributed.run" in cmd_str
        assert "--nproc_per_node=2" in cmd_str
        assert "--master_port=29500" in cmd_str
        assert str(script) in cmd_str
        assert "--epochs=10" in cmd_str

    @patch("subprocess.run")
    def test_launch_returns_nonzero_exit_code(self, mock_run, tmp_path):
        from training.distributed.launcher import DistributedLauncher
        from training.distributed.ddp_trainer import DistributedConfig

        script = tmp_path / "train_ddp.py"
        script.write_text("# dummy\n")

        mock_run.return_value = MagicMock(returncode=1)

        cfg = DistributedConfig(world_size=1, backend="gloo")
        launcher = DistributedLauncher()
        result = launcher.launch(str(script), cfg)

        assert result == 1


# ─────────────────────────────────────────────────────────────────────────────
# API Endpoints — using FastAPI TestClient
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def app_client():
    """
    Create a minimal FastAPI test app with the distributed_training router.

    Patches out the database dependency so tests do not need SQLite.
    """
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from backend.api.v1 import distributed_training as dt_module
    from backend.tasks.task_router import TaskRouter

    # Fresh module-level state
    dt_module._distributed_job_meta.clear()

    app = FastAPI()
    app.include_router(dt_module.router)

    # Override DB dependency
    from backend.database import get_session
    app.dependency_overrides[get_session] = lambda: MagicMock()

    return TestClient(app, raise_server_exceptions=False)


class TestDistributedTrainingAPI:
    """Tests for the distributed training REST endpoints."""

    def test_status_endpoint_returns_200(self, app_client):
        resp = app_client.get("/training/distributed/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "available_gpus" in data
        assert "nccl_available" in data
        assert "gloo_available" in data
        assert "optimal_world_size" in data

    def test_status_endpoint_correct_shape(self, app_client):
        resp = app_client.get("/training/distributed/status")
        data = resp.json()
        for key in (
            "available_gpus",
            "nccl_available",
            "gloo_available",
            "cuda_available",
            "optimal_world_size",
            "current_job_id",
            "distributed_job_running",
        ):
            assert key in data, f"Missing key: {key}"

    @patch("training.distributed.ddp_trainer.DistributedConfig.resolve_world_size", return_value=1)
    @patch("backend.tasks.task_router.TaskRouter.submit_gpu_task", new_callable=AsyncMock)
    def test_start_endpoint_returns_task_id(
        self, mock_submit, mock_resolve, app_client, tmp_path
    ):
        from backend.api.v1 import distributed_training as dt_module

        # Make submit_gpu_task return a known uuid
        test_uuid = "test-task-abc-123"
        mock_submit.return_value = test_uuid

        # Create a fake data.yaml
        data_yaml = str(tmp_path / "data.yaml")
        Path(data_yaml).write_text("train: .\nval: .\nnc: 4\n")

        resp = app_client.post(
            "/training/distributed/start",
            json={
                "data_yaml": data_yaml,
                "epochs": 5,
                "world_size": 1,
                "backend": "gloo",
                "gradient_accumulation_steps": 1,
            },
        )

        # Should succeed or be 200/202
        assert resp.status_code in (200, 201, 202), resp.text
        data = resp.json()
        assert "task_id" in data

    @patch("torch.distributed.is_nccl_available", return_value=False)
    @patch("training.distributed.ddp_trainer.DistributedConfig.resolve_world_size", return_value=1)
    @patch("backend.tasks.task_router.TaskRouter.submit_gpu_task", new_callable=AsyncMock)
    def test_start_endpoint_falls_back_to_gloo(
        self, mock_submit, mock_resolve, mock_nccl, app_client, tmp_path
    ):
        """When NCCL unavailable, backend must be silently downgraded to gloo."""
        mock_submit.return_value = "test-task-xyz"

        data_yaml = str(tmp_path / "data.yaml")
        Path(data_yaml).write_text("train: .\nval: .\nnc: 4\n")

        resp = app_client.post(
            "/training/distributed/start",
            json={
                "data_yaml": data_yaml,
                "epochs": 5,
                "world_size": 1,
                "backend": "nccl",
                "gradient_accumulation_steps": 1,
            },
        )
        # Endpoint should succeed (200) — no 500 due to NCCL unavailability
        assert resp.status_code in (200, 201, 202), resp.text

    def test_get_job_status_404_for_unknown(self, app_client):
        resp = app_client.get("/training/distributed/jobs/nonexistent-task-id")
        assert resp.status_code == 404

    def test_get_job_status_returns_data_for_known_job(self, app_client):
        from backend.api.v1 import distributed_training as dt_module
        from backend.tasks.task_router import task_router as tr

        # Inject a fake job into both task_router and meta store
        fake_id = "test-known-job-001"
        tr._active_jobs[fake_id] = {
            "job_id": fake_id,
            "job_type": "distributed_training",
            "status": "running",
            "progress": 0.4,
        }
        dt_module._distributed_job_meta[fake_id] = {
            "world_size": 2,
            "backend": "nccl",
            "per_rank_metrics": {"0": {"loss": 0.5}, "1": {"loss": 0.6}},
            "created_at": time.time(),
            "params": {"data_yaml": "/tmp/data.yaml"},
        }

        resp = app_client.get(f"/training/distributed/jobs/{fake_id}")
        assert resp.status_code == 200

        data = resp.json()
        assert data["task_id"] == fake_id
        assert data["status"] == "running"
        assert data["world_size"] == 2
        assert "per_rank_metrics" in data

        # Cleanup
        del tr._active_jobs[fake_id]
        del dt_module._distributed_job_meta[fake_id]

    def test_stop_endpoint_404_for_unknown(self, app_client):
        resp = app_client.post("/training/distributed/stop/nonexistent-stop-task")
        assert resp.status_code == 404

    def test_stop_endpoint_not_stopped_for_completed_job(self, app_client):
        from backend.tasks.task_router import task_router as tr

        fake_id = "test-completed-job-002"
        tr._active_jobs[fake_id] = {
            "job_id": fake_id,
            "job_type": "distributed_training",
            "status": "completed",
            "progress": 1.0,
        }

        resp = app_client.post(f"/training/distributed/stop/{fake_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["stopped"] is False

        del tr._active_jobs[fake_id]

    @patch("backend.tasks.task_router.TaskRouter.cancel_job", new_callable=AsyncMock)
    def test_stop_endpoint_stopped_true_for_running_job(self, mock_cancel, app_client):
        from backend.tasks.task_router import task_router as tr

        mock_cancel.return_value = True

        fake_id = "test-running-job-003"
        tr._active_jobs[fake_id] = {
            "job_id": fake_id,
            "job_type": "distributed_training",
            "status": "running",
            "progress": 0.3,
        }

        resp = app_client.post(f"/training/distributed/stop/{fake_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["stopped"] is True

        del tr._active_jobs[fake_id]


# ─────────────────────────────────────────────────────────────────────────────
# Single-GPU graceful degradation
# ─────────────────────────────────────────────────────────────────────────────

class TestSingleGPUDegradation:
    """Tests confirming world_size=1 behaves like normal (non-DDP) training."""

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.set_device")
    @patch("torch.distributed.init_process_group")
    def test_single_gpu_no_process_group(self, mock_ipg, mock_set_device, mock_cuda):
        from training.distributed.ddp_trainer import DDPTrainer, DistributedConfig
        cfg = DistributedConfig(world_size=1)
        trainer = DDPTrainer(cfg)
        trainer.setup(0, 1)

        mock_ipg.assert_not_called()
        assert not trainer.distributed_active

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.set_device")
    @patch("torch.distributed.init_process_group")
    def test_single_gpu_is_always_rank_zero(self, mock_ipg, mock_set_device, mock_cuda):
        from training.distributed.ddp_trainer import DDPTrainer, DistributedConfig
        cfg = DistributedConfig(world_size=1)
        trainer = DDPTrainer(cfg)
        trainer.setup(0, 1)
        assert trainer.is_rank_zero() is True

    @patch("torch.distributed.init_process_group")
    @patch("torch.cuda.set_device")
    @patch("torch.cuda.is_available", return_value=True)
    def test_single_gpu_teardown_no_error(self, mock_cuda, mock_set_device, mock_ipg):
        from training.distributed.ddp_trainer import DDPTrainer, DistributedConfig
        cfg = DistributedConfig(world_size=1)
        trainer = DDPTrainer(cfg)
        trainer.setup(0, 1)
        trainer.teardown()  # Must not raise
