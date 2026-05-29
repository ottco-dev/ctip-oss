"""
training.distributed.ddp_trainer — PyTorch DistributedDataParallel wrapper.

DESIGN PRINCIPLES
-----------------
- Rank 0 is authoritative: all logging, checkpointing, and MLflow calls happen
  only on rank 0. Non-zero ranks are silent to avoid log spam.
- Single-GPU graceful degradation: when world_size == 1 the class skips
  init_process_group / DDP wrapping entirely and behaves like a plain trainer.
- AMP (fp16 / bf16): GradScaler is only used for fp16; bf16 uses torch.autocast
  without a scaler. AMP is skipped when mixed_precision=="no".
- Gradient accumulation: loss is divided by gradient_accumulation_steps before
  backward. Optimizer.step() is called only on the accumulation boundary.
- SyncBatchNorm: when sync_batchnorm=True, convert_sync_batchnorm() is applied
  after DDP wrapping so BN statistics are synchronized across all ranks.
- Checkpoint safety: save_checkpoint() only runs on rank 0. All ranks call
  load_checkpoint() independently using map_location=f"cuda:{rank}" so each
  rank loads directly onto its own device without bouncing through CPU
  (avoids 2x VRAM spike on large models).

TORCHRUN INTEGRATION
---------------------
This class expects the process to have been launched via torchrun (or
torch.distributed.launch), which sets the following environment variables:
  LOCAL_RANK, RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT

Usage:
    # In your torchrun-launched script:
    import os
    from training.distributed.ddp_trainer import DDPTrainer, DistributedConfig

    config = DistributedConfig(world_size=2, mixed_precision="fp16")
    trainer = DDPTrainer(config, base_trainer=my_yolo_trainer)
    rank = int(os.environ["LOCAL_RANK"])
    trainer.setup(rank, config.world_size)

    for epoch in range(config.epochs):
        metrics = trainer.train_epoch(dataloader, model, optimizer, scaler)
        if trainer.is_rank_zero():
            trainer.save_checkpoint(f"ckpt_{epoch}.pt", model, optimizer, epoch, metrics)

    trainer.teardown()
"""

from __future__ import annotations

import math
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generator

from shared.logging.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# DistributedConfig
# ---------------------------------------------------------------------------

_VALID_BACKENDS = frozenset({"nccl", "gloo", "mpi"})
_VALID_PRECISION = frozenset({"fp16", "bf16", "no"})


@dataclass
class DistributedConfig:
    """
    Configuration for DistributedDataParallel training.

    Validates itself on construction so that bad values are caught early,
    before any process-group initialization.
    """

    backend: str = "nccl"
    """
    Process-group communication backend.
    - nccl: fastest for GPU-to-GPU (NVLink / PCIe). Requires CUDA.
    - gloo: CPU fallback or when NCCL is unavailable.
    - mpi: multi-node clusters with MPI installed.
    """

    world_size: int = -1
    """
    Total number of processes (one per GPU).
    -1 = auto-detect from CUDA_VISIBLE_DEVICES / torch.cuda.device_count().
    """

    master_addr: str = "127.0.0.1"
    """IP or hostname of rank-0 process. Override for multi-node."""

    master_port: int = 29500
    """TCP port for the rendezvous. Must be free on master_addr host."""

    find_unused_parameters: bool = False
    """
    DDP flag. Set True when your model has conditional branches that may skip
    some parameters. Adds overhead — only enable if you see DDP hangs.
    """

    gradient_accumulation_steps: int = 1
    """
    Number of micro-batches to accumulate before calling optimizer.step().
    Effective batch size = per-GPU batch_size × gradient_accumulation_steps × world_size.
    """

    sync_batchnorm: bool = True
    """
    Convert all BatchNorm layers to SyncBatchNorm before DDP wrapping.
    Required for correct BN statistics when batch size per GPU is small.
    """

    mixed_precision: str = "fp16"
    """
    AMP dtype:
    - fp16: GradScaler + autocast. Best throughput on Turing/Ampere.
    - bf16: autocast only, no scaler. Requires Ampere+ (e.g. A100, RTX 30xx+).
    - no:  full float32, no autocast.
    """

    gradient_checkpointing: bool = False
    """
    Enable gradient checkpointing to trade compute for VRAM.
    Reduces activation memory at the cost of ~33 % extra forward passes.
    """

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        if self.backend not in _VALID_BACKENDS:
            raise ValueError(
                f"Invalid backend {self.backend!r}. "
                f"Must be one of {sorted(_VALID_BACKENDS)}"
            )
        if self.mixed_precision not in _VALID_PRECISION:
            raise ValueError(
                f"Invalid mixed_precision {self.mixed_precision!r}. "
                f"Must be one of {sorted(_VALID_PRECISION)}"
            )
        if self.world_size != -1 and self.world_size < 1:
            raise ValueError(
                f"world_size must be -1 (auto) or >= 1, got {self.world_size}"
            )
        if not (1 <= self.master_port <= 65535):
            raise ValueError(
                f"master_port must be in [1, 65535], got {self.master_port}"
            )
        if self.gradient_accumulation_steps < 1:
            raise ValueError(
                f"gradient_accumulation_steps must be >= 1, "
                f"got {self.gradient_accumulation_steps}"
            )

    def resolve_world_size(self) -> int:
        """
        Return the effective world_size.

        If world_size == -1, derive from CUDA_VISIBLE_DEVICES or
        torch.cuda.device_count().
        """
        if self.world_size != -1:
            return self.world_size

        # Honour CUDA_VISIBLE_DEVICES if set
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if cvd and cvd.lower() not in ("", "-1", "nodevfiles"):
            return len([d for d in cvd.split(",") if d.strip()])

        try:
            import torch
            return max(1, torch.cuda.device_count())
        except Exception:
            return 1


# ---------------------------------------------------------------------------
# DDPTrainer
# ---------------------------------------------------------------------------


class DDPTrainer:
    """
    PyTorch DistributedDataParallel wrapper for YOLO / morphology training.

    Follows the torchrun (torch.distributed.run) launch pattern where each
    process receives its LOCAL_RANK via environment variable.  This class
    does NOT spawn processes — use DistributedLauncher for that.

    Single-GPU graceful degradation
    --------------------------------
    When world_size resolves to 1, setup() skips init_process_group and
    DDP wrapping.  All public methods still work correctly, so callers need
    not branch on world_size themselves.

    Rank-0 privilege
    ----------------
    Only rank-0 calls logger.info/warning, saves checkpoints, and interacts
    with MLflow.  Other ranks use logger.debug for reduced noise.
    """

    def __init__(
        self,
        config: DistributedConfig,
        base_trainer: Any = None,
    ) -> None:
        """
        Args:
            config:       DistributedConfig instance.
            base_trainer: Optional reference to the underlying training object
                          (e.g. a YOLOTrainer).  DDPTrainer does not call it
                          directly — callers may use it via self.base_trainer.
        """
        self._config = config
        self.base_trainer = base_trainer
        self._rank: int = 0
        self._world_size: int = 1
        self._device: Any = None
        self._is_setup: bool = False
        self._distributed_active: bool = False
        """True only when a real process-group (world_size > 1) is initialized."""

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def setup(self, rank: int, world_size: int) -> None:
        """
        Initialize the process group, set the CUDA device, and wrap model in DDP.

        For single-GPU (world_size==1), the process group is skipped and the
        model is not wrapped.

        Args:
            rank:       Local process rank (0 … world_size-1).
            world_size: Total number of processes.
        """
        import torch

        self._rank = rank
        self._world_size = world_size

        # Determine effective backend — fall back to gloo when NCCL unavailable
        backend = self._config.backend
        if backend == "nccl" and not torch.distributed.is_nccl_available():
            logger.warning(
                "NCCL backend requested but not available — falling back to gloo",
                rank=rank,
            )
            backend = "gloo"

        if world_size > 1:
            # Set device before init_process_group (required for NCCL)
            if torch.cuda.is_available():
                torch.cuda.set_device(rank)
                self._device = torch.device(f"cuda:{rank}")
            else:
                self._device = torch.device("cpu")

            os.environ.setdefault("MASTER_ADDR", self._config.master_addr)
            os.environ.setdefault("MASTER_PORT", str(self._config.master_port))

            torch.distributed.init_process_group(
                backend=backend,
                rank=rank,
                world_size=world_size,
            )
            self._distributed_active = True

            if self.is_rank_zero():
                logger.info(
                    "Distributed process group initialized",
                    backend=backend,
                    world_size=world_size,
                    device=str(self._device),
                )
        else:
            # Single-GPU or CPU — no DDP, no process group
            if torch.cuda.is_available():
                torch.cuda.set_device(0)
                self._device = torch.device("cuda:0")
            else:
                self._device = torch.device("cpu")

            logger.info(
                "Single-process mode (world_size=1) — DDP skipped",
                device=str(self._device),
            )

        self._is_setup = True

    def teardown(self) -> None:
        """Destroy the process group (if active) and release resources."""
        import torch

        if self._distributed_active:
            try:
                torch.distributed.destroy_process_group()
                if self.is_rank_zero():
                    logger.info("Distributed process group destroyed")
            except Exception as exc:
                logger.warning("destroy_process_group error", error=str(exc))
            self._distributed_active = False

        self._is_setup = False

    # ------------------------------------------------------------------
    # Model helpers
    # ------------------------------------------------------------------

    def wrap_model(self, model: Any) -> Any:
        """
        Wrap a PyTorch model in DDP (multi-GPU) or return it unchanged (single-GPU).

        Should be called after setup().  When sync_batchnorm=True, converts all
        BatchNorm layers to SyncBatchNorm *before* DDP wrapping.

        Args:
            model: nn.Module to wrap.

        Returns:
            DDP-wrapped model (or the original model when world_size==1).
        """
        import torch
        import torch.nn as nn

        if not self._is_setup:
            raise RuntimeError("DDPTrainer.setup() must be called before wrap_model()")

        model = model.to(self._device)

        if not self._distributed_active:
            return model  # single-GPU path

        if self._config.sync_batchnorm:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            if self.is_rank_zero():
                logger.info("BatchNorm layers converted to SyncBatchNorm")

        if self._config.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()

        try:
            from torch.nn.parallel import DistributedDataParallel as DDP
            model = DDP(
                model,
                device_ids=[self._rank],
                find_unused_parameters=self._config.find_unused_parameters,
            )
        except Exception as exc:
            logger.error("DDP wrapping failed", error=str(exc))
            raise

        return model

    # ------------------------------------------------------------------
    # Context managers
    # ------------------------------------------------------------------

    @contextmanager
    def rank_zero_first(self) -> Generator[None, None, None]:
        """
        Context manager that lets rank 0 execute first, then synchronizes all
        other ranks via a barrier.

        Useful for one-time operations such as dataset preparation or cache
        creation that should only run once (on rank 0) before other ranks
        proceed.

        Example::

            with trainer.rank_zero_first():
                if trainer.is_rank_zero():
                    prepare_dataset_cache()
        """
        import torch

        if not self._distributed_active:
            yield
            return

        if self.is_rank_zero():
            yield
            torch.distributed.barrier()
        else:
            torch.distributed.barrier()
            yield

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_epoch(
        self,
        dataloader: Any,
        model: Any,
        optimizer: Any,
        scaler: Any,
    ) -> dict[str, float]:
        """
        Run one training epoch with gradient accumulation and AMP.

        Handles:
        - Mixed precision autocast (fp16 / bf16 / no)
        - GradScaler for fp16 (not needed for bf16)
        - Gradient accumulation across micro-batches
        - DDP gradient synchronization (no_sync context for accumulation steps)

        Args:
            dataloader: Iterable that yields (inputs, targets) tuples.
            model:      Model (possibly DDP-wrapped).
            optimizer:  PyTorch optimizer.
            scaler:     GradScaler instance (pass None for bf16 / fp16-less).

        Returns:
            Dictionary of aggregated metrics: {"loss": float, ...}.
        """
        import torch

        model.train()
        total_loss = 0.0
        num_batches = 0
        accumulation_steps = self._config.gradient_accumulation_steps

        use_amp = self._config.mixed_precision != "no"
        amp_dtype = (
            torch.float16 if self._config.mixed_precision == "fp16" else torch.bfloat16
        )
        use_scaler = self._config.mixed_precision == "fp16" and scaler is not None

        # Determine autocast device type
        device_type = "cuda" if str(self._device).startswith("cuda") else "cpu"

        for step, batch in enumerate(dataloader):
            # Move batch to device
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                inputs, targets = batch[0], batch[1]
                if hasattr(inputs, "to"):
                    inputs = inputs.to(self._device, non_blocking=True)
                if hasattr(targets, "to"):
                    targets = targets.to(self._device, non_blocking=True)
            else:
                inputs = batch
                targets = None

            is_accumulating = (step + 1) % accumulation_steps != 0
            is_last_batch = (step + 1) == len(dataloader) if hasattr(dataloader, "__len__") else False

            # Use no_sync during accumulation steps to avoid premature all-reduce
            if self._distributed_active and is_accumulating and not is_last_batch:
                sync_ctx = model.no_sync()
            else:
                from contextlib import nullcontext
                sync_ctx = nullcontext()

            with sync_ctx:
                if use_amp:
                    with torch.autocast(device_type=device_type, dtype=amp_dtype):
                        loss = self._compute_loss(model, inputs, targets)
                else:
                    loss = self._compute_loss(model, inputs, targets)

                # Scale loss for gradient accumulation
                scaled_loss = loss / accumulation_steps

                if use_scaler:
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

            total_loss += loss.item()
            num_batches += 1

            # Optimizer step at accumulation boundary or last batch
            if not is_accumulating or is_last_batch:
                if use_scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=10.0
                    )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=10.0
                    )
                    optimizer.step()

                optimizer.zero_grad()

        avg_loss = total_loss / max(num_batches, 1)
        metrics: dict[str, float] = {"loss": avg_loss}

        return metrics

    def _compute_loss(self, model: Any, inputs: Any, targets: Any) -> Any:
        """
        Forward pass returning a scalar loss.

        Handles three common model output signatures:
        1. Model returns a loss tensor directly.
        2. Model returns (loss, logits) tuple.
        3. Model returns an object with .loss attribute (HuggingFace-style).

        Args:
            model:   The (possibly DDP-wrapped) model.
            inputs:  Input tensor or dict.
            targets: Target tensor or None.
        """
        import torch

        if targets is not None:
            output = model(inputs, targets)
        else:
            output = model(inputs)

        if isinstance(output, tuple) and len(output) >= 1:
            loss = output[0]
        elif hasattr(output, "loss") and output.loss is not None:
            loss = output.loss
        elif isinstance(output, torch.Tensor):
            loss = output
        else:
            raise RuntimeError(
                f"Cannot extract loss from model output of type {type(output)!r}. "
                "Ensure your model returns a loss tensor, (loss, ...) tuple, "
                "or an object with a .loss attribute."
            )

        if not isinstance(loss, torch.Tensor) or loss.ndim != 0:
            raise RuntimeError(
                f"Expected scalar loss tensor, got shape {getattr(loss, 'shape', 'N/A')}"
            )

        return loss

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(
        self,
        path: str | Path,
        model: Any,
        optimizer: Any,
        epoch: int,
        metrics: dict[str, float],
    ) -> None:
        """
        Save training checkpoint.

        Only rank 0 writes to disk.  Unwraps DDP before saving so the
        checkpoint can be loaded without a distributed environment.

        Args:
            path:      Destination file path (only rank 0 uses this).
            model:     The (possibly DDP-wrapped) model.
            optimizer: Optimizer whose state_dict will be saved.
            epoch:     Current epoch number.
            metrics:   Metrics dict to embed in the checkpoint.
        """
        import torch
        from torch.nn.parallel import DistributedDataParallel as DDP

        if not self.is_rank_zero():
            return

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Unwrap DDP to get the raw module state_dict
        raw_model = model.module if isinstance(model, DDP) else model
        model_state = raw_model.state_dict()

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model_state,
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "world_size": self._world_size,
            "mixed_precision": self._config.mixed_precision,
        }

        torch.save(checkpoint, str(path))
        logger.info(
            "Checkpoint saved",
            path=str(path),
            epoch=epoch,
            metrics=metrics,
        )

    def load_checkpoint(self, path: str | Path) -> dict[str, Any]:
        """
        Load a checkpoint on all ranks.

        Uses map_location=f"cuda:{rank}" so each rank loads directly onto its
        assigned GPU, avoiding the VRAM spike from loading on CPU first.

        Args:
            path: Path to the checkpoint file (.pt / .pth).

        Returns:
            The checkpoint dictionary. Callers should apply
            model.load_state_dict(ckpt["model_state_dict"]).

        Raises:
            FileNotFoundError: If the checkpoint path does not exist.
        """
        import torch

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        if self._device is not None:
            map_location = str(self._device)
        else:
            map_location = f"cuda:{self._rank}" if self._rank >= 0 else "cpu"

        checkpoint: dict[str, Any] = torch.load(
            str(path),
            map_location=map_location,
            weights_only=False,
        )

        logger.debug(
            "Checkpoint loaded",
            path=str(path),
            rank=self._rank,
            epoch=checkpoint.get("epoch", "?"),
        )
        return checkpoint

    # ------------------------------------------------------------------
    # Distributed utilities
    # ------------------------------------------------------------------

    def is_rank_zero(self) -> bool:
        """Return True when this process is the coordinator (rank 0)."""
        return self._rank == 0

    def barrier(self) -> None:
        """
        Block all processes until every rank reaches this point.

        No-op when world_size == 1 (single-GPU mode).
        """
        import torch

        if self._distributed_active:
            torch.distributed.barrier()

    def all_reduce_mean(self, tensor: Any) -> Any:
        """
        Average a scalar tensor across all ranks.

        Returns the original tensor unchanged when world_size == 1.

        Args:
            tensor: A 0-d or 1-d CUDA tensor.

        Returns:
            Tensor with global mean.
        """
        import torch

        if not self._distributed_active:
            return tensor

        dist_tensor = tensor.clone()
        torch.distributed.all_reduce(dist_tensor, op=torch.distributed.ReduceOp.SUM)
        dist_tensor /= self._world_size
        return dist_tensor

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def world_size(self) -> int:
        return self._world_size

    @property
    def device(self) -> Any:
        return self._device

    @property
    def config(self) -> DistributedConfig:
        return self._config

    @property
    def distributed_active(self) -> bool:
        """True when a multi-rank process group is initialized."""
        return self._distributed_active
