"""
training.distributed — Multi-GPU DistributedDataParallel support.

Provides:
  - DistributedConfig: dataclass for DDP + AMP configuration
  - DDPTrainer: PyTorch DDP wrapper with rank-0 privilege, AMP, gradient accumulation
  - DistributedLauncher: torchrun-based process spawner

Designed for the RTX 4060 (single-GPU) → multi-GPU scale-out path.
When world_size == 1, DDPTrainer falls back to plain training (no DDP overhead).

Usage:
    from training.distributed import DDPTrainer, DistributedConfig, DistributedLauncher

    config = DistributedConfig(world_size=2, mixed_precision="fp16")
    launcher = DistributedLauncher()
    launcher.launch("training/scripts/train_ddp.py", config)
"""

from training.distributed.ddp_trainer import DDPTrainer, DistributedConfig
from training.distributed.launcher import DistributedLauncher

__all__ = [
    "DDPTrainer",
    "DistributedConfig",
    "DistributedLauncher",
]
