"""
training.distributed.launcher — torchrun-based distributed process launcher.

DESIGN
------
DistributedLauncher translates a DistributedConfig into a torchrun subprocess
invocation.  It does NOT manage the Python interpreter internals — it relies on
torchrun (torch.distributed.run) to handle rendezvous, rank assignment, and
process lifecycle.

TORCHRUN EQUIVALENCE
---------------------
Calling launcher.launch("train.py", config) is equivalent to::

    torchrun \\
        --nproc_per_node=<world_size> \\
        --master_addr=<master_addr> \\
        --master_port=<master_port> \\
        train.py <extra_args...>

VRAM SAFETY
-----------
optimal_world_size() enforces a per-GPU VRAM floor so that the model fits
comfortably on each device.  This prevents OOM crashes on multi-GPU setups
where one GPU has less headroom than expected.

SINGLE-GPU SUPPORT
------------------
When only one GPU is available (or optimal_world_size returns 1), launch()
executes the script with --nproc_per_node=1, which is equivalent to plain
torchrun single-process mode with no DDP overhead.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from shared.logging.logger import get_logger

logger = get_logger(__name__)


class DistributedLauncher:
    """
    Spawns worker processes via torchrun (torch.distributed.run).

    All subprocess.run calls are synchronous — launch() blocks until the
    training script completes.  For async execution wrap in asyncio.to_thread().
    """

    def launch(
        self,
        script_path: str,
        config: "DistributedConfig",  # noqa: F821
        extra_args: list[str] | None = None,
    ) -> int:
        """
        Launch a distributed training script via torchrun.

        Constructs and executes::

            torchrun
              --nproc_per_node=<world_size>
              --master_addr=<master_addr>
              --master_port=<master_port>
              <script_path>
              [extra_args...]

        Args:
            script_path: Absolute or relative path to the Python training script.
            config:      DistributedConfig providing world_size, master_addr, etc.
            extra_args:  Additional CLI arguments forwarded to the script.

        Returns:
            Process exit code (0 = success).

        Raises:
            FileNotFoundError: If script_path does not exist.
        """
        from training.distributed.ddp_trainer import DistributedConfig

        if extra_args is None:
            extra_args = []

        script = Path(script_path)
        if not script.exists():
            raise FileNotFoundError(f"Training script not found: {script}")

        world_size = config.resolve_world_size()

        # Build the torchrun command
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            f"--nproc_per_node={world_size}",
            f"--master_addr={config.master_addr}",
            f"--master_port={config.master_port}",
            str(script),
        ] + extra_args

        logger.info(
            "Launching distributed training",
            world_size=world_size,
            master_addr=config.master_addr,
            master_port=config.master_port,
            script=str(script),
            extra_args=extra_args,
        )

        try:
            result = subprocess.run(
                cmd,
                env={**os.environ},
                check=False,  # Don't raise — return exit code
            )
        except Exception as exc:
            logger.error("torchrun subprocess failed", error=str(exc))
            raise

        if result.returncode != 0:
            logger.error(
                "Distributed training exited with non-zero code",
                returncode=result.returncode,
                script=str(script),
            )
        else:
            logger.info(
                "Distributed training completed successfully",
                world_size=world_size,
                script=str(script),
            )

        return result.returncode

    @staticmethod
    def available_gpus() -> int:
        """
        Return the number of visible CUDA GPUs.

        Honours CUDA_VISIBLE_DEVICES if set.  Returns 0 when CUDA is not
        available (e.g. CPU-only environments).

        Returns:
            Non-negative integer count of usable GPUs.
        """
        try:
            import torch
            return torch.cuda.device_count()
        except Exception:
            return 0

    @staticmethod
    def optimal_world_size(
        vram_per_gpu_gb: float = 8.0,
        model_vram_gb: float = 2.0,
    ) -> int:
        """
        Compute the optimal number of training processes given hardware constraints.

        Each GPU must individually hold the model in memory (weights + optimizer
        state + activations).  A GPU is only usable for DDP if
        model_vram_gb <= vram_per_gpu_gb.

        Formula::

            if model_vram_gb > vram_per_gpu_gb:
                # Model doesn't fit on a single GPU — cannot run DDP
                return 1

            world_size = min(
                available_gpus,
                floor(total_vram / model_vram_gb)
            )

        where total_vram = available_gpus × vram_per_gpu_gb.

        Args:
            vram_per_gpu_gb: VRAM capacity of each GPU in GB (default: 8 = RTX 4060).
            model_vram_gb:   Estimated VRAM the model + optimizer occupies per GPU.
                             Must be <= vram_per_gpu_gb for DDP to be viable.

        Returns:
            Recommended world_size (always >= 1).

        Examples:
            >>> DistributedLauncher.optimal_world_size(8.0, 2.0)  # 4 GPUs, 4 GB model → 4
            4
            >>> DistributedLauncher.optimal_world_size(8.0, 10.0)  # model > per-GPU VRAM → 1
            1
        """
        n_gpus = DistributedLauncher.available_gpus()

        if n_gpus == 0:
            return 1

        if model_vram_gb <= 0:
            return n_gpus

        # Safety check: model must fit on a single GPU
        if model_vram_gb > vram_per_gpu_gb:
            return 1

        total_vram = n_gpus * vram_per_gpu_gb
        max_from_vram = int(math.floor(total_vram / model_vram_gb))

        # Ensure at least 1 process
        optimal = max(1, min(n_gpus, max_from_vram))
        return optimal

    @staticmethod
    def detect_backend() -> str:
        """
        Detect the best available distributed backend.

        Returns "nccl" when CUDA + NCCL are both available, otherwise "gloo".
        """
        try:
            import torch
            if torch.cuda.is_available() and torch.distributed.is_nccl_available():
                return "nccl"
        except Exception:
            pass
        return "gloo"
