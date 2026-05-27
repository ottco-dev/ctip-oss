"""
shared.utils.seed — Reproducibility utilities.

Reproducibility is a first-class requirement for scientific computing.
All experiments MUST be reproducible given the same seed.

Known non-determinism sources on RTX 4060 / CUDA:
1. cuDNN algorithms — some have non-deterministic implementations
2. Multi-threaded data loading — random batch order
3. Atomic GPU operations — non-deterministic accumulation

Performance impact of deterministic mode: ~5-15% slower.
Acceptable for scientific reproducibility.
"""
from __future__ import annotations
import os, random
import numpy as np

GLOBAL_SEED = 42

def set_global_seed(seed: int = GLOBAL_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

def get_worker_init_fn(seed: int = GLOBAL_SEED):
    def worker_init_fn(worker_id: int) -> None:
        worker_seed = seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
    return worker_init_fn
