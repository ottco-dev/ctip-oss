"""
shared.compute — the contract between the CTIP coordinator (VPS) and compute agents (volunteer PCs).

Two kinds of work, deliberately kept apart:
- parallel jobs: one job runs entirely on one worker (training runs, sweeps, folds, evaluation, benchmarks).
  Works over the internet and across CUDA / ROCm / Apple MPS / CPU workers.
- distributed training (DDP): only inside one worker with several GPUs of one backend (`gpus > 1`). Synchronising
  gradients every step across home internet links is orders of magnitude too slow, and NCCL does not cover MPS or
  CPU, so multi-machine DDP is not offered.
"""
