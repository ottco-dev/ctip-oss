## Distributed Training

CTIP supports multi-GPU training via **PyTorch Distributed Data Parallel (DDP)** launched with `torchrun`. On the default target hardware (RTX 4060, single GPU), `world_size=1` produces identical results to standard training with no overhead. The DDP code path is ready for future multi-GPU expansion without code changes.

---

## Architecture

```
torchrun --nproc_per_node=<world_size> backend/training/distributed/train_ddp.py

    Rank 0 (main)           Rank 1 … N-1
    ┌──────────────┐        ┌──────────────┐
    │ DataLoader   │        │ DataLoader   │
    │ (shard 0)    │        │ (shard k)    │
    │      │       │        │      │       │
    │  DDP model   │◄──────►│  DDP model   │
    │  (GPU 0)     │  NCCL  │  (GPU k)     │
    └──────┬───────┘        └──────────────┘
           │
    Checkpoint / MLflow logging (rank 0 only)
```

**Gradient synchronisation** happens automatically across all ranks via `torch.nn.parallel.DistributedDataParallel`. Only rank 0 writes checkpoints and logs metrics to MLflow.

---

## DistributedConfig

| Parameter | Type | Default | Description |
|---|---|---|---|
| `backend` | `"nccl"` \| `"gloo"` | `"nccl"` | Process-group backend; use `gloo` for CPU-only |
| `world_size` | int | 1 | Number of GPUs; `1` = single-GPU, no communication overhead |
| `mixed_precision` | bool | `true` | Enable `torch.cuda.amp` automatic mixed precision (FP16) |
| `gradient_accumulation_steps` | int | 1 | Accumulate gradients before optimizer step (effective batch multiplier) |
| `sync_batchnorm` | bool | `false` | Convert `BatchNorm` → `SyncBatchNorm` across ranks |
| `gradient_checkpointing` | bool | `false` | Trade compute for VRAM by recomputing activations during backward pass |

---

## API reference

### Get distributed training status

```bash
GET /api/v1/training/distributed/status
```

```json
{
  "available_gpus": 1,
  "active_jobs": 0,
  "nccl_available": true
}
```

### Start a distributed training job

```bash
POST /api/v1/training/distributed/start
Content-Type: application/json

{
  "model": "yolo11s",
  "dataset": "data/datasets/v2/",
  "epochs": 100,
  "batch_size": 16,
  "world_size": 1,
  "backend": "nccl",
  "mixed_precision": true,
  "gradient_accumulation_steps": 2,
  "sync_batchnorm": false,
  "gradient_checkpointing": false
}
```

Response:
```json
{
  "job_id": "ddp_k7m3",
  "status": "queued",
  "world_size": 1
}
```

### List distributed jobs

```bash
GET /api/v1/training/distributed/jobs/{job_id}
```

```json
{
  "job_id": "ddp_k7m3",
  "status": "running",
  "epoch": 42,
  "epochs_total": 100,
  "loss": 0.032,
  "map50": 0.81
}
```

### Stop a job

```bash
POST /api/v1/training/distributed/stop/{job_id}
```

---

## Single-GPU fallback (RTX 4060)

When `world_size=1` the DDP path collapses to standard single-process training:

- No inter-process communication
- No NCCL overhead
- Identical checkpoint format
- Identical MLflow run structure

This is the default on the RTX 4060 target system. The config flag is retained so that a future multi-GPU workstation can begin distributed training without any code modification.

---

## Frontend: Training page → Distributed tab

- **GPU inventory** — detected GPUs with VRAM per device.
- **World size selector** — dropdown limited to `≤ available_gpus`.
- **Config panel** — all `DistributedConfig` fields with inline tooltips.
- **Live metrics** — loss and mAP streamed over WebSocket `/ws/training`.
- **Job history** — past distributed runs with epoch count and final mAP.

---

## VRAM budget (single GPU, RTX 4060)

| Configuration | Approx. VRAM |
|---|---|
| `batch_size=16`, FP16, no gradient checkpointing | ~5.8 GB |
| `batch_size=16`, FP16, gradient checkpointing | ~4.2 GB |
| `batch_size=32`, FP16, gradient checkpointing | ~6.1 GB |

Enable `gradient_checkpointing=true` to free ~1.5 GB at the cost of ~20% longer backward pass.
