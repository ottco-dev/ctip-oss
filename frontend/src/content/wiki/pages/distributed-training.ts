import type { WikiPage } from '../types';

const en = `
## Distributed Training

CTIP supports multi-GPU training via **PyTorch Distributed Data Parallel (DDP)** launched with \`torchrun\`. On the default target hardware (RTX 4060, single GPU), \`world_size=1\` produces identical results to standard training with no overhead. The DDP code path is ready for future multi-GPU expansion without code changes.

---

## Architecture

\`\`\`
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
\`\`\`

---

## DistributedConfig

| Parameter | Default | Description |
|---|---|---|
| \`backend\` | \`"nccl"\` | Process-group backend; use \`gloo\` for CPU-only |
| \`world_size\` | 1 | Number of GPUs; \`1\` = single-GPU, no communication overhead |
| \`mixed_precision\` | \`true\` | Enable \`torch.cuda.amp\` automatic mixed precision (FP16) |
| \`gradient_accumulation_steps\` | 1 | Accumulate gradients before optimizer step |
| \`sync_batchnorm\` | \`false\` | Convert \`BatchNorm\` → \`SyncBatchNorm\` across ranks |
| \`gradient_checkpointing\` | \`false\` | Recompute activations during backward to save VRAM |

---

## API reference

### Get distributed training status

\`\`\`bash
GET /api/v1/training/distributed/status
\`\`\`

\`\`\`json
{
  "available_gpus": 1,
  "active_jobs": 0,
  "nccl_available": true
}
\`\`\`

### Start a distributed training job

\`\`\`bash
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
\`\`\`

### Get job status

\`\`\`bash
GET /api/v1/training/distributed/jobs/{job_id}
\`\`\`

\`\`\`json
{
  "job_id": "ddp_k7m3",
  "status": "running",
  "epoch": 42,
  "epochs_total": 100,
  "loss": 0.032,
  "map50": 0.81
}
\`\`\`

### Stop a job

\`\`\`bash
POST /api/v1/training/distributed/stop/{job_id}
\`\`\`

---

## Single-GPU fallback (RTX 4060)

When \`world_size=1\` the DDP path collapses to standard single-process training:

- No inter-process communication overhead
- No NCCL overhead
- Identical checkpoint format
- Identical MLflow run structure

This is the default on the RTX 4060 target system. A future multi-GPU workstation can begin distributed training without any code changes.

---

## Frontend: Training page → Distributed tab

- **GPU inventory** — detected GPUs with VRAM per device.
- **World size selector** — dropdown limited to ≤ available GPUs.
- **Config panel** — all \`DistributedConfig\` fields with inline tooltips.
- **Live metrics** — loss and mAP streamed over WebSocket \`/ws/training\`.
- **Job history** — past distributed runs with epoch count and final mAP.

---

## VRAM budget (single GPU, RTX 4060)

| Configuration | Approx. VRAM |
|---|---|
| \`batch_size=16\`, FP16, no gradient checkpointing | ~5.8 GB |
| \`batch_size=16\`, FP16, gradient checkpointing | ~4.2 GB |
| \`batch_size=32\`, FP16, gradient checkpointing | ~6.1 GB |

Enable \`gradient_checkpointing=true\` to free ~1.5 GB at the cost of ~20% longer backward pass.
`;

const de = `
## Verteiltes Training

CTIP unterstützt Multi-GPU-Training über **PyTorch Distributed Data Parallel (DDP)** mit \`torchrun\`. Mit \`world_size=1\` (Standard auf dem RTX 4060) verhält sich der DDP-Pfad identisch zum normalen Training — ohne Mehraufwand.

---

## DistributedConfig

| Parameter | Standard | Beschreibung |
|---|---|---|
| \`backend\` | \`"nccl"\` | Prozessgruppen-Backend; \`gloo\` für CPU-only |
| \`world_size\` | 1 | Anzahl GPUs |
| \`mixed_precision\` | \`true\` | AMP FP16 aktivieren |
| \`gradient_accumulation_steps\` | 1 | Gradienten akkumulieren |
| \`sync_batchnorm\` | \`false\` | BatchNorm über Ranks synchronisieren |
| \`gradient_checkpointing\` | \`false\` | Aktivierungen neu berechnen (spart VRAM) |

---

## API-Referenz

\`\`\`bash
GET  /api/v1/training/distributed/status
POST /api/v1/training/distributed/start
GET  /api/v1/training/distributed/jobs/{job_id}
POST /api/v1/training/distributed/stop/{job_id}
\`\`\`

---

## VRAM-Budget (Einzel-GPU, RTX 4060)

| Konfiguration | Ca. VRAM |
|---|---|
| \`batch_size=16\`, FP16, kein Gradient-Checkpointing | ~5,8 GB |
| \`batch_size=16\`, FP16, Gradient-Checkpointing | ~4,2 GB |
| \`batch_size=32\`, FP16, Gradient-Checkpointing | ~6,1 GB |

---

## Frontend: Training-Seite → Distributed-Tab

- **GPU-Inventar** — erkannte GPUs mit VRAM.
- **World-Size-Auswahl** — auf verfügbare GPUs begrenzt.
- **Konfigurationsfeld** — alle \`DistributedConfig\`-Parameter.
- **Live-Metriken** — Loss und mAP über WebSocket \`/ws/training\`.
`;

const es = `
## Entrenamiento Distribuido

CTIP soporta entrenamiento multi-GPU mediante **PyTorch DDP** con \`torchrun\`. Con \`world_size=1\` (predeterminado en RTX 4060), el camino DDP es idéntico al entrenamiento estándar sin sobrecarga.

---

## DistributedConfig

| Parámetro | Defecto | Descripción |
|---|---|---|
| \`backend\` | \`"nccl"\` | Backend del grupo de procesos |
| \`world_size\` | 1 | Número de GPUs |
| \`mixed_precision\` | \`true\` | Precisión mixta AMP FP16 |
| \`gradient_accumulation_steps\` | 1 | Pasos de acumulación de gradientes |
| \`sync_batchnorm\` | \`false\` | Sincronizar BatchNorm entre ranks |
| \`gradient_checkpointing\` | \`false\` | Recomputar activaciones para ahorrar VRAM |

---

## Referencia de API

\`\`\`bash
GET  /api/v1/training/distributed/status
POST /api/v1/training/distributed/start
GET  /api/v1/training/distributed/jobs/{job_id}
POST /api/v1/training/distributed/stop/{job_id}
\`\`\`

---

## Presupuesto de VRAM (GPU única, RTX 4060)

| Configuración | VRAM aprox. |
|---|---|
| \`batch_size=16\`, FP16, sin gradient checkpointing | ~5.8 GB |
| \`batch_size=16\`, FP16, gradient checkpointing | ~4.2 GB |
`;

const page: WikiPage = {
  slug: 'distributed-training',
  title: {
    en: 'Distributed Training',
    de: 'Verteiltes Training',
    es: 'Entrenamiento Distribuido',
  },
  description: {
    en: 'PyTorch DDP via torchrun for multi-GPU training, with single-GPU fallback for RTX 4060.',
    de: 'PyTorch DDP via torchrun für Multi-GPU-Training, mit Single-GPU-Fallback für den RTX 4060.',
    es: 'PyTorch DDP via torchrun para entrenamiento multi-GPU, con fallback a GPU única para RTX 4060.',
  },
  content: { en, de, es },
  section: 'reference',
  icon: '⚡',
};

export default page;
