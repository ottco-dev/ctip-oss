# Compute network — lend a GPU to CTIP

CTIP can spread work over machines that volunteers connect to a coordinator: your RTX 4060 at home, a friend's
AMD card, a Mac, a workstation CPU. The coordinator is the normal CTIP backend (for example the hosted instance on
the VPS); each machine runs the **`ctip-worker`** agent.

```
            ┌──────────── CTIP coordinator (VPS, CTIP backend) ────────────┐
            │ job queue · leases · heartbeats · checkpoints · artifacts    │
            │ web UI: Compute page          admin API: /api/v1/compute     │
            └───────▲───────────────────────▲───────────────────────▲──────┘
           HTTPS +  │ worker token          │                       │
          ┌─────────┴───────┐     ┌─────────┴───────┐     ┌─────────┴───────┐
          │ ctip-worker     │     │ ctip-worker     │     │ ctip-worker     │
          │ RTX 4060 (CUDA) │     │ RX 7900 (ROCm)  │     │ M3 (MPS) / CPU  │
          └─────────────────┘     └─────────────────┘     └─────────────────┘
```

## Two kinds of work

| | Parallel jobs | Distributed training (DDP) |
|---|---|---|
| What | one job runs completely on one machine | one training run uses several GPUs at once |
| Examples | training runs with different settings or data, cross-validation folds, evaluation, benchmarks | one large training job |
| Where | any worker, over the internet, mixed hardware | **inside one worker** with ≥ 2 GPUs of one backend (`spec.distributed: true`) |
| Why | the network is touched only for inputs, heartbeats and checkpoints | DDP synchronises gradients after every step |

Multi-machine DDP over the internet is deliberately **not** offered. A YOLO11s step exchanges ~40 MB of gradients;
at home upload speeds (10–50 Mbit/s) one step would take seconds to minutes instead of ~0.1 s, and NCCL does not
cover Apple MPS or CPUs, so mixed volunteers could not form one process group. Scaling across volunteers happens by
running many parallel jobs.

## Job kinds

Workers run only these kinds — the coordinator cannot send code. Every parameter is validated on both sides
(`shared/compute/schemas.py`).

| Kind | Does | Main parameters |
|---|---|---|
| `yolo_train` | trains YOLO11 with CTIP's training config; uploads `last.pt` every `checkpoint_every` epochs and `best.pt` + `results.csv` at the end | `dataset` (artifact id), `model` (yolo11n…x), `epochs`, `imgsz`, `batch` (0 = from free VRAM), `distributed` |
| `yolo_eval` | evaluates weights on a dataset split | `dataset`, `model_artifact`, `split` |
| `detection_benchmark` | measures YOLO11 inference on synthetic frames (p50/p95 ms, FPS, peak VRAM) | `model`, `imgsz`, `runs` |

Datasets are uploaded as a zip with `dataset.yaml` (Ultralytics format) at its root.

## Hardware

| Backend | Detected by | Notes |
|---|---|---|
| NVIDIA CUDA | `torch.cuda` | AutoBatch sizes the batch to the allowed share of free VRAM; "owner busy" detection via `nvidia-smi pmon` |
| AMD ROCm | `torch.cuda` with `torch.version.hip` | same code path as CUDA |
| Apple MPS | `torch.backends.mps` | unified memory: 60 % of free RAM counts as VRAM |
| CPU | fallback | batch 2, threads limited; good for benchmarks and small jobs |

## Security

- **Transport:** HTTPS only (Caddy, TLS). The agent refuses plain `http://` except to `localhost`; a private CA or
  self-signed certificate can be pinned with `enroll --ca`.
- **Enrolment:** the admin creates a one-time token (24 h by default) on the Compute page; the agent exchanges it for
  its own 256-bit worker token. Tokens are stored only as SHA-256 hashes on the server and in an owner-only file
  (`~/.config/ctip-worker/config.json`, mode 600) on the worker. Revoking a worker stops its token at once.
- **Separation:** agent endpoints (`/api/v1/compute/agent/*`) accept only worker tokens, never the admin API token;
  admin endpoints stay behind the CTIP API token.
- **Least privilege:** a worker can download only the inputs and checkpoint of the job it currently holds.
- **Integrity:** every download and upload carries a SHA-256 that the other side checks. Dataset archives are
  unpacked only for image/label/config files, without paths that leave the cache (zip slip) and with a size cap.
- **No remote code:** only the job kinds above exist; unknown kinds or out-of-range parameters are rejected.

## Resource limits and load adaptation (agent side)

`ctip-worker limits --set key=value`:

| Limit | Default | Effect |
|---|---|---|
| `max_vram_fraction` | 0.8 | share of GPU memory a job may fill (AutoBatch target) |
| `reserve_vram_gb` | 1.0 | always left free for the desktop |
| `min_free_ram_gb` | 1.0 | jobs are not taken below it; a running job is stopped when free RAM stays below it for 20 s |
| `max_ram_gb` | 0 | optional hard cap on what the worker offers |
| `cpu_threads` | half the cores | threads for PyTorch/OpenMP; jobs run at lower OS priority |
| `yield_when_busy` | true | when other programs use the GPU (≥ `busy_gpu_util` % for `busy_grace_s` s), the job is handed back with its checkpoint |
| `cache_gb` | 20 | cache for datasets/weights, least recently used entries are removed |
| `kinds` | all | which job kinds this machine accepts |

After an out-of-memory failure the next attempt on that machine uses half the batch and no extra loader processes.

## Reliability

- **Heartbeats** every 20 s carry progress and metrics; a lease lasts 90 s and is extended by each heartbeat.
- **Lost workers:** the coordinator requeues the job when a lease runs out; the next worker **resumes from the last
  uploaded checkpoint** (Ultralytics resume with the stored optimizer state, paths rewritten for the new machine).
  After `max_attempts` losses or failures the job is marked failed.
- **Yield** (owner needs the machine, or Ctrl+C) does not count as a failed attempt.
- **Cancel** from the dashboard reaches the worker at its next heartbeat.

---

## Step by step: your RTX 4060 PC as the first worker

The coordinator is the hosted instance `https://ctip.178-18-251-107.sslip.io` (any CTIP backend works the same).

**1. Install CTIP on the PC** (once; skip if the repo is already there):
```bash
git clone https://github.com/ottco-dev/ctip-oss.git && cd ctip-oss
python3.12 -m venv .venv && source .venv/bin/activate
pip install uv && uv pip install -e .
```

**2. Check the machine**
```bash
ctip-worker doctor
```
Expect `backend cuda (NVIDIA GeForce RTX 4060, 1 GPU, 7.6 GB VRAM)`. *free now* shows what the agent would offer.
If free RAM is below ~1 GB, close a few programs first — the agent will not take jobs when the machine is short of memory.

**3. Create a one-time token** — CTIP web UI → **Compute** → **Connect a worker** → *Create one-time token*.
The page shows the full command.

**4. Connect the PC**
```bash
ctip-worker enroll --server https://ctip.178-18-251-107.sslip.io --token ctipe_… --name "ottco RTX 4060"
```

**5. Optional: set limits** (example: leave more VRAM to the desktop, use 6 threads)
```bash
ctip-worker limits --set max_vram_fraction=0.7 --set cpu_threads=6
```

**6. Start working**
```bash
ctip-worker run
```
The PC appears on the Compute page as *idle* within seconds. Stop with Ctrl+C — a running training job is handed
back with its checkpoint.

**7. First test run** — on the Compute page click **Queue benchmark**. Within ~30 s the worker takes it; after about
10 s the job is *completed* with p50 latency, FPS and peak VRAM. (Reference: YOLO11s at 1280 px FP16 on the
RTX 4060 — about 12 ms, 82 FPS, 137 MB.)

**8. A training job** (optional): upload a zipped dataset and queue a run with the API token:
```bash
curl -H "Authorization: Bearer $API_TOKEN" -F kind=dataset -F file=@dataset.zip https://ctip…/api/v1/compute/artifacts
curl -H "Authorization: Bearer $API_TOKEN" -H 'content-type: application/json' \
  -d '{"name":"first training","spec":{"kind":"yolo_train","dataset":"<id>","model":"yolo11n","epochs":20,"imgsz":640}}' \
  https://ctip…/api/v1/compute/jobs
```
Progress, epoch and mAP appear live on the Compute page; `best.pt` is downloadable from the job when it finishes.

Nothing of this changes existing CTIP functions: the coordinator adds its own tables and routes, the agent is a
separate program, and jobs on a worker run in their own processes at low priority.
