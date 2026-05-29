# NVIDIA Container Toolkit Setup

Required for GPU access inside Docker containers (inference, training, TensorRT).

---

## What it does

The NVIDIA Container Toolkit injects the host's GPU drivers into Docker containers so that CUDA, cuDNN, and TensorRT work without installing drivers inside the image. The container image only needs the CUDA runtime libraries.

**Without toolkit:** `torch.cuda.is_available()` returns `False` inside containers.  
**With toolkit:** Full GPU access, same VRAM as bare-metal.

---

## Installation

### Ubuntu 22.04 / 24.04 (bare-metal or WSL2)

```bash
# Add NVIDIA package repository
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker runtime
sudo nvidia-ctk runtime configure --runtime=docker

# Restart Docker
sudo systemctl restart docker
```

### WSL2 (Windows)

WSL2 uses the Windows NVIDIA driver via the CUDA-over-wsl2 path. The Container Toolkit still needs to be installed inside WSL2 (same commands as Ubuntu above). The Windows NVIDIA driver must be ≥ 527.41.

```bash
# Verify GPU is visible in WSL2 before installing toolkit
nvidia-smi
```

---

## Verification

```bash
# Check toolkit is correctly configured
sudo docker run --rm --runtime=nvidia --gpus all nvidia/cuda:12.8.1-base-ubuntu22.04 nvidia-smi

# Expected output: table showing your GPU (e.g. RTX 4060, 8192 MiB)
```

---

## Docker Compose configuration

CTIP uses both the modern (`deploy.resources`) and legacy (`runtime: nvidia`) approaches for maximum compatibility:

```yaml
services:
  backend:
    runtime: nvidia                          # legacy — Docker Engine standalone
    environment:
      - NVIDIA_VISIBLE_DEVICES=all           # expose all GPUs
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility,video
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu, compute, utility]
```

| Approach | Works with | Notes |
|---|---|---|
| `runtime: nvidia` | Docker Engine (standalone), WSL2 | Requires `nvidia-ctk runtime configure` |
| `deploy.resources` | Docker Compose v3.8+, Docker Desktop | Preferred for Swarm / rootless mode |

**Both are set in all CTIP compose files.** Docker uses whichever the host supports.

---

## TensorRT inside containers

CTIP ships with TensorRT support built in (`inference/tensorrt_engine/`). When running inside a Docker container with the toolkit:

1. Use a CUDA base image that includes TensorRT:
   ```dockerfile
   FROM nvcr.io/nvidia/tensorrt:24.10-py3
   ```
   Or install via pip inside any CUDA image:
   ```dockerfile
   RUN pip install tensorrt==10.6.0 --extra-index-url https://pypi.nvidia.com \
       && pip install pycuda
   ```

2. Build engines on the deployment machine (engines are GPU-architecture-specific):
   ```bash
   # Via CTIP API
   curl -X POST http://localhost:8000/api/v1/tensorrt/build \
     -H "Content-Type: application/json" \
     -d '{"onnx_path": "/models/yolo11s.onnx", "fp16": true}'

   # Via CLI
   python -m apps.cli.main benchmark --backend tensorrt
   ```

3. Engines are stored in `models/engines/` and served via `GET /tensorrt/engines`.

---

## ulimits for TensorRT

TensorRT engine builds and CUDA stream allocations require relaxed memory limits. All CTIP inference containers set:

```yaml
ulimits:
  memlock:
    soft: -1
    hard: -1
  stack: 67108864    # 64 MB stack — required for some TRT ops
```

---

## Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `docker: Error response from daemon: unknown runtime` | Toolkit not configured | Run `sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker` |
| `CUDA not available inside container` | `NVIDIA_VISIBLE_DEVICES` not set | Add `environment: - NVIDIA_VISIBLE_DEVICES=all` |
| `Failed to initialize NVML: Driver/library version mismatch` | Host driver too old | Update NVIDIA driver on host (≥ 525.x for CUDA 12.x) |
| WSL2: `nvidia-smi` fails inside container | WSL2 CUDA path not mounted | Install `nvidia-container-toolkit` inside WSL2 (not on Windows) |
| TensorRT engine build OOM | Not enough VRAM for workspace | Reduce `workspace_gb` in build config (default 4.0 GB → try 2.0 GB) |

---

## RTX 4060 reference

| Setting | Value |
|---|---|
| VRAM | 8 GB GDDR6 |
| CUDA Compute Capability | 8.9 (Ada Lovelace) |
| TensorRT FP16 speedup | ~4× vs ONNX Runtime CPU |
| Max TRT workspace | 4 GB (safe), 6 GB (aggressive) |
| Recommended CUDA image | `nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04` |
