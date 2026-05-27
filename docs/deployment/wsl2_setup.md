# WSL2 Setup Guide — Trichome Analysis Platform

**Target**: Windows 11 host, RTX 4060 GPU, running the full CTIP stack inside WSL2.  
**Distribution**: Ubuntu 22.04 LTS  
**Status**: Development / lab workstation  

---

## 1. WSL2 Prerequisites (Windows side)

### 1.1 Enable WSL2 and GPU passthrough

```powershell
# PowerShell (Administrator)
wsl --install
wsl --set-default-version 2
wsl --install -d Ubuntu-22.04

# Verify GPU passthrough is active (requires NVIDIA driver ≥ 530 on Windows)
wsl nvidia-smi
```

Expected output should show your RTX 4060 with CUDA 12.x.

### 1.2 Required Windows components

| Component | Minimum version |
|-----------|----------------|
| Windows 11 | 22H2 or later |
| NVIDIA driver (Windows) | 530.xx or later |
| WSL2 Linux kernel | 5.10.102.1 or later (`wsl --update`) |
| Docker Desktop | 4.28 or later (with WSL2 backend) |

> **Note**: Do NOT install CUDA toolkit on the Windows side. WSL2 shares the
> Windows NVIDIA driver. Only install CUDA libraries inside WSL2.

---

## 2. WSL2 System Setup (Ubuntu side)

### 2.1 Base packages

```bash
sudo apt-get update && sudo apt-get upgrade -y
sudo apt-get install -y \
    build-essential cmake git curl wget \
    python3.12 python3.12-dev python3.12-venv python3-pip \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 \
    libtiff5-dev libjpeg-dev libpng-dev \
    ffmpeg libavcodec-dev libavformat-dev \
    libhdf5-dev
```

### 2.2 CUDA toolkit (inside WSL2 — do NOT install drivers)

```bash
# CUDA 12.1 keyring
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update

# Install CUDA runtime only (no driver — WSL2 kernel provides it)
sudo apt-get install -y cuda-toolkit-12-1

# Add to PATH
echo 'export PATH=/usr/local/cuda-12.1/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify
nvcc --version
nvidia-smi    # should show GPU via WSL2 passthrough
```

### 2.3 uv (Python package manager)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
uv --version
```

---

## 3. Repository Setup

```bash
# Clone
git clone https://github.com/<your-org>/trichome-analysis.git
cd trichome-analysis

# Create virtualenv (Python 3.12)
uv venv --python python3.12
source .venv/bin/activate

# Install with all extras
uv pip install -e ".[dev,vlm,sam]"

# Copy and configure .env
cp .env.example .env
# Edit .env — set:
#   TRICHOME_ROOT=/home/<wsl-user>/trichome-analysis
#   DATA_ROOT=/mnt/wsl/data/trichome          # or WSL internal path
#   MODELS_ROOT=/mnt/wsl/models/trichome
#   CUDA_VISIBLE_DEVICES=0
```

### 3.1 WSL2-specific path note

WSL2 mounts the Windows filesystem at `/mnt/c/`, `/mnt/d/`, etc.
For best I/O performance with large datasets (TIFF stacks), keep data **inside
the WSL2 filesystem** (`/home/<user>/...`) rather than on a Windows mount.

```bash
# Example: store large datasets on a second Windows drive via WSL2 mount
sudo mkdir -p /mnt/data
sudo mount -t drvfs D: /mnt/data    # mount D:\ into WSL2

# Or use WSL2's own ext4 VHD for maximum performance:
mkdir -p ~/datasets/trichome
# Point DATA_ROOT in .env to ~/datasets/trichome
```

---

## 4. Docker with GPU in WSL2

Docker Desktop for Windows automatically integrates WSL2 backends.
No separate Docker install inside WSL2 is needed when using Docker Desktop.

### 4.1 Docker Desktop configuration

1. Open Docker Desktop → **Settings** → **General**  
   ✓ "Use the WSL 2 based engine"

2. **Settings** → **Resources** → **WSL Integration**  
   ✓ Enable integration with your Ubuntu-22.04 distro

3. **Settings** → **Docker Engine** → add GPU support:
   ```json
   {
     "runtimes": {
       "nvidia": {
         "path": "nvidia-container-runtime"
       }
     },
     "default-runtime": "runc"
   }
   ```

### 4.2 NVIDIA Container Toolkit (inside WSL2)

```bash
# Add NVIDIA container repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
```

Restart Docker Desktop after this step.

### 4.3 Verify GPU in Docker

```bash
# Should print GPU info
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

---

## 5. Running the CTIP Stack

### 5.1 Build base image (one time)

```bash
cd /home/<user>/trichome-analysis
docker build -f docker/base/Dockerfile.base -t trichome-base:latest .
```

### 5.2 Core stack (API + frontend + MLflow)

```bash
cd docker
docker compose up -d

# Verify all services healthy
docker compose ps
```

Access:
- Frontend: http://localhost:3003
- API docs: http://localhost:3002/docs
- MLflow: http://localhost:3004

### 5.3 Inference-only stack (lighter footprint)

```bash
cd docker

# Ensure external volumes exist (created by main stack on first run):
docker volume create trichome-models
docker volume create trichome-mlflow

docker compose -f docker-compose.inference.yml up -d
```

### 5.4 Training stack (GPU — one run at a time)

```bash
cd docker
docker compose -f docker-compose.yml -f docker-compose.training.yml up -d trainer mlflow
```

---

## 6. WSL2 Performance Tuning

### 6.1 `.wslconfig` (Windows user home: `C:\Users\<you>\.wslconfig`)

```ini
[wsl2]
# Allow WSL2 to use up to 12 GB RAM (leave 4 GB for Windows)
memory=12GB

# Logical processors (i5-13400F: 16 threads; give WSL2 most of them)
processors=12

# Swap (optional — helpful when loading large VLM models)
swap=8GB

# Disable page reporting (keeps RAM allocated, avoids stutter during inference)
pageReporting=false

# Localhost forwarding — makes services on WSL2 ports accessible as localhost
localhostForwarding=true
```

Apply: `wsl --shutdown` then restart WSL2.

### 6.2 NVIDIA `expandable_segments` (prevents OOM fragmentation)

Already set in `.env.example`:
```
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Verify it's in your `.env` — critical for RTX 4060 (8 GB VRAM).

### 6.3 Disable Windows Defender scanning of WSL2 paths

WSL2 I/O goes through `\\wsl$` — Windows Defender can slow filesystem access.

1. Open **Windows Security** → **Virus & Threat Protection** → **Exclusions**
2. Add folder: `\\wsl$\Ubuntu-22.04\home\<user>\trichome-analysis`
3. Add folder: `\\wsl$\Ubuntu-22.04\home\<user>\datasets`

---

## 7. Known WSL2 Issues

| Issue | Cause | Fix |
|-------|-------|-----|
| `nvidia-smi` works but `torch.cuda.is_available()` = False | CUDA toolkit version mismatch | Ensure CUDA 12.1 matches PyTorch build index |
| Docker GPU not available | Toolkit not configured | Re-run `nvidia-ctk runtime configure --runtime=docker` |
| Slow dataset reads (>100ms/image) | Data on `/mnt/c/` Windows mount | Move data to WSL2 ext4 (`~/datasets/`) |
| Port not accessible from Windows browser | WSL2 network bridge issue | Use `localhost` (not the WSL2 IP) with `localhostForwarding=true` |
| `wsl: command not found` in Docker | Docker not using WSL2 backend | Docker Desktop → Settings → Use WSL2 backend |
| Memory exhausted during VLM loading | 4-bit quant still needs ~5 GB | Ensure `memory=12GB` in `.wslconfig`; close Windows apps |

---

## 8. Quick Smoke Test

```bash
# Activate virtualenv
source .venv/bin/activate

# 1. Check GPU
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0))"

# 2. Run unit tests (no GPU required for most)
pytest tests/ -m "not gpu" -q

# 3. Start backend
uvicorn backend.main:app --reload --port 8000

# 4. Test health endpoint
curl http://localhost:8000/health
# → {"status": "ok", "version": "..."}

# 5. Test maturity analysis
curl -X POST http://localhost:8000/api/v1/maturity/health
# → {"status": "ok", "module": "maturity"}
```

---

## 9. Updating WSL2 Kernel

```bash
# From Windows PowerShell (Administrator)
wsl --update
wsl --shutdown
```

Then reopen your WSL2 terminal. The new kernel takes effect immediately.

---

*Last updated: 2026-05-25 — RTX 4060 / i5-13400F / Windows 11 / WSL2 Ubuntu 22.04*
