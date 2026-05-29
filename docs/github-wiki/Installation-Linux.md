> Tested on: Ubuntu 22.04 LTS, Ubuntu 24.04 LTS, Debian 12, Fedora 39, Arch Linux

## 1. GPU drivers

CTIP supports three GPU backends. Pick the one that matches your hardware:

| Backend | Hardware | Performance |
|---------|----------|-------------|
| **CUDA** | NVIDIA GTX 1080+ / RTX series | Full — recommended |
| **ROCm** | AMD RX 6000 / RX 7000 series | ~70% of CUDA equivalent |
| **CPU** | Any x86-64 (no GPU required) | ~20× slower; dev/annotation only |

---

### 1a. NVIDIA CUDA

```bash
# Ubuntu / Debian
sudo apt update
sudo apt install -y nvidia-driver-535 nvidia-cuda-toolkit

# Fedora
sudo dnf install -y akmod-nvidia xorg-x11-drv-nvidia-cuda

# Arch
sudo pacman -S nvidia nvidia-utils cuda

# Verify (after reboot)
nvidia-smi
```

Expected output shows GPU name, VRAM, and CUDA version.

---

### 1b. AMD ROCm (RX 6000 / RX 7000)

> ROCm 6.x supports RDNA2 (RX 6000) and RDNA3 (RX 7000). Older GCN/RDNA1 cards are not officially supported.

```bash
# Ubuntu 22.04 / 24.04 — install ROCm 6.1
wget https://repo.radeon.com/amdgpu-install/6.1.3/ubuntu/jammy/amdgpu-install_6.1.60103-1_all.deb
sudo apt install -y ./amdgpu-install_6.1.60103-1_all.deb
sudo amdgpu-install --usecase=rocm
sudo usermod -aG render,video $USER
# Reboot required after this step
```

After reboot, install the ROCm build of PyTorch:

```bash
# In the CTIP virtualenv (replace rocm6.1 with your installed ROCm version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.1
```

Verify:

```bash
python -c "
import torch
print('PyTorch :', torch.__version__)
print('ROCm/HIP:', torch.cuda.is_available())   # True if ROCm is working
print('GPU     :', torch.cuda.get_device_name(0))
print('VRAM    :', round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1), 'GB')
"
```

Configure `.env`:

```bash
CUDA_DEVICE="cuda:0"          # PyTorch uses cuda: namespace even for ROCm
CUDA_VISIBLE_DEVICES="0"
VRAM_LIMIT_GB="8.0"           # set to your card's actual VRAM
VRAM_INFERENCE_BUDGET_GB="2.0"
```

**ROCm limitations**:
- TensorRT not available — ONNX Runtime with MIGraphX is the alternative
- fp16 is fully supported on RDNA3 (RX 7000); RDNA2 has partial support
- Some PyTorch ops fall back to CPU on RDNA2 — this is expected and logged
- Kernel and ROCm versions must match exactly; mixing versions causes crashes

---

### 1c. CPU-only (no GPU)

Suitable for development, UI work, dataset management, and running annotation tools.  
Not recommended for extended training or VLM auto-labeling (impractical without GPU acceleration).

No driver installation needed. Skip to step 2.

Configure `.env`:

```bash
CUDA_DEVICE="cpu"
CUDA_VISIBLE_DEVICES=""
VRAM_LIMIT_GB="0"
VRAM_INFERENCE_BUDGET_GB="0"
```

Install the lightweight extra set (omits VLM and SAM2):

```bash
uv pip install -e ".[dev]"
```

**Expected performance (i5-13400F)**:

| Task | CPU | RTX 4060 (CUDA) |
|------|-----|-----------------|
| YOLO11s inference (1280px tile) | ~900ms | ~45ms |
| SAM2-tiny segmentation | ~8s | ~80ms |
| Training 1 epoch / 100 images | ~45 min | ~2 min |

## 2. System dependencies

```bash
# Ubuntu / Debian
sudo apt install -y \\
    git curl wget build-essential \\
    python3.12 python3.12-venv python3.12-dev \\
    nodejs npm nginx \\
    libgl1-mesa-glx libglib2.0-0

# Fedora
sudo dnf install -y git curl wget gcc gcc-c++ python3.12 python3.12-devel nodejs npm nginx mesa-libGL

# Arch
sudo pacman -S git curl wget base-devel python nodejs npm nginx
```

### Node.js LTS ≥ 20 (via nvm)

```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
source ~/.bashrc
nvm install 20 && nvm use 20
node --version   # v20.x.x
```

### uv (fast Python package manager)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
```

## 3. Clone repository

```bash
git clone https://github.com/ottco-dev/ctip-oss.git
cd ctip-oss
```

## 4. Python environment

```bash
uv venv --python 3.12
source .venv/bin/activate

# All extras (VLM, SAM2, training, dev tools)
uv pip install -e ".[all]"

# Lightweight (no VLM / SAM2 — for weak hardware)
uv pip install -e ".[dev]"

# Verify GPU access
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# Expected: 2.x.x True
```

### Manual PyTorch with CUDA 12.1

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## 5. Frontend

```bash
cd frontend && npm install && cd ..
```

## 6. Docker (for Label Studio & CVAT)

```bash
# Ubuntu / Debian
sudo apt install -y ca-certificates gnupg
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list
sudo apt update && sudo apt install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Add user to docker group (requires logout/login)
sudo usermod -aG docker $USER
newgrp docker

docker run hello-world   # verify
```

## 7. Configure .env

```bash
cp .env.example .env
```

Minimum configuration:

```bash
DATA_ROOT="/home/youruser/ctip-oss/data"
MODELS_DIR="/home/youruser/ctip-oss/data/models"
OUTPUTS_DIR="/home/youruser/ctip-oss/data/outputs"
CUDA_DEVICE="cuda:0"
CUDA_VISIBLE_DEVICES="0"
VRAM_LIMIT_GB="8.0"
VRAM_INFERENCE_BUDGET_GB="2.0"
LABEL_STUDIO_URL="http://localhost:3005"
MLFLOW_TRACKING_URI="http://localhost:3004"
ENVIRONMENT="development"
```

Alternatively use the [Setup Wizard](setup-wizard) — it writes `.env` automatically.

## 8. Start all services

```bash
chmod +x scripts/dev-start.sh
./scripts/dev-start.sh

# Check status
./scripts/dev-start.sh status
```

| Service | URL |
|---------|-----|
| Main UI | http://localhost:3001 |
| API Docs | http://localhost:8000/docs |
| MLflow | http://localhost:3004 |

```bash
# Optional: start annotation stack
cd docker && docker compose --profile annotation up -d
```

## Common issues

### `nvidia-smi` works but CUDA not found in PyTorch

```bash
nvidia-smi | grep "CUDA Version"
python -c "import torch; print(torch.version.cuda)"
# Must be compatible: PyTorch CUDA ≤ driver CUDA
```

### Port already in use

```bash
ss -tlnp | grep -E ':3001|:8000|:3000'
kill $(lsof -t -i:3001)
```

### Permission denied on docker socket

```bash
sudo usermod -aG docker $USER
# Fully log out and back in, or:
newgrp docker
```

### `libGL.so.1: cannot open shared object`

```bash
sudo apt install -y libgl1-mesa-glx libglib2.0-0
```
