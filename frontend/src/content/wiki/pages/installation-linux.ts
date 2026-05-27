import type { WikiPage } from '../types';

const en = `
> Tested on: Ubuntu 22.04 LTS, Ubuntu 24.04 LTS, Debian 12, Fedora 39, Arch Linux

## 1. NVIDIA drivers

\`\`\`bash
# Ubuntu / Debian
sudo apt update
sudo apt install -y nvidia-driver-535 nvidia-cuda-toolkit

# Fedora
sudo dnf install -y akmod-nvidia xorg-x11-drv-nvidia-cuda

# Arch
sudo pacman -S nvidia nvidia-utils cuda

# Verify (after reboot)
nvidia-smi
\`\`\`

Expected output shows GPU name, VRAM, and CUDA version.

## 2. System dependencies

\`\`\`bash
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
\`\`\`

### Node.js LTS ≥ 20 (via nvm)

\`\`\`bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
source ~/.bashrc
nvm install 20 && nvm use 20
node --version   # v20.x.x
\`\`\`

### uv (fast Python package manager)

\`\`\`bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
\`\`\`

## 3. Clone repository

\`\`\`bash
git clone https://github.com/ottco-dev/ctip-oss.git
cd ctip-oss
\`\`\`

## 4. Python environment

\`\`\`bash
uv venv --python 3.12
source .venv/bin/activate

# All extras (VLM, SAM2, training, dev tools)
uv pip install -e ".[all]"

# Lightweight (no VLM / SAM2 — for weak hardware)
uv pip install -e ".[dev]"

# Verify GPU access
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# Expected: 2.x.x True
\`\`\`

### Manual PyTorch with CUDA 12.1

\`\`\`bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
\`\`\`

## 5. Frontend

\`\`\`bash
cd frontend && npm install && cd ..
\`\`\`

## 6. Docker (for Label Studio & CVAT)

\`\`\`bash
# Ubuntu / Debian
sudo apt install -y ca-certificates gnupg
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list
sudo apt update && sudo apt install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Add user to docker group (requires logout/login)
sudo usermod -aG docker $USER
newgrp docker

docker run hello-world   # verify
\`\`\`

## 7. Configure .env

\`\`\`bash
cp .env.example .env
\`\`\`

Minimum configuration:

\`\`\`bash
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
\`\`\`

Alternatively use the [Setup Wizard](setup-wizard) — it writes \`.env\` automatically.

## 8. Start all services

\`\`\`bash
chmod +x scripts/dev-start.sh
./scripts/dev-start.sh

# Check status
./scripts/dev-start.sh status
\`\`\`

| Service | URL |
|---------|-----|
| Main UI | http://localhost:3001 |
| API Docs | http://localhost:8000/docs |
| MLflow | http://localhost:3004 |

\`\`\`bash
# Optional: start annotation stack
cd docker && docker compose --profile annotation up -d
\`\`\`

## Common issues

### \`nvidia-smi\` works but CUDA not found in PyTorch

\`\`\`bash
nvidia-smi | grep "CUDA Version"
python -c "import torch; print(torch.version.cuda)"
# Must be compatible: PyTorch CUDA ≤ driver CUDA
\`\`\`

### Port already in use

\`\`\`bash
ss -tlnp | grep -E ':3001|:8000|:3000'
kill $(lsof -t -i:3001)
\`\`\`

### Permission denied on docker socket

\`\`\`bash
sudo usermod -aG docker $USER
# Fully log out and back in, or:
newgrp docker
\`\`\`

### \`libGL.so.1: cannot open shared object\`

\`\`\`bash
sudo apt install -y libgl1-mesa-glx libglib2.0-0
\`\`\`
`;

const de = `
> Getestet auf: Ubuntu 22.04 LTS, Ubuntu 24.04 LTS, Debian 12, Fedora 39, Arch Linux

## 1. NVIDIA-Treiber

\`\`\`bash
# Ubuntu / Debian
sudo apt update
sudo apt install -y nvidia-driver-535 nvidia-cuda-toolkit

# Fedora
sudo dnf install -y akmod-nvidia xorg-x11-drv-nvidia-cuda

# Arch
sudo pacman -S nvidia nvidia-utils cuda

# Prüfen (nach Neustart)
nvidia-smi
\`\`\`

## 2. Systemabhängigkeiten

\`\`\`bash
# Ubuntu / Debian
sudo apt install -y \\
    git curl wget build-essential \\
    python3.12 python3.12-venv python3.12-dev \\
    nodejs npm nginx \\
    libgl1-mesa-glx libglib2.0-0
\`\`\`

### Node.js LTS ≥ 20 (via nvm)

\`\`\`bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
source ~/.bashrc
nvm install 20 && nvm use 20
\`\`\`

### uv (schneller Python Package Manager)

\`\`\`bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
\`\`\`

## 3. Repository klonen

\`\`\`bash
git clone https://github.com/ottco-dev/ctip-oss.git
cd ctip-oss
\`\`\`

## 4. Python-Umgebung

\`\`\`bash
uv venv --python 3.12
source .venv/bin/activate

# Alle Extras (VLM, SAM2, Training, Dev-Tools)
uv pip install -e ".[all]"

# Leichtgewichtig (ohne VLM/SAM2 — für schwächere Hardware)
uv pip install -e ".[dev]"

# GPU-Zugriff prüfen
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# Erwartet: 2.x.x True
\`\`\`

## 5. Frontend

\`\`\`bash
cd frontend && npm install && cd ..
\`\`\`

## 6. Docker (für Label Studio & CVAT)

\`\`\`bash
# Ubuntu / Debian — Docker installieren
sudo apt install -y ca-certificates gnupg
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list
sudo apt update && sudo apt install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Benutzer zur docker-Gruppe hinzufügen (danach abmelden + wieder anmelden!)
sudo usermod -aG docker $USER
newgrp docker
\`\`\`

## 7. .env konfigurieren

\`\`\`bash
cp .env.example .env
\`\`\`

Minimalkonfiguration:

\`\`\`bash
DATA_ROOT="/home/deinuser/ctip-oss/data"
MODELS_DIR="/home/deinuser/ctip-oss/data/models"
OUTPUTS_DIR="/home/deinuser/ctip-oss/data/outputs"
CUDA_DEVICE="cuda:0"
VRAM_LIMIT_GB="8.0"
LABEL_STUDIO_URL="http://localhost:3005"
MLFLOW_TRACKING_URI="http://localhost:3004"
ENVIRONMENT="development"
\`\`\`

Alternativ: [Setup-Wizard](setup-wizard) verwenden — schreibt \`.env\` automatisch.

## 8. Alle Services starten

\`\`\`bash
chmod +x scripts/dev-start.sh
./scripts/dev-start.sh
./scripts/dev-start.sh status  # Status prüfen
\`\`\`

\`\`\`bash
# Optional: Annotation-Stack starten
cd docker && docker compose --profile annotation up -d
\`\`\`

## Häufige Probleme

### \`nvidia-smi\` funktioniert, CUDA in PyTorch nicht gefunden

\`\`\`bash
# CUDA-Versionen müssen kompatibel sein
nvidia-smi | grep "CUDA Version"
python -c "import torch; print(torch.version.cuda)"
\`\`\`

### Port bereits belegt

\`\`\`bash
ss -tlnp | grep -E ':3001|:8000|:3000'
kill $(lsof -t -i:3001)
\`\`\`

### Permission denied bei Docker

\`\`\`bash
sudo usermod -aG docker $USER
newgrp docker   # oder komplett abmelden und wieder anmelden
\`\`\`
`;

const es = `
> Probado en: Ubuntu 22.04 LTS, Ubuntu 24.04 LTS, Debian 12, Fedora 39, Arch Linux

## 1. Controladores NVIDIA

\`\`\`bash
# Ubuntu / Debian
sudo apt update && sudo apt install -y nvidia-driver-535 nvidia-cuda-toolkit

# Arch
sudo pacman -S nvidia nvidia-utils cuda

# Verificar (después de reiniciar)
nvidia-smi
\`\`\`

## 2. Dependencias del sistema

\`\`\`bash
# Ubuntu / Debian
sudo apt install -y git curl wget build-essential python3.12 python3.12-venv nodejs npm nginx libgl1-mesa-glx

# Node.js LTS ≥ 20
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
source ~/.bashrc && nvm install 20

# uv
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.bashrc
\`\`\`

## 3. Clonar el repositorio

\`\`\`bash
git clone https://github.com/ottco-dev/ctip-oss.git
cd ctip-oss
\`\`\`

## 4. Entorno Python

\`\`\`bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[all]"

# Verificar GPU
python -c "import torch; print(torch.cuda.is_available())"
\`\`\`

## 5. Frontend & Docker

\`\`\`bash
cd frontend && npm install && cd ..

# Docker
sudo usermod -aG docker $USER && newgrp docker
\`\`\`

## 6. Iniciar servicios

\`\`\`bash
cp .env.example .env   # editar con tus rutas
chmod +x scripts/dev-start.sh && ./scripts/dev-start.sh
# http://localhost:3001
\`\`\`
`;

const page: WikiPage = {
  slug: 'installation-linux',
  title: { en: 'Linux Installation', de: 'Linux Installation', es: 'Instalación Linux' },
  description: {
    en: 'Full installation guide for Ubuntu, Debian, Fedora, and Arch Linux.',
    de: 'Vollständige Installationsanleitung für Ubuntu, Debian, Fedora und Arch Linux.',
    es: 'Guía completa de instalación para Ubuntu, Debian, Fedora y Arch Linux.',
  },
  content: { en, de, es },
  section: 'setup',
  icon: '🐧',
};

export default page;
