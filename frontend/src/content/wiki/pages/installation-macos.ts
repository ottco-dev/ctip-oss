import type { WikiPage } from '../types';

const en = `
> **Apple Silicon (M1/M2/M3)**: GPU acceleration via Metal/MPS (no CUDA).
> **Intel Mac**: CPU-only recommended (eGPU with CUDA not officially supported).
> Tested: macOS 13 Ventura, macOS 14 Sonoma

## 1. Xcode Command Line Tools

\`\`\`bash
xcode-select --install
\`\`\`

## 2. Homebrew

\`\`\`bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Apple Silicon — add to PATH
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zshrc
source ~/.zshrc
\`\`\`

## 3. System dependencies

\`\`\`bash
brew install git python@3.12 node nginx

# uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.zshrc
\`\`\`

## 4. Clone & install

\`\`\`bash
git clone https://github.com/ottco-dev/ctip-oss.git
cd ctip-oss

uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[all]"

# Verify MPS (Apple Silicon GPU)
python -c "
import torch
print('PyTorch:', torch.__version__)
print('MPS available:', torch.backends.mps.is_available())
"
\`\`\`

## 5. Apple Silicon GPU (MPS)

CTIP auto-detects MPS. Set explicitly in \`.env\`:

\`\`\`bash
CUDA_DEVICE="mps"
CUDA_VISIBLE_DEVICES=""
VRAM_LIMIT_GB="8.0"     # MPS shares RAM — on 16 GB Mac, ~8–10 GB usable for GPU
\`\`\`

**MPS limitations**:
- No half-precision (fp16) in all ops — some ops fall back to CPU
- Shared memory: GPU tasks compete with system RAM
- No CUDA streams → sequential inference only

## 6. Frontend

\`\`\`bash
cd frontend && npm install && cd ..
\`\`\`

## 7. Docker Desktop for Mac

1. Download [Docker Desktop for Mac](https://www.docker.com/products/docker-desktop/)
   → Apple Silicon: select Arm64 version
2. Install and start
3. Verify: \`docker run hello-world\`

## 8. nginx (Homebrew-specific)

Homebrew nginx runs as your user (not root):

\`\`\`bash
# Fix hardcoded paths in nginx config
sed -i '' "s|/home/ottcouture/trichome-analysis|$HOME/ctip-oss|g" nginx-local/nginx.conf

# Start nginx
nginx -c "$(pwd)/nginx-local/nginx.conf"
\`\`\`

## 9. Start

\`\`\`bash
cp .env.example .env   # edit with your paths
./scripts/dev-start.sh
# http://localhost:3001
\`\`\`

## Performance comparison

| Task | Apple M2 (MPS) | RTX 4060 (CUDA) |
|------|----------------|-----------------|
| YOLO11s inference (1280px tile) | ~180ms | ~45ms |
| SAM2-tiny segmentation | ~350ms | ~80ms |
| Training 1 epoch / 100 images | ~12 min | ~2 min |

macOS is good for **development and annotation**, not for extended training runs.

## Common issues

### \`Error: The brew link step did not complete successfully\`

\`\`\`bash
brew link --overwrite python@3.12
\`\`\`

### \`RuntimeError: MPS backend out of memory\`

\`\`\`bash
# Reduce in .env:
VRAM_INFERENCE_BUDGET_GB="4.0"
# Or reduce tile size:
TILE_SIZE="640"    # default 1280
\`\`\`

### nginx: permission denied on port 3001

\`\`\`bash
sudo nginx -c "$(pwd)/nginx-local/nginx.conf"
\`\`\`

### \`SSL: CERTIFICATE_VERIFY_FAILED\` during model download

\`\`\`bash
/Applications/Python\\ 3.12/Install\\ Certificates.command
\`\`\`
`;

const de = `
> **Apple Silicon (M1/M2/M3)**: GPU-Beschleunigung über Metal/MPS (kein CUDA).
> **Intel Mac**: CPU-only empfohlen.
> Getestet: macOS 13 Ventura, macOS 14 Sonoma

## 1. Xcode Command Line Tools

\`\`\`bash
xcode-select --install
\`\`\`

## 2. Homebrew

\`\`\`bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Apple Silicon — zu PATH hinzufügen
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zshrc
source ~/.zshrc
\`\`\`

## 3. Systemabhängigkeiten

\`\`\`bash
brew install git python@3.12 node nginx
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.zshrc
\`\`\`

## 4. Klonen & installieren

\`\`\`bash
git clone https://github.com/ottco-dev/ctip-oss.git
cd ctip-oss

uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[all]"

# MPS prüfen (Apple Silicon GPU)
python -c "import torch; print('MPS:', torch.backends.mps.is_available())"
\`\`\`

## 5. Apple Silicon GPU (MPS)

In \`.env\` setzen:

\`\`\`bash
CUDA_DEVICE="mps"
CUDA_VISIBLE_DEVICES=""
VRAM_LIMIT_GB="8.0"     # MPS teilt RAM — bei 16 GB Mac ~8–10 GB nutzbar
\`\`\`

**MPS-Einschränkungen**: Kein volles fp16, geteilter Speicher mit System-RAM, sequentielle Inferenz.

## 6. nginx-Konfiguration (macOS-spezifisch)

\`\`\`bash
# Hardkodierte Pfade anpassen
sed -i '' "s|/home/ottcouture/trichome-analysis|$HOME/ctip-oss|g" nginx-local/nginx.conf
nginx -c "$(pwd)/nginx-local/nginx.conf"
\`\`\`

## 7. Starten

\`\`\`bash
cp .env.example .env
./scripts/dev-start.sh
# http://localhost:3001
\`\`\`

## Leistungsvergleich

| Aufgabe | Apple M2 (MPS) | RTX 4060 (CUDA) |
|---------|----------------|-----------------|
| YOLO11s Inferenz (1280px) | ~180ms | ~45ms |
| SAM2-tiny Segmentierung | ~350ms | ~80ms |
| Training 1 Epoche / 100 Bilder | ~12 min | ~2 min |

macOS eignet sich für **Entwicklung und Annotation** — nicht für lange Trainingsläufe.

## Häufige Probleme

### MPS Out of Memory

\`\`\`bash
VRAM_INFERENCE_BUDGET_GB="4.0"
TILE_SIZE="640"
\`\`\`

### SSL-Fehler beim Modell-Download

\`\`\`bash
/Applications/Python\\ 3.12/Install\\ Certificates.command
\`\`\`
`;

const es = `
> **Apple Silicon (M1/M2/M3)**: Aceleración GPU vía Metal/MPS (sin CUDA).
> **Mac Intel**: Solo CPU recomendado.
> Probado: macOS 13 Ventura, macOS 14 Sonoma

## 1. Xcode Command Line Tools

\`\`\`bash
xcode-select --install
\`\`\`

## 2. Homebrew & dependencias

\`\`\`bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zshrc && source ~/.zshrc

brew install git python@3.12 node nginx
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.zshrc
\`\`\`

## 3. Instalar CTIP

\`\`\`bash
git clone https://github.com/ottco-dev/ctip-oss.git && cd ctip-oss
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[all]"
cd frontend && npm install && cd ..
\`\`\`

## 4. GPU Apple Silicon (MPS)

\`\`\`bash
# En .env:
CUDA_DEVICE="mps"
CUDA_VISIBLE_DEVICES=""
VRAM_LIMIT_GB="8.0"
\`\`\`

## 5. Iniciar

\`\`\`bash
sed -i '' "s|/home/ottcouture/trichome-analysis|$HOME/ctip-oss|g" nginx-local/nginx.conf
cp .env.example .env && ./scripts/dev-start.sh
# http://localhost:3001
\`\`\`
`;

const page: WikiPage = {
  slug: 'installation-macos',
  title: { en: 'macOS Installation', de: 'macOS Installation', es: 'Instalación macOS' },
  description: {
    en: 'Installation for Apple Silicon (M1/M2/M3) with MPS and Intel Macs.',
    de: 'Installation für Apple Silicon (M1/M2/M3) mit MPS und Intel Macs.',
    es: 'Instalación para Apple Silicon (M1/M2/M3) con MPS y Macs Intel.',
  },
  content: { en, de, es },
  section: 'setup',
  icon: '🍎',
};

export default page;
