import type { WikiPage } from '../types';

const en = `
> **Apple Silicon (M1/M2/M3/M4)**: GPU acceleration via Metal Performance Shaders (MPS). No CUDA required.
> **Intel Mac**: CPU-only. eGPU with CUDA is not officially supported by PyTorch on macOS.
> Tested: macOS 13 Ventura, macOS 14 Sonoma, macOS 15 Sequoia

| Mac type | GPU backend | Suitable for |
|----------|-------------|--------------|
| Apple Silicon (M1/M2/M3/M4) | MPS | Development, annotation, inference |
| Intel Mac (any) | CPU only | Development, annotation only |

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

## 5. GPU backend selection

### Apple Silicon — MPS

CTIP auto-detects MPS. Set explicitly in \`.env\`:

\`\`\`bash
CUDA_DEVICE="mps"
CUDA_VISIBLE_DEVICES=""
VRAM_LIMIT_GB="8.0"          # MPS shares system RAM — on 16 GB Mac ~8–10 GB usable
VRAM_INFERENCE_BUDGET_GB="4.0"
\`\`\`

Verify:

\`\`\`bash
python -c "
import torch
print('PyTorch :', torch.__version__)
print('MPS     :', torch.backends.mps.is_available())
print('Built   :', torch.backends.mps.is_built())
"
\`\`\`

**MPS limitations**:
- Not all PyTorch ops support fp16 on MPS — affected ops fall back to CPU automatically (logged, not fatal)
- Memory is shared with system RAM — GPU tasks compete with the OS and other applications
- No CUDA streams, no concurrent GPU kernels → strictly sequential inference
- TensorRT not available; ONNX Runtime CoreML execution provider is an alternative

**Recommended tile size for MPS** (reduces peak memory usage):

\`\`\`bash
TILE_SIZE="960"    # default 1280 — reduce if you see out-of-memory errors
\`\`\`

---

### Intel Mac — CPU only

Intel Macs run CTIP in CPU mode. eGPU acceleration via CUDA is not supported by PyTorch on macOS.

\`\`\`bash
CUDA_DEVICE="cpu"
CUDA_VISIBLE_DEVICES=""
VRAM_LIMIT_GB="0"
VRAM_INFERENCE_BUDGET_GB="0"
\`\`\`

Install the lightweight profile (no VLM, no SAM2):

\`\`\`bash
uv pip install -e ".[dev]"
\`\`\`

CPU mode is suitable for:
- Frontend development and UI work
- Dataset management and Label Studio annotation
- Running the API and inspecting results

Extended training and VLM auto-labeling are impractical on CPU.

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
sed -i '' "s|/path/to/trichome-analysis|$HOME/ctip-oss|g" nginx-local/nginx.conf

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
> **Apple Silicon (M1/M2/M3/M4)**: GPU-Beschleunigung über Metal Performance Shaders (MPS). Kein CUDA erforderlich.
> **Intel Mac**: Nur CPU. eGPU mit CUDA wird von PyTorch auf macOS nicht unterstützt.
> Getestet: macOS 13 Ventura, macOS 14 Sonoma, macOS 15 Sequoia

| Mac-Typ | GPU-Backend | Geeignet für |
|---------|-------------|--------------|
| Apple Silicon (M1/M2/M3/M4) | MPS | Entwicklung, Annotation, Inferenz |
| Intel Mac (beliebig) | Nur CPU | Entwicklung, Annotation |

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

## 5. GPU-Backend-Auswahl

### Apple Silicon — MPS

In \`.env\` setzen:

\`\`\`bash
CUDA_DEVICE="mps"
CUDA_VISIBLE_DEVICES=""
VRAM_LIMIT_GB="8.0"          # MPS teilt System-RAM — bei 16 GB Mac ~8–10 GB nutzbar
VRAM_INFERENCE_BUDGET_GB="4.0"
\`\`\`

Prüfen:

\`\`\`bash
python -c "
import torch
print('MPS verfügbar:', torch.backends.mps.is_available())
print('MPS gebaut   :', torch.backends.mps.is_built())
"
\`\`\`

**MPS-Einschränkungen**:
- Nicht alle PyTorch-Ops unterstützen fp16 auf MPS — betroffene Ops fallen automatisch auf CPU zurück (wird geloggt, kein Fehler)
- Speicher wird mit System-RAM geteilt — GPU-Tasks konkurrieren mit dem Betriebssystem
- Keine CUDA Streams, keine parallelen GPU-Kernel → streng sequentielle Inferenz
- TensorRT nicht verfügbar; ONNX Runtime mit CoreML als Alternative

Empfohlene Tile-Größe für MPS (reduziert Peak-Speicherverbrauch):

\`\`\`bash
TILE_SIZE="960"    # Standard 1280 — reduzieren bei Out-of-Memory-Fehlern
\`\`\`

---

### Intel Mac — Nur CPU

Intel-Macs laufen im CPU-Modus. eGPU via CUDA wird von PyTorch auf macOS nicht unterstützt.

\`\`\`bash
CUDA_DEVICE="cpu"
CUDA_VISIBLE_DEVICES=""
VRAM_LIMIT_GB="0"
VRAM_INFERENCE_BUDGET_GB="0"
\`\`\`

Leichtgewichtiges Profil installieren (ohne VLM/SAM2):

\`\`\`bash
uv pip install -e ".[dev]"
\`\`\`

CPU-Modus eignet sich für: Frontend-Entwicklung, Datensatzverwaltung, Label Studio Annotation. Nicht empfohlen für Training oder VLM-Auto-Labeling.

## 6. nginx-Konfiguration (macOS-spezifisch)

\`\`\`bash
# Hardkodierte Pfade anpassen
sed -i '' "s|/path/to/trichome-analysis|$HOME/ctip-oss|g" nginx-local/nginx.conf
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
> **Apple Silicon (M1/M2/M3/M4)**: Aceleración GPU vía Metal Performance Shaders (MPS). Sin CUDA necesario.
> **Mac Intel**: Solo CPU. eGPU con CUDA no está soportado por PyTorch en macOS.
> Probado: macOS 13 Ventura, macOS 14 Sonoma, macOS 15 Sequoia

| Tipo de Mac | Backend GPU | Adecuado para |
|-------------|-------------|---------------|
| Apple Silicon (M1/M2/M3/M4) | MPS | Desarrollo, anotación, inferencia |
| Mac Intel (cualquiera) | Solo CPU | Desarrollo, anotación |

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

## 4. Selección del backend GPU

### Apple Silicon — MPS

\`\`\`bash
# En .env:
CUDA_DEVICE="mps"
CUDA_VISIBLE_DEVICES=""
VRAM_LIMIT_GB="8.0"          # MPS comparte RAM del sistema — en Mac 16 GB ~8–10 GB usables
VRAM_INFERENCE_BUDGET_GB="4.0"
TILE_SIZE="960"               # reducir si hay errores de memoria
\`\`\`

Verificar:

\`\`\`bash
python -c "import torch; print('MPS:', torch.backends.mps.is_available())"
\`\`\`

**Limitaciones MPS**: No todas las ops soportan fp16 (caen a CPU automáticamente). Memoria compartida con RAM del sistema. Inferencia secuencial únicamente.

---

### Mac Intel — Solo CPU

\`\`\`bash
# En .env:
CUDA_DEVICE="cpu"
CUDA_VISIBLE_DEVICES=""
VRAM_LIMIT_GB="0"
\`\`\`

Instalar perfil ligero (sin VLM ni SAM2):

\`\`\`bash
uv pip install -e ".[dev]"
\`\`\`

Adecuado para desarrollo de UI, gestión de datos y anotación. No recomendado para entrenamiento.

---

## 5. Iniciar

\`\`\`bash
sed -i '' "s|/path/to/trichome-analysis|$HOME/ctip-oss|g" nginx-local/nginx.conf
cp .env.example .env && ./scripts/dev-start.sh
# http://localhost:3001
\`\`\`
`;

const page: WikiPage = {
  slug: 'installation-macos',
  title: { en: 'macOS Installation', de: 'macOS Installation', es: 'Instalación macOS' },
  description: {
    en: 'Installation for Apple Silicon (M1–M4) with MPS acceleration and Intel Macs (CPU-only).',
    de: 'Installation für Apple Silicon (M1–M4) mit MPS-Beschleunigung und Intel Macs (nur CPU).',
    es: 'Instalación para Apple Silicon (M1–M4) con aceleración MPS y Macs Intel (solo CPU).',
  },
  content: { en, de, es },
  section: 'setup',
  icon: '🍎',
};

export default page;
