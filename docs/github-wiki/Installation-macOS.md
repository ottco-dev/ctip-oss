> **Apple Silicon (M1/M2/M3)**: GPU acceleration via Metal/MPS (no CUDA).
> **Intel Mac**: CPU-only recommended (eGPU with CUDA not officially supported).
> Tested: macOS 13 Ventura, macOS 14 Sonoma

## 1. Xcode Command Line Tools

```bash
xcode-select --install
```

## 2. Homebrew

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Apple Silicon — add to PATH
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zshrc
source ~/.zshrc
```

## 3. System dependencies

```bash
brew install git python@3.12 node nginx

# uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.zshrc
```

## 4. Clone & install

```bash
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
```

## 5. Apple Silicon GPU (MPS)

CTIP auto-detects MPS. Set explicitly in `.env`:

```bash
CUDA_DEVICE="mps"
CUDA_VISIBLE_DEVICES=""
VRAM_LIMIT_GB="8.0"     # MPS shares RAM — on 16 GB Mac, ~8–10 GB usable for GPU
```

**MPS limitations**:
- No half-precision (fp16) in all ops — some ops fall back to CPU
- Shared memory: GPU tasks compete with system RAM
- No CUDA streams → sequential inference only

## 6. Frontend

```bash
cd frontend && npm install && cd ..
```

## 7. Docker Desktop for Mac

1. Download [Docker Desktop for Mac](https://www.docker.com/products/docker-desktop/)
   → Apple Silicon: select Arm64 version
2. Install and start
3. Verify: `docker run hello-world`

## 8. nginx (Homebrew-specific)

Homebrew nginx runs as your user (not root):

```bash
# Fix hardcoded paths in nginx config
sed -i '' "s|/path/to/trichome-analysis|$HOME/ctip-oss|g" nginx-local/nginx.conf

# Start nginx
nginx -c "$(pwd)/nginx-local/nginx.conf"
```

## 9. Start

```bash
cp .env.example .env   # edit with your paths
./scripts/dev-start.sh
# http://localhost:3001
```

## Performance comparison

| Task | Apple M2 (MPS) | RTX 4060 (CUDA) |
|------|----------------|-----------------|
| YOLO11s inference (1280px tile) | ~180ms | ~45ms |
| SAM2-tiny segmentation | ~350ms | ~80ms |
| Training 1 epoch / 100 images | ~12 min | ~2 min |

macOS is good for **development and annotation**, not for extended training runs.

## Common issues

### `Error: The brew link step did not complete successfully`

```bash
brew link --overwrite python@3.12
```

### `RuntimeError: MPS backend out of memory`

```bash
# Reduce in .env:
VRAM_INFERENCE_BUDGET_GB="4.0"
# Or reduce tile size:
TILE_SIZE="640"    # default 1280
```

### nginx: permission denied on port 3001

```bash
sudo nginx -c "$(pwd)/nginx-local/nginx.conf"
```

### `SSL: CERTIFICATE_VERIFY_FAILED` during model download

```bash
/Applications/Python\\ 3.12/Install\\ Certificates.command
```
