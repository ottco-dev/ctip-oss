> CTIP runs on Windows via **WSL2** (Windows Subsystem for Linux 2).
> Native Windows installation is not supported — GPU access works through WSL2 with CUDA.
> Tested: Windows 10 21H2+, Windows 11 22H2+

## 1. Enable WSL2

```powershell
# Run as Administrator in PowerShell
wsl --install -d Ubuntu-22.04

# After reboot — set version 2
wsl --set-default-version 2
wsl --status
```

## 2. NVIDIA drivers for WSL2

**Important**: Install the Windows driver — NOT a Linux driver inside WSL2!

1. Download [NVIDIA driver](https://www.nvidia.com/Download/index.aspx) — Game Ready or Studio
2. Install, reboot Windows
3. In WSL2 terminal, verify:

```bash
nvidia-smi   # must show your GPU
```

The CUDA runtime is provided by the Windows driver. Only nvcc needs separate install if required:

```bash
sudo apt install -y nvidia-cuda-toolkit
```

## 3. Docker Desktop

1. Download [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop/)
2. During install: enable **"Use WSL 2 based engine"**
3. After install: Docker Desktop → Settings → Resources → WSL Integration → enable Ubuntu-22.04
4. Verify in WSL2:

```bash
docker run hello-world
```

## 4. Windows Terminal (recommended)

Install [Windows Terminal](https://aka.ms/terminal) from Microsoft Store.
Set Ubuntu as default profile.

## 5. Inside WSL2: follow Linux guide

Everything from here runs in the **WSL2 Ubuntu terminal**.
Follow the [Linux installation guide](installation-linux) from step 2 onward.

### Important path note

```bash
# Windows drives are mounted at:
ls /mnt/c/Users/YourName/

# RECOMMENDED: Clone CTIP into the Linux filesystem, NOT /mnt/c/
# (I/O over the 9P protocol is very slow for build tools)
cd ~
git clone https://github.com/ottco-dev/ctip-oss.git
```

### Port forwarding

WSL2 automatically binds ports to Windows localhost.
`http://localhost:3001` works directly in the Windows browser.

## 6. Performance tips

### Limit WSL2 memory

Create `C:\\Users\\YourName\\.wslconfig`:

```ini
[wsl2]
memory=8GB
processors=6
swap=4GB
localhostForwarding=true
```

### Verify GPU inside WSL2

```bash
python -c "
import torch
print('CUDA:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0))
print('VRAM:', round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1), 'GB')
"
```

### Windows Defender exclusions

Windows Defender can slow WSL2 file access significantly:
`Windows Security → Virus & threat protection → Exclusions`
→ Add `%USERPROFILE%\\AppData\\Local\\Packages\\CanonicalGroupLimited.Ubuntu*`

## Common issues

### `nvidia-smi` not found in WSL2

```powershell
# In Windows PowerShell (Admin) — check driver version (must be ≥ 470.76)
nvidia-smi --query-gpu=driver_version --format=csv,noheader
```

```bash
# In WSL2 — check device nodes
ls /dev/nvidia*
# If empty: wsl --shutdown → reboot Windows
```

### Port 3001 already taken on Windows side

```powershell
netstat -ano | findstr :3001
taskkill /PID <PID> /F
```

### Docker Desktop won't start

```powershell
# Enable Hyper-V
Enable-WindowsOptionalFeature -Online -FeatureName Microsoft-Hyper-V -All
# Also enable virtualization in BIOS (Intel VT-x / AMD-V)
```

### WSL2 has no internet access

```bash
echo "nameserver 8.8.8.8" | sudo tee /etc/resolv.conf
```

Permanently, in `/etc/wsl.conf`:
```ini
[network]
generateResolvConf = false
```
