#!/usr/bin/env bash
# CTIP worker installer for Linux and macOS - lend this machine's GPU/CPU to a CTIP instance.
#
#   curl -fsSL https://<ctip-server>/install-worker.sh | bash -s -- --token ctipe_...
#
# Options:  --server URL   coordinator (default: the server this script was downloaded from)
#           --token TOKEN  one-time enrolment token from the Compute page ("Connect a worker")
#           --name NAME    how this machine appears in the dashboard (default: hostname)
#           --backend B    auto (default) | cuda (NVIDIA) | rocm (AMD, Linux) | mps (Apple Silicon) | cpu
#           --no-service   do not install the background service (start with: ctip-worker run)
#           --uninstall    stop the service and remove the worker (keeps nothing)
# Installs into ~/.ctip-worker (no root needed). Source: https://github.com/ottco-dev/ctip-oss
set -euo pipefail

SERVER="${CTIP_SERVER:-@SERVER@}"
TOKEN="" NAME="$(hostname -s 2>/dev/null || hostname)" SERVICE=1 UNINSTALL=0 BACKEND="${CTIP_BACKEND:-auto}"
REPO="${CTIP_REPO:-https://github.com/ottco-dev/ctip-oss.git}" REF="${CTIP_REF:-main}"
HOME_DIR="${CTIP_WORKER_HOME:-$HOME/.ctip-worker}"
while [ $# -gt 0 ]; do
  case "$1" in
    --server) SERVER="$2"; shift 2 ;;
    --token) TOKEN="$2"; shift 2 ;;
    --name) NAME="$2"; shift 2 ;;
    --backend) BACKEND="$2"; shift 2 ;;
    --no-service) SERVICE=0; shift ;;
    --uninstall) UNINSTALL=1; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done
say() { printf '\033[1;32m==>\033[0m %s\n' "$*"; }
die() { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }
OS="$(uname -s)"

if [ "$UNINSTALL" = 1 ]; then
  if [ "$OS" = Linux ]; then systemctl --user disable --now ctip-worker.service 2>/dev/null || true; rm -f "$HOME/.config/systemd/user/ctip-worker.service"; fi
  if [ "$OS" = Darwin ]; then launchctl unload "$HOME/Library/LaunchAgents/org.ctip.worker.plist" 2>/dev/null || true; rm -f "$HOME/Library/LaunchAgents/org.ctip.worker.plist"; fi
  rm -rf "$HOME_DIR" "$HOME/.local/bin/ctip-worker"
  say "removed. Ask the CTIP admin to revoke this machine on the Compute page."
  exit 0
fi

case "$SERVER" in https://*|http://localhost*|http://127.0.0.1*) ;; *) die "--server must be an https:// URL" ;; esac
command -v git >/dev/null || die "git is missing (Debian/Ubuntu: sudo apt install git; macOS: xcode-select --install)"

# 1. uv - a self-contained Python installer (no system Python changes, no root)
if ! command -v uv >/dev/null && [ ! -x "$HOME/.local/bin/uv" ]; then
  say "installing uv (Python package manager)"
  curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh >/dev/null
fi
UV="$(command -v uv || echo "$HOME/.local/bin/uv")"

# 2. hardware -> matching PyTorch build (auto-detected unless --backend is given)
if [ "$BACKEND" = auto ]; then
  if [ "$OS" = Darwin ]; then BACKEND=mps
  elif command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1; then BACKEND=cuda
  elif command -v rocminfo >/dev/null && rocminfo 2>/dev/null | grep -q gfx; then BACKEND=rocm
  else BACKEND=cpu; fi
fi
case "$OS/$BACKEND" in
  Linux/cuda) TORCH_INDEX="https://download.pytorch.org/whl/cu128"
              KIND="NVIDIA CUDA ($(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo 'driver not found'))" ;;
  Linux/rocm) TORCH_INDEX="https://download.pytorch.org/whl/rocm6.3" KIND="AMD ROCm" ;;
  */cpu)      TORCH_INDEX="https://download.pytorch.org/whl/cpu" KIND="CPU only"
              [ "$OS" = Darwin ] && TORCH_INDEX="" ;;
  Darwin/mps) TORCH_INDEX="" KIND="Apple Silicon (MPS)"
              [ "$(uname -m)" = arm64 ] || die "MPS needs an Apple Silicon Mac (M1 or newer); use --backend cpu" ;;
  Darwin/cuda|Darwin/rocm) die "macOS has no CUDA/ROCm - use --backend mps (Apple Silicon) or cpu" ;;
  Linux/mps)  die "MPS is Apple only - use --backend cuda, rocm or cpu" ;;
  *) die "unknown --backend '$BACKEND' (auto, cuda, rocm, mps, cpu)" ;;
esac
say "hardware: $KIND"

# 3. CTIP into its own environment
say "installing CTIP worker into $HOME_DIR (a few minutes, ~2-5 GB with PyTorch)"
mkdir -p "$HOME_DIR"
if [ -d "$HOME_DIR/src/.git" ]; then git -C "$HOME_DIR/src" fetch -q origin "$REF" && git -C "$HOME_DIR/src" reset -q --hard "origin/$REF"
else git clone -q --depth 1 --branch "$REF" "$REPO" "$HOME_DIR/src"; fi
"$UV" venv -q --python 3.12 "$HOME_DIR/venv"
PY="$HOME_DIR/venv/bin/python"
if [ -n "$TORCH_INDEX" ]; then
  "$UV" pip install -q --python "$PY" torch torchvision --index-url "$TORCH_INDEX" --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match
fi
(cd "$HOME_DIR/src" && "$UV" export -q --frozen --no-hashes --no-emit-project 2>/dev/null | grep -vE '^(torch|torchvision|triton|nvidia-)' > "$HOME_DIR/constraints.txt") || : > "$HOME_DIR/constraints.txt"
"$UV" pip install -q --python "$PY" -e "$HOME_DIR/src" -c "$HOME_DIR/constraints.txt"
mkdir -p "$HOME/.local/bin"
ln -sf "$HOME_DIR/venv/bin/ctip-worker" "$HOME/.local/bin/ctip-worker"
CW="$HOME_DIR/venv/bin/ctip-worker"
"$CW" doctor | sed -n 2,4p

# 4. connect to the coordinator
if [ -n "$TOKEN" ]; then
  say "connecting to $SERVER as '$NAME'"
  "$CW" enroll --server "$SERVER" --token "$TOKEN" --name "$NAME"
elif ! grep -q '"token": "ctipw_' "${XDG_CONFIG_HOME:-$HOME/.config}/ctip-worker/config.json" 2>/dev/null; then
  say "no --token given: connect later with  ctip-worker enroll --server $SERVER --token ctipe_..."
  SERVICE=0
fi

# 5. background service (starts with your login, lowest priority, hands jobs back on stop)
if [ "$SERVICE" = 1 ]; then
  if [ "$OS" = Linux ] && command -v systemctl >/dev/null && systemctl --user show-environment >/dev/null 2>&1; then
    mkdir -p "$HOME/.config/systemd/user"
    cat > "$HOME/.config/systemd/user/ctip-worker.service" <<UNIT
[Unit]
Description=CTIP compute worker ($SERVER)
After=network-online.target

[Service]
ExecStart=$CW run
KillSignal=SIGINT
TimeoutStopSec=120
Restart=on-failure
RestartSec=30
Nice=10
CPUSchedulingPolicy=batch
IOSchedulingClass=idle

[Install]
WantedBy=default.target
UNIT
    systemctl --user daemon-reload && systemctl --user enable --now ctip-worker.service
    say "running as a service:  systemctl --user status ctip-worker  ·  logs: journalctl --user -u ctip-worker -f"
  elif [ "$OS" = Darwin ]; then
    PL="$HOME/Library/LaunchAgents/org.ctip.worker.plist"; mkdir -p "$(dirname "$PL")"
    cat > "$PL" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>org.ctip.worker</string>
  <key>ProgramArguments</key><array><string>$CW</string><string>run</string></array>
  <key>RunAtLoad</key><true/><key>KeepAlive</key><dict><key>SuccessfulExit</key><false/></dict>
  <key>ProcessType</key><string>Background</string><key>LowPriorityIO</key><true/><key>Nice</key><integer>10</integer>
  <key>StandardOutPath</key><string>$HOME_DIR/worker.log</string><key>StandardErrorPath</key><string>$HOME_DIR/worker.log</string>
</dict></plist>
PLIST
    launchctl unload "$PL" 2>/dev/null || true; launchctl load "$PL"
    say "running in the background (launchd) · logs: tail -f $HOME_DIR/worker.log"
  else
    say "no service manager found - start it with:  ctip-worker run"
  fi
fi
say "done. Limits (VRAM share, threads, ...):  ctip-worker limits   ·   remove:  bash install-worker.sh --uninstall"
