# CTIP worker installer for Windows - lend this PC's GPU/CPU to a CTIP instance.
#
#   $env:CTIP_TOKEN="ctipe_..."; irm https://<ctip-server>/install-worker.ps1 | iex
#
# Variables: CTIP_TOKEN (one-time token from the Compute page), CTIP_NAME (default: computer name),
#            CTIP_BACKEND = auto (default) | cuda (NVIDIA) | cpu   - PyTorch has no ROCm builds for Windows,
#            CTIP_SERVER (default: the server this script came from), CTIP_NO_SERVICE=1, CTIP_UNINSTALL=1
# Installs into %USERPROFILE%\.ctip-worker without admin rights. Source: https://github.com/ottco-dev/ctip-oss
$ErrorActionPreference = "Stop"
$Server = if ($env:CTIP_SERVER) { $env:CTIP_SERVER } else { "@SERVER@" }
$Token = $env:CTIP_TOKEN
$Name = if ($env:CTIP_NAME) { $env:CTIP_NAME } else { $env:COMPUTERNAME }
$Home_ = Join-Path $env:USERPROFILE ".ctip-worker"
$Task = "CTIP Worker"
$Backend = if ($env:CTIP_BACKEND) { $env:CTIP_BACKEND.ToLower() } else { "auto" }
function Say($m) { Write-Host "==> $m" -ForegroundColor Green }

if ($env:CTIP_UNINSTALL -eq "1") {
  Unregister-ScheduledTask -TaskName $Task -Confirm:$false -ErrorAction SilentlyContinue
  Get-Process ctip-worker -ErrorAction SilentlyContinue | Stop-Process -Force
  Remove-Item -Recurse -Force $Home_ -ErrorAction SilentlyContinue
  Say "removed. Ask the CTIP admin to revoke this PC on the Compute page."
  return
}
if (-not $Server.StartsWith("https://")) { throw "CTIP_SERVER must be an https:// URL" }
if (-not (Get-Command git -ErrorAction SilentlyContinue)) { throw "git is missing - install it with: winget install Git.Git" }

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
  Say "installing uv (Python package manager)"
  $env:UV_NO_MODIFY_PATH = "1"
  Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
}
$Uv = (Get-Command uv -ErrorAction SilentlyContinue).Source
if (-not $Uv) { $Uv = Join-Path $env:USERPROFILE ".local\bin\uv.exe" }

# NVIDIA needs the CUDA build of PyTorch on Windows (PyPI only has the CPU build there)
if ($Backend -eq "auto") { $Backend = if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) { "cuda" } else { "cpu" } }
switch ($Backend) {
  "cuda" { $Index = "https://download.pytorch.org/whl/cu128"
           $Kind = "NVIDIA CUDA " + (& nvidia-smi --query-gpu=name --format=csv,noheader 2>$null | Select-Object -First 1) }
  "cpu"  { $Index = "https://download.pytorch.org/whl/cpu"; $Kind = "CPU only" }
  "rocm" { throw "PyTorch has no ROCm builds for Windows - use an AMD GPU under Linux, or CTIP_BACKEND=cpu" }
  "mps"  { throw "MPS is Apple only - use CTIP_BACKEND=cuda or cpu" }
  default { throw "unknown CTIP_BACKEND '$Backend' (auto, cuda, cpu)" }
}
Say "hardware: $Kind"

Say "installing CTIP worker into $Home_ (a few minutes, ~2-5 GB with PyTorch)"
New-Item -ItemType Directory -Force $Home_ | Out-Null
$Src = Join-Path $Home_ "src"
if (Test-Path (Join-Path $Src ".git")) { git -C $Src fetch -q origin main; git -C $Src reset -q --hard origin/main }
else { git clone -q --depth 1 https://github.com/ottco-dev/ctip-oss.git $Src }
& $Uv venv -q --allow-existing --python 3.12 (Join-Path $Home_ "venv")
$Py = Join-Path $Home_ "venv\Scripts\python.exe"
& $Uv pip install -q --python $Py torch torchvision --index-url $Index --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match
Push-Location $Src
$C = Join-Path $Home_ "constraints.txt"
(& $Uv export -q --frozen --no-hashes --no-emit-project) | Where-Object { $_ -notmatch '^(torch|torchvision|triton|nvidia-)' } | Set-Content $C
Pop-Location
& $Uv pip install -q --python $Py -e $Src -c $C
$Cw = Join-Path $Home_ "venv\Scripts\ctip-worker.exe"
& $Cw doctor | Select-Object -Skip 1 -First 3

if ($Token) {
  Say "connecting to $Server as '$Name'"
  & $Cw enroll --server $Server --token $Token --name $Name
  if ($LASTEXITCODE -ne 0) { throw "enrolment failed" }
} else {
  Say "no CTIP_TOKEN: connect later with  $Cw enroll --server $Server --token ctipe_..."
  $env:CTIP_NO_SERVICE = "1"
}

if ($env:CTIP_NO_SERVICE -ne "1") {
  $Action = New-ScheduledTaskAction -Execute $Cw -Argument "run" -WorkingDirectory $Home_
  $Trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
  $Settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit 0 `
    -RestartCount 5 -RestartInterval (New-TimeSpan -Minutes 1) -Priority 7
  Register-ScheduledTask -TaskName $Task -Action $Action -Trigger $Trigger -Settings $Settings -Force | Out-Null
  Start-ScheduledTask -TaskName $Task
  Say "running in the background (task '$Task', starts at login)"
}
Say "done. Limits: $Cw limits   ·   remove: `$env:CTIP_UNINSTALL='1'; irm $Server/install-worker.ps1 | iex"
