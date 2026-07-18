#Requires -Version 5.1
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Model tiers — Choose-Model sets $LLAVA_MODEL based on detected hardware.
#   fast     minicpm-v:latest   low-VRAM / CPU-only    ~2-4s/photo
#   balanced qwen2.5vl:7b       6-8 GB VRAM / mid GPU  ~4-6s/photo  (default)
#   quality  qwen3-vl:32b       20+ GB VRAM            ~90-120s/photo
$MODEL_FAST     = "minicpm-v:latest"
$MODEL_BALANCED = "qwen2.5vl:7b"
$MODEL_QUALITY  = "qwen3-vl:32b"

$MIN_DISK_FAST     = 8
$MIN_DISK_BALANCED = 12
$MIN_DISK_QUALITY  = 32

$OLLAMA_TIMEOUT  = 60   # seconds to wait for Ollama service
$OLLAMA_TIMEOUT  = 60   # seconds to wait for Ollama service
$SCRIPT_DIR      = Split-Path -Parent $MyInvocation.MyCommand.Path
$LOG_DIR         = Join-Path $env:USERPROFILE ".needlestack"
$LOG_FILE        = Join-Path $LOG_DIR "install.log"

# --- logging ---

New-Item -ItemType Directory -Force -Path $LOG_DIR | Out-Null
Start-Transcript -Path $LOG_FILE -Append | Out-Null

# --- helpers ---

function Step($msg)  { Write-Host "`n── $msg" -ForegroundColor White }
function Ok($msg)    { Write-Host "  ✓  $msg" -ForegroundColor Green }
function Warn($msg)  { Write-Host "  !  $msg" -ForegroundColor Yellow }
function Fail($msg)  {
    Write-Host "`n  ✗  $msg" -ForegroundColor Red
    Write-Host ""
    Write-Host "  Install log: $LOG_FILE"
    Write-Host "  Please send that file to whoever gave you needlestack."
    Write-Host ""
    Stop-Transcript | Out-Null
    exit 1
}

# Catch unexpected errors
trap {
    Write-Host ""
    Write-Host "  ✗  Something went wrong: $_" -ForegroundColor Red
    Write-Host "  Install log: $LOG_FILE"
    Write-Host "  Please send that file to whoever gave you needlestack."
    Stop-Transcript | Out-Null
    exit 1
}

# --- header ---

Clear-Host
Write-Host ""
Write-Host "  needlestack installer" -ForegroundColor White
Write-Host "  Find your photos by describing what's in them."
Write-Host ""

# --- Windows version ---

Step "Checking system"

$osVersion = [System.Environment]::OSVersion.Version
if ($osVersion.Major -lt 10) {
    Fail "Windows 10 or later is required."
}

$buildNumber = (Get-ItemProperty "HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion").CurrentBuild
if ([int]$buildNumber -lt 19041) {
    Warn "Windows 10 version 2004 or later is recommended. Some features may not work."
} else {
    Ok "Windows build $buildNumber"
}

$arch = $env:PROCESSOR_ARCHITECTURE
Ok "Architecture: $arch"

# --- choose vision model based on hardware ---

Step "Choosing vision model"

# Detect GPU: prefer nvidia-smi for accurate VRAM; fall back to WMI (often capped at 4 GB).
$vramGB  = 0
$gpuName = "none detected"
$nvidiaSmi = Join-Path $env:SystemRoot "System32\nvidia-smi.exe"
if (Test-Path $nvidiaSmi) {
    try {
        $vramMB = & $nvidiaSmi --query-gpu=memory.total --format=csv,noheader,nounits 2>$null |
                  Select-Object -First 1
        $vramGB = [math]::Round([int]$vramMB / 1024)
        $gpuName = (& $nvidiaSmi --query-gpu=name --format=csv,noheader 2>$null |
                    Select-Object -First 1).Trim()
    } catch {}
}
if ($vramGB -eq 0) {
    # WMI fallback — AdapterRAM is unreliable above 4 GB but usable for low-end detection
    try {
        $gpu = Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue |
               Where-Object { $_.Name -match "NVIDIA|GeForce|RTX|GTX|Quadro|Radeon|AMD" } |
               Select-Object -First 1
        if ($gpu) {
            $vramGB = [math]::Round($gpu.AdapterRAM / 1GB)
            $gpuName = $gpu.Name
        }
    } catch {}
}

$systemRamGB = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB)

if ($vramGB -ge 20) {
    $LLAVA_MODEL   = $MODEL_QUALITY
    $MODEL_TIER    = "quality"
    $MODEL_REASON  = "${gpuName} with ${vramGB} GB VRAM comfortably fits the 32B model (~20 GB)."
    $MODEL_SPEED   = "~90-120 seconds per photo"
    $MIN_DISK_GB   = $MIN_DISK_QUALITY
} elseif ($vramGB -ge 6) {
    $LLAVA_MODEL   = $MODEL_BALANCED
    $MODEL_TIER    = "balanced"
    $MODEL_REASON  = "${gpuName} with ${vramGB} GB VRAM is a good fit for the 7B model (~5 GB)."
    $MODEL_SPEED   = "~4-6 seconds per photo"
    $MIN_DISK_GB   = $MIN_DISK_BALANCED
} else {
    $LLAVA_MODEL   = $MODEL_FAST
    $MODEL_TIER    = "fast"
    if ($vramGB -gt 0) {
        $MODEL_REASON = "${gpuName} has ${vramGB} GB VRAM — using the fast model to stay within limits."
    } else {
        $MODEL_REASON = "No GPU detected or VRAM too low — using the fast model for CPU inference."
    }
    $MODEL_SPEED   = "~15-30 seconds per photo (CPU) or ~3-5s with a capable GPU"
    $MIN_DISK_GB   = $MIN_DISK_FAST
}

Write-Host ""
Write-Host "  ┌─────────────────────────────────────────────────────────────┐" -ForegroundColor Cyan
Write-Host ("  │  Hardware: GPU={0,-20} VRAM={1,2} GB  RAM={2,3} GB      │" -f ($gpuName.Substring(0, [math]::Min(20,$gpuName.Length))), $vramGB, $systemRamGB) -ForegroundColor Cyan
Write-Host "  │                                                             │" -ForegroundColor Cyan
Write-Host ("  │  Selected model:  {0,-44}│" -f "$LLAVA_MODEL  ($MODEL_TIER)") -ForegroundColor Cyan
Write-Host ("  │  Speed estimate:  {0,-44}│" -f $MODEL_SPEED) -ForegroundColor Cyan
Write-Host "  │                                                             │" -ForegroundColor Cyan
Write-Host ("  │  Why: {0,-55}│" -f $MODEL_REASON) -ForegroundColor Cyan
Write-Host "  │                                                             │" -ForegroundColor Cyan
Write-Host "  │  Other options (switch any time after install):             │" -ForegroundColor Cyan
Write-Host ("  │    fast     {0,-49}│" -f $MODEL_FAST) -ForegroundColor Cyan
Write-Host ("  │    balanced {0,-49}│" -f $MODEL_BALANCED) -ForegroundColor Cyan
Write-Host ("  │    quality  {0,-49}│" -f $MODEL_QUALITY) -ForegroundColor Cyan
Write-Host "  │                                                             │" -ForegroundColor Cyan
Write-Host "  │  To switch:  needlestack index /photos --preset quality     │" -ForegroundColor Cyan
Write-Host "  └─────────────────────────────────────────────────────────────┘" -ForegroundColor Cyan
Write-Host ""

Ok "Model: $LLAVA_MODEL ($MODEL_TIER tier)"

# --- internet ---

Step "Checking internet connection"

try {
    $null = Invoke-WebRequest -Uri "https://ollama.com" -UseBasicParsing -TimeoutSec 5
    Ok "Internet reachable"
} catch {
    Fail "No internet connection detected. Please connect and try again."
}

# --- disk space ---

Step "Checking disk space"

try {
    $drive = (Get-Item $env:USERPROFILE).PSDrive.Name + ":"
    $disk = Get-PSDrive -Name (Get-Item $env:USERPROFILE).PSDrive.Name
    $freeGB = [math]::Floor($disk.Free / 1GB)
} catch {
    Warn "Could not check disk space — continuing anyway."
    $freeGB = 999
}

if ($freeGB -lt $MIN_DISK_GB) {
    Fail "Not enough disk space. Need at least ${MIN_DISK_GB} GB free, only ${freeGB} GB available."
}
Ok "${freeGB} GB available (need ${MIN_DISK_GB} GB)"

# --- Ollama ---

Step "Checking Ollama"

$ollamaExe = $null
$ollamaCandidates = @(
    (Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe"),
    (Join-Path $env:ProgramFiles "Ollama\ollama.exe"),
    "ollama"
)
foreach ($candidate in $ollamaCandidates) {
    if (Get-Command $candidate -ErrorAction SilentlyContinue) {
        $ollamaExe = $candidate
        break
    }
}

if (-not $ollamaExe) {
    Write-Host ""
    Write-Host "  Ollama is not installed. It's needed to understand your photos."
    Write-Host ""
    Write-Host "  To install it:"
    Write-Host "    1. Go to https://ollama.com"
    Write-Host "    2. Click Download for Windows and run the installer"
    Write-Host "    3. Run this installer again"
    Write-Host ""
    Fail "Stopping — please install Ollama first, then run this again."
}
Ok "Ollama found"

# Start Ollama if not running
$ollamaRunning = $false
try {
    $null = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -TimeoutSec 3
    $ollamaRunning = $true
} catch {}

if (-not $ollamaRunning) {
    Write-Host "  Starting Ollama..."
    Start-Process $ollamaExe -ArgumentList "serve" -WindowStyle Hidden -ErrorAction SilentlyContinue

    $waited = 0
    while ($waited -lt $OLLAMA_TIMEOUT) {
        Start-Sleep -Seconds 2
        $waited += 2
        try {
            $null = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -TimeoutSec 2
            $ollamaRunning = $true
            break
        } catch {}
        Write-Host "  Waiting for Ollama... ${waited}s" -NoNewline
        Write-Host "`r" -NoNewline
    }

    if (-not $ollamaRunning) {
        Fail "Ollama did not start after ${OLLAMA_TIMEOUT}s. Try running Ollama manually, then run this installer again."
    }
}
Ok "Ollama is running"

# Pull model if not present
$modelBase = $LLAVA_MODEL.Split(":")[0]
$modelList = & $ollamaExe list 2>&1
$modelPresent = $modelList | Where-Object { $_ -match "^$modelBase" }

if ($modelPresent) {
    Ok "Model $LLAVA_MODEL already downloaded"
} else {
    Write-Host ""
    $dlSize = switch ($MODEL_TIER) { "fast" {"~2 GB"} "balanced" {"~5 GB"} "quality" {"~20 GB"} default {"several GB"} }
    Write-Host "  Downloading $LLAVA_MODEL ($MODEL_TIER tier) — $dlSize, this will take a while."
    Write-Host "  You can leave this running and come back."
    Write-Host ""
    try {
        & $ollamaExe pull $LLAVA_MODEL
        if ($LASTEXITCODE -ne 0) { throw "ollama pull exited with code $LASTEXITCODE" }
    } catch {
        Write-Host ""
        Write-Host "  The model download failed. Common causes:"
        Write-Host "    * Internet connection dropped"
        Write-Host "    * Not enough disk space"
        Write-Host ""
        Write-Host "  You can try again by running in a terminal:  ollama pull $LLAVA_MODEL"
        Fail "Model download failed."
    }
    Ok "Model $LLAVA_MODEL downloaded"
}

# Verify model responds
Write-Host "  Verifying model works..."
try {
    $body = '{"model":"' + $LLAVA_MODEL + '","prompt":"Reply with just the word OK.","stream":false}'
    $resp = Invoke-RestMethod -Uri "http://localhost:11434/api/generate" `
        -Method Post -Body $body -ContentType "application/json" -TimeoutSec 30
    Ok "Model responded: $($resp.response.Trim().Substring(0, [Math]::Min(40, $resp.response.Trim().Length)))"
} catch {
    Warn "Could not verify model response — it may still work. Continuing."
}

# --- pixi ---

Step "Checking pixi (Python environment manager)"

$pixiExe = $null
$pixiCandidates = @(
    (Join-Path $env:USERPROFILE ".pixi\bin\pixi.exe"),
    "pixi"
)
foreach ($candidate in $pixiCandidates) {
    if (Get-Command $candidate -ErrorAction SilentlyContinue) {
        $pixiExe = $candidate
        break
    }
}

if (-not $pixiExe) {
    Write-Host "  Installing pixi..."
    try {
        Invoke-Expression (Invoke-WebRequest -Uri "https://pixi.sh/install.ps1" -UseBasicParsing).Content
    } catch {
        Fail "Failed to install pixi. Check your internet connection and try again."
    }
    $pixiExe = Join-Path $env:USERPROFILE ".pixi\bin\pixi.exe"
}

if (-not (Get-Command $pixiExe -ErrorAction SilentlyContinue)) {
    Fail "pixi installed but could not be found. Please restart this installer."
}

$pixiVersion = & $pixiExe --version 2>&1
Ok "pixi $pixiVersion"

# Add pixi to PATH for this session
$env:PATH = (Join-Path $env:USERPROFILE ".pixi\bin") + ";" + $env:PATH

# --- Python environment ---

Step "Setting up Python environment (a few minutes first time)"

Set-Location $SCRIPT_DIR

try {
    & $pixiExe install
    if ($LASTEXITCODE -ne 0) { throw "pixi install failed" }
} catch {
    Write-Host ""
    Write-Host "  Failed to set up the Python environment. Common causes:"
    Write-Host "    * Internet connection dropped"
    Write-Host "    * Not enough disk space"
    Write-Host ""
    Fail "Environment setup failed."
}
Ok "Python environment ready"

# --- smoke test ---

Step "Verifying installation"

$smokeScript = @"
import sys
failures = []
for mod in ['torch', 'open_clip', 'fastapi', 'rawpy', 'PIL', 'numpy', 'httpx', 'rich', 'click']:
    try:
        __import__(mod)
    except ImportError as e:
        failures.append(f'{mod}: {e}')
if failures:
    print('FAIL: ' + '; '.join(failures))
else:
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'OK device={device}')
"@

$smokeOutput = & $pixiExe run python -c $smokeScript 2>&1

if ($smokeOutput -match "^FAIL:") {
    $missing = $smokeOutput -replace "^FAIL: ", ""
    Write-Host ""
    Write-Host "  Some required components did not install correctly:"
    Write-Host "  $missing"
    Write-Host ""
    Fail "Installation incomplete."
}

$device = if ($smokeOutput -match "device=(\S+)") { $Matches[1] } else { "unknown" }
Ok "All components verified (using $device)"

# --- launcher ---

Step "Creating launcher"

$launcherPath = Join-Path $SCRIPT_DIR "Start Needlestack.bat"
$launcherContent = @"
@echo off
cd /d "%~dp0"
set PATH=%USERPROFILE%\.pixi\bin;%PATH%

set PORT=8484

REM If needlestack is already running on the port, just open the browser
curl -sf --max-time 2 http://localhost:%PORT%/ 2>nul | findstr /i "needlestack" >nul 2>&1
if not errorlevel 1 (
    echo needlestack already running -- opening browser.
    start "" "http://localhost:%PORT%"
    exit /b 0
)

netstat -an 2>nul | findstr ":%PORT% " | findstr "LISTENING" >nul 2>&1
if not errorlevel 1 (
    echo Port %PORT% is in use by another application -- needlestack will try a nearby port.
)

echo Starting needlestack...

REM Check Ollama
curl -sf http://localhost:11434/api/tags >nul 2>&1
if errorlevel 1 (
    echo Starting Ollama...
    start "" "$ollamaExe" serve
    timeout /t 5 /nobreak >nul
)

curl -sf http://localhost:11434/api/tags >nul 2>&1
if errorlevel 1 (
    echo.
    echo Ollama does not appear to be running.
    echo Please start Ollama and try again.
    pause
    exit /b 1
)

pixi run needlestack serve
if errorlevel 1 (
    echo.
    echo needlestack exited unexpectedly.
    echo.
    echo If this keeps happening, please send this log file to whoever gave you needlestack:
    echo %USERPROFILE%\.needlestack\needlestack.log
    echo.
    pause
)
"@

Set-Content -Path $launcherPath -Value $launcherContent -Encoding ASCII
Ok "Created 'Start Needlestack.bat'"

# --- start menu shortcut ---

Step "Adding Start Menu shortcut"

$iconPath = Join-Path $SCRIPT_DIR "resources\windows\AppIcon.ico"
$startMenuDir = Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs"
$shortcutPath = Join-Path $startMenuDir "Needlestack.lnk"

try {
    $WshShell = New-Object -ComObject WScript.Shell
    $shortcut = $WshShell.CreateShortcut($shortcutPath)
    $shortcut.TargetPath = $launcherPath
    $shortcut.WorkingDirectory = $SCRIPT_DIR
    $shortcut.Description = "Find your photos by describing what's in them"
    if (Test-Path $iconPath) {
        $shortcut.IconLocation = $iconPath
    }
    $shortcut.Save()
    Ok "Added 'Needlestack' to the Start Menu"
} catch {
    Warn "Could not create Start Menu shortcut: $_"
    Warn "You can still launch needlestack from 'Start Needlestack.bat' in $SCRIPT_DIR"
}

# --- done ---

Stop-Transcript | Out-Null

Write-Host ""
Write-Host "════════════════════════════════════════════════════"
Write-Host "  Installation complete!" -ForegroundColor White
Write-Host "════════════════════════════════════════════════════"
Write-Host ""
Write-Host "  !   Do not move this folder after installing."
Write-Host "      The Start Menu shortcut expects needlestack to stay here:"
Write-Host "      $SCRIPT_DIR"
Write-Host ""
Write-Host "  Step 1 — Index your photos (run once, takes an hour or two):"
Write-Host ""
Write-Host "    Open a terminal in this folder and run:"
Write-Host "    pixi run needlestack index C:\path\to\your\photos"
Write-Host ""
Write-Host "  Step 2 — Search:"
Write-Host ""
Write-Host "    Press the Windows key, type 'Needlestack', and press Enter"
Write-Host "    (Click 'More info' then 'Run anyway' if Windows warns you)"
Write-Host ""
Write-Host "  Install log saved to: $LOG_FILE"
Write-Host ""
