#Requires -Version 5.1
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$LLAVA_MODEL     = "llava:13b"
$MIN_DISK_GB     = 14
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
    Write-Host "  Downloading $LLAVA_MODEL — about 8 GB, this will take a while."
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

# --- done ---

Stop-Transcript | Out-Null

Write-Host ""
Write-Host "════════════════════════════════════════════════════"
Write-Host "  Installation complete!" -ForegroundColor White
Write-Host "════════════════════════════════════════════════════"
Write-Host ""
Write-Host "  Step 1 — Index your photos (run once, takes an hour or two):"
Write-Host ""
Write-Host "    Open a terminal in this folder and run:"
Write-Host "    pixi run needlestack index C:\path\to\your\photos"
Write-Host ""
Write-Host "  Step 2 — Search:"
Write-Host ""
Write-Host "    Double-click 'Start Needlestack.bat' in File Explorer"
Write-Host "    (Click 'More info' then 'Run anyway' if Windows warns you)"
Write-Host ""
Write-Host "  Install log saved to: $LOG_FILE"
Write-Host ""
