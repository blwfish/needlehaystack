#!/bin/bash
set -euo pipefail

# Model tiers — do not hardcode here; choose_model() sets LLAVA_MODEL below.
#   fast     minicpm-v:latest   low-RAM / CPU-only machines  ~2-4s/photo
#   balanced qwen2.5vl:7b       16 GB+ unified / mid GPU    ~4-6s/photo  (default)
#   quality  qwen3-vl:32b       32 GB+ unified / high-VRAM  ~90-120s/photo
MODEL_FAST="minicpm-v:latest"
MODEL_BALANCED="qwen2.5vl:7b"
MODEL_QUALITY="qwen3-vl:32b"

# Disk requirements per tier (model + pixi env + index buffer)
MIN_DISK_FAST=8
MIN_DISK_BALANCED=12
MIN_DISK_QUALITY=32
OLLAMA_TIMEOUT=30     # seconds to wait for Ollama to start
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$HOME/.needlestack"
LOG_FILE="$LOG_DIR/install.log"

# --- helpers ---

bold()   { printf "\033[1m%s\033[0m\n" "$*"; }
green()  { printf "  \033[32m✓\033[0m  %s\n" "$*"; }
red()    { printf "\n  \033[31m✗\033[0m  %s\n" "$*"; }
warn()   { printf "  \033[33m!\033[0m  %s\n" "$*"; }
step()   { printf "\n\033[1m── %s\033[0m\n" "$*"; }
die()    { red "$*"; echo ""; echo "  Install log: $LOG_FILE"; echo ""; exit 1; }

# Redirect all output to log file AND terminal
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "=== needlestack install $(date) ==="

# Trap unexpected errors
trap 'echo ""; red "Something went wrong on line $LINENO."; echo "  Please send this file to whoever gave you needlestack:"; echo "  $LOG_FILE"; echo ""; exit 1' ERR

# --- header ---

clear 2>/dev/null || true
echo ""
bold "  needlestack installer"
echo "  Find your photos by describing what's in them."
echo ""

# --- macOS check ---

step "Checking system"

if [[ "$(uname)" != "Darwin" ]]; then
    die "This installer is for macOS only."
fi

MACOS_VERSION=$(sw_vers -productVersion)
MACOS_MAJOR=$(echo "$MACOS_VERSION" | cut -d. -f1)
if [[ "$MACOS_MAJOR" -lt 12 ]]; then
    die "macOS 12 (Monterey) or later is required. You have $MACOS_VERSION."
fi
green "macOS $MACOS_VERSION"

ARCH=$(uname -m)
RAM_BYTES=$(sysctl -n hw.memsize 2>/dev/null || echo 0)
RAM_GB=$(( RAM_BYTES / 1024 / 1024 / 1024 ))

# --- choose vision model based on hardware ---

step "Choosing vision model"

if [[ "$ARCH" == "arm64" ]]; then
    # Apple Silicon: unified memory is shared CPU + GPU.
    # The whole pool is available to Ollama for model weights.
    HW_DESC="Apple Silicon, ${RAM_GB} GB unified memory"
    if [[ "$RAM_GB" -ge 32 ]]; then
        LLAVA_MODEL="$MODEL_QUALITY"
        MODEL_TIER="quality"
        MODEL_REASON="32 GB+ unified memory comfortably fits the 32B quality model (~20 GB)."
        MODEL_SPEED="~90-120 seconds per photo"
        MIN_DISK_GB="$MIN_DISK_QUALITY"
    elif [[ "$RAM_GB" -ge 16 ]]; then
        LLAVA_MODEL="$MODEL_BALANCED"
        MODEL_TIER="balanced"
        MODEL_REASON="16 GB unified memory is a comfortable fit for the 7B model (~5 GB)."
        MODEL_SPEED="~4-6 seconds per photo"
        MIN_DISK_GB="$MIN_DISK_BALANCED"
    else
        LLAVA_MODEL="$MODEL_FAST"
        MODEL_TIER="fast"
        MODEL_REASON="8 GB unified memory is tight for the 7B model alongside the OS. Using the fast model leaves breathing room."
        MODEL_SPEED="~2-4 seconds per photo"
        MIN_DISK_GB="$MIN_DISK_FAST"
    fi
else
    # Intel Mac: Ollama runs on CPU (no Metal acceleration).
    HW_DESC="Intel Mac, ${RAM_GB} GB RAM (CPU inference)"
    LLAVA_MODEL="$MODEL_FAST"
    MODEL_TIER="fast"
    MODEL_REASON="Intel Macs run Ollama on CPU — smaller model means much faster indexing."
    MODEL_SPEED="~15-30 seconds per photo (CPU)"
    MIN_DISK_GB="$MIN_DISK_FAST"
fi

# Print the explanation box
printf "\n"
printf "  ┌─────────────────────────────────────────────────────────────┐\n"
printf "  │  Hardware detected:  %-40s│\n" "$HW_DESC"
printf "  │                                                             │\n"
printf "  │  Selected model:     %-40s│\n" "$LLAVA_MODEL  ($MODEL_TIER)"
printf "  │  Speed estimate:     %-40s│\n" "$MODEL_SPEED"
printf "  │                                                             │\n"
# wrap reason at ~56 chars
printf "  │  Why: %-54s│\n" "$MODEL_REASON"
printf "  │                                                             │\n"
printf "  │  Other options you can switch to any time:                  │\n"
printf "  │    fast     %-47s│\n" "$MODEL_FAST"
printf "  │    balanced %-47s│\n" "$MODEL_BALANCED"
printf "  │    quality  %-47s│\n" "$MODEL_QUALITY"
printf "  │                                                             │\n"
printf "  │  To use a different model after install:                    │\n"
printf "  │    needlestack index /your/photos --preset quality          │\n"
printf "  │    needlestack index /your/photos --model your-model-name   │\n"
printf "  │  Re-indexing picks up the new model automatically.          │\n"
printf "  └─────────────────────────────────────────────────────────────┘\n"
printf "\n"

green "Model: $LLAVA_MODEL ($MODEL_TIER tier)"

# --- internet ---

step "Checking internet connection"

if ! curl -sf --max-time 5 https://ollama.com >/dev/null 2>&1; then
    die "No internet connection detected. Please connect and try again."
fi
green "Internet reachable"

# --- disk space ---

step "Checking disk space"

AVAILABLE_KB=$(df -k "$HOME" | awk 'NR==2 {print $4}')
AVAILABLE_GB=$(( AVAILABLE_KB / 1024 / 1024 ))

if [[ "$AVAILABLE_GB" -lt "$MIN_DISK_GB" ]]; then
    die "Not enough disk space. Need at least ${MIN_DISK_GB} GB free, only ${AVAILABLE_GB} GB available."
fi
green "${AVAILABLE_GB} GB available (need ${MIN_DISK_GB} GB)"

# --- Ollama ---

step "Checking Ollama"

if ! command -v ollama &>/dev/null; then
    echo ""
    echo "  Ollama is not installed. It's needed to understand your photos."
    echo ""
    echo "  To install it:"
    echo "    1. Go to https://ollama.com"
    echo "    2. Click Download and open the .dmg file"
    echo "    3. Drag Ollama to your Applications folder and open it"
    echo "    4. Run this installer again"
    echo ""
    die "Stopping — please install Ollama first, then run this again."
fi
green "Ollama installed"

# Start Ollama app if not already running
if ! curl -sf --max-time 3 http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo "  Starting Ollama..."
    open -a Ollama 2>/dev/null || ollama serve >/dev/null 2>&1 &

    WAITED=0
    while ! curl -sf --max-time 2 http://localhost:11434/api/tags >/dev/null 2>&1; do
        sleep 2
        WAITED=$(( WAITED + 2 ))
        if [[ "$WAITED" -ge "$OLLAMA_TIMEOUT" ]]; then
            die "Ollama did not start after ${OLLAMA_TIMEOUT}s. Try opening the Ollama app manually, then run this installer again."
        fi
        printf "  waiting for Ollama... %ds\r" "$WAITED"
    done
    printf "%-40s\r" ""
fi
green "Ollama is running"

# Pull model — handle partial downloads and failures
MODEL_BASE="${LLAVA_MODEL%%:*}"
if ollama list 2>/dev/null | grep -q "^${MODEL_BASE}"; then
    MODEL_SIZE=$(ollama list 2>/dev/null | grep "^${MODEL_BASE}" | awk '{print $3}' | head -1)
    green "Model $LLAVA_MODEL already downloaded ($MODEL_SIZE)"
else
    echo ""
    case "$MODEL_TIER" in
        fast)     MODEL_DL_SIZE="~2 GB" ;;
        balanced) MODEL_DL_SIZE="~5 GB" ;;
        quality)  MODEL_DL_SIZE="~20 GB" ;;
        *)        MODEL_DL_SIZE="several GB" ;;
    esac
    echo "  Downloading $LLAVA_MODEL ($MODEL_TIER tier) — $MODEL_DL_SIZE, this will take a while."
    echo "  You can leave this running and come back."
    echo ""
    if ! ollama pull "$LLAVA_MODEL"; then
        echo ""
        echo "  The model download failed. Common causes:"
        echo "    • Internet connection dropped"
        echo "    • Not enough disk space"
        echo ""
        echo "  You can try again by running:  ollama pull $LLAVA_MODEL"
        echo "  Then run this installer again."
        die "Model download failed."
    fi
    green "Model $LLAVA_MODEL downloaded"
fi

# Verify model responds to a test inference
echo "  Verifying model works..."
TEST_RESPONSE=$(curl -sf --max-time 30 http://localhost:11434/api/generate \
    -d "{\"model\":\"$LLAVA_MODEL\",\"prompt\":\"Reply with just the word OK.\",\"stream\":false}" \
    2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin).get('response',''))" 2>/dev/null || true)

if [[ -z "$TEST_RESPONSE" ]]; then
    warn "Could not verify model response — it may still work. Continuing."
else
    green "Model responded correctly"
fi

# --- pixi ---

step "Checking pixi (Python environment manager)"

if ! command -v pixi &>/dev/null && [[ ! -f "$HOME/.pixi/bin/pixi" ]]; then
    echo "  Installing pixi..."
    if ! curl -fsSL https://pixi.sh/install.sh | bash; then
        die "Failed to install pixi. Check your internet connection and try again."
    fi
fi

export PATH="$HOME/.pixi/bin:$PATH"

if ! command -v pixi &>/dev/null; then
    die "pixi installed but could not be found. Please restart Terminal and run this installer again."
fi
green "pixi $(pixi --version)"

# --- Python environment ---

step "Setting up Python environment (a few minutes first time)"

cd "$SCRIPT_DIR"

if ! pixi install; then
    echo ""
    echo "  Failed to set up the Python environment. Common causes:"
    echo "    • Internet connection dropped"
    echo "    • Not enough disk space"
    echo ""
    die "Environment setup failed."
fi
green "Python environment ready"

# --- smoke test ---

step "Verifying installation"

SMOKE_OUTPUT=$(pixi run python -c "
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
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'OK device={device}')
" 2>&1)

if echo "$SMOKE_OUTPUT" | grep -q "^FAIL:"; then
    MISSING=$(echo "$SMOKE_OUTPUT" | sed 's/^FAIL: //')
    echo ""
    echo "  Some required components did not install correctly:"
    echo "  $MISSING"
    echo ""
    echo "  Try running:  pixi install"
    echo "  If it still fails, send the install log to whoever gave you needlestack:"
    echo "  $LOG_FILE"
    die "Installation incomplete."
fi

DEVICE=$(echo "$SMOKE_OUTPUT" | grep -o 'device=[^ ]*' | cut -d= -f2)
green "All components verified (using $DEVICE)"

# --- app bundle ---

step "Installing app"

APP_NAME="Needlestack"
if [[ -w "/Applications" ]]; then
    APPS_DIR="/Applications"
else
    APPS_DIR="$HOME/Applications"
    mkdir -p "$APPS_DIR"
fi
APP_BUNDLE="$APPS_DIR/$APP_NAME.app"

rm -rf "$APP_BUNDLE"
mkdir -p "$APP_BUNDLE/Contents/MacOS" "$APP_BUNDLE/Contents/Resources"

VERSION=$(grep -m1 '^version' "$SCRIPT_DIR/pyproject.toml" | sed -E 's/.*"([^"]+)".*/\1/')
sed "s/__VERSION__/${VERSION:-0.0.0}/g" "$SCRIPT_DIR/resources/macos/Info.plist" > "$APP_BUNDLE/Contents/Info.plist"
cp "$SCRIPT_DIR/resources/macos/AppIcon.icns" "$APP_BUNDLE/Contents/Resources/AppIcon.icns"

LAUNCHER="$APP_BUNDLE/Contents/MacOS/needlestack-launcher"
cat > "$LAUNCHER" << LAUNCHER_EOF
#!/bin/bash
# needlestack's Python/pixi environment lives where it was installed, not
# inside this app bundle — this path is baked in at install time.
INSTALL_DIR="$SCRIPT_DIR"
LOG_DIR="\$HOME/.needlestack"
LOG_FILE="\$LOG_DIR/needlestack.log"
mkdir -p "\$LOG_DIR"
echo "=== needlestack started \$(date) ===" >> "\$LOG_FILE"
exec > >(tee -a "\$LOG_FILE") 2>&1

cd "\$INSTALL_DIR"
export PATH="\$HOME/.pixi/bin:\$PATH"

PORT=8484

dialog() {
    osascript -e "display dialog \"\$1\" with title \"Needlestack\" buttons {\"OK\"} default button 1 with icon caution" >/dev/null 2>&1
}

# If needlestack is already running on the port, just open the browser
if lsof -i ":\$PORT" -sTCP:LISTEN -t >/dev/null 2>&1; then
    if curl -sf --max-time 2 "http://localhost:\$PORT/" | grep -qi needlestack; then
        echo "needlestack already running — opening browser."
        open "http://localhost:\$PORT"
        exit 0
    else
        echo "Port \$PORT is in use by another application — needlestack will try a nearby port."
    fi
fi

# Start Ollama if needed
if ! curl -sf --max-time 3 http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo "Starting Ollama..."
    open -a Ollama 2>/dev/null || true
    sleep 4
fi

if ! curl -sf --max-time 3 http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo "Ollama does not appear to be running."
    dialog "Ollama does not appear to be running. Please open the Ollama app and try again."
    exit 1
fi

# There's no Terminal window in a double-clicked .app, so pop the browser
# tab ourselves once the server has had a moment to bind the port.
(sleep 2 && open "http://localhost:\$PORT") &

# exec replaces this script with the server process itself, so quitting
# the app from the Dock (or Cmd+Q) sends its signal straight to the
# server instead of orphaning it — there's no wrapper process left to
# catch a later crash, but the log file still captures it.
exec pixi run needlestack serve
dialog "needlestack could not start. If this keeps happening, send this log to whoever gave you needlestack: \$LOG_FILE"
LAUNCHER_EOF
chmod +x "$LAUNCHER"

# A locally-built bundle shouldn't carry quarantine flags, but the icon we
# just copied came from a possibly-downloaded checkout — strip defensively
# so Gatekeeper doesn't report the app as "damaged."
xattr -cr "$APP_BUNDLE" 2>/dev/null || true

# Ad-hoc sign so a corrupted/missing signature never triggers Gatekeeper's
# "app is damaged" dialog (e.g. if quarantine gets reapplied later by
# cloud sync or AV scanning). This does NOT grant Apple's trusted/notarized
# status — that requires a paid Developer ID — so the first-launch
# "unidentified developer" prompt (see GUIDE.md) is still expected.
if command -v codesign &>/dev/null; then
    if codesign --deep --force -s - "$APP_BUNDLE" 2>>"$LOG_FILE"; then
        green "App signed (ad-hoc)"
    else
        warn "Ad-hoc signing failed — app may show extra Gatekeeper warnings"
    fi
else
    warn "codesign not found — app may show extra Gatekeeper warnings"
fi

# Refresh Launch Services so Spotlight/Launchpad see the app immediately
# instead of waiting for their next periodic scan.
/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister -f "$APP_BUNDLE" >/dev/null 2>&1 || true

green "Installed 'Needlestack' to $APPS_DIR"

# --- done ---

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
bold "  Installation complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  ⚠   Do not move this folder after installing."
echo "      The launcher expects needlestack to stay here:"
echo "      $SCRIPT_DIR"
echo ""
echo "  Step 1 — Index your photos (run once, takes an hour or two):"
echo ""
echo "    Open Terminal and run:"
echo "    cd \"$SCRIPT_DIR\""
echo "    pixi run needlestack index /path/to/your/photos"
echo ""
echo "  Step 2 — Search:"
echo ""
echo "    Open 'Needlestack' from Launchpad or Spotlight (⌘-Space, type Needlestack)"
echo "    (Right-click → Open the first time to allow it — it's not signed)"
echo ""
echo "  Install log saved to: $LOG_FILE"
echo ""
