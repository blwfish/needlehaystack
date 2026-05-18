#!/bin/bash
set -euo pipefail

LLAVA_MODEL="llava:13b"
MIN_DISK_GB=14        # ~8GB model + ~3GB pixi env + 3GB buffer
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
if [[ "$ARCH" == "arm64" ]]; then
    green "Apple Silicon (fast)"
else
    warn "Intel Mac detected — indexing will be slower, but everything will work."
fi

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
    echo "  Downloading $LLAVA_MODEL — this is about 8 GB and will take a while."
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

# --- launcher ---

step "Creating launcher"

LAUNCHER="$SCRIPT_DIR/Start Needlestack.command"
cat > "$LAUNCHER" << LAUNCHER_EOF
#!/bin/bash
SCRIPT_DIR="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="\$HOME/.needlestack"
LOG_FILE="\$LOG_DIR/needlestack.log"
mkdir -p "\$LOG_DIR"
echo "=== needlestack started \$(date) ===" >> "\$LOG_FILE"
exec > >(tee -a "\$LOG_FILE") 2>&1

cd "\$SCRIPT_DIR"
export PATH="\$HOME/.pixi/bin:\$PATH"

PORT=8484

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
    echo ""
    echo "Ollama does not appear to be running."
    echo "Please open the Ollama app and try again."
    read -r -p "Press Enter to close..."
    exit 1
fi

# Run and catch crashes
if ! pixi run needlestack serve; then
    echo ""
    echo "needlestack exited unexpectedly."
    echo "If this keeps happening, send this log to whoever gave you needlestack:"
    echo "\$LOG_FILE"
    echo ""
    read -r -p "Press Enter to close..."
    exit 1
fi
LAUNCHER_EOF
chmod +x "$LAUNCHER"
green "Created 'Start Needlestack.command'"

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
echo "    Double-click 'Start Needlestack.command' in Finder"
echo "    (Right-click → Open the first time to allow it)"
echo ""
echo "  Install log saved to: $LOG_FILE"
echo ""
