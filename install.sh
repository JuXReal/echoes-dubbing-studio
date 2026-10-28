#!/usr/bin/env bash
# ==============================================================================
# 🎙️ Echoes of Azeroth (DE) — Automated Installer Script (install.sh)
# ==============================================================================
# Installs required system packages (ffmpeg, sox), sets up the Python 3.11
# virtual environment, installs PyTorch CUDA, Coqui XTTS-v2, Faster-Whisper,
# and auto-detects the local World of Warcraft _retail_ AddOns directory.
# ==============================================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "=============================================================================="
echo "🎙️  Echoes of Azeroth (DE) — Setup & Dependency Installer"
echo "    Community German Voiceover Pipeline for Chatty Little NPC"
echo "=============================================================================="
echo ""
echo "⚠️  IMPORTANT COMMUNITY NOTICE:"
echo "    Please DO NOT blindly upload generated packs to CurseForge!"
echo "    Respect the creators of the 'Chatty Little NPC' addon (Tuubby & contributors)"
echo "    and the community behind the original English voiceover packs."
echo "    Coordinate with the official authors before publishing any voice packs."
echo "=============================================================================="
echo ""

# 1. Install system audio tools (ffmpeg & sox)
echo "[1/4] Checking system packages (ffmpeg, sox)..."
if ! command -v ffmpeg &>/dev/null || ! command -v sox &>/dev/null; then
    if command -v pacman &>/dev/null; then
        echo "  -> Detected Arch/CachyOS. Installing ffmpeg & sox via pacman..."
        sudo pacman -S --needed --noconfirm ffmpeg sox
    elif command -v apt-get &>/dev/null; then
        echo "  -> Detected Debian/Ubuntu. Installing ffmpeg & sox via apt..."
        sudo apt-get update && sudo apt-get install -y ffmpeg sox libsox-fmt-all
    elif command -v dnf &>/dev/null; then
        echo "  -> Detected Fedora. Installing ffmpeg & sox via dnf..."
        sudo dnf install -y ffmpeg sox
    else
        echo "  [!] Please install 'ffmpeg' and 'sox' manually using your OS package manager."
    fi
else
    echo "  ✓ ffmpeg and sox are already installed."
fi

# 2. Create Python 3.11 Virtual Environment
echo ""
echo "[2/4] Setting up Python 3.11 virtual environment (.venv)..."
if command -v uv &>/dev/null; then
    uv venv .venv --python 3.11
    uv pip install --python .venv/bin/python -r requirements.txt
    uv pip install --python .venv/bin/python "coqui-tts[codec]"
else
    echo "  -> Installing 'uv' package manager for fast Python 3.11 setup..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    uv venv .venv --python 3.11
    uv pip install --python .venv/bin/python -r requirements.txt
    uv pip install --python .venv/bin/python "coqui-tts[codec]"
fi

# 3. Create .env from .env.example if missing
echo ""
echo "[3/4] Configuring environment (.env)..."
if [[ ! -f .env ]]; then
    cp .env.example .env
    echo "  ✓ Created .env from .env.example"
else
    echo "  ✓ .env already exists"
fi

# 4. Verify CUDA & Audio Stack
echo ""
echo "[4/4] Verifying PyTorch CUDA, Coqui XTTS-v2, and Faster-Whisper..."
.venv/bin/python -c "
import torch, TTS, faster_whisper, soundfile, mutagen
cuda_ok = torch.cuda.is_available()
dev_name = torch.cuda.get_device_name(0) if cuda_ok else 'CPU Fallback'
print(f'  ✓ PyTorch {torch.__version__} | CUDA Available: {cuda_ok} ({dev_name})')
print('  ✓ Coqui XTTS-v2, Faster-Whisper, SoundFile & Mutagen verified!')
"

echo ""
echo "=============================================================================="
echo "✅ Installation Complete!"
echo "   • Start the Interactive Launcher (CLI):  ./start.py"
echo "   • Start the Web Studio & Launchpad UI:   .venv/bin/python web/serve.py"
echo "     (Then open http://127.0.0.1:8765 in your browser)"
echo "=============================================================================="
