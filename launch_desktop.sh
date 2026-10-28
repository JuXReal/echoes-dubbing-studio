#!/usr/bin/env bash
# ==============================================================================
# 🎙️ Echoes of Azeroth — Desktop One-Click Launcher
# Starts the Web Studio & Launchpad server (if not already running) and opens
# http://127.0.0.1:8765 in the default browser.
#
# Stale-server protection: if a serve.py process from an OLD version is still
# running (recognizable by the missing /api/version endpoint), restart_server.sh
# stops every process on port 8765 and starts the current version.
# ==============================================================================
set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

URL="http://127.0.0.1:8765"

if ! curl -s "${URL}/api/version" >/dev/null 2>&1; then
    # No (or outdated) server running — replace it with the current version
    "$PROJECT_DIR/restart_server.sh"
fi

xdg-open "$URL" >/dev/null 2>&1 &
