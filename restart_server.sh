#!/usr/bin/env bash
# ==============================================================================
# Echoes of Azeroth — Server-Neustart (restart_server.sh)
# ------------------------------------------------------------------------------
# Stops EVERY process that currently listens on port 8765 (found via /proc —
# works regardless of process name, path or how the old server was started)
# and starts the current version of the Web Studio. No sudo required.
#
# Usage:  ./restart_server.sh
# ==============================================================================
set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"
PY="$PROJECT_DIR/.venv/bin/python"

echo "== Echoes of Azeroth: Server-Neustart =="

# 1) Stop everything listening on 8765 (visible in this namespace)
"$PY" - <<'EOF'
import os
import signal

PORT_HEX = "223D"  # 8765

inodes = set()
for netf in ("/proc/net/tcp", "/proc/net/tcp6"):
    try:
        with open(netf) as f:
            for line in f:
                p = line.split()
                if len(p) < 10 or ":" not in p[1]:
                    continue
                if p[1].split(":")[1] == PORT_HEX and p[3] == "0A":
                    inodes.add(p[9])
    except FileNotFoundError:
        pass

killed = []
for pid in (int(x) for x in os.listdir("/proc") if x.isdigit()):
    try:
        fds = os.listdir(f"/proc/{pid}/fd")
    except OSError:
        continue
    for fd in fds:
        try:
            link = os.readlink(f"/proc/{pid}/fd/{fd}")
        except OSError:
            continue
        if any(f"socket:[{i}]" in link for i in inodes):
            try:
                os.kill(pid, signal.SIGTERM)
                killed.append(pid)
            except (OSError, PermissionError):
                pass

print(f"  beendete Prozesse auf Port 8765: {killed or 'keine'}")
EOF

sleep 1

# 1b) Stop orphaned pipeline jobs — children of a killed server keep running
#     and would keep writing state/catalog files in parallel with a new job.
if pkill -f "pipeline/0[1-7]_" 2>/dev/null; then
    echo "  laufende Pipeline-Jobs beendet"
fi
sleep 1

# 2) Start the current server version in the background
nohup "$PY" "$PROJECT_DIR/web/serve.py" --host 127.0.0.1 --port 8765 >/tmp/echoes-of-azeroth-web.log 2>&1 &
sleep 1.5

# 3) Verify
if curl -s http://127.0.0.1:8765/api/version >/dev/null 2>&1; then
    echo "  Server läuft: http://127.0.0.1:8765  (aktuelle Version)"
else
    echo "  FEHLER: Server konnte nicht gestartet werden."
    echo "  Log: /tmp/echoes-of-azeroth-web.log"
    exit 1
fi
