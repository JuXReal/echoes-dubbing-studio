#!/usr/bin/env python3
"""
A/B Listening Studio & Launchpad Web Server (web/serve.py)
-----------------------------------------------------------
Serves the interactive HTML5 A/B comparison & One-Click Launchpad UI (`web/index.html`)
and provides:
  - `GET  /api/playlist`        -> Full JSON metadata of all localized quests, scripts, QA scores & FX
  - `GET  /api/audit`           -> Independent 100-Quest Quality Audit report
  - `GET  /api/state`           -> LIVE pipeline state (current quest, step, counts, ETA, AI calls)
  - `GET  /api/job_status`      -> Job status & console log tail (compat)
  - `POST /api/start_job`       -> Starts a pipeline run or 100-Quest Audit
  - `POST /api/regenerate_entry`-> Regenerates a single entry (TTS + QA + FX)
  - `GET  /api/settings`        -> Safe AI settings (NO API keys returned)
  - `POST /api/settings`        -> Saves AI settings to the git-ignored .env file
  - `POST /api/settings/test`   -> Tests the AI provider connection
  - `GET  /audio/en/<file>`     -> Original English OGG
  - `GET  /audio/de/<file>`     -> Synthesized & mastered German OGG
  - `GET  /audio/seed/<file>`   -> Extracted 24kHz Mono reference WAV voice seed
"""

import argparse
import json
import mimetypes
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import yaml
from rich.console import Console

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.ai_client import (
    PROVIDERS,
    get_settings as get_ai_settings,
    is_available as ai_is_available,
    list_models as ai_list_models,
    test_connection as ai_test_connection,
)
from pipeline.live_state import (
    ack_topup as live_ack_topup,
    begin_job as live_begin_job,
    finish_job as live_finish_job,
    get_expansion_status,
    get_usage,
    load as load_live_state,
    set_ai_config as live_set_ai_config,
)

console = Console()

# Bumped whenever the web API surface changes — launch_desktop.sh and the UI
# use this to detect (and restart) stale server processes that predate the
# current endpoints (/api/state, /api/settings, /api/settings/models, ...).
SERVER_VERSION = 4

ENV_MANAGED_KEYS = (
    "AI_PROVIDER",
    "AI_API_KEY",
    "AI_BASE_URL",
    "AI_MODEL",
    "AI_ENABLED",
    "AI_REPAIR_AFTER_FAILURES",
)

# Global background job state for the Web UI Start Button
JOB_LOCK = threading.Lock()
CURRENT_JOB = {
    "running": False,
    "mode": None,
    "expansion": None,
    "started_at": None,
    "finished_at": None,
    "exit_code": None,
    "logs": [],
}

# Playlist file cache (mtime-based) — the JSON is several MB and changes per step
PLAYLIST_CACHE = {"mtime_ns": None, "data": None}


def _env_file(project_root: Path) -> Path:
    return project_root / ".env"


def read_env_lines(project_root: Path) -> dict:
    """Returns {key: (line_with_key, value)} for managed keys present in .env."""
    lines = {}
    path = _env_file(project_root)
    if not path.exists():
        return lines
    for raw in path.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        k, _, v = stripped.partition("=")
        k = k.strip()
        if k in ENV_MANAGED_KEYS:
            lines[k] = {"raw": raw, "value": v.strip().strip('"').strip("'")}
    return lines


def write_env_settings(project_root: Path, updates: dict) -> None:
    """Atomically updates AI_* keys in .env, preserving everything else. No key logging."""
    path = _env_file(project_root)
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    existing_lines = existing.splitlines()

    out_lines = []
    written = set()
    for raw in existing_lines:
        stripped = raw.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            k = stripped.split("=", 1)[0].strip()
            if k in updates:
                if updates[k] is not None:
                    out_lines.append(f'{k}="{updates[k]}"')
                written.add(k)
                continue
        out_lines.append(raw)

    for k in ENV_MANAGED_KEYS:
        if k in updates and updates[k] is not None and k not in written:
            out_lines.append(f'{k}="{updates[k]}"')

    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    tmp.replace(path)



def update_config_yaml(project_root: Path, qa_retries: int) -> None:
    """Schreibt max_retries in config.yaml (gezielte Zeilenersetzung, Kommentare bleiben)."""
    import re as _re
    cfg_path = project_root / "config.yaml"
    text = cfg_path.read_text(encoding="utf-8")
    text = _re.sub(r"(\n  max_retries: )\d+", rf"\g<1>{int(qa_retries)}", text, count=1)
    cfg_path.write_text(text, encoding="utf-8")


def _run_background_commands(commands: list, project_root: Path, mode: str, expansion: str):
    global CURRENT_JOB
    env = os.environ.copy()
    env["COQUI_TOS_AGREED"] = "1"
    env["PYTHONUNBUFFERED"] = "1"

    with JOB_LOCK:
        CURRENT_JOB["running"] = True
        CURRENT_JOB["mode"] = mode
        CURRENT_JOB["expansion"] = expansion
        CURRENT_JOB["started_at"] = time.strftime("%H:%M:%S")
        CURRENT_JOB["finished_at"] = None
        CURRENT_JOB["exit_code"] = None
        CURRENT_JOB["logs"] = [f"[{time.strftime('%H:%M:%S')}] Starting job: {mode} ({expansion})..."]

    live_begin_job(mode=mode, expansion=expansion)

    # Ausgabe in eine Datei statt in eine Pipe: überlebt Server-Neustarts,
    # weil der Job-Prozess bei einem getöteten Server nicht an SIGPIPE stirbt.
    log_path = project_root / "output" / "qa_reports" / "job_log.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    exit_code = 0
    with open(log_path, "a", encoding="utf-8") as logf:
        logf.write(f"\n[{time.strftime('%H:%M:%S')}] Starting job: {mode} ({expansion})\n")
        for cmd in commands:
            with JOB_LOCK:
                CURRENT_JOB["logs"].append(f"\n$ {' '.join(cmd)}")
            logf.write(f"\n$ {' '.join(cmd)}\n")
            logf.flush()
            proc = subprocess.Popen(
                cmd,
                cwd=str(project_root),
                env=env,
                stdout=logf,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            with JOB_LOCK:
                CURRENT_JOB["proc"] = proc
            while proc.poll() is None:
                time.sleep(0.5)
                try:
                    with open(log_path, "rb") as f:
                        f.seek(0, 2)
                        f.seek(max(0, f.tell() - 8192))
                        tail = f.read().decode("utf-8", errors="replace")
                    lines = [
                        l.rstrip() for l in tail.splitlines()
                        if l.strip()
                        and "[transformers] Model config:" not in l
                        and "Fetching 9 files:" not in l
                    ]
                    with JOB_LOCK:
                        CURRENT_JOB["logs"] = lines[-400:]
                except Exception:
                    pass
            if proc.returncode != 0:
                exit_code = proc.returncode
                break

    with JOB_LOCK:
        CURRENT_JOB["proc"] = None
        CURRENT_JOB["running"] = False
        CURRENT_JOB["finished_at"] = time.strftime("%H:%M:%S")
        CURRENT_JOB["exit_code"] = exit_code
        CURRENT_JOB["logs"].append(f"\n[{time.strftime('%H:%M:%S')}] Job finished with exit code {exit_code}.")

    live_finish_job(ok=exit_code == 0)


class ABComparisonHandler(BaseHTTPRequestHandler):
    project_root: Path = Path(".")
    cfg: dict = {}

    def log_message(self, fmt, *args):
        pass

    def _safe_ai_settings(self) -> dict:
        """AI settings WITHOUT the API key — safe to send to the browser."""
        settings = get_ai_settings(self.cfg)
        try:
            qa_retries = int((self.cfg.get("qa_validation") or {}).get("max_retries", 3))
        except (TypeError, ValueError):
            qa_retries = 3
        return {
            "provider": settings["provider"],
            "model": settings["model"],
            "base_url": settings["base_url"],
            "enabled": settings["enabled"],
            "trigger_after_failures": settings["trigger_after_failures"],
            "temperature": settings["temperature"],
            "timeout_sec": settings["timeout_sec"],
            "available": ai_is_available(settings),
            "api_key_set": bool(settings["api_key"]),
            "providers": PROVIDERS,
            "pricing": settings.get("pricing", {}),
            "qa_retries": qa_retries,
        }

    def _send_json(self, data: dict, status: int = 200):
        raw = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(raw)

    def _send_file(self, file_path: Path, send_body: bool = True):
        if not file_path.exists() or not file_path.is_file():
            self.send_error(404, f"File not found: {file_path.name}")
            return

        ctype, _ = mimetypes.guess_type(str(file_path))
        is_html = False
        is_audio = False
        if file_path.suffix.lower() == ".ogg":
            ctype = "audio/ogg"
            is_audio = True
        elif file_path.suffix.lower() == ".wav":
            ctype = "audio/wav"
            is_audio = True
        elif file_path.suffix.lower() == ".html":
            ctype = "text/html; charset=utf-8"
            is_html = True

        size = file_path.stat().st_size
        range_header = self.headers.get("Range")

        if range_header and range_header.startswith("bytes="):
            try:
                byte_range = range_header.split("=")[1].split("-")
                start = int(byte_range[0]) if byte_range[0] else 0
                end = int(byte_range[1]) if (len(byte_range) > 1 and byte_range[1]) else size - 1
                end = min(end, size - 1)
                length = max(0, end - start + 1)

                self.send_response(206)
                self.send_header("Content-Type", ctype or "application/octet-stream")
                if is_html or is_audio:
                    self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                self.send_header("Content-Length", str(length))
                self.send_header("Accept-Ranges", "bytes")
                self.end_headers()
                if send_body:
                    with open(file_path, "rb") as f:
                        f.seek(start)
                        self.wfile.write(f.read(length))
                return
            except Exception:
                pass

        self.send_response(200)
        self.send_header("Content-Type", ctype or "application/octet-stream")
        if is_html or is_audio:
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Content-Length", str(size))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        if send_body:
            self.wfile.write(file_path.read_bytes())

    def do_HEAD(self):
        self._handle_request(send_body=False)

    def do_GET(self):
        self._handle_request(send_body=True)

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/approve_entry":
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8")) if length > 0 else {}
            entry_id = body.get("entry_id")
            if not entry_id:
                self._send_json({"error": "entry_id is required"}, status=400)
                return
            pilot_path = self.project_root / self.cfg["paths"]["data_raw_de_text"] / "midnight_de_pilot_20.json"
            if not pilot_path.exists():
                self._send_json({"error": "Katalog nicht vorhanden"}, status=404)
                return
            try:
                data = json.loads(pilot_path.read_text(encoding="utf-8"))
            except Exception:
                self._send_json({"error": "Katalog nicht lesbar"}, status=500)
                return
            found = None
            for e in data.get("entries", []):
                if e.get("entry_id") == entry_id:
                    found = e
                    break
            if found is None:
                self._send_json({"error": f"Eintrag {entry_id} nicht gefunden"}, status=404)
                return
            qa = found.setdefault("qa_result", {})
            qa["passed"] = True
            qa["approved_by_user"] = True
            pilot_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
            self._send_json({"status": "approved", "entry_id": entry_id})
            return

        if parsed.path == "/api/regenerate_entry":
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8")) if length > 0 else {}
            entry_id = body.get("entry_id")
            if not entry_id:
                self._send_json({"error": "entry_id is required"}, status=400)
                return

            with JOB_LOCK:
                if CURRENT_JOB["running"]:
                    self._send_json({"error": "A job is already running!"}, status=409)
                    return

            script = str(self.project_root / "pipeline" / "regenerate_single.sh")
            retries = body.get("retries") or body.get("qa_retries") or 3
            ai_trigger = body.get("ai_trigger") or body.get("trigger_after_failures") or 3
            cmds = [["bash", script, str(entry_id), str(int(retries)), str(int(ai_trigger))]]

            t = threading.Thread(
                target=_run_background_commands,
                args=(cmds, self.project_root, f"regenerate_{entry_id}", "Single Entry"),
                daemon=True,
            )
            t.start()
            self._send_json({"status": "started", "entry_id": entry_id})
            return

        if parsed.path == "/api/settings":
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8")) if length > 0 else {}
            updates = {}
            provider = body.get("provider")
            if provider is not None:
                provider = str(provider).strip().lower()
                if provider not in PROVIDERS:
                    self._send_json({"error": f"Unknown provider: {provider}"}, status=400)
                    return
                updates["AI_PROVIDER"] = provider
            if body.get("model") is not None:
                updates["AI_MODEL"] = str(body.get("model", "")).strip()
            if body.get("base_url") is not None:
                updates["AI_BASE_URL"] = str(body.get("base_url", "")).strip().rstrip("/")
            if body.get("enabled") is not None:
                updates["AI_ENABLED"] = "true" if bool(body.get("enabled")) else "false"
            if body.get("trigger_after_failures") is not None:
                try:
                    trigger = max(1, int(body.get("trigger_after_failures")))
                except (TypeError, ValueError):
                    self._send_json({"error": "trigger_after_failures must be an integer >= 1"}, status=400)
                    return
                updates["AI_REPAIR_AFTER_FAILURES"] = str(trigger)
            # API key: only stored if a non-empty value is submitted; NEVER returned
            if body.get("api_key"):
                updates["AI_API_KEY"] = str(body["api_key"]).strip()

            if body.get("qa_retries") is not None:
                try:
                    qa_retries = max(1, min(10, int(body.get("qa_retries"))))
                except (TypeError, ValueError):
                    self._send_json({"error": "qa_retries must be an integer 1-10"}, status=400)
                    return
                update_config_yaml(self.project_root, qa_retries)

            write_env_settings(self.project_root, updates)
            # Apply to this process so subsequently spawned pipeline jobs inherit it
            for k, v in updates.items():
                if v is not None:
                    os.environ[k] = v
            self._send_json({"status": "saved", **self._safe_ai_settings()})
            return

        if parsed.path == "/api/settings/test":
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8")) if length > 0 else {}
            settings = get_ai_settings(self.cfg)
            # Allow testing with unsaved values (e.g. a freshly typed key)
            if body.get("provider") is not None:
                settings["provider"] = str(body["provider"]).strip().lower()
            if body.get("model") is not None:
                settings["model"] = str(body["model"]).strip()
            if body.get("base_url") is not None:
                settings["base_url"] = str(body["base_url"]).strip().rstrip("/")
            if body.get("api_key"):
                settings["api_key"] = str(body["api_key"]).strip()
            if body.get("enabled") is not None:
                settings["enabled"] = bool(body["enabled"])
            ok, message = ai_test_connection(settings)
            self._send_json({"ok": ok, "message": message})
            return

        if parsed.path == "/api/settings/models":
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8")) if length > 0 else {}
            settings = get_ai_settings(self.cfg)
            # List models with form values (fresh key allowed, never stored)
            if body.get("provider") is not None:
                settings["provider"] = str(body["provider"]).strip().lower()
            if body.get("base_url") is not None:
                settings["base_url"] = str(body["base_url"]).strip().rstrip("/")
            if body.get("api_key"):
                settings["api_key"] = str(body["api_key"]).strip()
            ok, models, message = ai_list_models(settings)
            self._send_json({"ok": ok, "models": models, "message": message})
            return

        if parsed.path == "/api/ack_topup":
            # Bestätigt das DeepSeek-Guthaben-TopUp — beendet die Warteschleife
            # eines pausierten Jobs (Marker-Datei für alle Prozesse sichtbar).
            marker = live_ack_topup()
            self._send_json({"status": "acked", "at": marker.get("at")})
            return

        if parsed.path == "/api/stop_job":
            with JOB_LOCK:
                proc = CURRENT_JOB.get("proc")
            if proc is None or proc.poll() is not None:
                self._send_json({"status": "idle", "message": "Kein laufender Job"}, status=409)
                return
            try:
                pgid = os.getpgid(proc.pid)
                os.killpg(pgid, signal.SIGTERM)
                time.sleep(1.5)
                if proc.poll() is None:
                    os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except Exception as e:
                self._send_json({"error": str(e)}, status=500)
                return
            self._send_json({"status": "stopping"})
            return

        if parsed.path == "/api/start_job":
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8")) if length > 0 else {}
            expansions = body.get("expansions") or [body.get("expansion", "Midnight")]
            if isinstance(expansions, str):
                expansions = [expansions]
            expansions = [str(e) for e in expansions if e]
            mode = body.get("mode", "audit_100")

            with JOB_LOCK:
                if CURRENT_JOB["running"]:
                    self._send_json({"error": "A job is already running!"}, status=409)
                    return

            py = str(self.project_root / ".venv" / "bin" / "python")
            if not Path(py).exists():
                py = sys.executable

            cmds = []
            if mode in ("audit_100", "audit_full"):
                # Audits beziehen sich auf den Katalog der ERSTEN Expansion
                sample = "100" if mode == "audit_100" else "0"
                cmds = [[py, "pipeline/07_independent_audit_100.py", "--sample-size", sample]]
                mode_label = f"{mode} ({expansions[0]})"
            elif mode == "rebuild_bundle":
                # Nur die Addon-Pakete neu bauen + in WoW installieren (Schritt 6)
                cmds = [[py, "pipeline/06_audio_fx.py", "--install-to-wow", "--skip-audit"]]
                mode_label = f"{mode} ({len(expansions)} Addons)"
            elif mode == "repair_failed":
                # Fehlgeschlagene Einträge mit bis zu 20 Versuchen reparieren, danach Bundle bauen
                cmds = [
                    [py, "pipeline/05_qa_validator.py", "--only-failed", "--max-retries", "20", "--ai-trigger", "3"],
                    [py, "pipeline/06_audio_fx.py", "--install-to-wow", "--skip-audit"],
                ]
                mode_label = f"{mode} ({len(expansions)} Addons)"
            else:
                for expansion in expansions:
                    if mode == "pilot_20":
                        cmds.extend([
                            [py, "pipeline/01_addon_analyzer.py", "--expansion", expansion, "--limit", "20"],
                            [py, "pipeline/02_voice_extractor.py"],
                            [py, "pipeline/03_script_agent.py"],
                            [py, "pipeline/04_tts_generator.py"],
                            [py, "pipeline/05_qa_validator.py"],
                            [py, "pipeline/06_audio_fx.py", "--install-to-wow", "--skip-audit"],
                            [py, "pipeline/07_independent_audit_100.py", "--sample-size", "100"],
                        ])
                    elif mode == "full_expansion":
                        cmds.extend([
                            [py, "pipeline/01_addon_analyzer.py", "--expansion", expansion, "--limit", "0", "--include-gossip", "--scrape-all-quests"],
                            [py, "pipeline/02_voice_extractor.py"],
                            [py, "pipeline/03_script_agent.py"],
                            [py, "pipeline/04_tts_generator.py"],
                            [py, "pipeline/05_qa_validator.py"],
                            [py, "pipeline/06_audio_fx.py", "--install-to-wow", "--skip-audit"],
                            [py, "pipeline/07_independent_audit_100.py", "--sample-size", "100"],
                        ])
                    else:
                        self._send_json({"error": f"Unknown mode: {mode}"}, status=400)
                        return
                mode_label = f"{mode} ({len(expansions)} Addons)"

            t = threading.Thread(
                target=_run_background_commands,
                args=(cmds, self.project_root, mode_label, ", ".join(expansions)),
                daemon=True,
            )
            t.start()
            self._send_json({"status": "started", "mode": mode, "expansions": expansions})
            return

        self.send_error(404, "Not found")

    def _handle_request(self, send_body: bool = True):
        parsed = urllib.parse.urlparse(self.path)
        path = urllib.parse.unquote(parsed.path)

        if path == "/favicon.ico":
            svg = (
                b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">'
                b'<rect width="64" height="64" rx="14" fill="#0b0c10"/>'
                b'<circle cx="32" cy="26" r="9" fill="none" stroke="#f0b93c" stroke-width="4"/>'
                b'<path d="M18 46c4-8 8-12 14-12s10 4 14 12" fill="none" stroke="#f0b93c" stroke-width="4" stroke-linecap="round"/>'
                b'</svg>'
            )
            self.send_response(200)
            self.send_header("Content-Type", "image/svg+xml")
            self.send_header("Content-Length", str(len(svg)))
            self.send_header("Cache-Control", "max-age=86400")
            self.end_headers()
            if send_body:
                self.wfile.write(svg)
            return

        if path in ("/", "/index.html"):
            self._send_file(self.project_root / "web" / "index.html", send_body=send_body)
            return

        if path == "/api/playlist":
            pilot_path = self.project_root / self.cfg["paths"]["data_raw_de_text"] / "midnight_de_pilot_20.json"
            if not pilot_path.exists():
                self._send_json({"error": "Playlist not generated yet."}, status=404)
                return
            try:
                mtime_ns = pilot_path.stat().st_mtime_ns
            except OSError:
                mtime_ns = None
            if PLAYLIST_CACHE["mtime_ns"] == mtime_ns and PLAYLIST_CACHE["data"] is not None:
                self._send_json(PLAYLIST_CACHE["data"])
                return
            data = json.loads(pilot_path.read_text(encoding="utf-8"))
            PLAYLIST_CACHE["mtime_ns"] = mtime_ns
            PLAYLIST_CACHE["data"] = data
            self._send_json(data)
            return

        if path == "/api/expansions":
            packs = []
            wow_addons = Path(os.getenv("WOW_ADDONS_DIR") or self.cfg["paths"]["wow_addons_dir"])
            try:
                for d in sorted(wow_addons.iterdir()):
                    n = d.name
                    if not (n.startswith("ChattyLittleNpc_") and n.endswith("_voiceovers")):
                        continue
                    mid = n[len("ChattyLittleNpc_"):-len("_voiceovers")]
                    # Lokalisierte Pakete (z. B. Midnight_deDE) ausschließen
                    if re.match(r"^[a-z]{2}[A-Z]{2}$", mid.split("_")[-1]):
                        continue
                    packs.append({"id": mid, "name": mid.replace("_", " ")})
            except Exception:
                packs = []
            if not packs:
                # Fallback: Standard-Liste
                for e in ("Midnight", "The_War_Within", "Dragonflight", "Battle_for_Azeroth", "Wrath_of_the_Lich_King", "Vanilla"):
                    packs.append({"id": e, "name": e.replace("_", " ")})
            self._send_json({"expansions": packs})
            return

        if path == "/api/audit":
            audit_path = self.project_root / self.cfg["paths"]["output_qa_reports"] / "independent_audit_100_report.json"
            if audit_path.exists():
                data = json.loads(audit_path.read_text(encoding="utf-8"))
                self._send_json(data)
            else:
                self._send_json({"audited_count": 0, "passed_count": 0, "results": []})
            return

        if path == "/api/version":
            self._send_json({
                "server_version": SERVER_VERSION,
                "endpoints": ["playlist", "audit", "state", "settings", "settings/test", "settings/models", "start_job", "regenerate_entry"],
            })
            return

        if path == "/api/state":
            with JOB_LOCK:
                job = {k: v for k, v in CURRENT_JOB.items() if k != "proc"}
                job["logs"] = list(job.get("logs", []))[-400:]
            # Nach einem Server-Neustart: Job-Log-Datei als Konsole verwenden
            if not job["logs"]:
                try:
                    log_path = self.project_root / "output" / "qa_reports" / "job_log.txt"
                    if log_path.exists():
                        tail = log_path.read_text(encoding="utf-8", errors="replace")
                        lines = [
                            l.rstrip() for l in tail.splitlines()
                            if l.strip()
                            and "[transformers] Model config:" not in l
                            and "Fetching 9 files:" not in l
                        ]
                        job["logs"] = lines[-400:]
                except Exception:
                    pass
            job["state"] = load_live_state()
            job["ai"] = self._safe_ai_settings()
            job["usage"] = get_usage()
            job["expansion_status"] = get_expansion_status()
            self._send_json(job)
            return

        if path == "/api/settings":
            self._send_json(self._safe_ai_settings())
            return

        if path == "/api/job_status":
            with JOB_LOCK:
                job = {k: v for k, v in CURRENT_JOB.items() if k != "proc"}
                job["logs"] = list(job.get("logs", []))[-400:]
            job["state"] = load_live_state()
            self._send_json(job)
            return

        if path.startswith("/audio/en/"):
            fname = os.path.basename(path[len("/audio/en/") :])
            cfg_source = Path(self.cfg["paths"]["source_pack_dir"])
            candidates = [cfg_source / "voiceovers"]
            wow_addons = Path(os.getenv("WOW_ADDONS_DIR") or self.cfg["paths"]["wow_addons_dir"])
            if cfg_source.name:
                candidates.insert(0, wow_addons / cfg_source.name / "voiceovers")
            if wow_addons.exists():
                # Search all ENGLISH packs (names without locale suffix,
                # e.g. ChattyLittleNpc_Midnight_voiceovers — not *_deDE_*)
                for pack in sorted(wow_addons.glob("ChattyLittleNpc_*_voiceovers")):
                    if pack.name.count("_") == 2:
                        candidates.append(pack / "voiceovers")
            for en_dir in candidates:
                cand = en_dir / fname
                if cand.is_file():
                    self._send_file(cand, send_body=send_body)
                    return
            self.send_error(404, f"File not found: {fname}")
            return

        if path.startswith("/audio/de/"):
            fname = os.path.basename(path[len("/audio/de/") :])
            de_dir = self.project_root / self.cfg["paths"]["output_final_ogg"]
            self._send_file(de_dir / fname, send_body=send_body)
            return

        if path.startswith("/audio/raw/"):
            fname = os.path.basename(path[len("/audio/raw/") :])
            raw_dir = self.project_root / self.cfg["paths"]["output_raw_tts"]
            self._send_file(raw_dir / fname, send_body=send_body)
            return

        if path.startswith("/audio/seed/"):
            fname = os.path.basename(path[len("/audio/seed/") :])
            seed_dir = self.project_root / self.cfg["paths"]["data_reference_voices"]
            for sub in ("per_entry", "unique_npcs", "archetypes"):
                cand = seed_dir / sub / fname
                if cand.exists():
                    self._send_file(cand, send_body=send_body)
                    return
            self.send_error(404, "Voice seed not found")
            return

        self.send_error(404, "Not found")


def main():
    parser = argparse.ArgumentParser(description="Serve A/B Audio Comparison & Launchpad Web UI.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--host", default=None, help="Bind host")
    parser.add_argument("--port", type=int, default=None, help="Bind port")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    with open(project_root / args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Make .env (AI keys, WoW paths) available to this process and spawned jobs
    from pipeline.ai_client import load_env
    load_env(project_root / ".env")

    host = args.host or cfg.get("web", {}).get("host", "127.0.0.1")
    port = args.port or int(cfg.get("web", {}).get("port", 8765))

    ABComparisonHandler.project_root = project_root
    ABComparisonHandler.cfg = cfg

    try:
        settings = get_ai_settings(cfg)
        live_set_ai_config(
            enabled=ai_is_available(settings),
            provider=settings.get("provider"),
            model=settings.get("model") if ai_is_available(settings) else None,
        )
    except Exception:
        pass

    server = ThreadingHTTPServer((host, port), ABComparisonHandler)
    console.print(f"[bold green]🎙️  Echoes of Azeroth (DE) Studio & Launchpad running at:[/bold green] [bold cyan]http://{host}:{port}[/bold cyan]")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        console.print("\n[yellow]Shutting down web server...[/yellow]")
        server.server_close()


if __name__ == "__main__":
    main()
