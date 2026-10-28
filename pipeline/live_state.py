#!/usr/bin/env python3
"""
Live Progress State Tracker (pipeline/live_state.py)
----------------------------------------------------
Single source of truth for the Web UI while the pipeline is running.

Every pipeline step reports its progress here (atomic JSON writes to
`output/qa_reports/live_state.json`, git-ignored) so the Web Studio can show
LIVE:
  - which step is currently running (1-7),
  - WHICH quest/entry is currently being validated & its attempt/status/CER,
  - counts: processed / passed / failed / retried / AI-repaired,
  - ETA (computed from the measured per-item throughput),
  - every external AI call (provider, model, entry, result).

Multiple processes (web server + pipeline steps) may write concurrently —
all writes are atomic (tmp + replace) and lock-protected.
"""

import json
import os
import threading
import time
from pathlib import Path

_LOCK = threading.RLock()

STEP_DEFS = {
    "analyze": {"index": 1, "label": "Analyse & Scraping"},
    "voices":  {"index": 2, "label": "Voice-Seed Extraktion"},
    "script":  {"index": 3, "label": "Script & Phonetik"},
    "tts":     {"index": 4, "label": "KI-Sprachsynthese"},
    "qa":      {"index": 5, "label": "Whisper-Validierung"},
    "fx":      {"index": 6, "label": "Mastering & Export"},
    "audit":   {"index": 7, "label": "Quality Audit"},
}
TOTAL_STEPS = len(STEP_DEFS)


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def state_file() -> Path:
    env_path = os.environ.get("LIVE_STATE_FILE")
    if env_path:
        return Path(env_path)
    return project_root() / "output" / "qa_reports" / "live_state.json"


def expansion_status_file() -> Path:
    return project_root() / "output" / "qa_reports" / "expansion_status.json"


def usage_file() -> Path:
    return project_root() / "output" / "qa_reports" / "ai_usage.json"


def topup_ack_file() -> Path:
    return project_root() / "output" / "qa_reports" / "topup_ack.json"


def topup_acked() -> bool:
    """True, sobald der Nutzer das Guthaben-TopUp bestätigt hat (Marker-Datei)."""
    return topup_ack_file().exists()


def ack_topup() -> dict:
    """Setzt den TopUp-Bestätigungs-Marker (Web-Button oder manuell anlegbar)."""
    data = {"acked": True, "at": time.strftime("%Y-%m-%d %H:%M:%S"), "ts": time.time()}
    _write(data, topup_ack_file())
    return data


def clear_topup_ack() -> None:
    """Entfernt den Marker — wird bei jedem neuen Job-Start aufgerufen."""
    try:
        topup_ack_file().unlink()
    except OSError:
        pass


def set_paused(reason: str = "") -> dict:
    """Markiert den Job als pausiert (z. B. warten auf Guthaben-TopUp)."""
    with _LOCK:
        data = load()
        data["paused"] = True
        data["pause_reason"] = reason
        data["updated_at"] = time.time()
        _write(data)
        return data


def set_resumed() -> dict:
    with _LOCK:
        data = load()
        data["paused"] = False
        data["pause_reason"] = None
        data["updated_at"] = time.time()
        _write(data)
        return data


def _blank_usage() -> dict:
    return {
        "calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cost_usd": 0.0,
    }


def get_usage() -> dict:
    """Lifetime AI token & cost totals (persists across jobs, git-ignored)."""
    try:
        with open(usage_file(), "r", encoding="utf-8") as f:
            data = json.load(f)
        base = _blank_usage()
        base.update(data)
        return base
    except Exception:
        return _blank_usage()


def add_usage(prompt_tokens: int = 0, completion_tokens: int = 0, cost_usd: float = 0.0, calls: int = 0) -> dict:
    """Accumulates token/cost totals into ai_usage.json (atomic write)."""
    with _LOCK:
        data = get_usage()
        data["calls"] += int(calls)
        data["prompt_tokens"] += int(prompt_tokens)
        data["completion_tokens"] += int(completion_tokens)
        data["total_tokens"] += int(prompt_tokens) + int(completion_tokens)
        data["cost_usd"] = round(float(data["cost_usd"]) + float(cost_usd), 6)
        path = usage_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        tmp.replace(path)
        return data


def _blank() -> dict:
    return {
        "job_id": None,
        "mode": None,
        "expansion": None,
        "step": None,
        "total_steps": TOTAL_STEPS,
        "started_at": None,
        "updated_at": None,
        "finished_at": None,
        "ok": None,
        "current": None,
        "counts": {
            "total": 0,
            "processed": 0,
            "passed": 0,
            "failed": 0,
            "retried": 0,
            "ai_repaired": 0,
        },
        "eta_seconds": None,
        "paused": False,
        "pause_reason": None,
        "ai": {
            "enabled": False,
            "provider": None,
            "model": None,
            "calls": [],
            "tokens": {"prompt": 0, "completion": 0, "total": 0},
            "cost_usd": 0.0,
        },
    }


def load() -> dict:
    try:
        with open(state_file(), "r", encoding="utf-8") as f:
            data = json.load(f)
        base = _blank()
        base.update(data)
        return base
    except Exception:
        return _blank()


def _deep_merge(base: dict, patch: dict) -> dict:
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def _write(data: dict, path: Path = None) -> None:
    path = path or state_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Eindeutiger Temp-Name: mehrere Prozesse (Pipeline + Watcher) schreiben
    # parallel — ein geteilter .tmp-Name führt sonst zu Replace-Crashes.
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        tmp.replace(path)
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass


def _compute_eta(data: dict) -> None:
    counts = data.get("counts", {})
    total = counts.get("total", 0)
    processed = counts.get("processed", 0)
    started = data.get("started_at")
    if not started or processed <= 0 or total <= processed:
        data["eta_seconds"] = None
        return
    try:
        elapsed = time.time() - float(started)
    except (TypeError, ValueError):
        data["eta_seconds"] = None
        return
    if elapsed <= 0:
        data["eta_seconds"] = None
        return
    rate = elapsed / max(1, processed)
    remaining = max(0, total - processed)
    data["eta_seconds"] = int(round(rate * remaining))


def update(**changes) -> dict:
    """Merges changes into the live state and writes it atomically."""
    with _LOCK:
        data = load()
        data = _deep_merge(data, changes)
        data["updated_at"] = time.time()
        if "counts" in changes:
            _compute_eta(data)
        _write(data)
        return data


def reset() -> dict:
    with _LOCK:
        data = _blank()
        data["updated_at"] = time.time()
        _write(data)
        return data


def begin_job(mode: str = None, expansion: str = None) -> dict:
    """Starts a fresh job (clears previous progress and any stale TopUp marker)."""
    with _LOCK:
        data = _blank()
        data["job_id"] = f"job-{int(time.time() * 1000)}"
        data["mode"] = mode
        data["expansion"] = expansion
        data["started_at"] = time.time()
        data["updated_at"] = time.time()
        _write(data)
    clear_topup_ack()
    return data


def begin_step(key: str, total: int = 0, mode: str = None, expansion: str = None) -> dict:
    """Marks the start of a pipeline step and resets per-step counters."""
    with _LOCK:
        data = load()
        if not data.get("started_at"):
            data = begin_job(mode=mode, expansion=expansion)
        step_def = STEP_DEFS.get(key, {"index": 0, "label": key})
        data["step"] = {
            "key": key,
            "index": step_def["index"],
            "label": step_def["label"],
            "done": False,
            "ok": None,
        }
        data["counts"] = {
            "total": int(total),
            "processed": 0,
            "passed": 0,
            "failed": 0,
            "retried": 0,
            "ai_repaired": 0,
        }
        # ETA-Basis je Schritt zurücksetzen (sonst rechnet ein alter
        # started_at aus einem früheren Lauf die Rate kaputt)
        data["started_at"] = time.time()
        data["current"] = None
        data["eta_seconds"] = None
        data["finished_at"] = None
        data["ok"] = None
        data["updated_at"] = time.time()
        _write(data)
        return data


def update_current(entry: dict) -> dict:
    """Sets the entry currently being processed (no counter changes)."""
    with _LOCK:
        data = load()
        data["current"] = entry
        data["updated_at"] = time.time()
        _write(data)
        return data


def increment(
    processed: int = 0,
    passed: int = 0,
    failed: int = 0,
    retried: int = 0,
    ai_repaired: int = 0,
    current: dict = None,
) -> dict:
    """Adds to per-step counters, recomputes ETA, optionally updates current entry."""
    with _LOCK:
        data = load()
        c = data.setdefault("counts", {})
        c["processed"] = int(c.get("processed", 0)) + int(processed)
        c["passed"] = int(c.get("passed", 0)) + int(passed)
        c["failed"] = int(c.get("failed", 0)) + int(failed)
        c["retried"] = int(c.get("retried", 0)) + int(retried)
        c["ai_repaired"] = int(c.get("ai_repaired", 0)) + int(ai_repaired)
        if current is not None:
            data["current"] = current
        _compute_eta(data)
        data["updated_at"] = time.time()
        _write(data)
        return data


def get_expansion_status() -> dict:
    """Persistenter Fortschritt je Expansion (übersteht Job-Wechsel)."""
    try:
        with open(expansion_status_file(), "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _record_expansion_status(data: dict) -> None:
    """Schreibt den Step-Abschluss in den persistenten Expansions-Status."""
    try:
        pilot = project_root() / "data" / "raw_de_text" / "midnight_de_pilot_20.json"
        if not pilot.exists():
            return
        catalog = json.loads(pilot.read_text(encoding="utf-8"))
        expansion = (catalog.get("addon_architecture") or {}).get("expansion")
        if not expansion:
            return
        step = data.get("step") or {}
        status = get_expansion_status()
        entry = status.setdefault(str(expansion), {})
        entry["step"] = step.get("key")
        entry["step_done"] = bool(step.get("done"))
        entry["updated_at"] = time.time()
        # QA-Zahlen dauerhaft behalten (Audit-Zähler überschreiben sie nicht)
        entry["step_counts"] = entry.get("step_counts") or {}
        entry["step_counts"][step.get("key") or "?"] = dict(data.get("counts") or {})
        if step.get("key") == "qa":
            entry["counts"] = dict(data.get("counts") or {})
        entry["finished"] = bool(entry.get("finished") or (step.get("key") == "audit" and step.get("done")))
        _write(status, expansion_status_file())
    except Exception:
        pass


def finish_step(ok: bool = True) -> dict:
    with _LOCK:
        data = load()
        if data.get("step"):
            data["step"] = dict(data["step"])
            data["step"]["done"] = True
            data["step"]["ok"] = bool(ok)
        data["updated_at"] = time.time()
        _write(data)
    # Expansions-Fortschritt persistieren (Expansion steht im Katalog)
    _record_expansion_status(data)
    return data


def finish_job(ok: bool = True) -> dict:
    with _LOCK:
        data = load()
        data["finished_at"] = time.time()
        data["ok"] = bool(ok)
        data["updated_at"] = time.time()
        _write(data)
        return data


def set_ai_config(enabled: bool, provider: str = None, model: str = None) -> dict:
    with _LOCK:
        data = load()
        data["ai"]["enabled"] = bool(enabled)
        data["ai"]["provider"] = provider
        data["ai"]["model"] = model
        data["updated_at"] = time.time()
        _write(data)
        return data


def log_ai_call(
    provider: str,
    model: str,
    entry_id: str,
    note: str = "",
    ok: bool = True,
    tokens: dict = None,
    cost_usd: float = 0.0,
) -> dict:
    """Logs an AI call (per-job feed) and accumulates lifetime token/cost totals."""
    tokens = tokens or {}
    prompt_t = int(tokens.get("prompt_tokens") or 0)
    completion_t = int(tokens.get("completion_tokens") or 0)
    total_t = int(tokens.get("total_tokens") or 0) or (prompt_t + completion_t)
    cost = float(cost_usd or tokens.get("cost_usd") or 0.0)

    with _LOCK:
        data = load()
        data["ai"]["enabled"] = True
        data["ai"]["provider"] = provider
        data["ai"]["model"] = model
        data["ai"]["tokens"]["prompt"] = int(data["ai"]["tokens"].get("prompt", 0)) + prompt_t
        data["ai"]["tokens"]["completion"] = int(data["ai"]["tokens"].get("completion", 0)) + completion_t
        data["ai"]["tokens"]["total"] = int(data["ai"]["tokens"].get("total", 0)) + total_t
        data["ai"]["cost_usd"] = round(float(data["ai"].get("cost_usd", 0.0)) + cost, 6)
        data["ai"]["calls"].append({
            "at": time.strftime("%H:%M:%S"),
            "ts": time.time(),
            "provider": provider,
            "model": model,
            "entry_id": entry_id,
            "note": (note or "")[:200],
            "ok": bool(ok),
            "tokens": {
                "prompt": prompt_t,
                "completion": completion_t,
                "total": total_t,
            },
            "cost_usd": round(cost, 6),
        })
        data["ai"]["calls"] = data["ai"]["calls"][-50:]
        data["updated_at"] = time.time()
        _write(data)

    add_usage(prompt_tokens=prompt_t, completion_tokens=completion_t, cost_usd=cost, calls=1)
    return data
