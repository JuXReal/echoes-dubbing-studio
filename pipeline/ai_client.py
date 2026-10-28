#!/usr/bin/env python3
"""
AI-Provider-Gateway (pipeline/ai_client.py)
-------------------------------------------
Single gateway for AI repair / optimization calls to external providers:

    - OpenAI        (api.openai.com)
    - OpenRouter    (openrouter.ai)
    - Google Gemini (generativelanguage.googleapis.com)
    - DeepSeek      (api.deepseek.com)
    - Ollama        (local, http://127.0.0.1:11434)
    - none          (disable external AI — deterministic local repair only)

SECURITY / SECRET HANDLING
--------------------------
API keys are read EXCLUSIVELY from the environment (.env — git-ignored).
Keys are NEVER written to config.yaml, JSON reports, the live state or any
web API response. The web settings UI only receives a boolean "api_key_set".
"""

import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

PROVIDERS = {
    "none":       {"label": "Keine externe KI (nur lokale Regeln)", "base_url": "",                                              "default_model": ""},
    "openai":     {"label": "OpenAI",                             "base_url": "https://api.openai.com/v1",                       "default_model": "gpt-4o"},
    "openrouter": {"label": "OpenRouter",                         "base_url": "https://openrouter.ai/api/v1",                    "default_model": "anthropic/claude-3.5-sonnet"},
    "google":     {"label": "Google Gemini",                      "base_url": "https://generativelanguage.googleapis.com/v1beta", "default_model": "gemini-2.0-flash"},
    "deepseek":   {"label": "DeepSeek",                           "base_url": "https://api.deepseek.com/v1",                     "default_model": "deepseek-chat"},
    "ollama":     {"label": "Lokales Ollama",                     "base_url": "http://127.0.0.1:11434",                          "default_model": "qwen2.5:7b"},
}

# Estimated list prices in USD per 1M tokens (overridable via config.yaml ai.pricing).
# OpenRouter varies per model — its actual "cost" from the API response wins if present.
DEFAULT_PRICING_USD_PER_1M = {
    "none":       {"input": 0.0,  "output": 0.0},
    "openai":     {"input": 2.50, "output": 10.00},
    "openrouter": {"input": 0.80, "output": 3.50},
    "google":     {"input": 0.10, "output": 0.40},
    "deepseek":   {"input": 0.27, "output": 1.10},
    "ollama":     {"input": 0.0,  "output": 0.0},
}

REPAIR_SYSTEM_PROMPT = (
    "Du bist der QA-Reparatur-Agent einer World-of-Warcraft-Voiceover-Pipeline "
    "(Coqui XTTS-v2, deutsche Sprache). Ein gesprochener Take wurde mit Faster-Whisper "
    "transkribiert und weicht vom Zieltext ab. Erstelle daraus eine optimierte, phonetisch "
    "robuste deutsche TTS-Vorlage:\n"
    "- Warcraft-Eigennamen lautgetreu eingedeutscht und ohne Apostrophe schreiben "
    "(z. B. \"Xal'atath\" -> \"Xallatath\", \"Lor'themar\" -> \"Lorthemar\").\n"
    "- Lange Komposita mit Leerzeichen trennen (z. B. \"Zwielichtklingenkultisten\" -> "
    "\"Zwielichtklingen Kultisten\").\n"
    "- Kurze, klare Sätze mit natürlichen Atempausen durch Kommas und Punkte; keine "
    "Semikolons, keine Doppelpunkte.\n"
    "- Keine Anführungszeichen und keine ALL-CAPS-Wörter verwenden.\n"
    "Antworte NUR mit der fertigen TTS-Vorlage als Fließtext, ohne Kommentare, ohne Markdown."
)


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_env(env_file: Optional[Path] = None) -> None:
    """Loads .env into os.environ without overwriting already-set variables."""
    env_file = env_file or (project_root() / ".env")
    if not env_file.exists():
        return
    try:
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v
    except Exception:
        pass


def get_settings(cfg: Optional[Dict] = None) -> Dict:
    """Resolves the active AI settings from .env + config.yaml (no secrets in config!)."""
    load_env()
    ai_cfg = (cfg or {}).get("ai", {})

    provider = (os.getenv("AI_PROVIDER") or ai_cfg.get("provider") or "none").strip().lower()
    api_key = os.getenv("AI_API_KEY", "").strip()
    base_url = (
        os.getenv("AI_BASE_URL")
        or ai_cfg.get("base_url")
        or PROVIDERS.get(provider, {}).get("base_url", "")
    ).rstrip("/")
    model = os.getenv("AI_MODEL") or ai_cfg.get("model") or PROVIDERS.get(provider, {}).get("default_model", "")

    enabled = ai_cfg.get("enabled", True)
    env_enabled = os.getenv("AI_ENABLED")
    if env_enabled is not None:
        enabled = env_enabled.strip().lower() in ("1", "true", "yes", "on")

    # Backwards-compatible fallback for the legacy .env keys
    if provider == "none" or not api_key:
        if os.getenv("OPENAI_API_KEY"):
            provider, api_key = "openai", os.getenv("OPENAI_API_KEY", "").strip()
            base_url = base_url or "https://api.openai.com/v1"
            model = model or "gpt-4o"
        elif os.getenv("GEMINI_API_KEY"):
            provider, api_key = "google", os.getenv("GEMINI_API_KEY", "").strip()
            base_url = base_url or "https://generativelanguage.googleapis.com/v1beta"
            model = model or "gemini-2.0-flash"

    if provider == "ollama":
        base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
        model = model or os.getenv("OLLAMA_MODEL", "qwen2.5:7b")

    if provider not in PROVIDERS:
        provider = "none"

    try:
        trigger = int(os.getenv("AI_REPAIR_AFTER_FAILURES") or ai_cfg.get("trigger_after_failures", 3))
    except (TypeError, ValueError):
        trigger = 3

    # Per-provider pricing (USD per 1M tokens), overridable in config.yaml ai.pricing
    pricing = {}
    pricing_cfg = ai_cfg.get("pricing") or {}
    for prov, defaults in DEFAULT_PRICING_USD_PER_1M.items():
        override = pricing_cfg.get(prov) or {}
        try:
            pricing[prov] = {
                "input": float(override.get("input", defaults["input"])),
                "output": float(override.get("output", defaults["output"])),
            }
        except (TypeError, ValueError):
            pricing[prov] = dict(defaults)

    return {
        "provider": provider,
        "api_key": api_key,
        "base_url": base_url,
        "model": model,
        "enabled": bool(enabled),
        "trigger_after_failures": max(1, trigger),
        "temperature": float(ai_cfg.get("temperature", 0.3)),
        "timeout_sec": int(ai_cfg.get("timeout_sec", 45)),
        "max_tokens": int(ai_cfg.get("max_tokens", 1500)),
        "pricing": pricing,
    }


def is_available(settings: Dict) -> bool:
    if not settings.get("enabled") or settings.get("provider") in (None, "none"):
        return False
    if settings.get("provider") == "ollama":
        return bool(settings.get("base_url"))
    return bool(settings.get("api_key"))


def _estimate_cost(settings: Dict, prompt_t: int, completion_t: int) -> float:
    """Computes an estimated cost in USD from the configured pricing table."""
    pricing = (settings.get("pricing") or {}).get(settings.get("provider") or "", {})
    p_in = float(pricing.get("input", 0.0))
    p_out = float(pricing.get("output", 0.0))
    return round((prompt_t * p_in + completion_t * p_out) / 1_000_000.0, 6)


def _chat_openai_compatible(messages: List[Dict], s: Dict, temperature: float, timeout: int, max_tokens: int) -> Tuple[str, Dict]:
    resp = requests.post(
        f"{s['base_url']}/chat/completions",
        headers={"Authorization": f"Bearer {s['api_key']}", "Content-Type": "application/json"},
        json={
            "model": s["model"],
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    message = data["choices"][0]["message"] or {}
    text = (message.get("content") or "").strip()
    reasoning = (message.get("reasoning_content") or "").strip()
    usage_raw = data.get("usage") or {}
    prompt_t = int(usage_raw.get("prompt_tokens") or 0)
    completion_t = int(usage_raw.get("completion_tokens") or 0)
    total_t = int(usage_raw.get("total_tokens") or 0) or (prompt_t + completion_t)
    # OpenRouter returns the real cost; otherwise estimate from the pricing table
    cost = float(usage_raw.get("cost") or data.get("total_cost") or _estimate_cost(s, prompt_t, completion_t))
    return text, {
        "prompt_tokens": prompt_t,
        "completion_tokens": completion_t,
        "total_tokens": total_t,
        "cost_usd": round(cost, 6),
        # Reasoning models (e.g. deepseek-reasoner) sometimes spend the whole
        # output budget on reasoning and return an empty content field.
        "empty_content": bool(reasoning and not text),
    }


def _chat_google(messages: List[Dict], s: Dict, temperature: float, timeout: int, max_tokens: int) -> Tuple[str, Dict]:
    system = next((m["content"] for m in messages if m["role"] == "system"), "")
    user = "\n\n".join(m["content"] for m in messages if m["role"] == "user")
    text = f"{system}\n\n{user}" if system else user
    resp = requests.post(
        f"{s['base_url']}/models/{s['model']}:generateContent",
        params={"key": s["api_key"]},
        json={
            "contents": [{"role": "user", "parts": [{"text": text}]}],
            "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens},
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    reply = data["candidates"][0]["content"]["parts"][0]["text"].strip()
    meta = data.get("usageMetadata") or {}
    prompt_t = int(meta.get("promptTokenCount") or 0)
    completion_t = int(meta.get("candidatesTokenCount") or 0)
    total_t = int(meta.get("totalTokenCount") or 0) or (prompt_t + completion_t)
    return reply, {
        "prompt_tokens": prompt_t,
        "completion_tokens": completion_t,
        "total_tokens": total_t,
        "cost_usd": _estimate_cost(s, prompt_t, completion_t),
    }


def _chat_ollama(messages: List[Dict], s: Dict, temperature: float, timeout: int) -> Tuple[str, Dict]:
    resp = requests.post(
        f"{s['base_url']}/api/chat",
        json={"model": s["model"], "messages": messages, "stream": False,
              "options": {"temperature": temperature}},
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    reply = data["message"]["content"].strip()
    prompt_t = int(data.get("prompt_eval_count") or 0)
    completion_t = int(data.get("eval_count") or 0)
    return reply, {
        "prompt_tokens": prompt_t,
        "completion_tokens": completion_t,
        "total_tokens": prompt_t + completion_t,
        "cost_usd": 0.0,
    }


def chat_with_usage(
    messages: List[Dict],
    settings: Dict,
    temperature: Optional[float] = None,
    timeout: Optional[int] = None,
    max_tokens: Optional[int] = None,
) -> Tuple[str, Dict]:
    """Sends a chat request; returns (reply_text, usage_dict). Raises on failure."""
    provider = settings.get("provider")
    if provider in (None, "none"):
        raise RuntimeError("No AI provider configured")
    temp = settings.get("temperature", 0.3) if temperature is None else temperature
    t_out = timeout or settings.get("timeout_sec", 45)
    t_max = max_tokens or settings.get("max_tokens", 1500)

    if provider == "google":
        return _chat_google(messages, settings, temp, t_out, t_max)
    if provider == "ollama" and "/v1" not in (settings.get("base_url") or ""):
        return _chat_ollama(messages, settings, temp, t_out)
    return _chat_openai_compatible(messages, settings, temp, t_out, t_max)


def chat(
    messages: List[Dict],
    settings: Dict,
    temperature: Optional[float] = None,
    timeout: Optional[int] = None,
    max_tokens: Optional[int] = None,
) -> str:
    """Convenience wrapper returning only the reply text."""
    text, _ = chat_with_usage(messages, settings, temperature, timeout, max_tokens)
    return text


def test_connection(settings: Dict) -> Tuple[bool, str]:
    """Checks the configured AI connection with a minimal ping request."""
    if not settings.get("enabled"):
        return False, "KI-Reparatur ist deaktiviert."
    if not is_available(settings):
        if settings.get("provider") in (None, "none"):
            return False, "Kein KI-Provider ausgewählt."
        return False, "API-Key fehlt (oder Base-URL leer bei Ollama)."
    try:
        reply, usage = chat_with_usage(
            [{"role": "system", "content": "Antworte exakt mit: OK"},
             {"role": "user", "content": "Ping"}],
            settings, temperature=0, timeout=15, max_tokens=8,
        )
        if reply and reply.strip():
            tokens = usage.get("total_tokens", 0)
            return True, f"Verbindung OK — Antwort: {reply.strip()[:60]} ({tokens} Tokens)"
        return False, "Leere Antwort vom Provider erhalten."
    except Exception as e:
        msg = f"Verbindungsfehler: {e}"
        if "404" in str(e):
            msg += " — Modell oder Base-URL prüfen ('Modelle laden' verwenden)."
        return False, msg


TRANSLATE_SYSTEM_PROMPT = (
    "Du bist Übersetzer für World-of-Warcraft-Questtexte. Übersetze den folgenden "
    "englischen Text in natürliches, idiomatisches Deutsch. Eigennamen und Ortsnamen "
    "beibehalten (z. B. 'Lor'themar', 'Silvermoon', \"Zul'Aman\"). Zahlen ausschreiben "
    "ist NICHT nötig. Antworte NUR mit der deutschen Übersetzung, ohne Kommentare."
)

# HTTP-/Fehler-Signaturen, die auf leeres Guthaben/Kontingent hindeuten.
BALANCE_EXHAUSTED_MARKERS = (
    "402",
    "insufficient balance",
    "payment required",
    "not enough quota",
    "no enough quota",
    "quota exceeded",
    "billing",
)


def is_balance_exhausted(note: str) -> bool:
    """Erkennt anhand der Fehler-/Statusmeldung, ob das Provider-Guthaben leer ist."""
    if not note:
        return False
    low = str(note).lower()
    return any(m in low for m in BALANCE_EXHAUSTED_MARKERS)


def wait_for_topup_ack(reason: str = "", poll_seconds: int = 60) -> None:
    """
    Blockiert den Job, bis der Nutzer das Guthaben-TopUp bestätigt hat.

    Die Bestätigung erfolgt über den Web-Studio-Button „TopUp erledigt"
    (POST /api/ack_topup) oder manuell durch Anlegen der Marker-Datei
    output/qa_reports/topup_ack.json (z. B. `touch`). Danach wird der
    fehlgeschlagene Aufruf vom Aufrufer nahtlos wiederholt.
    """
    from pipeline.live_state import set_paused, set_resumed, topup_acked

    set_paused(reason or "deepseek_balance_empty")
    print(
        "\n[AI] DeepSeek-Guthaben erschöpft (402 Insufficient Balance).\n"
        "[AI] Der Job pausiert hier und wartet auf Deine Bestätigung.\n"
        "[AI] Sobald Du aufgeladen hast: im Web-Studio auf „TopUp erledigt\" klicken\n"
        f"[AI] (oder Marker-Datei anlegen). Geprüft wird alle {poll_seconds} s — es geht automatisch weiter.",
        flush=True,
    )
    try:
        while not topup_acked():
            time.sleep(poll_seconds)
    except KeyboardInterrupt:
        print("[AI] Abbruch durch Nutzer während TopUp-Warteschleife.", flush=True)
        raise
    set_resumed()
    print("[AI] TopUp bestätigt — Übersetzung wird nahtlos fortgesetzt.", flush=True)


def translate_to_german(settings: Dict, text: str) -> Tuple[Optional[str], str, Dict]:
    """Übersetzt einen englischen Quest-/Gossip-Text ins Deutsche.
    Returns (translation, note, usage); (None, reason, usage) bei Fehlern.
    Bei leerem Guthaben wartet die Funktion auf die TopUp-Bestätigung und
    wiederholt dann denselben Text automatisch."""
    if not is_available(settings):
        return None, "ai_disabled", {}
    while True:
        try:
            raw, usage = chat_with_usage(
                [{"role": "system", "content": TRANSLATE_SYSTEM_PROMPT},
                 {"role": "user", "content": text}],
                settings,
            )
        except Exception as e:
            note = f"call_failed: {e}"
            if is_balance_exhausted(note):
                wait_for_topup_ack(reason="deepseek_balance_empty")
                continue
            return None, note, {}
        break
    cleaned = re.sub(r"```[a-zA-Z]*\n?", "", raw or "").strip().strip('"').strip()
    if not cleaned or len(cleaned) < 2:
        if usage.get("empty_content"):
            return None, "empty_reply (Reasoning-Modell — Chat-Modell wie 'deepseek-chat' nutzen)", usage
        return None, "empty_reply", usage
    return cleaned, "translated", usage


def repair_script_with_ai(
    settings: Dict,
    target_text: str,
    transcript: str,
    entry_context: str = "",
) -> Tuple[Optional[str], str, Dict]:
    """
    KI-Reparatur: lässt die konfigurierte KI die fehlgeschlagene Zeile in ein
    phonetically robust German TTS script.
    Returns (repaired_text, note, usage) — usage is {} when the call failed.
    Returns (None, reason, {}) when the call fails or the reply is unusable.
    """
    if not is_available(settings):
        return None, "ai_disabled", {}
    user = f"Zieltext:\n{target_text}\n\nWhisper-Transkript (fehlerhafter Take):\n{transcript}"
    if entry_context:
        user = f"Kontext: {entry_context}\n\n" + user
    while True:
        try:
            raw, usage = chat_with_usage(
                [{"role": "system", "content": REPAIR_SYSTEM_PROMPT},
                 {"role": "user", "content": user}],
                settings,
            )
        except Exception as e:
            note = f"call_failed: {e}"
            if is_balance_exhausted(note):
                wait_for_topup_ack(reason="deepseek_balance_empty")
                continue
            return None, note, {}
        break

    cleaned = re.sub(r"```[a-zA-Z]*\n?", "", raw or "").strip().strip("`").strip()
    if not cleaned:
        if usage.get("empty_content"):
            return None, "empty_reply (Reasoning-Modell: nur Gedankengänge, keine Antwort — Chat-Modell wie 'deepseek-chat' nutzen)", usage
        return None, "empty_reply", usage
    if len(cleaned) < 10 or len(cleaned) > 4000:
        return None, "suspicious_reply_length", usage
    return cleaned, "ai_phonetic_repair", usage


def _openrouter_pricing_to_per_1m(entry: Dict) -> Optional[Dict]:
    """Converts OpenRouter's per-token pricing strings into USD per 1M tokens."""
    p = entry.get("pricing") or {}
    try:
        return {
            "input": round(float(p.get("prompt") or 0) * 1_000_000, 4),
            "output": round(float(p.get("completion") or 0) * 1_000_000, 4),
        }
    except (TypeError, ValueError):
        return None


def list_models(settings: Dict) -> Tuple[bool, List[Dict], str]:
    """
    Fetches the model catalog of the configured provider.
    Returns (ok, models, message); models = [{"id": str, "pricing": {input, output}|None}].
    """
    provider = settings.get("provider")
    if provider in (None, "none"):
        return False, [], "Kein Provider ausgewählt."
    if provider == "ollama":
        if not settings.get("base_url"):
            return False, [], "Ollama Base-URL fehlt."
        try:
            resp = requests.get(f"{settings['base_url']}/api/tags", timeout=settings.get("timeout_sec", 45))
            resp.raise_for_status()
            models = [{"id": m.get("name", ""), "pricing": None} for m in (resp.json().get("models") or [])]
            models = [m for m in models if m["id"]]
            return (True, models, f"{len(models)} Modelle von Ollama geladen.") if models else (False, [], "Ollama meldet keine installierten Modelle.")
        except Exception as e:
            return False, [], f"Fehler beim Laden der Modelle: {e}"
    if provider == "google":
        if not settings.get("api_key"):
            return False, [], "API-Key fehlt."
        try:
            resp = requests.get(
                f"{settings['base_url']}/models",
                params={"key": settings["api_key"], "pageSize": 200},
                timeout=settings.get("timeout_sec", 45),
            )
            resp.raise_for_status()
            out = []
            for m in resp.json().get("models") or []:
                methods = m.get("supportedGenerationMethods") or []
                if "generateContent" not in methods:
                    continue
                name = m.get("name") or ""
                if "gemini" not in name.lower():
                    continue
                out.append({"id": name.split("/", 1)[-1], "pricing": None})
            return (True, out, f"{len(out)} Modelle von Google Gemini geladen.") if out else (False, [], "Keine Gemini-Modelle gefunden.")
        except Exception as e:
            return False, [], f"Fehler beim Laden der Modelle: {e}"
    # OpenAI-compatible (openai, deepseek, openrouter, custom /v1 endpoints)
    if not settings.get("api_key"):
        return False, [], "API-Key fehlt."
    try:
        resp = requests.get(
            f"{settings['base_url']}/models",
            headers={"Authorization": f"Bearer {settings['api_key']}"},
            timeout=settings.get("timeout_sec", 45),
        )
        resp.raise_for_status()
        data = resp.json().get("data") or []
        models = []
        for m in data:
            entry = {"id": m.get("id") or "", "pricing": None}
            if provider == "openrouter":
                entry["pricing"] = _openrouter_pricing_to_per_1m(m)
            if entry["id"]:
                models.append(entry)
        if not models:
            return False, [], "Der Provider meldet keine Modelle."
        return True, models, f"{len(models)} Modelle geladen."
    except Exception as e:
        hint = ""
        if "404" in str(e):
            hint = " — Prüfe Base-URL und Modell (bei DeepSeek z. B. 'deepseek-chat')."
        return False, [], f"Fehler beim Laden der Modelle: {e}{hint}"
