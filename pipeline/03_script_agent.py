#!/usr/bin/env python3
"""
Step 3: Skript-Aufbereitung (03_script_agent.py)
------------------------------------------------
Stufe 1 — Lokale Normalisierung (regelbasiert):
  - Zahlen (num2words de), Abkürzungen und Dialektkontraktionen
    (Amani-Apostrophe wie "is'", "nich'", "helf'n", "aus'm") in sauberes
    gesprochenes Deutsch auflösen, damit XTTS-v2 nicht buchstabiert.

Stufe 2 — Regelbasiertes Scripting:
  - Satzlängen für XTTS-v2 begrenzen (< 180 Zeichen), Pausen setzen,
    Prosodie-Hinweise (Tempo/Temperatur) ableiten und Warcraft-Eigennamen
    phonetisch umschreiben (z. B. "Kwell Thalas", "Sul Aman", "Srohl").
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
import yaml
from dotenv import load_dotenv
from num2words import num2words
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.ai_client import get_settings as get_ai_settings, is_available as ai_is_available
from pipeline.live_state import begin_step, finish_step, increment, set_ai_config, update_current

console = Console()


# ==============================================================================
# STAGE 1: Small / Local Model — Routine Text & Dialect Normalization
# ==============================================================================
DIALECT_EXPANSIONS_DE = [
    # Common German / Amani Troll colloquial contractions in Midnight quests
    # Expanded into clean phonetic German words so XTTS-v2 pronounces them fluidly
    # without pausing at apostrophes or pronouncing isolated 'n' as "En".
    (r"(?<!\w)is'(?!\w)", "ist"),
    (r"(?<!\w)nich'(?!\w)", "nicht"),
    (r"(?<!\w)un'(?!\w)", "und"),
    (r"(?<!\w)sin'(?!\w)", "sind"),
    (r"(?<!\w)ham(?!\w)", "haben"),
    (r"(?<!\w)wär's(?!\w)", "wäre es"),
    (r"(?<!\w)zeig's(?!\w)", "zeige es"),
    (r"(?<!\w)sein's(?!\w)", "seien es"),
    (r"(?<!\w)aus'm(?!\w)", "aus dem"),
    (r"(?<!\w)in'n(?!\w)", "in den"),
    (r"(?<!\w)auf'm(?!\w)", "auf dem"),
    (r"(?<!\w)'ne(?!\w)", "eine"),
    (r"(?<!\w)'nen(?!\w)", "einen"),
    (r"(?<!\w)'n(?!\w)", "ein"),
    (r"(?<!\w)'was(?!\w)", "etwas"),
    (r"(?<!\w)helf'n(?!\w)", "helfen"),
    (r"(?<!\w)geh'n(?!\w)", "gehen"),
    (r"(?<!\w)sag'n(?!\w)", "sagen"),
    (r"(?<!\w)erhalt'n(?!\w)", "erhalten"),
    (r"(?<!\w)halt'n(?!\w)", "halten"),
    (r"(?<!\w)zahlreich'n(?!\w)", "zahlreichen"),
    (r"(?<!\w)Gab'n(?!\w)", "Gaben"),
    (r"(?<!\w)geb'n(?!\w)", "geben"),
    (r"(?<!\w)erst'n(?!\w)", "ersten"),
    (r"(?<!\w)verlor'n(?!\w)", "verloren"),
    (r"(?<!\w)anfang'n(?!\w)", "anfangen"),
    (r"(?<!\w)belast'n(?!\w)", "belasten"),
    (r"(?<!\w)jemand'n(?!\w)", "jemanden"),
    (r"(?<!\w)schick'n(?!\w)", "schicken"),
    (r"(?<!\w)fehl'n(?!\w)", "fehlen"),
    (r"(?<!\w)dies'n(?!\w)", "diesen"),
    (r"(?<!\w)kriech'n(?!\w)", "kriechen"),
    (r"(?<!\w)erblick'n(?!\w)", "erblicken"),
    (r"(?<!\w)ganz'n(?!\w)", "ganzen"),
    (r"(?<!\w)entgegenwerf'n(?!\w)", "entgegenwerfen"),
    (r"(?<!\w)groß'n(?!\w)", "großen"),
    (r"(?<!\w)bös'n(?!\w)", "bösen"),
    (r"(?<!\w)dazwisch'n(?!\w)", "dazwischen"),
    (r"(?<!\w)unsre(?!\w)", "unsere"),
    (r"(?<!\w)drauß'n(?!\w)", "draußen"),
    (r"(?<!\w)gestoß'n(?!\w)", "gestoßen"),
    (r"(?<!\w)geseh'n(?!\w)", "gesehen"),
    (r"(?<!\w)schrecklich'n(?!\w)", "schrecklichen"),
    (r"(?<!\w)sterb'n(?!\w)", "sterben"),
    (r"(?<!\w)aufgehalt'n(?!\w)", "aufgehalten"),
    (r"(?<!\w)werd'(?!\w)", "werde"),
    (r"(?<!\w)würd(?!\w)", "würde"),
    (r"(?<!\w)Käpt'n(?!\w)", "Käpten"),
]

ABBREVIATIONS_DE = {
    r"\bz\.\s*B\.": "zum Beispiel",
    r"\bd\.\s*h\.": "das heißt",
    r"\bu\.\s*a\.": "unter anderem",
    r"\bbzw\.": "beziehungsweise",
    r"\bSt\.\b": "Sankt",
}


def tier1_routine_normalize(text: str) -> str:
    """
    Stage 1 (Routine Agent):
    - Strips remaining stage directions / UI tags
    - Expands abbreviations and numbers to German words
    - Normalizes apostrophe-heavy dialect forms into clean spoken German
    - Normalizes punctuation spacing.
    """
    out = text.strip()
    # Replace double quotes with nothing (TTS sometimes glitches on quotation marks)
    out = out.replace('"', "").replace("„", "").replace("“", "")

    # Expand abbreviations
    for pat, repl in ABBREVIATIONS_DE.items():
        out = re.sub(pat, repl, out, flags=re.IGNORECASE)

    # Expand dialect contractions so TTS doesn't spell out "'n"
    for pat, repl in DIALECT_EXPANSIONS_DE:
        out = re.sub(pat, repl, out)

    # Convert standalone numbers to German words
    def _num_repl(match):
        try:
            val = int(match.group(0))
            return num2words(val, lang="de")
        except Exception:
            return match.group(0)

    out = re.sub(r"\b\d+\b", _num_repl, out)
    out = re.sub(r"\s+", " ", out).strip()
    return out


def normalize_punctuation(text: str) -> str:
    """
    Satzzeichen für XTTS-v2 glätten: "....." oder "…" lassen das Modell sonst
    "Punkt" vorlesen oder in Wiederholungsschleifen geraten.
    """
    out = text.replace("…", "...")
    out = re.sub(r"\.{4,}", "...", out)
    # Ellipse mitten im Satz (gefolgt von Kleinbuchstabe) -> Sprechpause
    out = re.sub(r"\.{2,}\s*(?=[a-zäöüß])", ", ", out)
    # Restliche Punktfolgen -> ein Satzpunkt
    out = re.sub(r"\.{2,}", ". ", out)
    # Doppelte Ausrufe-/Fragezeichen -> einzeln
    out = re.sub(r"!{2,}", "! ", out)
    out = re.sub(r"\?{2,}", "? ", out)
    # Trennzeichen, die XTTS aussprechen könnte
    out = out.replace("·", ", ")
    # Gedankenstriche -> Sprechpause (XTTS voziert sie sonst teils als "Bindestrich")
    out = out.replace("—", ", ").replace("–", ", ")
    # Keine Leerzeichen vor Satzzeichen
    out = re.sub(r"\s+([,.;:!?])", r"\1", out)
    # Verwaiste einzelne Punkte entfernen (z. B. " . " oder ".." Reste)
    out = re.sub(r"(?<!\w)\.(?!\w)", "", out)
    out = re.sub(r"\s{2,}", " ", out).strip()
    return out


# ==============================================================================
# STAGE 2: Regelbasiertes Scripting — Pausen & Warcraft-Phonetik
# ==============================================================================
def apply_phonetic_lexicon(text: str, lexicon: Dict[str, str]) -> Tuple[str, List[str]]:
    """
    Replaces complex Warcraft proper nouns (including genitive forms like Xal'ataths
    or Ula'teks) with smooth German phonetic spellings without hard hyphens.
    """
    applied = []
    # Sort keys by length descending so longer names ("Lor'themar Theron") match first
    for src, target in sorted(lexicon.items(), key=lambda x: -len(x[0])):
        # Check genitive 's' first
        gen_pattern = re.compile(re.escape(src) + r"s\b", flags=re.IGNORECASE)
        if gen_pattern.search(text):
            gen_target = target + ("s" if not target.lower().endswith(("s", "x", "z")) else "")
            text = gen_pattern.sub(gen_target, text)
            applied.append(f"{src}s -> {gen_target}")

        base_pattern = re.compile(re.escape(src) + r"\b", flags=re.IGNORECASE)
        if base_pattern.search(text):
            text = base_pattern.sub(target, text)
            applied.append(f"{src} -> {target}")

    return text, applied


def infer_emotion_and_prosody(entry: Dict, cfg: Optional[Dict] = None) -> Dict:
    """
    Ermittelt Sprechparameter (Tempo, Temperatur) aus Archetyp, Regieanweisungen
    und Stimmhöhe des Voice-Seeds.
    """
    cues = " ".join(entry.get("stage_directions", [])).lower()
    text = entry.get("clean_de_text", "").lower()
    arch = entry.get("voice_archetype", "")

    # Default prosody
    prosody = {
        "emotion": "neutral_authoritative",
        "tts_temperature": 0.62,
        "tts_speed": 1.02,
        "acting_note": "Klare, natürliche und charaktergetreue Betonung.",
    }

    if any(k in cues for k in ("zorn", "verhärtet", "wütend", "schreit", "brüllt", "brodelt", "wut")) or "falle" in text or "blind" in text:
        prosody = {
            "emotion": "grim_determined_anger",
            "tts_temperature": 0.65,
            "tts_speed": 1.00,
            "acting_note": "Angespannt, entschlossen, leicht grollender Unterton mit markanten Pausen.",
        }
    elif any(k in cues for k in ("duckt sich", "zittert", "flüstert", "ängstlich", "fleht", "verzweifelt")) or "tut mir nicht weh" in text or "kreatur wie mich" in text:
        prosody = {
            "emotion": "fearful_pleading_wretch",
            "tts_temperature": 0.68,
            "tts_speed": 0.98,
            "acting_note": "Flehend, nervös zögernd, geheimnisvoll flüsternd.",
        }
    elif "lippen" in cues or "dekadent" in text or "korrosion" in text or arch == "lore_erinye":
        prosody = {
            "emotion": "dark_mystical_alluring",
            "tts_temperature": 0.64,
            "tts_speed": 0.98,
            "acting_note": "Düster, langsam, verführerisch-unheimlich (Geisthexe von Atal'Utek).",
        }
    elif any(k in cues for k in ("lächelt", "freundlich", "sanft", "herzlich", "lacht")):
        prosody = {
            "emotion": "warm_friendly",
            "tts_temperature": 0.60,
            "tts_speed": 0.99,
            "acting_note": "Warm, freundlich, leicht gelöst.",
        }
    elif arch == "lore_lorthemar_theron":
        prosody = {
            "emotion": "regal_commanding",
            "tts_temperature": 0.60,
            "tts_speed": 1.00,
            "acting_note": "Hoheitsvoll, ruhig, staatsmännisch und tief (Lordregent von Silbermond).",
        }
    elif arch == "lore_magister_umbric":
        prosody = {
            "emotion": "analytical_concerned",
            "tts_temperature": 0.60,
            "tts_speed": 1.02,
            "acting_note": "Bedacht, gelehrt, besorgt über die Leeren-Anomalien.",
        }
    elif arch == "lore_abdumati":
        prosody = {
            "emotion": "battle_hardened_commander",
            "tts_temperature": 0.64,
            "tts_speed": 1.01,
            "acting_note": "Rau, direkt, kampferprobter Amani-Kriegsanführer.",
        }

    # Hohe Stimmlagen klont XTTS-v2 instabiler (z. B. Alleria/Vereesa):
    # konservativere Parameter für stabilere, wiederholungsärmere Synthese.
    tts_cfg = (cfg or {}).get("tts", {})
    try:
        threshold = float(tts_cfg.get("high_voice_f0_threshold_hz", 240.0))
    except (TypeError, ValueError):
        threshold = 240.0
    f0 = entry.get("seed_mean_f0_hz")
    if f0 and float(f0) > threshold:
        prosody["tts_temperature"] = min(
            prosody.get("tts_temperature", 0.62),
            float(tts_cfg.get("high_voice_temperature", 0.55)),
        )
        prosody["tts_speed"] = float(tts_cfg.get("high_voice_speed", 0.98))
        prosody["high_voice"] = True
        prosody["acting_note"] = (prosody.get("acting_note", "") + " Hohe Stimmlage: stabilere Synthese.").strip()

    return prosody


def stage2_frontier_dramaturgy(text_tier1: str, entry: Dict, lexicon: Dict[str, str], cfg: Optional[Dict] = None) -> Tuple[str, str, List[str], Dict]:
    """
    Stufe 2 (regelbasiertes Scripting):
    1. Erzeugt `qa_reference_text`: sauberer deutscher Text mit Standard-
       Satzzeichen für den Whisper-WER-Vergleich (inkl. phonetischer Namen).
    2. Erzeugt `tts_script_text`: für Coqui XTTS-v2 optimierter Text —
       saubere Satzgrenzen (je Satz <= 180 Zeichen wegen des Token-Limits),
       natürliche Atempausen via Kommas/Punkte, phonetische Warcraft-Namen.
    """
    # Satzzeichen glätten ("....." / "…" lassen XTTS sonst "Punkt" vorlesen)
    staged = normalize_punctuation(text_tier1)

    # Apply Warcraft phonetic lexicon
    phonetic_text, applied_phonetics = apply_phonetic_lexicon(staged, lexicon)

    # Split overly long sentences (> 180 chars) at natural conjunctions (", aber ", ", doch ", ", und ", ", damit ")
    sentences = re.split(r"(?<=[.!?])\s+", phonetic_text)
    refined_sentences = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if len(s) > 175:
            # Split at a natural middle clause boundary for optimal XTTS prosody & breathing
            parts = re.split(r",\s+(?=(?:aber|doch|und|damit|denn|während|weshalb|sodass)\b)", s, maxsplit=1)
            if len(parts) == 2 and len(parts[0]) > 40:
                p1 = parts[0].strip().rstrip(".") + "."
                p2 = parts[1].strip()
                p2 = p2[0].upper() + p2[1:] if p2 else ""
                refined_sentences.extend([p1, p2])
            else:
                refined_sentences.append(s)
        else:
            refined_sentences.append(s)

    tts_script = " ".join(refined_sentences)
    tts_script = re.sub(r"\s+", " ", tts_script).strip()

    # Härte-Split: Sätze > 240 Zeichen an Wortgrenzen teilen (XTTS-Tokenlimit)
    hard_split = []
    for s in refined_sentences:
        if len(s) <= 240:
            hard_split.append(s)
            continue
        part, parts = "", []
        for w in s.split(" "):
            if part and len(part) + len(w) + 1 > 200:
                parts.append(part)
                part = w
            else:
                part = f"{part} {w}".strip()
        if part:
            parts.append(part)
        hard_split.extend(parts)
    refined_sentences = hard_split
    tts_script = " ".join(refined_sentences).strip()

    prosody = infer_emotion_and_prosody(entry, cfg)
    tts_script, prosody = apply_emotion_punctuation(tts_script, prosody)
    return tts_script, phonetic_text, applied_phonetics, prosody


def apply_emotion_punctuation(script: str, prosody: Dict) -> Tuple[str, Dict]:
    """
    Übersetzt die erkannte Emotion in Text-/Tempo-Hinweise für XTTS-v2
    (Satzzeichen sind der einzige direkte Ausdrucks-Kanal des Modells):
      - Zorn: Satzpunkte -> Ausrufezeichen, minimal schneller
      - Ängstlich/düster: minimal langsamer
    """
    emotion = prosody.get("emotion", "")
    if emotion == "grim_determined_anger":
        sents = re.split(r"(?<=[.!?])\s+", script)
        out = []
        for s in sents:
            if s.endswith(".") and len(s) >= 12:
                s = s[:-1] + "!"
            out.append(s)
        script = " ".join(out)
        prosody["tts_speed"] = max(float(prosody.get("tts_speed", 1.0)), 1.04)
        prosody["acting_note"] = (prosody.get("acting_note", "") + " Sätze mit Nachdruck.").strip()
    elif emotion in ("fearful_pleading_wretch", "dark_mystical_alluring"):
        prosody["tts_speed"] = min(float(prosody.get("tts_speed", 1.0)), 0.96)
    return script, prosody


def try_external_frontier_api(prompt: str) -> Optional[str]:
    """Optional external LLM call if OPENAI_API_KEY is configured in .env."""
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    if not api_key:
        return None
    try:
        resp = requests.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": os.getenv("FRONTIER_MODEL", "gpt-4o"),
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
            },
            timeout=15,
        )
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        pass
    return None


def main():
    parser = argparse.ArgumentParser(description="Step 3: Skript-Aufbereitung & Phonetik.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    load_dotenv(project_root / ".env")
    with open(project_root / args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    pilot_path = project_root / cfg["paths"]["data_raw_de_text"] / "midnight_de_pilot_20.json"
    pilot_data = json.loads(pilot_path.read_text(encoding="utf-8"))
    entries = pilot_data["entries"]
    lexicon = cfg["llm_routing"]["tier2_frontier"].get("phonetic_lexicon", {})

    console.rule("[bold cyan]Step 3: Skript-Aufbereitung (Normalisierung + Warcraft-Phonetik)[/bold cyan]")
    begin_step("script", total=len(entries))

    try:
        ai_settings = get_ai_settings(cfg)
        set_ai_config(
            enabled=ai_is_available(ai_settings),
            provider=ai_settings.get("provider"),
            model=ai_settings.get("model") if ai_is_available(ai_settings) else None,
        )
    except Exception:
        pass

    table = Table(title="Skripte & Warcraft-Phonetik (Midnight DE)")
    table.add_column("Entry", style="cyan")
    table.add_column("Speaker", style="green")
    table.add_column("Prosodie-Hinweis", style="magenta")
    table.add_column("Phonetic Adaptations", style="yellow")
    table.add_column("Final TTS Script Preview", style="white")

    for e in entries:
        update_current({
            "entry_id": e["entry_id"],
            "npc_name": e["npc_name"],
            "quest_title_de": e.get("quest_title_de", ""),
            "phase": e.get("phase", ""),
            "status": "scripting",
            "cer": None,
        })
        raw_clean = e["clean_de_text"]
        # Tier 1: Routine normalizer (numbers, abbreviations, dialect apostrophes)
        tier1_text = tier1_routine_normalize(raw_clean)

        # Stufe 2: Pausen, Satzgrenzen, Warcraft-Phonetik, Prosodie
        tts_script, qa_target, phonetics, prosody = stage2_frontier_dramaturgy(tier1_text, e, lexicon, cfg)

        e["tier1_normalized_text"] = tier1_text
        e["tts_script_text"] = tts_script
        e["qa_reference_text"] = qa_target
        e["phonetic_replacements"] = phonetics
        e["prosody"] = prosody

        preview = tts_script if len(tts_script) <= 85 else tts_script[:82] + "..."
        table.add_row(
            e["entry_id"],
            e["npc_name"],
            prosody["emotion"],
            ", ".join(phonetics) if phonetics else "—",
            preview,
        )
        increment(processed=1)

    console.print(table)
    pilot_path.write_text(json.dumps(pilot_data, indent=2, ensure_ascii=False), encoding="utf-8")
    console.print(f"[bold green]✓ {len(entries)} deutsche Voiceover-Skripte vorbereitet -> {pilot_path}[/bold green]")
    finish_step(ok=True)


if __name__ == "__main__":
    main()
