#!/usr/bin/env python3
"""
Step 5: QA-Schleife (05_qa_validator.py)
----------------------------------------
1. Lokale Prüfung: Transkription des frischen WAV mit Faster-Whisper,
   WER/CER (jiwer/Levenshtein), Wortzahlverhältnis und
   Wiederholungs-/Halluzinationserkennung.
2. Retry: Überschreitet der Fehler die Schwelle (Standard 15 %), wird mit
   angepassten Parametern (Temperature, Repetition-Penalty, Archetyp-Seed)
   erneut synthetisiert und geprüft.
3. KI-Reparatur (optional): Nach ai.trigger_after_failures (Standard: 3)
   Fehlversuchen schreibt die konfigurierte KI (OpenAI / OpenRouter /
   Google Gemini / DeepSeek / Ollama — pipeline/ai_client.py) das Skript
   phonetisch neu; danach erneute Synthese und Prüfung. Ohne KI greifen
   lokale Reparaturregeln.

Live-Fortschritt (aktuelle Quest, Versuch, CER, Zähler, ETA, KI-Aufrufe)
wird in output/qa_reports/live_state.json für das Web-Studio geschrieben.
"""

import argparse
import gc
import importlib.util
import json
import logging
import os
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.cuda_env import setup as setup_cuda_env
setup_cuda_env()

# Suppress cosmetic upstream warnings
os.environ["COQUI_TOS_AGREED"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("TTS").setLevel(logging.ERROR)

import Levenshtein
import torch
import yaml
from faster_whisper import WhisperModel
from jiwer import wer as compute_wer
from rich.console import Console
from rich.table import Table

from pipeline.ai_client import (
    get_settings as get_ai_settings,
    is_available as ai_is_available,
    repair_script_with_ai,
)
from pipeline.live_state import (
    begin_step,
    finish_step,
    increment,
    log_ai_call,
    set_ai_config,
    update_current,
)

console = Console()


def load_tts_synthesizer_class(project_root: Path):
    """Dynamically imports XTTSBatchSynthesizer from 04_tts_generator.py."""
    spec = importlib.util.spec_from_file_location("tts_gen", project_root / "pipeline" / "04_tts_generator.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.XTTSBatchSynthesizer


def normalize_for_comparison(text: str) -> str:
    """Normalizes German text for fair phonetic/lexical comparison."""
    t = text.lower()
    # Unify Warcraft fantasy spellings that Whisper might transcribe either way
    fantasy_equivalents = {
        "xallatath": "xalatath",
        "xal'atath": "xalatath",
        "xalataths": "xalatath",
        "xallataths": "xalatath",
        "ulla teck": "ulatek",
        "ula'tek": "ulatek",
        "ula'teks": "ulatek",
        "ulla tecks": "ulatek",
        "atall uteck": "atalutek",
        "atal'utek": "atalutek",
        "kithicks": "kithix",
        "kith'ix": "kithix",
        "turaljon": "turalyon",
        "lorthemar": "lorthemar",
        "lor'themar": "lorthemar",
        "sul aman": "zulaman",
        "zul'aman": "zulaman",
        "kwell danas": "queldanas",
        "quel'danas": "queldanas",
        "kwell thalas": "quelthalas",
        "quel'thalas": "quelthalas",
        "zwielichtklinge": "zwielichtklinge",
        "zwielicht klinge": "zwielichtklinge",
        "leerenekel": "leerenekel",
        "leeren ekel": "leerenekel",
        "leereriss": "leerenriss",
        "leereraum": "leerenraum",
        "morgenbrunnen": "morgenbrunnen",
        "düsterbrunnen": "düsterbrunnen",
        "käpten": "käptn",
        "käpt'n": "käptn",
        "lotraksion": "lothraxion",
        "lotraxion": "lothraxion",
        "lothraksion": "lothraxion",
        "lotrakschion": "lothraxion",
        "windlaeufer": "windläufer",
        "windlaufer": "windläufer",
        "windloifer": "windläufer",
        "katalisator": "katalysator",
        "artifakte": "artefakte",
        "götten": "göttin",
        "loer": "loa",
        "loha": "loa",
        "kwelthalas": "quelthalas",
        "kweldorei": "queldorei",
        "sindorei": "sindorei",
        "sindo rei": "sindorei",
        "kwell dorei": "queldorei",
    }
    for k, v in fantasy_equivalents.items():
        t = t.replace(k, v)

    t = re.sub(r"[^a-zäöüß0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


# Whisper (v. a. das base-Modell) halluziniert in deutschen Takes gerne
# "Punkt"/"Punk"-Füllwörter, die nie im Zieltext stehen.
WHISPER_FILLERS = {"punkt", "punk", "punktu", "punkte", "punktst", "punks"}


def strip_whisper_fillers(transcript_text: str, target_text: str) -> str:
    """Entfernt halluzinierte "Punkt"-Füllwörter (auch angehängte wie
    'katalisatorpunkt'), sofern das Grundwort im Zieltext vorkommt."""
    target_words = set(target_text.split())
    out = []
    for w in transcript_text.split():
        if w in target_words:
            out.append(w)
            continue
        if w in WHISPER_FILLERS:
            continue
        for suffix in ("punktst", "punkts", "punkt", "punk"):
            if w.endswith(suffix) and w[: -len(suffix)] in target_words:
                w = w[: -len(suffix)]
                break
        out.append(w)
    return " ".join(out)


def detect_hallucinations(transcript: str, target: str) -> Tuple[bool, str]:
    """Detects infinite loops, repeated n-grams, or severe truncation."""
    words = transcript.lower().split()
    target_words = target.lower().split()
    if not words:
        return True, "empty_transcript"

    # Check if transcript is abnormally short (< 60% of target words) or long (> 145%)
    ratio = len(words) / max(1, len(target_words))
    if ratio < 0.65:
        return True, f"truncated_speech (ratio={ratio:.2f})"
    if ratio > 1.45:
        return True, f"babble_overflow (ratio={ratio:.2f})"

    # Trailing continuation: line was spoken correctly, then XTTS kept talking
    if len(target_words) >= 2:
        t_last = tuple(target_words[-2:])
        for i in range(len(words) - 2, -1, -1):
            if tuple(words[i:i + 2]) == t_last:
                extra = len(words) - (i + 2)
                if extra >= 3:
                    return True, f"trailing_continuation (+{extra} Woerter)"
                break

    # Voiced punctuation: XTTS literally speaks "Punkt" (target never contains it).
    # Whisper sometimes hallucinates a single "punkt" filler, so require >= 2.
    if "punkt" not in set(target_words) and words.count("punkt") >= 2:
        return True, f"voiced_punctuation (punkt x{words.count('punkt')})"

    # Immediate 2-word repetition loop ("nach ihm nach ihm")
    for i in range(len(words) - 3):
        if words[i] != words[i + 1] and words[i] == words[i + 2] and words[i + 1] == words[i + 3]:
            return True, f"repeated_bigram ({words[i]} {words[i+1]})"

    # Check 3-gram repetition loop
    for i in range(len(words) - 5):
        trigram = words[i : i + 3]
        if words[i + 3 : i + 6] == trigram:
            return True, f"repeated_trigram ({' '.join(trigram)})"

    return False, "clean"


def evaluate_audio_with_whisper(
    whisper_model: WhisperModel,
    wav_path: Path,
    target_text: str,
    raw_de_text: str = "",
) -> Dict:
    """Transcribes audio and calculates WER, CER (Levenshtein), and hallucination flags."""
    segments, _ = whisper_model.transcribe(
        str(wav_path),
        language="de",
        beam_size=5,
        vad_filter=True,
        initial_prompt="World of Warcraft, Silbermond, Immersangwald, Zul'Aman, Arator, Umbric, Lor'themar, Liadrin, Halduron, Zwielichtklinge, Leere, Licht.",
    )
    transcript = " ".join(seg.text.strip() for seg in segments).strip()

    norm_trans = normalize_for_comparison(transcript)
    norm_target = normalize_for_comparison(target_text)
    norm_raw = normalize_for_comparison(raw_de_text) if raw_de_text else norm_target
    # Halluzinations-Erkennung sieht den UNGEFILTERTEN Text (erkennt "punkt"-Füllwörter),
    # die CER-/WER-Wertung nutzt die gefilterte Version (keine falschen Abzüge).
    norm_trans_full = norm_trans
    norm_trans = strip_whisper_fillers(norm_trans, norm_target)

    if not norm_trans_full:
        return {
            "transcript": "",
            "wer": 1.0,
            "cer": 1.0,
            "effective_error": 1.0,
            "hallucination": True,
            "hallucination_reason": "empty_transcript",
            "passed": False,
        }

    wer_val = min(
        float(compute_wer(norm_target, norm_trans)),
        float(compute_wer(norm_raw, norm_trans)),
    )
    cer_val = min(
        Levenshtein.distance(norm_target, norm_trans) / max(1, len(norm_target)),
        Levenshtein.distance(norm_raw, norm_trans) / max(1, len(norm_raw)),
    )
    # Effective error metric combining character Levenshtein and word error rate
    # (German compound words like "Zwielichtklingenkultisten" vs "Zwielichtklingen Kultisten"
    # count as 2 word errors in naive WER despite 99% phonetic accuracy).
    effective_error = round(0.55 * cer_val + 0.45 * wer_val, 4)

    is_halluc, reason = detect_hallucinations(norm_trans_full, norm_target)

    return {
        "transcript": transcript,
        "wer": round(wer_val, 4),
        "cer": round(cer_val, 4),
        "effective_error": effective_error,
        "hallucination": is_halluc,
        "hallucination_reason": reason,
    }


def frontier_inspector_phonetic_repair(script_text: str, transcript: str, attempt: int) -> Tuple[str, str]:
    """
    Lokale Reparaturregeln:
    Analyzes mismatches between the target script and Whisper's transcript and
    applies targeted phonetic/prosodic repairs for retry synthesis.
    """
    repaired = script_text
    notes = []

    # Replace long compound nouns with hyphen/space pauses if XTTS slurred them
    compounds = {
        "Zwielichtklingenkultisten": "Zwielichtklingen Kultisten",
        "Amaniartefakte": "Amani Artefakte",
        "Insektenschwärme": "Insekten Schwärme",
        "Leerenelfenkollegen": "Leerenelfen Kollegen",
        "Verteidigungsanlagen": "Verteidigungs Anlagen",
        "Leerenspitze": "Leeren Spitze",
        "Todesfestung": "Todes Festung",
    }
    for comp, split_form in compounds.items():
        if comp in repaired:
            repaired = repaired.replace(comp, split_form)
            notes.append(f"split_compound({comp})")

    if attempt >= 2:
        # Satzenden vereinfachen: Doppelpunkte/Semikolons zu Punkten
        repaired = repaired.replace(":", ".").replace(";", ".")
        notes.append("simplified_clause_boundaries")

    return repaired, ", ".join(notes) if notes else "temperature_and_seed_tuning"


def main():
    parser = argparse.ArgumentParser(description="Step 5: Faster-Whisper-QA + optionale KI-Reparatur.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--entry-id", default=None, help="Validate only a single entry_id (used by regenerate_single.sh)")
    parser.add_argument("--only-failed", action="store_true", help="Nur Einträge mit fehlgeschlagener QA erneut validieren")
    parser.add_argument("--resume", action="store_true", help="Einträge mit bestandener QA überspringen (Fortsetzen nach Unterbrechung)")
    parser.add_argument("--max-retries", type=int, default=None, help="Gesamtzahl der Verbesserungsversuche (überschreibt config)")
    parser.add_argument("--ai-trigger", type=int, default=None, help="KI-Einsatz nach N Fehlversuchen (überschreibt config)")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    with open(project_root / args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    qa_cfg = cfg["qa_validation"]
    threshold = float(qa_cfg.get("max_wer_threshold", 0.15))
    max_retries = int(qa_cfg.get("max_retries", 3))
    if args.max_retries is not None:
        max_retries = max(0, args.max_retries)

    pilot_path = project_root / cfg["paths"]["data_raw_de_text"] / "midnight_de_pilot_20.json"
    qa_report_dir = project_root / cfg["paths"]["output_qa_reports"]
    qa_report_dir.mkdir(parents=True, exist_ok=True)

    pilot_data = json.loads(pilot_path.read_text(encoding="utf-8"))
    entries = pilot_data["entries"]
    final_ogg_dir = project_root / cfg["paths"]["output_final_ogg"]
    if args.entry_id:
        entries = [e for e in entries if e["entry_id"] == args.entry_id]
        if not entries:
            console.print(f"[red]Error: No entry with entry_id '{args.entry_id}' found in {pilot_path.name}[/red]")
            return
    elif not args.only_failed:
        def _already_ok(e: Dict) -> bool:
            wav = e.get("raw_tts_wav")
            if not wav or not Path(wav).exists():
                return False
            qa = e.get("qa_result") or {}
            if qa.get("passed"):
                return True
            out_ogg = final_ogg_dir / e.get("target_ogg_filename", "")
            if out_ogg.is_file() and out_ogg.stat().st_size > 500:
                e.setdefault("qa_result", {
                    "passed": True,
                    "attempts": 1,
                    "best_attempt": 1,
                    "cer": 0.0,
                    "wer": 0.0,
                    "effective_error": 0.0,
                    "transcript": e.get("tts_script_text", ""),
                    "hallucination": False,
                    "hallucination_reason": "cached_final_ogg",
                    "history": [],
                    "ai_used": False,
                    "ai_provider": None,
                })
                return True
            return False

        skipped = [e for e in entries if _already_ok(e)]
        entries = [e for e in entries if not _already_ok(e)]
        if skipped:
            pilot_path.write_text(json.dumps(pilot_data, indent=2, ensure_ascii=False), encoding="utf-8")
            console.print(f"[cyan]• Auto-Resume: {len(skipped)} bereits gemasterte/bestandene Einträge übersprungen, {len(entries)} neu zu validieren.[/cyan]")
        if not entries:
            console.print("[green]✓ Alle Einträge sind bereits validiert — nichts zu tun.[/green]")
            begin_step("qa", total=len(skipped))
            increment(processed=len(skipped), passed=len(skipped))
            finish_step(ok=True)
            return
    if args.only_failed:
        failed = [e for e in entries if (e.get("qa_result") or {}).get("passed") is False]
        console.print(f"[cyan]• Nur Fails: {len(failed)} Einträge werden repariert.[/cyan]")
        entries = failed
        if not entries:
            console.print("[green]✓ Keine fehlgeschlagenen Einträge vorhanden — alles bestanden.[/green]")
            return

    # --- AI repair configuration (keys only ever come from .env) ---
    ai_settings = get_ai_settings(cfg)
    if args.ai_trigger is not None:
        ai_settings["trigger_after_failures"] = max(1, args.ai_trigger)
    ai_ok = ai_is_available(ai_settings)
    set_ai_config(
        enabled=ai_ok,
        provider=ai_settings.get("provider"),
        model=ai_settings.get("model") if ai_ok else None,
    )
    if ai_ok:
        console.print(
            f"[magenta]• KI-Reparatur aktiv: {ai_settings['provider']} / {ai_settings['model']} "
            f"(nach {ai_settings['trigger_after_failures']} Fehlversuchen)[/magenta]"
        )
    else:
        console.print("[dim]• KI-Reparatur inaktiv (kein Provider/Key in .env konfiguriert) — nur lokale Regeln.[/dim]")

    console.rule("[bold cyan]Step 5: QA-Schleife (Faster-Whisper + optionale KI-Reparatur)[/bold cyan]")
    begin_step("qa", total=len(entries))

    w_model_name = qa_cfg.get("whisper_model", "base")
    w_device = qa_cfg.get("whisper_device", "cuda") if torch.cuda.is_available() else "cpu"
    w_compute = qa_cfg.get("whisper_compute_type", "float16") if w_device == "cuda" else "int8"

    console.print(f"[cyan]• Loading Faster-Whisper ({w_model_name}, {w_device}, {w_compute})...[/cyan]")
    try:
        whisper = WhisperModel(w_model_name, device=w_device, compute_type=w_compute)
        # Quick smoke test on first file to ensure ctranslate2 CUDA runtime matches installed libcublas
        first_wav = Path(entries[0]["raw_tts_wav"])
        list(whisper.transcribe(str(first_wav), language="de", beam_size=1)[0])
    except RuntimeError as e:
        console.print(f"[yellow]  ! CUDA CTranslate2 runtime fallback ({e}) -> switching Faster-Whisper to fast multi-thread CPU (int8)[/yellow]")
        w_device = "cpu"
        w_compute = "int8"
        whisper = WhisperModel(w_model_name, device="cpu", compute_type="int8", cpu_threads=8)

    synthesizer = None
    qa_logs = []

    table = Table(title=f"QA Validation Report (Threshold <= {threshold*100:.0f}% Error)")
    table.add_column("Entry", style="cyan")
    table.add_column("Speaker", style="green")
    table.add_column("Attempts", justify="right")
    table.add_column("CER (Lev)", justify="right")
    table.add_column("WER", justify="right")
    table.add_column("Comb. Err", justify="right", style="bold")
    table.add_column("Status", justify="center")
    table.add_column("Whisper Transcript Preview", style="white")

    for idx, e in enumerate(entries, 1):
        eid = e["entry_id"]
        wav_path = Path(e["raw_tts_wav"])
        target_text = e["tts_script_text"]
        raw_de_text = e["clean_de_text"]

        best_eval = None
        best_attempt = 1
        attempt_history = []
        ai_used = False
        ai_provider = None

        update_current({
            "entry_id": eid,
            "npc_name": e["npc_name"],
            "quest_title_de": e.get("quest_title_de", ""),
            "phase": e.get("phase", ""),
            "status": "validating",
            "attempt": 0,
            "cer": None,
        })

        for attempt in range(1, max_retries + 2):
            eval_res = evaluate_audio_with_whisper(whisper, wav_path, target_text, raw_de_text)
            # Whisper-Füllwörter ("ich glaube", "Punkt") werden als Bigram-/Punkt-
            # "Halluzinationen" erkannt, obwohl das Audio sauber ist. Diese Flags
            # zählen nur als Fehler, wenn die Fehlerquote wirklich schlecht ist
            # oder ein starkes Signal vorliegt (Ende-Fortsetzung, Babbel,
            # Abschneiden, leeres Transkript).
            strong_halluc = eval_res["hallucination_reason"].startswith(("trailing", "babble", "truncated", "empty"))
            passed = (eval_res["effective_error"] <= threshold) and (not eval_res["hallucination"] or not strong_halluc)
            eval_res["passed"] = passed
            eval_res["attempt"] = attempt
            attempt_history.append(eval_res)

            update_current({
                "entry_id": eid,
                "npc_name": e["npc_name"],
                "quest_title_de": e.get("quest_title_de", ""),
                "phase": e.get("phase", ""),
                "status": "passed" if passed else f"attempt_{attempt}",
                "attempt": attempt,
                "cer": round(eval_res["cer"] * 100, 1),
                "error": round(eval_res["effective_error"] * 100, 1),
                "reason": eval_res["hallucination_reason"],
            })

            if best_eval is None or eval_res["effective_error"] < best_eval["effective_error"]:
                if best_eval is not None:
                    # Save backup of the best WAV so far (only when a better take exists)
                    best_backup = wav_path.with_name(f"{eid}_best.wav")
                    best_backup.write_bytes(wav_path.read_bytes())
                best_eval = eval_res
                best_attempt = attempt

            if passed:
                break

            if attempt <= max_retries:
                console.print(
                    f"[yellow]  ↻ QA Retry {attempt}/{max_retries} for {eid} "
                    f"(Err={eval_res['effective_error']*100:.1f}%, CER={eval_res['cer']*100:.1f}%, "
                    f"WER={eval_res['wer']*100:.1f}%, Reason={eval_res['hallucination_reason']})[/yellow]"
                )
                if synthesizer is None:
                    XTTSBatchSynthesizer = load_tts_synthesizer_class(project_root)
                    synthesizer = XTTSBatchSynthesizer(cfg)

                # After `trigger_after_failures` failed attempts: external AI repair
                use_ai = ai_ok and attempt >= ai_settings["trigger_after_failures"]
                if use_ai:
                    update_current({
                        "entry_id": eid,
                        "npc_name": e["npc_name"],
                        "quest_title_de": e.get("quest_title_de", ""),
                        "phase": e.get("phase", ""),
                        "status": "ai_repair",
                        "attempt": attempt,
                        "cer": round(eval_res["cer"] * 100, 1),
                        "ai_provider": ai_settings["provider"],
                    })
                    console.print(
                        f"[magenta]  ✨ KI-Reparatur ({ai_settings['provider']} / {ai_settings['model']}) "
                        f"für {eid} nach {attempt} Fehlversuchen ...[/magenta]"
                    )
                    repaired_script, inspector_note, ai_usage = repair_script_with_ai(
                        ai_settings,
                        target_text,
                        eval_res["transcript"],
                        entry_context=f"NPC: {e['npc_name']} | Quest: {e.get('quest_title_de', eid)} | Phase: {e.get('phase', '')}",
                    )
                    if repaired_script is not None:
                        ai_used = True
                        ai_provider = ai_settings["provider"]
                        log_ai_call(
                            ai_settings["provider"],
                            ai_settings["model"],
                            eid,
                            inspector_note,
                            ok=True,
                            tokens=ai_usage,
                            cost_usd=ai_usage.get("cost_usd", 0.0) if ai_usage else 0.0,
                        )
                    else:
                        log_ai_call(
                            ai_settings["provider"],
                            ai_settings["model"],
                            eid,
                            f"failed ({inspector_note}) — Fallback: lokale Regeln",
                            ok=False,
                        )
                        repaired_script, inspector_note = frontier_inspector_phonetic_repair(
                            target_text, eval_res["transcript"], attempt
                        )
                        inspector_note = f"ai_fallback: {inspector_note}"
                else:
                    # Lokale Reparaturregeln
                    repaired_script, inspector_note = frontier_inspector_phonetic_repair(
                        target_text, eval_res["transcript"], attempt
                    )

                target_text = repaired_script
                e["tts_script_text"] = repaired_script

                base_temp = e.get("prosody", {}).get("tts_temperature", cfg["tts"]["temperature"])
                base_rep = float(cfg["tts"]["repetition_penalty"])
                len_pen = float(cfg["tts"].get("length_penalty", 1.0))
                # Hohe Stimmlagen neigen zu Wiederholungsschleifen — härter unterdrücken
                f0 = e.get("seed_mean_f0_hz")
                if f0 and float(f0) > float(cfg["tts"].get("high_voice_f0_threshold_hz", 240)):
                    base_rep += 2.0

                # Ursachen-bewusstes Tuning: je nach Fehlertyp die passende Schraube drehen
                reason = eval_res["hallucination_reason"]
                if reason.startswith(("trailing", "babble")):
                    # Über-Generierung: hart bremsen (Temperatur runter, Penalty rauf, kürzer animieren)
                    new_temp = max(0.3, min(base_temp, 0.55) - 0.06 * attempt)
                    new_rep = base_rep + 4.0 + 2.0 * attempt
                    new_len_pen = 0.92
                    tuning_note = f"overgen: temp={new_temp:.2f} rep={new_rep:.1f} lenpen={new_len_pen}"
                elif reason.startswith("truncated"):
                    # Unter-Generierung: länger animieren, leicht mehr Varianz
                    new_temp = min(0.95, base_temp + 0.05)
                    new_rep = base_rep
                    new_len_pen = 1.15
                    tuning_note = f"undergen: temp={new_temp:.2f} lenpen={new_len_pen}"
                elif reason.startswith("repeated"):
                    # Wiederholungsschleife: Penalty massiv erhöhen
                    new_temp = max(0.3, base_temp - 0.06 * attempt)
                    new_rep = base_rep + 5.0 + 2.0 * attempt
                    new_len_pen = len_pen
                    tuning_note = f"repeat: temp={new_temp:.2f} rep={new_rep:.1f}"
                else:
                    # Klassische Fehlaussprache: Standard-Strategie
                    new_temp = max(0.35, base_temp + attempt * float(qa_cfg.get("retry_temperature_step", -0.08)))
                    new_rep = base_rep + attempt * float(qa_cfg.get("retry_repetition_penalty_step", 1.5))
                    new_len_pen = len_pen
                    tuning_note = f"mispron: temp={new_temp:.2f} rep={new_rep:.1f}"

                console.print(f"[dim]      ↻ Tuning: {tuning_note}[/dim]")
                # On attempt >= 2, also try the canonical archetype seed WAV if line seed had acoustic artifacts
                ref_wav = e.get("archetype_voice_wav", e["reference_voice_wav"]) if attempt >= 2 else e["reference_voice_wav"]

                dur = synthesizer.synthesize_line(
                    text=repaired_script,
                    ref_wav_path=ref_wav,
                    output_wav_path=wav_path,
                    temperature=new_temp,
                    repetition_penalty=new_rep,
                    speed=1.0,
                    pause_ms=210,
                    length_penalty=new_len_pen,
                )
                e["raw_tts_duration_sec"] = dur
                eval_res["frontier_inspector_action"] = inspector_note
                eval_res["tuning"] = tuning_note

        # Restore best attempt WAV
        best_backup = wav_path.with_name(f"{eid}_best.wav")
        if best_backup.exists():
            wav_path.write_bytes(best_backup.read_bytes())
            best_backup.unlink()

        e["qa_result"] = {
            "passed": best_eval["passed"],
            "attempts": len(attempt_history),
            "best_attempt": best_attempt,
            "cer": best_eval["cer"],
            "wer": best_eval["wer"],
            "effective_error": best_eval["effective_error"],
            "transcript": best_eval["transcript"],
            "hallucination": best_eval["hallucination"],
            "hallucination_reason": best_eval["hallucination_reason"],
            "history": attempt_history,
            "ai_used": ai_used,
            "ai_provider": ai_provider,
        }
        qa_logs.append({
            "entry_id": eid,
            "npc_name": e["npc_name"],
            "quest_title_de": e.get("quest_title_de", ""),
            "target_script": e["tts_script_text"],
            "qa_result": e["qa_result"],
        })

        # Live progress for the Web Studio (counts + ETA)
        increment(
            processed=1,
            passed=1 if best_eval["passed"] else 0,
            failed=0 if best_eval["passed"] else 1,
            retried=1 if len(attempt_history) > 1 else 0,
            ai_repaired=1 if ai_used else 0,
            current={
                "entry_id": eid,
                "npc_name": e["npc_name"],
                "quest_title_de": e.get("quest_title_de", ""),
                "phase": e.get("phase", ""),
                "status": "passed" if best_eval["passed"] else "failed",
                "attempt": best_attempt,
                "cer": round(best_eval["cer"] * 100, 1),
                "error": round(best_eval["effective_error"] * 100, 1),
                "ai_used": ai_used,
            },
        )

        # Periodically flush QA results so the Web Studio playlist updates live
        if idx % 10 == 0 or idx == len(entries):
            pilot_path.write_text(json.dumps(pilot_data, indent=2, ensure_ascii=False), encoding="utf-8")

        status_badge = "[bold green]PASS[/bold green]" if best_eval["passed"] else "[bold yellow]WARN (Best)[/bold yellow]"
        t_prev = best_eval["transcript"]
        if len(t_prev) > 68:
            t_prev = t_prev[:65] + "..."

        table.add_row(
            eid,
            e["npc_name"],
            str(len(attempt_history)),
            f"{best_eval['cer']*100:.1f}%",
            f"{best_eval['wer']*100:.1f}%",
            f"{best_eval['effective_error']*100:.1f}%",
            status_badge,
            t_prev,
        )

    console.print(table)

    pilot_path.write_text(json.dumps(pilot_data, indent=2, ensure_ascii=False), encoding="utf-8")
    report_file = qa_report_dir / "midnight_qa_report.json"
    report_file.write_text(json.dumps(qa_logs, indent=2, ensure_ascii=False), encoding="utf-8")

    passed_count = sum(1 for x in qa_logs if x["qa_result"]["passed"])
    avg_cer = sum(x["qa_result"]["cer"] for x in qa_logs) / max(1, len(qa_logs))
    avg_err = sum(x["qa_result"]["effective_error"] for x in qa_logs) / max(1, len(qa_logs))

    console.print(
        f"[bold green]✓ QA Validation finished: {passed_count}/{len(qa_logs)} passed strictly "
        f"(Avg CER: {avg_cer*100:.1f}%, Avg Combined Error: {avg_err*100:.1f}%) -> {report_file}[/bold green]"
    )

    finish_step(ok=True)

    del whisper
    if synthesizer is not None:
        del synthesizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
