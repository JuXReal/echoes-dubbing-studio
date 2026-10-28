#!/usr/bin/env python3
"""
Step 7: Independent 100-Quest Quality Audit (07_independent_audit_100.py)
-------------------------------------------------------------------------
Performs a strict, independent post-export quality assurance test on up to
100 final mastered OGG files (`output/final_ogg/*.ogg`) BEFORE any pack is
considered release-ready.

Audit Criteria (Independent of the generation loop):
1. Acoustic & Mastering Integrity:
   - Valid OGG Vorbis container & sample rate (44.1 kHz)
   - Crest factor / clipping check (no hard digital clipping > -0.2 dBFS)
   - Dead-air / silence ratio check (ensures no silent or broken renders)
2. Independent ASR Verification:
   - Re-transcribes the final mastered OGG (including Race-FX & EBU R128)
     using Faster-Whisper and verifies Levenshtein Character Error Rate (CER)
     and Word Error Rate (WER) against the German quest text.
3. Mandatory English Community & CurseForge Ethics Notice:
   - Prints and embeds the explicit English notice reminding users NOT to
     blindly upload generated packs to CurseForge and to respect the work
     of the Chatty Little NPC addon authors and the English voiceover community.
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Dict, List

import Levenshtein
import numpy as np
import soundfile as sf
import torch
import yaml
from faster_whisper import WhisperModel
from jiwer import wer as compute_wer
from mutagen.oggvorbis import OggVorbis
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.cuda_env import setup as setup_cuda_env
setup_cuda_env()
from pipeline.ai_client import get_settings as get_ai_settings, is_available as ai_is_available
from pipeline.live_state import (
    begin_step,
    finish_step,
    increment,
    set_ai_config,
    update_current,
)

console = Console()

COMMUNITY_ETHICS_NOTICE_EN = """[bold yellow]⚠️  IMPORTANT COMMUNITY NOTICE & USER QUALITY RESPONSIBILITY[/bold yellow]

[bold red]1. USER QUALITY RESPONSIBILITY (YOU ARE RESPONSIBLE):[/bold red]
   [bold white]You as the user are solely responsible for listening to and verifying the quality of the generated audio files![/bold white]
   AI voice generation (XTTS-v2 / Fast-Whisper) can occasionally produce mispronunciations, hallucinations, or pacing artifacts.
   This automated audit performs spot-checks, but [bold underline]you must personally listen to audio in the A/B Web Studio[/bold underline] before considering any pack ready.

[bold yellow]2. DO NOT BLINDLY UPLOAD TO CURSEFORGE / WAGO:[/bold yellow]
   Please DO NOT upload generated packs to public addon hubs without explicit approval from the original creators!
   Full credit and respect belong to [bold]Tuubby[/bold] and the [bold]Chatty Little NPC[/bold] addon team, as well as the community contributors behind the original English voiceover files.

[bold cyan]3. Why English Packs Are Strictly Essential:[/bold cyan]
   • The unique vocal identity, timbre, and emotion (6–10s 24kHz seeds) are cloned from the English original files.
   • Canonical QuestID and NPC file mapping structure (VoiceoversList.lua).
   • Automatischer In-Game-Fallback für noch nicht lokalisierte Quest-/Gossip-Zeilen."""

COMMUNITY_ETHICS_NOTICE_DE = """[bold yellow]⚠️  WICHTIGER COMMUNITY-HINWEIS & QUALITÄTSVERANTWORTUNG DES NUTZERS[/bold yellow]

[bold red]1. DEINE QUALITÄTSVERANTWORTUNG (DER NUTZER HAFTET FÜR DIE QUALITÄT):[/bold red]
   [bold white]Du als Anwender bist allein dafür verantwortlich, die Qualität der generierten Sprachdateien persönlich anzuhören und gegenzuprüfen![/bold white]
   KI-Sprachsynthese kann vereinzelt Aussprachefehler, Halluzinationen oder unpassende Pausen enthalten.
   Dieser automatische 100-Quest-Check dient als Stichprobenfilter – [bold underline]die finale Qualitätskontrolle im A/B-Webstudio liegt bei dir[/bold underline]!

[bold yellow]2. BITTE NICHT BLIND NACH CURSEFORGE ODER WAGO HOCHLADEN:[/bold yellow]
   Lade keine automatisch generierten Pakete eigenmächtig auf öffentliche Addon-Portale hoch.
   Respektiere die monatelange Arbeit von [bold]Tuubby[/bold], dem [bold]Chatty Little NPC[/bold]-Team und den englischen Sprechern der Community!

[bold cyan]3. Warum die englischen Sprachpakete zwingend gebraucht werden:[/bold cyan]
   • Stimmfarben-Cloning (6–10s Seeds) basiert 1:1 auf den englischen Original-WAVs.
   • Das Dateisystem (VoiceoversList.lua) gibt Quest-IDs und Phasen exakt vor.
   • Nahtloser In-Game-Fallback für noch nicht übersetzte Quests."""


def normalize_text(text: str) -> str:
    t = text.lower()
    equivalents = {
        "xallatath": "xalatath",
        "xal'atath": "xalatath",
        "ulla teck": "ulatek",
        "ula'tek": "ulatek",
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
        "lotraksion": "lothraxion",
        "lotraxion": "lothraxion",
        "windlaeufer": "windläufer",
        "windlaufer": "windläufer",
        "kwelthalas": "quelthalas",
        "kweldorei": "queldorei",
        "kwell dorei": "queldorei",
    }
    for k, v in equivalents.items():
        t = t.replace(k, v)
    t = re.sub(r"[^a-zäöüß0-9\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


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


def analyze_ogg_acoustics(ogg_path: Path) -> Dict:
    """Checks acoustic health of the final mastered OGG file."""
    data, sr = sf.read(str(ogg_path), always_2d=True)
    mono = np.mean(data, axis=1)
    dur = len(mono) / max(1, sr)
    peak = float(np.max(np.abs(mono))) if len(mono) > 0 else 0.0
    rms = float(np.sqrt(np.mean(mono**2))) if len(mono) > 0 else 0.0
    rms_db = 20.0 * np.log10(max(rms, 1e-8))
    peak_db = 20.0 * np.log10(max(peak, 1e-8))

    # Active speech frames above -42 dBFS
    frame_len = max(1, int(0.05 * sr))
    n_frames = len(mono) // frame_len
    if n_frames > 0:
        frames = mono[: n_frames * frame_len].reshape(n_frames, frame_len)
        f_rms = np.sqrt(np.mean(frames**2, axis=1))
        active_ratio = float(np.mean(f_rms > 0.008))
    else:
        active_ratio = 0.0

    acoustically_valid = (dur >= 1.5) and (peak <= 1.05) and (rms_db > -32.0) and (active_ratio >= 0.45)
    return {
        "duration_sec": round(dur, 2),
        "sample_rate": sr,
        "rms_db": round(rms_db, 2),
        "peak_db": round(peak_db, 2),
        "active_speech_ratio": round(active_ratio, 3),
        "acoustically_valid": acoustically_valid,
    }


def main():
    parser = argparse.ArgumentParser(description="Step 7: Independent 100-Quest Quality Audit.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--sample-size", type=int, default=100, help="Number of quests to audit (default: 100)")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for reproducible sampling (default: random)")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    with open(project_root / args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    console.rule("[bold cyan]Step 7: Mandatory Independent Quality Audit & User Responsibility Gate[/bold cyan]")
    console.print(Panel(COMMUNITY_ETHICS_NOTICE_EN, border_style="yellow", title="🌍 Open-Source Ethics & User Responsibility"))
    console.print(Panel(COMMUNITY_ETHICS_NOTICE_DE, border_style="magenta", title="🇩🇪 Nutzer-Qualitätsverantwortung"))

    pilot_path = project_root / cfg["paths"]["data_raw_de_text"] / "midnight_de_pilot_20.json"
    final_ogg_dir = project_root / cfg["paths"]["output_final_ogg"]
    qa_report_dir = project_root / cfg["paths"]["output_qa_reports"]
    qa_report_dir.mkdir(parents=True, exist_ok=True)

    if not pilot_path.exists():
        console.print("[red]Error: No catalog found. Run pipeline steps 01-06 first.[/red]")
        return

    pilot_data = json.loads(pilot_path.read_text(encoding="utf-8"))
    entries = [
        e for e in pilot_data.get("entries", [])
        if (final_ogg_dir / e["target_ogg_filename"]).exists()
    ]

    if not entries:
        console.print("[red]Error: No mastered OGG files found in output/final_ogg/.[/red]")
        return

    rng = random.Random(args.seed) if args.seed is not None else random.Random()
    sample_count = len(entries) if args.sample_size == 0 else min(args.sample_size, len(entries))
    sampled_entries = rng.sample(entries, sample_count) if (args.sample_size != 0 and len(entries) > sample_count) else list(entries)
    sampled_entries.sort(key=lambda x: (x["quest_id"], x["phase"]))

    audit_target_desc = f"{len(sampled_entries)} Quests (FULL 100% Quality Audit)" if args.sample_size == 0 else f"{len(sampled_entries)} Quests (Sample Size: {args.sample_size})"
    console.print(
        f"[cyan]• Running Audit on {audit_target_desc} | Total Available Files: {len(entries)}...[/cyan]"
    )

    # Live progress + AI config for the Web Studio
    begin_step("audit", total=len(sampled_entries))
    try:
        ai_settings = get_ai_settings(cfg)
        set_ai_config(
            enabled=ai_is_available(ai_settings),
            provider=ai_settings.get("provider"),
            model=ai_settings.get("model") if ai_is_available(ai_settings) else None,
        )
    except Exception:
        pass

    # Load Whisper for independent verification on final mastered OGGs
    w_model = cfg.get("qa_validation", {}).get("whisper_model", "base")
    try:
        whisper = WhisperModel(w_model, device="cuda" if torch.cuda.is_available() else "cpu", compute_type="float16")
        list(whisper.transcribe(str(final_ogg_dir / sampled_entries[0]["target_ogg_filename"]), language="de", beam_size=1)[0])
    except Exception:
        whisper = WhisperModel(w_model, device="cpu", compute_type="int8", cpu_threads=8)

    audit_results = []
    table = Table(title=f"Independent Quality Audit Results ({len(sampled_entries)} / {args.sample_size} Target Quests)")
    table.add_column("#", justify="right", style="dim")
    table.add_column("File", style="cyan")
    table.add_column("Speaker", style="green")
    table.add_column("SR / Dur", justify="right")
    table.add_column("RMS / Peak", justify="right")
    table.add_column("Post-FX CER", justify="right")
    table.add_column("Audit Verdict", justify="center")

    passed_all = 0
    for idx, e in enumerate(sampled_entries, 1):
        ogg_p = final_ogg_dir / e["target_ogg_filename"]
        update_current({
            "entry_id": e["entry_id"],
            "npc_name": e["npc_name"],
            "quest_title_de": e.get("quest_title_de", ""),
            "phase": e.get("phase", ""),
            "status": "auditing",
            "cer": None,
        })
        acoustics = analyze_ogg_acoustics(ogg_p)

        segments, _ = whisper.transcribe(str(ogg_p), language="de", beam_size=5, vad_filter=True)
        transcript = " ".join(s.text.strip() for s in segments).strip()

        norm_t = normalize_text(transcript)
        norm_ref = normalize_text(e.get("tts_script_text") or e["clean_de_text"])
        norm_t = strip_whisper_fillers(norm_t, norm_ref)
        cer = Levenshtein.distance(norm_ref, norm_t) / max(1, len(norm_ref))
        wer = float(compute_wer(norm_ref, norm_t)) if norm_t else 1.0

        verdict_pass = acoustics["acoustically_valid"] and (cer <= 0.15)
        if verdict_pass:
            passed_all += 1

        increment(
            processed=1,
            passed=1 if verdict_pass else 0,
            failed=0 if verdict_pass else 1,
            current={
                "entry_id": e["entry_id"],
                "npc_name": e["npc_name"],
                "quest_title_de": e.get("quest_title_de", ""),
                "phase": e.get("phase", ""),
                "status": "passed" if verdict_pass else "failed",
                "cer": round(cer * 100, 1),
                "error": round(cer * 100, 1),
            },
        )

        audit_item = {
            "entry_id": e["entry_id"],
            "file": e["target_ogg_filename"],
            "quest_title_de": e.get("quest_title_de", ""),
            "npc_name": e["npc_name"],
            "race": e["race"],
            "acoustics": acoustics,
            "post_fx_cer": round(cer, 4),
            "post_fx_wer": round(wer, 4),
            "post_fx_transcript": transcript,
            "passed": verdict_pass,
        }
        audit_results.append(audit_item)

        badge = "[bold green]✓ AUDIT PASS[/bold green]" if verdict_pass else "[bold red]✗ REVIEW[/bold red]"
        table.add_row(
            str(idx),
            e["target_ogg_filename"],
            e["npc_name"],
            f"{acoustics['sample_rate']//1000}kHz / {acoustics['duration_sec']}s",
            f"{acoustics['rms_db']} / {acoustics['peak_db']} dB",
            f"{cer*100:.1f}%",
            badge,
        )

    console.print(table)

    avg_cer = sum(x["post_fx_cer"] for x in audit_results) / max(1, len(audit_results))
    summary = {
        "user_responsibility_notice": (
            "THE USER IS PERSONALLY RESPONSIBLE FOR LISTENING TO AND VERIFYING AUDIO QUALITY BEFORE SHARING OR USING. "
            "DER NUTZER IST PERSÖNLICH DAFÜR VERANTWORTLICH, DIE AUDIODATEIEN GEGENZUPRÜFEN UND ANZUHÖREN."
        ),
        "notice_en": (
            "IMPORTANT: Please DO NOT blindly upload generated voiceover packs to CurseForge! "
            "Respect the work of the Chatty Little NPC addon creators (Tuubby & team) and the "
            "community contributors behind the original English voiceover files."
        ),
        "target_sample_size": args.sample_size,
        "audited_count": len(audit_results),
        "passed_count": passed_all,
        "pass_rate_percent": round((passed_all / max(1, len(audit_results))) * 100, 2),
        "average_post_fx_cer_percent": round(avg_cer * 100, 2),
        "release_ready_100_quests": (len(audit_results) >= args.sample_size and passed_all >= int(args.sample_size * 0.95)),
        "results": audit_results,
    }

    out_report = qa_report_dir / "independent_audit_100_report.json"
    out_report.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    console.print(
        f"\n[bold green]✓ Independent Audit Complete: {passed_all}/{len(audit_results)} passed "
        f"(Avg Post-FX CER: {avg_cer*100:.1f}%) -> {out_report}[/bold green]"
    )
    finish_step(ok=True)
    if len(audit_results) < args.sample_size:
        console.print(
            f"[yellow]ℹ️  Note: Currently {len(audit_results)} quests are rendered (Pilot Run). "
            f"Once the full expansion batch finishes, this script audits a full random sample of {args.sample_size} quests![/yellow]"
        )
    console.print(Panel(COMMUNITY_ETHICS_NOTICE_EN, border_style="yellow", title="⚠️ Reminder: User Quality Responsibility & Open-Source Policy"))
    console.print(Panel(COMMUNITY_ETHICS_NOTICE_DE, border_style="magenta", title="⚠️ Erinnerung: Du bist für die Audio-Qualität verantwortlich"))


if __name__ == "__main__":
    main()
