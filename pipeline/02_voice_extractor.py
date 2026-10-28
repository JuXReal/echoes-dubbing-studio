#!/usr/bin/env python3
"""
Step 2: Voice Extractor & Embedding Pool Builder (02_voice_extractor.py)
------------------------------------------------------------------------
Extracts clean, dry 6–10s reference voice seeds from the English original OGGs:
1. Groups entries by NPC / Voice Archetype (Unique Lore NPCs vs. Race/Gender archetypes)
   AND extracts a dedicated per-line voice seed so every quest phase can use either
   its canonical character voice seed or its exact line voice seed.
2. Uses librosa/soundfile energy windowing to locate the cleanest, most expressive
   6–10 second speech segment without long pauses or clipping.
3. Cleans and normalizes each voice seed via FFmpeg:
   - Mono (1 channel), 24,000 Hz PCM 16-bit WAV (`pcm_s16le`)
   - High-pass (70 Hz) & Low-pass (11,500 Hz) filtering to remove rumble/clicks
   - Edge silence removal (`silenceremove`) & micro fade-in/out (15ms) to prevent clicks.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import yaml
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.live_state import begin_step, finish_step, increment, update_current

console = Console()

_F0_CACHE: Dict[str, Optional[float]] = {}


def estimate_mean_f0(wav_path: str) -> Optional[float]:
    """
    Mittlere Grundfrequenz (Hz) eines Voice-Seeds über die stimmhaften Frames.
    Dient als Indikator für hohe Stimmlagen, die XTTS-v2 instabiler klont.
    """
    if wav_path in _F0_CACHE:
        return _F0_CACHE[wav_path]
    try:
        import librosa
    except ImportError:
        return None
    mean = None
    try:
        y, sr = sf.read(wav_path, dtype="float32")
        if y.ndim > 1:
            y = y.mean(axis=1)
        if len(y) < int(0.5 * sr):
            _F0_CACHE[wav_path] = None
            return None
        # 2-Sekunden-Fenster aus der Mitte (schnell & repräsentativ)
        start = max(0, (len(y) - int(2 * sr)) // 2)
        seg = y[start : start + int(2 * sr)]
        f0, voiced, _ = librosa.pyin(
            seg, fmin=60, fmax=500, sr=sr,
            frame_length=2048, hop_length=256,
        )
        if f0 is not None:
            v = f0[voiced]
            mean = float(np.nanmean(v)) if len(v) else None
    except Exception:
        mean = None
    _F0_CACHE[wav_path] = mean
    return mean


def find_best_speech_window(
    audio_path: Path,
    min_sec: float = 6.0,
    max_sec: float = 9.5,
    ideal_sec: float = 8.5,
) -> Tuple[float, float, float]:
    """
    Sucht das sauberste UND ausdrucksstärkste Fenster (Sprachdichte +
    Lautstärke-Dynamik) zwischen min_sec und max_sec (Ziel: ideal_sec).
    Impulsive, emotionale Passagen der Originale werden so als Klon-Seed
    bevorzugt und übertragen mehr Ausdruck auf die Synthese.
    Returns (start_sec, duration_sec, score).
    """
    data, sr = sf.read(str(audio_path), always_2d=True)
    mono = np.mean(data, axis=1)
    total_sec = len(mono) / sr

    if total_sec <= max_sec:
        return 0.0, total_sec, float(np.sqrt(np.mean(mono**2)))

    target_dur = min(max_sec, max(min_sec, min(ideal_sec, total_sec * 0.85)))
    win_samples = int(target_dur * sr)
    step_samples = int(0.25 * sr)

    best_score = -1.0
    best_start = 0

    # Frame-level energy to penalize long internal silences
    frame_len = int(0.05 * sr)
    for start in range(0, max(1, len(mono) - win_samples), step_samples):
        segment = mono[start : start + win_samples]
        rms = float(np.sqrt(np.mean(segment**2)))
        peak = float(np.max(np.abs(segment))) + 1e-8
        # Compute active speech ratio (frames above -38 dB relative)
        n_frames = len(segment) // frame_len
        if n_frames > 0:
            frames = segment[: n_frames * frame_len].reshape(n_frames, frame_len)
            frame_rms = np.sqrt(np.mean(frames**2, axis=1))
            active_ratio = float(np.mean(frame_rms > (rms * 0.25)))
        else:
            active_ratio = 1.0

        # Score bevorzugt kräftiges RMS, hohe Sprachdichte, Ausdrucksdynamik
        # (Varianz der Frame-Lautstärke) und bestraft hartes Clipping
        clip_penalty = 0.7 if peak > 0.99 else 1.0
        if n_frames > 1:
            dyn = float(np.std(frame_rms) / max(rms, 1e-8))
            dyn_score = min(1.0, max(0.0, (dyn - 0.1) * 1.8))
        else:
            dyn_score = 0.0
        score = rms * (0.35 + 0.45 * active_ratio + 0.20 * dyn_score) * clip_penalty
        if score > best_score:
            best_score = score
            best_start = start

    return round(best_start / sr, 2), round(target_dur, 2), best_score


def extract_clean_wav_ffmpeg(
    input_ogg: Path,
    output_wav: Path,
    start_sec: float,
    duration_sec: float,
    sample_rate: int = 24000,
    highpass_hz: int = 70,
    lowpass_hz: int = 11500,
) -> float:
    """
    Extracts and cleans a reference voice seed via FFmpeg:
    - Trims to [start_sec, start_sec + duration_sec]
    - Filters rumble/clicks (highpass + lowpass)
    - Removes leading/trailing silence and applies 15ms de-click fades
    - Outputs 24,000 Hz Mono 16-bit PCM WAV.
    """
    output_wav.parent.mkdir(parents=True, exist_ok=True)

    af_filters = (
        f"highpass=f={highpass_hz},"
        f"lowpass=f={lowpass_hz},"
        f"silenceremove=start_periods=1:start_duration=0.05:start_threshold=-45dB:"
        f"stop_periods=1:stop_duration=0.35:stop_threshold=-45dB,"
        f"afade=t=in:st=0:d=0.015,"
        f"dynaudnorm=f=150:g=11:p=0.90"
    )

    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        str(start_sec),
        "-t",
        str(duration_sec),
        "-i",
        str(input_ogg),
        "-af",
        af_filters,
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-c:a",
        "pcm_s16le",
        str(output_wav),
    ]
    subprocess.run(cmd, check=True)

    info = sf.info(str(output_wav))
    # If silence removal trimmed it below 5.5s and original file is longer, re-extract without aggressive stop trim
    if info.duration < 5.5 and sf.info(str(input_ogg)).duration >= 6.0:
        fallback_af = (
            f"highpass=f={highpass_hz},"
            f"lowpass=f={lowpass_hz},"
            f"afade=t=in:st=0:d=0.015,"
            f"dynaudnorm=f=150:g=11:p=0.90"
        )
        cmd_fb = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            str(start_sec),
            "-t",
            str(max(duration_sec, 7.5)),
            "-i",
            str(input_ogg),
            "-af",
            fallback_af,
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-c:a",
            "pcm_s16le",
            str(output_wav),
        ]
        subprocess.run(cmd_fb, check=True)
        info = sf.info(str(output_wav))

    return round(info.duration, 2)


def main():
    parser = argparse.ArgumentParser(description="Step 2: Extract 6-10s 24kHz Mono WAV Voice Seeds.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    with open(project_root / args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    ve_cfg = cfg["voice_extraction"]
    pilot_path = project_root / cfg["paths"]["data_raw_de_text"] / "midnight_de_pilot_20.json"
    full_path = project_root / cfg["paths"]["data_raw_de_text"] / "midnight_de_full_catalog.json"
    ref_dir = project_root / cfg["paths"]["data_reference_voices"]

    unique_dir = ref_dir / "unique_npcs"
    archetypes_dir = ref_dir / "archetypes"
    per_entry_dir = ref_dir / "per_entry"
    for d in (unique_dir, archetypes_dir, per_entry_dir):
        d.mkdir(parents=True, exist_ok=True)

    console.rule("[bold cyan]Step 2: Voice Seed Extraction & Embedding Pool (24kHz Mono WAV)[/bold cyan]")

    pilot_data = json.loads(pilot_path.read_text(encoding="utf-8"))
    all_entries = json.loads(full_path.read_text(encoding="utf-8"))
    entries = pilot_data["entries"]

    begin_step("voices", total=len(entries))

    # 1. Build Canonical Archetype / Lore-NPC Voice Seeds from the best 8-10s source OGG per archetype
    archetype_candidates: Dict[str, List[Dict]] = {}
    for e in all_entries:
        arch = e["voice_archetype"]
        archetype_candidates.setdefault(arch, []).append(e)

    archetype_seeds: Dict[str, Dict] = {}
    for arch, cand_list in archetype_candidates.items():
        is_lore = arch.startswith("lore_")
        sub_folder = unique_dir if is_lore else archetypes_dir
        seed_wav = sub_folder / f"{arch}.wav"
        if seed_wav.is_file() and seed_wav.stat().st_size > 1000:
            try:
                actual_dur = round(sf.info(str(seed_wav)).duration, 2)
                first_cand = cand_list[0] if cand_list else {}
                archetype_seeds[arch] = {
                    "archetype": arch,
                    "category": "unique_lore_npc" if is_lore else "race_gender_archetype",
                    "npc_name": first_cand.get("npc_name", arch),
                    "race": first_cand.get("race", "Default"),
                    "gender": first_cand.get("gender", "Neutral"),
                    "source_ogg": first_cand.get("source_ogg_filename", ""),
                    "slice_start_sec": 0.0,
                    "duration_sec": actual_dur,
                    "wav_path": str(seed_wav),
                }
                continue
            except Exception:
                pass

        # Pick candidate whose duration is closest to 12-25s (ideal for slicing a clean 8.5s window)
        best_cand = None
        best_win = (0.0, 8.0, -1.0)
        update_current({
            "entry_id": f"archetype:{arch}",
            "npc_name": (cand_list[0].get("npc_name") if cand_list else "?"),
            "quest_title_de": "",
            "phase": "Archetyp-Seed",
            "status": "extracting_seed",
            "cer": None,
        })
        for c in cand_list:
            ogg_p = Path(c["source_ogg_path"])
            if not ogg_p.exists():
                continue
            st, dur, score = find_best_speech_window(
                ogg_p,
                min_sec=ve_cfg["min_duration_sec"],
                max_sec=ve_cfg["max_duration_sec"],
                ideal_sec=ve_cfg["ideal_duration_sec"],
            )
            # Prefer candidates that are >= 7s
            if dur >= 6.0:
                score *= 1.5
            if score > best_win[2]:
                best_win = (st, dur, score)
                best_cand = c

        if best_cand:
            actual_dur = extract_clean_wav_ffmpeg(
                Path(best_cand["source_ogg_path"]),
                seed_wav,
                best_win[0],
                best_win[1],
                sample_rate=ve_cfg["sample_rate"],
                highpass_hz=ve_cfg["highpass_hz"],
                lowpass_hz=ve_cfg["lowpass_hz"],
            )
            archetype_seeds[arch] = {
                "archetype": arch,
                "category": "unique_lore_npc" if is_lore else "race_gender_archetype",
                "npc_name": best_cand["npc_name"],
                "race": best_cand["race"],
                "gender": best_cand["gender"],
                "source_ogg": best_cand["source_ogg_filename"],
                "slice_start_sec": best_win[0],
                "duration_sec": actual_dur,
                "wav_path": str(seed_wav),
            }
        else:
            # Voice Pool Fallback for silent / unvoiced mods & expansions:
            # Deterministically select an existing voice seed from archetypes_dir or unique_dir
            pool_wavs = sorted(list(archetypes_dir.glob("*.wav")) + list(unique_dir.glob("*.wav")))
            first_cand = cand_list[0] if cand_list else {}
            if pool_wavs:
                chosen_pool_wav = pool_wavs[abs(hash(arch)) % len(pool_wavs)]
                try:
                    actual_dur = round(sf.info(str(chosen_pool_wav)).duration, 2)
                except Exception:
                    actual_dur = 8.0
                archetype_seeds[arch] = {
                    "archetype": arch,
                    "category": "voice_pool_fallback",
                    "npc_name": first_cand.get("npc_name", arch),
                    "race": first_cand.get("race", "Default"),
                    "gender": first_cand.get("gender", "Neutral"),
                    "source_ogg": chosen_pool_wav.name,
                    "slice_start_sec": 0.0,
                    "duration_sec": actual_dur,
                    "wav_path": str(chosen_pool_wav),
                }

    # 2. Also extract per-entry specific seeds (or fall back to archetype seed if source < 6s or unvoiced)
    table = Table(title="Extracted 24kHz Mono Reference Voice Seeds")
    table.add_column("Entry", style="cyan")
    table.add_column("NPC / Archetype", style="green")
    table.add_column("Category", style="magenta")
    table.add_column("Seed WAV", style="white")
    table.add_column("Slice Window", justify="right")
    table.add_column("Seed Dur", justify="right", style="bold green")

    final_ogg_dir = project_root / cfg["paths"]["output_final_ogg"]
    raw_tts_dir = project_root / cfg["paths"]["output_raw_tts"]

    for e in entries:
        ogg_p = Path(e["source_ogg_path"])
        arch = e["voice_archetype"]
        arch_info = archetype_seeds[arch]

        entry_wav = per_entry_dir / f"{e['entry_id']}_seed.wav"
        if entry_wav.is_file() and entry_wav.stat().st_size > 1000:
            try:
                actual_dur = round(sf.info(str(entry_wav)).duration, 2)
                st, dur = 0.0, actual_dur
            except Exception:
                actual_dur = 0.0
                st, dur = 0.0, 0.0
        elif ogg_p.is_file():
            update_current({
                "entry_id": e["entry_id"],
                "npc_name": e["npc_name"],
                "quest_title_de": e.get("quest_title_de", ""),
                "phase": e.get("phase", ""),
                "status": "extracting_seed",
                "cer": None,
            })
            st, dur, _ = find_best_speech_window(
                ogg_p,
                min_sec=ve_cfg["min_duration_sec"],
                max_sec=ve_cfg["max_duration_sec"],
                ideal_sec=ve_cfg["ideal_duration_sec"],
            )
            actual_dur = extract_clean_wav_ffmpeg(
                ogg_p,
                entry_wav,
                st,
                dur,
                sample_rate=ve_cfg["sample_rate"],
                highpass_hz=ve_cfg["highpass_hz"],
                lowpass_hz=ve_cfg["lowpass_hz"],
            )
        else:
            actual_dur = 0.0
            st, dur = 0.0, 0.0

        # If this specific line's English audio is >= 6.0s, use its direct voice seed for exact emotional tone;
        # if it's shorter (< 6.0s, e.g. 4.5s), use the NPC's canonical 8.5s archetype seed!
        if actual_dur >= 6.0:
            chosen_wav = str(entry_wav)
            chosen_dur = actual_dur
            slice_info = f"{st:.1f}s + {dur:.1f}s (Line)"
        else:
            chosen_wav = arch_info["wav_path"]
            chosen_dur = arch_info["duration_sec"]
            slice_info = f"Archetype ({arch_info['source_ogg']})"

        e["reference_voice_wav"] = chosen_wav
        e["reference_voice_duration_sec"] = chosen_dur
        e["archetype_voice_wav"] = arch_info["wav_path"]

        out_ogg = final_ogg_dir / e["target_ogg_filename"]
        out_raw = raw_tts_dir / f"{e['entry_id']}_raw.wav"
        if out_ogg.is_file() and out_ogg.stat().st_size > 500 and out_raw.is_file():
            e["seed_mean_f0_hz"] = e.get("seed_mean_f0_hz")
        else:
            seed_f0 = estimate_mean_f0(chosen_wav)
            e["seed_mean_f0_hz"] = round(seed_f0, 1) if seed_f0 is not None else None

        increment(processed=1)

        table.add_row(
            e["entry_id"],
            f"{e['npc_name']} ({arch})",
            arch_info["category"],
            Path(chosen_wav).name,
            slice_info,
            f"{chosen_dur:.2f}s",
        )

    console.print(table)

    # Save updated pilot JSON and voice pool manifest
    pilot_path.write_text(json.dumps(pilot_data, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest_path = ref_dir / "voice_pool_manifest.json"
    manifest_path.write_text(json.dumps(archetype_seeds, indent=2, ensure_ascii=False), encoding="utf-8")
    console.print(f"[bold green]✓ Extracted {len(archetype_seeds)} canonical NPC/Archetype seeds & {len(entries)} line seeds -> {ref_dir}[/bold green]")
    finish_step(ok=True)


if __name__ == "__main__":
    main()
