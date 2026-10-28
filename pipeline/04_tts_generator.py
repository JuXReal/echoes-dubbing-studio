#!/usr/bin/env python3
"""
Step 4: Local GPU Batch Voice Synthesis (04_tts_generator.py)
-------------------------------------------------------------
Synthesizes high-quality German (deDE) voiceovers using Coqui XTTS-v2 on CUDA:
1. Runs on PyTorch CUDA (RTX 4070 SUPER) with automatic VRAM management.
2. Pre-computes and caches speaker conditioning latents (`gpt_cond_latent` and
   `speaker_embedding`) per reference WAV so multiple lines from the same voice
   seed synthesize at maximum speed without redundant encoding.
3. Splits prepared German scripts into natural sentences and inserts subtle
   breathing pauses (180ms between sentences) for authentic RPG dialogue pacing.
4. Exposes a reusable `XTTSBatchSynthesizer` class so `05_qa_validator.py` can
   trigger targeted re-synthesis with adjusted temperature/repetition_penalty
   when a line fails the Whisper QA gate.
"""

import argparse
import gc
import json
import logging
import os
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Suppress cosmetic upstream warnings from HuggingFace transformers / Coqui
os.environ["COQUI_TOS_AGREED"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("TTS").setLevel(logging.ERROR)

import numpy as np
import soundfile as sf
import torch
import yaml
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from TTS.api import TTS

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.live_state import begin_step, finish_step, increment, update_current

console = Console()


def _hard_split_sentence(sent: str, limit: int = 200) -> List[str]:
    """Teilt überlange Sätze an Wortgrenzen (verhindert XTTS-Token-Truncation)."""
    if len(sent) <= limit:
        return [sent]
    chunks: List[str] = []
    cur = ""
    for w in sent.split(" "):
        if cur and len(cur) + len(w) + 1 > limit:
            chunks.append(cur.strip())
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        chunks.append(cur)
    return chunks


class XTTSBatchSynthesizer:
    """High-efficiency CUDA XTTS-v2 synthesizer with latent caching & VRAM control."""

    def __init__(self, cfg: Dict):
        os.environ["COQUI_TOS_AGREED"] = "1"
        self.tts_cfg = cfg["tts"]
        self.device = self.tts_cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu"
        model_name = self.tts_cfg.get("model_name", "tts_models/multilingual/multi-dataset/xtts_v2")

        console.print(f"[cyan]• Loading XTTS-v2 model ({model_name}) on {self.device.upper()}...[/cyan]")
        self.tts = TTS(model_name).to(self.device)
        self.xtts_model = self.tts.synthesizer.tts_model
        self.sample_rate = 24000
        self._latent_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

    def get_speaker_latents(self, ref_wav_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes and caches (gpt_cond_latent, speaker_embedding) for a 24kHz reference WAV."""
        if ref_wav_path not in self._latent_cache:
            gpt_cond_latent, speaker_embedding = self.xtts_model.get_conditioning_latents(
                audio_path=[ref_wav_path],
                gpt_cond_len=30,
                gpt_cond_chunk_len=4,
                max_ref_length=15,
            )
            self._latent_cache[ref_wav_path] = (gpt_cond_latent, speaker_embedding)
        return self._latent_cache[ref_wav_path]

    def synthesize_line(
        self,
        text: str,
        ref_wav_path: str,
        output_wav_path: Path,
        temperature: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        speed: Optional[float] = None,
        pause_ms: int = 160,
        length_penalty: Optional[float] = None,
    ) -> float:
        """
        Synthesizes a German dialogue line sentence-by-sentence using cached speaker
        latents, stitches sentences with a natural pause, and writes 24kHz PCM WAV.
        Returns generated duration in seconds.
        """
        temp = float(temperature if temperature is not None else self.tts_cfg.get("temperature", 0.62))
        rep_pen = float(repetition_penalty if repetition_penalty is not None else self.tts_cfg.get("repetition_penalty", 5.0))
        spd = float(speed if speed is not None else self.tts_cfg.get("speed", 1.02))
        len_pen = float(length_penalty if length_penalty is not None else self.tts_cfg.get("length_penalty", 1.0))
        top_k = int(self.tts_cfg.get("top_k", 50))
        top_p = float(self.tts_cfg.get("top_p", 0.85))

        gpt_cond_latent, speaker_embedding = self.get_speaker_latents(ref_wav_path)

        # In Sätze aufteilen, damit XTTS lange Absätze nicht abschneidet
        raw_sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
        if not raw_sentences:
            raw_sentences = [text.strip()]
        # Hart-Split überlanger Sätze — XTTS' eigener Splitter bräuchte sonst spacy
        raw_sentences = [chunk for s in raw_sentences for chunk in _hard_split_sentence(s)]

        audio_chunks: List[np.ndarray] = []
        silence_gap = np.zeros(int(self.sample_rate * (pause_ms / 1000.0)), dtype=np.float32)

        with torch.inference_mode():
            for idx, sent in enumerate(raw_sentences):
                # Kurze Sätze halluzinieren häufiger ("Punkt" / Fortsetzung am Ende)
                # -> niedrigere Temperatur für mehr Stabilität
                sent_temp = temp
                if len(sent) < 30:
                    sent_temp = min(temp, 0.5)
                out = self.xtts_model.inference(
                    text=sent,
                    language=self.tts_cfg.get("language", "de"),
                    gpt_cond_latent=gpt_cond_latent,
                    speaker_embedding=speaker_embedding,
                    temperature=max(0.15, min(0.95, sent_temp)),
                    length_penalty=len_pen,
                    repetition_penalty=rep_pen,
                    top_k=top_k,
                    top_p=top_p,
                    speed=spd,
                    enable_text_splitting=False,
                )
                wav_chunk = np.asarray(out["wav"], dtype=np.float32)
                # Trim trailing artifact clicks if any
                if len(wav_chunk) > 240:
                    fade_len = min(240, len(wav_chunk) // 10)
                    wav_chunk[-fade_len:] *= np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
                audio_chunks.append(wav_chunk)
                if idx < len(raw_sentences) - 1:
                    audio_chunks.append(silence_gap)

        full_wav = np.concatenate(audio_chunks) if audio_chunks else np.zeros(self.sample_rate, dtype=np.float32)
        # Peak normalize to -1.5 dBFS before saving
        peak = float(np.max(np.abs(full_wav)))
        if peak > 1e-5:
            full_wav = full_wav * (0.84 / peak)

        output_wav_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_wav_path), full_wav, self.sample_rate, subtype="PCM_16")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return round(len(full_wav) / self.sample_rate, 2)


def main():
    parser = argparse.ArgumentParser(description="Step 4: Local GPU Batch TTS Synthesis (XTTS-v2).")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--entry-id", default=None, help="Synthesize only a specific entry_id")
    parser.add_argument("--force", action="store_true", help="Overwrite existing synthesized WAVs")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    with open(project_root / args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    pilot_path = project_root / cfg["paths"]["data_raw_de_text"] / "midnight_de_pilot_20.json"
    raw_tts_dir = project_root / cfg["paths"]["output_raw_tts"]
    raw_tts_dir.mkdir(parents=True, exist_ok=True)

    pilot_data = json.loads(pilot_path.read_text(encoding="utf-8"))
    entries = pilot_data["entries"]
    if args.entry_id:
        entries = [e for e in entries if e["entry_id"] == args.entry_id]

    console.rule("[bold cyan]Step 4: Local GPU Batch Voice Synthesis (Coqui XTTS-v2 CUDA)[/bold cyan]")
    begin_step("tts", total=len(entries))

    synthesizer = XTTSBatchSynthesizer(cfg)
    start_all = time.time()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[green]Synthesizing Midnight DE Quests...", total=len(entries))

        for idx, e in enumerate(entries, 1):
            eid = e["entry_id"]
            out_wav = raw_tts_dir / f"{eid}_raw.wav"
            progress.update(task, description=f"[green]Synthesizing {eid} ({e['npc_name']})...")
            update_current({
                "entry_id": eid,
                "npc_name": e["npc_name"],
                "quest_title_de": e.get("quest_title_de", ""),
                "phase": e.get("phase", ""),
                "status": "synth_cached" if (out_wav.exists() and not args.force) else "synthesizing",
                "cer": None,
            })

            if out_wav.exists() and not args.force:
                info = sf.info(str(out_wav))
                e["raw_tts_wav"] = str(out_wav)
                e["raw_tts_duration_sec"] = round(info.duration, 2)
                progress.advance(task)
                increment(processed=1)
                continue

            prosody = e.get("prosody", {})
            dur = synthesizer.synthesize_line(
                text=e["tts_script_text"],
                ref_wav_path=e["reference_voice_wav"],
                output_wav_path=out_wav,
                temperature=prosody.get("tts_temperature", cfg["tts"]["temperature"]),
                speed=prosody.get("tts_speed", cfg["tts"]["speed"]),
            )
            e["raw_tts_wav"] = str(out_wav)
            e["raw_tts_duration_sec"] = dur
            progress.advance(task)
            increment(processed=1)

    elapsed = time.time() - start_all
    pilot_path.write_text(json.dumps(pilot_data, indent=2, ensure_ascii=False), encoding="utf-8")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    console.print(f"[bold green]✓ Synthesized {len(entries)} German voiceover files in {elapsed:.1f}s -> {raw_tts_dir}[/bold green]")
    finish_step(ok=True)


if __name__ == "__main__":
    main()
