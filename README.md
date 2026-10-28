# 🎙️ Echoes Dubbing Studio

**Universal Multi-Game AI Voiceover & Localization Pipeline (Bring Your Own Data / BYOD)**

Echoes Dubbing Studio is an end-to-end, self-healing AI voiceover localization and audio mastering suite for game modders, indie developers, and localization teams. It transforms reference audio and dialogue tables into studio-mastered target-language voiceovers using **Coqui XTTS-v2** voice cloning, **Faster-Whisper** automated quality assurance (CER/WER & hallucination detection), **15-band Spectral EQ Matching**, character DSP effects, and **EBU R128 (−16 LUFS)** loudness normalization.

An integrated bilingual (EN/DE) **A/B Web Studio** (`http://127.0.0.1:8765`) provides real-time pipeline monitoring, side-by-side waveform playback (Original vs. RAW TTS vs. Mastered OGG), one-click single-line regeneration, and optional LLM phonetic repair (`OpenAI`, `OpenRouter`, `Google Gemini`, `DeepSeek`, or local `Ollama`).

[![License: MIT](https://img.shields.io/badge/License-MIT-amber.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![TTS: Coqui XTTS-v2](https://img.shields.io/badge/TTS-Coqui_XTTS--v2-purple.svg)](https://github.com/coqui-ai/TTS)
[![QA: Faster-Whisper](https://img.shields.io/badge/QA-Faster--Whisper-emerald.svg)](https://github.com/SYSTRAN/faster-whisper)
[![Architecture: BYOD](https://img.shields.io/badge/Data_Model-Bring_Your_Own_Data-cyan.svg)](#bring-your-own-data-byod--legal-notice)

---

- [English Documentation](#english-documentation)
- [Deutsche Dokumentation](#deutsche-dokumentation)

---

## English Documentation

### 🛡️ Bring Your Own Data (BYOD) & Legal Notice

**Echoes Dubbing Studio is a pure software tool and ships with ZERO proprietary game assets, ZERO copyrighted dialogue texts, and ZERO hardcoded third-party web scrapers.**

1. **No Bundled Assets:** This repository contains only open-source Python/HTML pipeline code and a 3-line synthetic example CSV (`data/custom_texts/example_lines.csv`).
2. **User-Controlled Data Sources (BYOD):** Users provide their own legal input data locally:
   - **Generic Multi-Game Mode (`--adapter generic`):** Drop any CSV or JSON dialogue table into `data/custom_texts/` and reference audio files into `data/original/` (or existing voice seeds into `data/reference_voices/`). Compatible with *Skyrim*, *Fallout*, *Baldur's Gate*, *Witcher*, *Cyberpunk*, *Unreal/Unity* indie games, or visual novels.
   - **Addon / SavedVariables Mode (`--adapter wow_chatty`):** Reads dialogue lines recorded locally by the user during their own gameplay (`SavedVariables`) or user-provided JSON/CSV files in `data/custom_texts/`.
   - **Optional Custom Provider Endpoint:** If you run a local database server or custom API, you can optionally set `QUEST_TEXT_PROVIDER_URL` in your local `.env` file (disabled by default).
3. **Non-Commercial & Respect for Creators:**
   - Coqui XTTS-v2 weights are subject to the **Coqui Public Model License (CPML)** (non-commercial use).
   - Never blindly upload auto-generated voice packs to public mod hubs (CurseForge, NexusMods, Wago) without explicit permission from the original mod/addon authors and voice actors.
   - This project is strictly non-commercial and is not affiliated with, endorsed, or sponsored by Blizzard Entertainment or any other game publisher.

---

### ✨ Key Features

| Stage | Script | What It Does |
| :--- | :--- | :--- |
| **Step 1** | `pipeline/01_addon_analyzer.py` | **Universal BYOD Ingest:** Reads generic CSV/JSON dialogue tables (`--adapter generic`) or addon packs (`--adapter wow_chatty`) and builds the active batch catalog. |
| **Step 2** | `pipeline/02_voice_extractor.py` | **Expressive Voice Seed Extraction & Voice Pool Router:** Slices clean 6–10 s, 24 kHz mono WAV reference seeds scored by speech density and dynamic range. Includes automatic **Voice Pool Fallback** for unvoiced/silent mods so every character receives a consistent voice from your seed pool even without source audio! |
| **Step 3** | `pipeline/03_script_agent.py` | **Prosody & Phonetic Script Director:** Normalizes numbers, Roman numerals, abbreviations, and custom phonetic lexicons; maps acting cues (`<shouts>`, `<whispers>`) to speaking speed and punctuation. |
| **Step 4** | `pipeline/04_tts_generator.py` | **GPU Batch Synthesis (XTTS-v2):** Synthesizes target-language audio with speaker-latent caching, pitch ($f_0$) stability guards for high-pitched voices, and sentence-level micro-fades. |
| **Step 5** | `pipeline/05_qa_validator.py` | **Self-Healing Whisper QA Loop:** Transcribes every generated WAV via **Faster-Whisper**, computes combined Character/Word Error Rate (`CER`/`WER`), detects hallucinations (babble, truncation, repetition loops), tunes synthesis parameters per failure cause, and optionally invokes LLM phonetic script repair. |
| **Step 6** | `pipeline/06_audio_fx.py` | **Studio DSP Mastering:** Applies **15-band Spectral EQ Matching** (`firequalizer`) against the original reference audio, character/race/comms DSP chains (`Undead`, `Radio`, `Helmet`, `Troll`, `Tauren`, `Void Elf`, etc.), and **EBU R128 (−16 LUFS)** loudness normalization. |
| **Step 7** | `pipeline/07_independent_audit_100.py` | **Independent Post-FX Audit:** Verifies container integrity, clipping, silence ratio, and post-FX Whisper ASR accuracy on a 100-line sample. |

---

### 🚀 Quick Start

#### 1. Installation
```bash
git clone https://github.com/JuXReal/echoes-dubbing-studio.git
cd echoes-dubbing-studio
./install.sh
```
`install.sh` creates the Python virtual environment (`.venv`), installs PyTorch with CUDA support, Coqui XTTS-v2, Faster-Whisper, and copies `.env.example` to `.env`.

#### 2. Mode A: Generic Multi-Game CSV / JSON Dubbing (`--adapter generic`)
Place your reference audio files (`.ogg`, `.wav`, or `.mp3`) in `data/original/` (or pre-extracted voice seeds in `data/reference_voices/archetypes/`) and create a CSV or JSON file in `data/custom_texts/lines.csv`:

```csv
entry_id,source_audio,target_audio,speaker,effect_profile,title,phase,text
line_001,line_001.ogg,line_001.ogg,Commander Vaelen,Default,Prologue,Line,"Hört her, Reisender! Die Grenzen sind gesichert."
line_002,line_002.ogg,line_002.ogg,Operator Sierra,Radio,Outpost 7,Line,"Hier Außenposten Sieben. Empfangen Ihr Signal klar und deutlich."
```

Run the pipeline:
```bash
.venv/bin/python pipeline/01_addon_analyzer.py --adapter generic --custom-data data/custom_texts/example_lines.csv --expansion MyGameMod --limit 0
.venv/bin/python pipeline/02_voice_extractor.py
.venv/bin/python pipeline/03_script_agent.py
.venv/bin/python pipeline/04_tts_generator.py
.venv/bin/python pipeline/05_qa_validator.py
.venv/bin/python pipeline/06_audio_fx.py --skip-audit
```

#### 3. Mode B: Interactive A/B Web Studio (`http://127.0.0.1:8765`)
```bash
./launch_desktop.sh
# or: .venv/bin/python web/serve.py
```

---

## Deutsche Dokumentation

### 🛡️ Bring Your Own Data (BYOD) & Rechtlicher Hinweis

**Echoes Dubbing Studio ist ein reines Software-Werkzeug. Dieses Repository enthält KEINE urheberrechtlich geschützten Spieldateien, KEINE Questtexte und KEINE fest eingebauten Web-Scraper für Drittseiten.**

1. **Sauberes BYOD-Prinzip (*Bring Your Own Data*):**
   Jeder Nutzer entscheidet lokal auf seinem eigenen System, welche eigenen Text- und Audiodaten er verarbeitet:
   - **Multi-Gaming CSV/JSON-Modus (`--adapter generic`):** Beliebige Spiele oder Mods über eine einfache CSV- oder JSON-Tabelle in `data/custom_texts/` und Referenz-Audios in `data/original/` (oder bestehende Stimm-Seeds in `data/reference_voices/`) vertonen.
   - **Lokaler Addon-Modus (`--adapter wow_chatty`):** Verarbeitet die beim eigenen Spielen lokal aufgezeichneten `SavedVariables` sowie eigene JSON/CSV-Textdateien in `data/custom_texts/`.
   - **Optionale eigene Schnittstelle:** Wer einen eigenen lokalen Text-Server oder eine eigene API betreibt, kann in seiner lokalen `.env` (die niemals auf GitHub hochgeladen wird) optional `QUEST_TEXT_PROVIDER_URL` hinterlegen.
2. **Stimm-Pool-Fallback für unvertonte Spiele & Erweiterungen:**
   Selbst wenn für ein Spiel, eine Mod oder eine Erweiterung überhaupt keine englischen Original-Audiodateien existieren, kann Schritt 2 (`02_voice_extractor.py`) jedem Sprecher automatisch und deterministisch eine konsistente Stimme aus deinem lokalen Stimm-Seed-Pool (`data/reference_voices/archetypes/`) zuweisen!
3. **Qualitätsverantwortung & Community-Respekt:**
   - Du als Nutzer trägst die alleinige Verantwortung für die Prüfung der lokal generierten Audiodateien im A/B-Webstudio.
   - Lade keine automatisch generierten Sprachpakete ohne Rücksprache mit den Original-Autoren auf öffentliche Portale (CurseForge, NexusMods, Wago) hoch.
   - Nicht-kommerzielles Open-Source-Werkzeug; steht in keinerlei Verbindung zu Blizzard Entertainment oder anderen Spieleherstellern.

---

### 📊 Was bedeutet „CER“ in der Qualitätskontrolle (Schritt 5 & 7)?

Jede generierte Sprachdatei wird nach der Synthese automatisch von **Faster-Whisper** (Spracherkennung) abgehört und mit dem Zielskript abgeglichen:
- **CER (*Character Error Rate* / Zeichenfehlerrate):** Misst über die Levenshtein-Distanz den prozentualen Anteil abweichender Buchstaben zwischen Zieltext und Transkript. Eine CER von `2,0 %` bedeutet 98 % Buchstabengenauigkeit.
- Im Deutschen ist die CER deutlich aussagekräftiger als die reine Wortfehlerrate (WER), da zusammengesetzte Nomen (z. B. *„Zwielichtklingenkultisten“* vs. *„Zwielichtklingen Kultisten“*) akustisch identisch klingen.
- Liegt der kombinierte Fehler über dem Schwellenwert (`max_wer_threshold: 0.15`), greift die **ursachenbewusste Selbstheilung** (Anpassung von Temperatur, Repetition-Penalty und Sprechtempo sowie ab Versuch 3 optionale phonetische Skript-Reparatur per LLM).

---

### 📄 Lizenz / License
Code released under the [MIT License](LICENSE). Third-party AI models (Coqui XTTS-v2, Faster-Whisper) are governed by their respective upstream licenses.
