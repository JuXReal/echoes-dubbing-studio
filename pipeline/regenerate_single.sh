#!/usr/bin/env bash
# Helper script to regenerate a single failed or selected voiceover entry
# Usage: ./regenerate_entry.sh <entry_id> (e.g. 86515_Comp)
set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"
PYTHON="$PROJECT_DIR/.venv/bin/python"

ENTRY_ID="$1"
RETRIES="${2:-3}"
AI_TRIGGER="${3:-3}"
if [ -z "$ENTRY_ID" ]; then
    echo "Usage: ./regenerate_single.sh <entry_id> [retries] [ai_trigger]"
    exit 1
fi

echo "🔄 Regenerating entry: $ENTRY_ID..."
# Delete old artifacts for this entry so pipeline steps will re-generate it
rm -f "output/raw_tts_wav/${ENTRY_ID}_raw.wav"
rm -f "output/final_ogg/${ENTRY_ID}.ogg"

# Run TTS Generator (synthesizes only missing raw WAVs)
"$PYTHON" pipeline/04_tts_generator.py

# Run the Whisper QA gate for this entry (with retry loop & optional AI repair)
"$PYTHON" pipeline/05_qa_validator.py --entry-id "$ENTRY_ID" --max-retries "$RETRIES" --ai-trigger "$AI_TRIGGER"

# Run Audio FX & WoW deployment (processes only missing or updated files)
"$PYTHON" pipeline/06_audio_fx.py --install-to-wow --skip-audit

echo "✅ Entry $ENTRY_ID successfully regenerated and installed!"
