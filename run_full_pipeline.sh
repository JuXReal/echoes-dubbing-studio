#!/usr/bin/env bash
# ==============================================================================
# WoW Chatty Little NPC — Full Batch Runner (DE Localization Pipeline)
# ==============================================================================
# Usage Examples:
#   1) Run ALL Midnight Quests (SavedVariables + BYOD Custom Texts) + Gossip:
#      ./run_full_pipeline.sh Midnight
#
#   2) Run another installed expansion (e.g. The_War_Within, Dragonflight,
#      Battle_for_Azeroth, Wrath_of_the_Lich_King, Vanilla):
#      ./run_full_pipeline.sh The_War_Within
#
#   3) Run ALL 6 installed expansions sequentially overnight:
#      ./run_full_pipeline.sh ALL
# ==============================================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

export COQUI_TOS_AGREED=1
PYTHON="$PROJECT_DIR/.venv/bin/python"

run_expansion() {
    local EXP="$1"
    echo "=============================================================================="
    echo "🚀 Starting Full German (deDE) Voiceover Pipeline for Expansion: $EXP"
    echo "=============================================================================="

    "$PYTHON" pipeline/01_addon_analyzer.py \
        --expansion "$EXP" \
        --limit 0 \
        --include-gossip \
        --scrape-all-quests

    "$PYTHON" pipeline/02_voice_extractor.py
    "$PYTHON" pipeline/03_script_agent.py
    "$PYTHON" pipeline/04_tts_generator.py
    "$PYTHON" pipeline/05_qa_validator.py
    "$PYTHON" pipeline/06_audio_fx.py --install-to-wow --skip-audit
    "$PYTHON" pipeline/07_independent_audit_100.py --sample-size 100

    echo "✅ Completed Expansion: $EXP -> Installed ChattyLittleNpc_${EXP}_deDE_voiceovers!"
}

TARGET="${1:-Midnight}"

if [[ "$TARGET" == "ALL" ]]; then
    for EXP in Midnight The_War_Within Dragonflight Battle_for_Azeroth Wrath_of_the_Lich_King Vanilla; do
        run_expansion "$EXP"
    done
else
    run_expansion "$TARGET"
fi
