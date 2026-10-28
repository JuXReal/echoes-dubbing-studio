#!/usr/bin/env python3
"""
🎙️ Echoes of Azeroth (DE) — Interactive Start Launcher (start.py)
------------------------------------------------------------------
Provides a simple, interactive menu where users just select what they want to do:
  [1] 🌐 Open Web Studio & One-Click Launchpad UI (http://127.0.0.1:8765)
  [2] 🧪 Run 20-Quest Pilot Test (Fast verification for any Expansion)
  [3] 🚀 Run Full Expansion Batch (Quests + Gossip + Auto-Install to WoW)
  [4] 🛡️ Run Independent 100-Quest Quality Audit (Post-Export Verification)
  [5] 🌌 Run ALL Installed Expansions Overnight
  [0] Exit
"""

import os
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
VENV_PYTHON = PROJECT_DIR / ".venv" / "bin" / "python"
PYTHON_BIN = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable

EXPANSIONS = [
    "Midnight",
    "The_War_Within",
    "Dragonflight",
    "Battle_for_Azeroth",
    "Wrath_of_the_Lich_King",
    "Vanilla",
]

ETHICS_BANNER = """================================================================================
🎙️  ECHOES OF AZEROTH — Interactive Voiceover Pipeline Launcher
    Multi-Language AI Voice Cloning & Mastering Suite (Tested with German / deDE)
================================================================================
⚠️  IMPORTANT COMMUNITY NOTICE (ENGLISH):
    Please DO NOT blindly upload generated voiceover packs to CurseForge!
    Always respect the hard work of the "Chatty Little NPC" addon creators
    (Tuubby & team) and the community contributors who created the original
    English voiceover files. Coordinate with the official maintainers first!
================================================================================"""


def pick_expansion() -> str:
    print("\nSelect Expansion:")
    for idx, exp in enumerate(EXPANSIONS, 1):
        print(f"  [{idx}] {exp.replace('_', ' ')} ({exp})")
    choice = input("\nEnter number [default=1 (Midnight)]: ").strip()
    if choice.isdigit() and 1 <= int(choice) <= len(EXPANSIONS):
        return EXPANSIONS[int(choice) - 1]
    return "Midnight"


def run_pipeline(expansion: str, limit: int, scrape_all: bool, include_gossip: bool):
    env = os.environ.copy()
    env["COQUI_TOS_AGREED"] = "1"

    cmd_step1 = [
        PYTHON_BIN,
        "pipeline/01_addon_analyzer.py",
        "--expansion",
        expansion,
        "--limit",
        str(limit),
    ]
    if include_gossip:
        cmd_step1.append("--include-gossip")
    if scrape_all:
        cmd_step1.append("--scrape-all-quests")

    steps = [
        cmd_step1,
        [PYTHON_BIN, "pipeline/02_voice_extractor.py"],
        [PYTHON_BIN, "pipeline/03_script_agent.py"],
        [PYTHON_BIN, "pipeline/04_tts_generator.py"],
        [PYTHON_BIN, "pipeline/05_qa_validator.py"],
        [PYTHON_BIN, "pipeline/06_audio_fx.py", "--install-to-wow", "--skip-audit"],
        [PYTHON_BIN, "pipeline/07_independent_audit_100.py", "--sample-size", "100"],
    ]

    for cmd in steps:
        subprocess.run(cmd, cwd=str(PROJECT_DIR), env=env, check=True)


def main():
    print(ETHICS_BANNER)
    while True:
        print("\nWhat would you like to do?")
        print("  [1] 🌐 Start Web Studio & Interactive Launchpad UI (http://127.0.0.1:8765)")
        print("  [2] 🧪 Run 20-Quest Pilot Test (Fast verification + 20-Quest Audit)")
        print("  [3] 🚀 Run Full Expansion Localization (All Quests + Gossip + 100-Quest Audit)")
        print("  [4] 🛡️  Run Independent 100-Quest Quality Audit Only")
        print("  [5] 🌌 Run ALL 6 Expansions Sequentially (Overnight Batch)")
        print("  [0] ❌ Exit")

        choice = input("\nSelect an option [1-5, 0]: ").strip()
        if choice == "1":
            print("\nStarting Web Studio at http://127.0.0.1:8765 (Press Ctrl+C to stop)...")
            subprocess.run([PYTHON_BIN, "web/serve.py"], cwd=str(PROJECT_DIR))
        elif choice == "2":
            exp = pick_expansion()
            run_pipeline(exp, limit=20, scrape_all=False, include_gossip=False)
        elif choice == "3":
            exp = pick_expansion()
            run_pipeline(exp, limit=0, scrape_all=True, include_gossip=True)
        elif choice == "4":
            subprocess.run(
                [PYTHON_BIN, "pipeline/07_independent_audit_100.py", "--sample-size", "100"],
                cwd=str(PROJECT_DIR),
            )
        elif choice == "5":
            for exp in EXPANSIONS:
                run_pipeline(exp, limit=0, scrape_all=True, include_gossip=True)
        elif choice == "0":
            print("Goodbye! Have fun in Azeroth!")
            break
        else:
            print("Invalid selection, please choose 1-5 or 0.")


if __name__ == "__main__":
    main()
