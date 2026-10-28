#!/usr/bin/env python3
"""
Übersetzungs-Fortschritts-Anzeige (pipeline/translation_progress_watcher.py)
---------------------------------------------------------------------------
Läuft parallel zu Schritt 1, wenn dessen Prozess noch ohne Live-Updates in
der Übersetzungsphase arbeitet, und spiegelt den Fortschritt aus den
Cache-Dateien (quest_translations_<exp>_cache.json) in live_state.json —
damit das Studio Anzeige, Zähler und ETA bekommt.

Usage:
  .venv/bin/python pipeline/translation_progress_watcher.py <total> [expansion]
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import live_state as ls


def main() -> None:
    total = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    expansion = (sys.argv[2] if len(sys.argv) > 2 else "Midnight").lower()
    project = Path(__file__).resolve().parent.parent
    trans_cache = project / "data" / "raw_de_text" / f"quest_translations_{expansion}_cache.json"

    if total <= 0:
        print("Bitte Gesamtzahl angeben: translation_progress_watcher.py <total>")
        return

    ls.begin_step("analyze", total=total)
    ls.update(**{
        "mode": "full_expansion",
        "expansion": sys.argv[2] if len(sys.argv) > 2 else "Midnight",
        "step": {"key": "analyze", "index": 1, "label": "KI-Übersetzung (EN→DE)", "done": False, "ok": None},
    })
    print(f"Watcher aktiv: {total} Übersetzungen im Visier -> {trans_cache.name}")

    last_mtime = 0.0
    idle_since = time.time()
    while True:
        processed = 0
        last_key = None
        if trans_cache.exists():
            try:
                data = json.loads(trans_cache.read_text(encoding="utf-8"))
                processed = len(data)
                if data:
                    last_key = max(map(int, data.keys()))
                m = trans_cache.stat().st_mtime
                if m != last_mtime:
                    last_mtime = m
                    idle_since = time.time()
            except Exception:
                pass

        ls.update(
            **{
                "counts": {"processed": processed},
                "current": {
                "entry_id": f"translate:{last_key}" if last_key else "translate:…",
                "npc_name": "DeepSeek",
                "quest_title_de": f"Quest #{last_key}" if last_key else "Übersetzung läuft",
                "phase": "Übersetzung",
                "status": "translating",
                "cer": None,
                },
            },
        )

        if processed >= total:
            print(f"Fertig: {processed}/{total} Übersetzungen gecacht.")
            return
        if time.time() - idle_since > 600:
            print("Seit 10 Minuten keine neuen Einträge — Watcher beendet.")
            return
        time.sleep(2)


if __name__ == "__main__":
    main()
