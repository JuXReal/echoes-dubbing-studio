#!/usr/bin/env python3
"""
Step 1: UniversalBYOD Ingest & Addon Analyzer (01_addon_analyzer.py)
--------------------------------------------------------------------
Strictly adheres to the "Bring Your Own Data" (BYOD) architecture:
No third-party websites or URLs are hardcoded in this repository.

Supported Ingest Modes:
1. Generic Multi-Game CSV / JSON Mode (`--adapter generic` or `--custom-data <path>`):
   - Reads user-provided dialogue tables (`lines.csv` or `lines.json`) from
     `data/custom_texts/` together with local reference audio in `data/original/`.
   - Works with any game (Skyrim, Fallout, Baldur's Gate, Witcher, indie RPGs, etc.).
2. World of Warcraft ChattyLittleNpc Mode (`--adapter wow_chatty`, default):
   - Parses `VoiceoversList.lua` from the user's locally installed voiceover pack.
   - Bit-exact Python port of `CLN.MD5:GenerateHash` for Gossip/GameObject/Item hashes.
   - Reads user-recorded texts from local `SavedVariables/ChattyLittleNpc.lua`.
   - Loads user-supplied quest text JSON/CSV files from `data/custom_texts/`.
   - Optional user-configured external endpoint (`QUEST_TEXT_PROVIDER_URL` in `.env`,
     disabled by default) and optional AI translation (`QUEST_EN_PROVIDER_URL` + AI key).
"""

import argparse
import concurrent.futures
import csv
import ctypes
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
import yaml
from bs4 import BeautifulSoup, NavigableString, Tag
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.live_state import begin_step, finish_step, increment, log_ai_call, update_current
from pipeline.ai_client import get_settings as get_ai_settings, is_available as ai_is_available, translate_to_german

console = Console()


# ==============================================================================
# ChattyLittleNpc Custom Lua MD5 & Text Cleaning Implementation
# ==============================================================================
CLN_MD5_K = [
    0xD76AA478, 0xE8C7B756, 0x242070DB, 0xC1BDCEEE,
    0xF57C0FAF, 0x4787C62A, 0xA8304613, 0xFD469501,
    0x698098D8, 0x8B44F7AF, 0xFFFF5BB1, 0x895CD7BE,
    0x6B901122, 0xFD987193, 0xA679438E, 0x49B40821,
    0xF61E2562, 0xC040B340, 0x265E5A51, 0xE9B6C7AA,
    0xD62F105D, 0x02441453, 0xD8A1E681, 0xE7D3FBC8,
    0x21E1CDE6, 0xC33707D6, 0xF4D50D87, 0x455A14ED,
    0xA9E3E905, 0xFCEFA3F8, 0x676F02D9, 0x8D2A4C8A,
    0xFFFA3942, 0x8771F681, 0x6D9D6122, 0xFDE5380C,
    0xA4BEEA44, 0x4BDECFA9, 0xF6BB4B60, 0xBEBFBC70,
    0x289B7EC6, 0xEAA127FA, 0xD4EF3085, 0x04881D05,
    0xD9D4D039, 0xE6DB99E5, 0x1FA27CF8, 0xC4AC5665,
    0xF4292244, 0x432AFF97, 0xAB9423A7, 0xFC93A039,
    0x655B59C3, 0x8F0CCC92, 0xFFEFF47D, 0x85845DD1,
    0x6FA87E4F, 0xFE2CE6E0, 0xA3014314, 0x4E0811A1,
    0xF7537E82, 0xBD3AF235, 0x2AD7D2BB, 0xEB86D391,
]


def _to_i32(val: float) -> int:
    return ctypes.c_int32(int(val) & 0xFFFFFFFF).value


def _to_u32(val: float) -> int:
    return ctypes.c_uint32(int(val) & 0xFFFFFFFF).value


def _lrot(x: float, n: int) -> float:
    lsh = _to_i32(_to_u32(x) << (n & 31))
    rsh = _to_i32(_to_u32(x) >> ((32 - n) & 31))
    return float(lsh) + float(rsh)


def generate_cln_lua_md5(message: str) -> str:
    """
    Bit-exact port of CLN.MD5:GenerateHash(message) from ChattyLittleNpc/Utils/Md5.lua.
    Reproduces Lua's exact float64 accumulation and LuaBitOp signed 32-bit shifts.
    """
    data = message.encode("utf-8")
    length = len(data)
    bits = length * 8
    pad_len = 56 - (length + 1) % 64
    if pad_len < 0:
        pad_len += 64
    padding = bytearray([128]) + bytearray(pad_len)
    b_tmp = bits
    for _ in range(8):
        padding.append(b_tmp % 256)
        b_tmp //= 256
    full = data + bytes(padding)

    h0, h1, h2, h3 = float(0x67452301), float(0xEFCDAB89), float(0x98BADCFE), float(0x10325476)
    for offset in range(0, len(full), 64):
        chunk = list(full[offset : offset + 64])
        a, b, c, d = h0, h1, h2, h3
        for i in range(64):
            if i < 16:
                f_val = c
                g_idx = i
            elif i < 32:
                f_val = b
                g_idx = (5 * i + 1) % 16
            elif i < 48:
                f_val = float(_to_i32(_to_i32(b) ^ _to_i32(c) ^ _to_i32(d)))
                g_idx = (3 * i + 5) % 16
            else:
                bor_bd = _to_i32(_to_i32(b) | _to_i32(~_to_i32(d)))
                f_val = float(_to_i32(_to_i32(c) ^ bor_bd))
                g_idx = (7 * i) % 16
            shift_amount = (7, 12, 17, 22)[i % 4]
            temp = d
            d = c
            c = b
            b = b + _lrot(a + f_val + float(CLN_MD5_K[i]) + float(chunk[g_idx]), shift_amount)
            a = temp
        h0 += a
        h1 += b
        h2 += c
        h3 += d

    return f"{_to_u32(h0):08x}{_to_u32(h1):08x}{_to_u32(h2):08x}{_to_u32(h3):08x}"


HTML_CLEAN_PATTERNS = [
    ("\r\n", " "),
    ("<HTML>", ""), ("</HTML>", ""),
    ("<BODY>", ""), ("</BODY>", ""),
    ("<BR/>", ""), ("<p>", ""), ("</p>", ""),
    ('<p align="center">', ""),
]


def load_dotenv_file(project_root: Path) -> None:
    env_file = project_root / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v


def clean_text_v1(text: str, player_name: Optional[str] = None, player_class: Optional[str] = None, player_race: Optional[str] = None) -> str:
    """Replicates CLN.Utils:CleanText(text) from Utils/Utils.lua."""
    if not text:
        return ""
    player_name = player_name or os.getenv("WOW_PLAYER_NAME", "Held")
    player_class = player_class or os.getenv("WOW_PLAYER_CLASS", "Paladin")
    player_race = player_race or os.getenv("WOW_PLAYER_RACE", "Blutelf")
    for token in (player_name, player_class, player_race):
        if token:
            for variant in (token, token.lower(), token.upper()):
                text = text.replace(variant, "Hero")
    text = text.replace("\n\n", " ")
    for find_str, repl_str in HTML_CLEAN_PATTERNS:
        text = text.replace(find_str, repl_str)
    return text


def clean_text_v2(text: str, player_name: Optional[str] = None, player_class: Optional[str] = None, player_race: Optional[str] = None) -> str:
    """Replicates CLN.Utils:CleanTextV2(text) from Utils/Utils.lua."""
    if not text:
        return ""
    player_name = player_name or os.getenv("WOW_PLAYER_NAME", "Held")
    player_class = player_class or os.getenv("WOW_PLAYER_CLASS", "Paladin")
    player_race = player_race or os.getenv("WOW_PLAYER_RACE", "Blutelf")
    replacements = [
        (player_name, "name"),
        (player_class, "class"),
        (player_race, "race"),
    ]
    for token, tag in replacements:
        if token:
            for variant in (token, token.lower(), token.upper()):
                text = text.replace(variant, f"{{{tag}|{variant}}}")
    for find_str, repl_str in HTML_CLEAN_PATTERNS:
        text = text.replace(find_str, repl_str)
    return text


def compute_cln_hashes(npc_id: int, text: str) -> List[str]:
    """Replicates CLN.Utils:GetHashes(npcId, text) -> [lua_md5_v1, lua_md5_v2]."""
    t1 = clean_text_v1(text)
    h1 = generate_cln_lua_md5(f"{npc_id}{t1}")
    t2 = clean_text_v2(text)
    h2 = generate_cln_lua_md5(f"{npc_id}{t2}")
    return [h1, h2]


def strip_stage_directions_and_placeholders(text: str, spoken_player_title: str = "Champion") -> Tuple[str, List[str]]:
    """
    Removes stage directions like <Arator brodelt vor Zorn.> from spoken dialogue
    (while extracting them as acting cues for the LLM director) and resolves
    {name|...} / {class|...} placeholders for natural target-language speech.
    """
    stage_cues = re.findall(r"<([^>]+)>", text)
    spoken = re.sub(r"<[^>]+>", " ", text)
    spoken = re.sub(r"\{name\|[^}]+\}", spoken_player_title, spoken)
    spoken = re.sub(r"\{class\|[^}]+\}", spoken_player_title, spoken)
    spoken = re.sub(r"\{race\|[^}]+\}", spoken_player_title, spoken)
    spoken = re.sub(r"\s+", " ", spoken).strip()
    return spoken, stage_cues


# ==============================================================================
# Known Character Voice Archetype & Effect Profile Enrichment
# ==============================================================================
KNOWN_NPC_ARCHETYPES = {
    235787: {"race": "Blood Elf", "archetype": "lore_lorthemar_theron", "role": "Noble Regent (Deep, majestic, authoritative)"},
    263465: {"race": "Blood Elf", "archetype": "lore_arator", "role": "Half-Elf Paladin (Determined, emotional)"},
    258020: {"race": "Blood Elf", "archetype": "lore_arator", "role": "Half-Elf Paladin (Determined, emotional)"},
    258029: {"race": "Blood Elf", "archetype": "lore_arator", "role": "Half-Elf Paladin (Determined, emotional)"},
    258022: {"race": "Blood Elf", "archetype": "lore_arator", "role": "Half-Elf Paladin (Determined, emotional)"},
    258150: {"race": "Blood Elf", "archetype": "lore_arator", "role": "Half-Elf Paladin (Determined, emotional)"},
    261878: {"race": "Blood Elf", "archetype": "lore_arator", "role": "Half-Elf Paladin (Determined, emotional)"},
    258019: {"race": "Void Elf", "archetype": "lore_magister_umbric", "role": "Void Scholar (Analytical, calm, mystic)"},
    258151: {"race": "Void Elf", "archetype": "lore_magister_umbric", "role": "Void Scholar (Analytical, calm, mystic)"},
    258021: {"race": "Void Elf", "archetype": "lore_magister_umbric", "role": "Void Scholar (Analytical, calm, mystic)"},
    262269: {"race": "Void Elf", "archetype": "lore_magister_umbric", "role": "Void Scholar (Analytical, calm, mystic)"},
    258538: {"race": "Void Elf", "archetype": "lore_magister_umbric", "role": "Void Scholar (Analytical, calm, mystic)"},
    262969: {"race": "Blood Elf", "archetype": "lore_lady_liadrin", "role": "Paladin Matriarch (Noble, fierce)"},
    262944: {"race": "Blood Elf", "archetype": "lore_halduron", "role": "Ranger-General (Alert, commanding)"},
    258024: {"race": "Entartung", "archetype": "creature_void_wretch", "role": "Void Creature (Hissing, pleading)"},
    262880: {"race": "Troll", "archetype": "lore_erinye", "role": "Troll Mystic (Dark, seductive)"},
    272056: {"race": "Troll", "archetype": "lore_erinye", "role": "Troll Mystic (Dark, seductive)"},
    262798: {"race": "Troll", "archetype": "lore_abdumati", "role": "Warlord (Rough, martial, deep)"},
    259416: {"race": "Troll", "archetype": "lore_zuljarra", "role": "Chieftain (Serious, weary)"},
    272197: {"race": "Undead", "archetype": "undead_amani_windcaller", "role": "Undead Spirit (Echoing, mystic)"},
    258755: {"race": "Troll", "archetype": "npc_captain_tokka", "role": "Sea Captain (Rough, determined)"},
    264258: {"race": "Blood Elf", "archetype": "bloodelf_female_farstrider", "role": "Ranger Recruit (Attentive, dutiful)"},
}


def infer_race_from_name_or_zone(npc_name: str, zone: str = "") -> str:
    """Heuristic fallback to assign DSP effect profile when character profile is not explicitly specified."""
    low = f"{npc_name} {zone}".lower()
    if any(k in low for k in ("umbric", "leeren", "void", "rissanbeter", "alleria")):
        return "Void Elf"
    if any(k in low for k in ("zul'", "amani", "atal'", "loa", "troll", "halazzi", "nalorakk", "akil'zon", "jan'alai")):
        return "Troll"
    if any(k in low for k in ("untot", "geißel", "geist", "knochen", "gruft", "todesritter", "lich", "undead", "ghoul")):
        return "Undead"
    if any(k in low for k in ("tauren", "mulgore", "donnerfels", "haranir")):
        return "Tauren"
    if any(k in low for k in ("gnom", "mechagnom", "tüftler", "gnome")):
        return "Gnome"
    if any(k in low for k in ("goblin", "bilgewasser", "kezan")):
        return "Goblin"
    if any(k in low for k in ("silbermond", "immersang", "quel'", "lor'themar", "liadrin", "halduron", "arator", "magister", "weltenwanderer", "blutritter")):
        return "Blood Elf"
    return "Default"


def parse_voiceovers_list(vo_list_path: Path) -> Tuple[List[str], Dict[str, float]]:
    """Parses VoiceoversList.lua from a locally installed voiceover pack."""
    if not vo_list_path.exists():
        return [], {}
    content = vo_list_path.read_text(encoding="utf-8")
    files = re.findall(r"'([^']+\.(?:ogg|mp3|wav))'", content)
    durations = {}
    for m in re.finditer(r'\["([^"]+)"\]\s*=\s*([0-9.]+)', content):
        durations[m.group(1)] = float(m.group(2))
    return files, durations


# ==============================================================================
# Generic Multi-Game CSV / JSON Ingest Adapter (Bring Your Own Data)
# ==============================================================================
def load_generic_custom_dataset(
    custom_path: Path,
    orig_audio_dir: Path,
) -> List[Dict]:
    """
    Loads a game-agnostic dubbing dataset from a CSV or JSON file (or a directory
    containing `lines.csv` / `lines.json`).

    Supported columns / keys per entry:
      - entry_id (optional, derived from filename if omitted)
      - source_audio (filename in `orig_audio_dir` or path to original reference audio)
      - target_audio (optional, defaults to source_audio with .ogg extension)
      - speaker / npc_name (character name)
      - archetype / voice_archetype (optional speaker grouping ID)
      - race / effect_profile (optional DSP profile key from config.yaml, e.g. Default, Undead, Radio)
      - title / quest_title_de (optional scene or quest title)
      - phase (optional, e.g. Line, Desc, Prog, Comp)
      - text / raw_de_text (the target language text to synthesize)
      - duration_sec (optional reference duration)
    """
    files_to_read: List[Path] = []
    if custom_path.is_file():
        files_to_read.append(custom_path)
    elif custom_path.is_dir():
        for pat in ("*.csv", "*.json"):
            files_to_read.extend(sorted(custom_path.glob(pat)))

    entries: List[Dict] = []
    for idx_file, fpath in enumerate(files_to_read):
        rows: List[Dict] = []
        if fpath.suffix.lower() == ".csv":
            with open(fpath, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
        elif fpath.suffix.lower() == ".json":
            data = json.loads(fpath.read_text(encoding="utf-8"))
            if isinstance(data, list):
                rows = data
            elif isinstance(data, dict) and "entries" in data:
                rows = data["entries"]
            elif isinstance(data, dict):
                # Quest-style dict: {"100": {"title": "...", "Desc": "...", "Prog": "...", "Comp": "..."}}
                for qid_key, qval in data.items():
                    if not isinstance(qval, dict):
                        continue
                    for ph in ("Desc", "Prog", "Comp"):
                        if (qval.get(ph) or "").strip():
                            rows.append({
                                "entry_id": f"{qid_key}_{ph}",
                                "quest_id": int(qid_key) if str(qid_key).isdigit() else idx_file + 1,
                                "phase": ph,
                                "title": qval.get("title", f"Entry #{qid_key}"),
                                "speaker": qval.get("start_npc_name" if ph == "Desc" else "end_npc_name", "Speaker"),
                                "npc_id": qval.get("start_npc_id" if ph == "Desc" else "end_npc_id", 0),
                                "source_audio": f"{qid_key}_{ph}.ogg",
                                "target_audio": f"{qid_key}_{ph}.ogg",
                                "text": qval[ph],
                            })

        for r_idx, row in enumerate(rows, 1):
            raw_text = (
                row.get("text")
                or row.get("raw_de_text")
                or row.get("clean_de_text")
                or ""
            ).strip()
            if not raw_text:
                continue
            src_name = (row.get("source_audio") or row.get("source_ogg_filename") or f"line_{r_idx:04d}.ogg").strip()
            tgt_name = (row.get("target_audio") or row.get("target_ogg_filename") or Path(src_name).with_suffix(".ogg").name).strip()
            eid = (row.get("entry_id") or Path(tgt_name).stem).strip()
            speaker = (row.get("speaker") or row.get("npc_name") or "Speaker").strip()
            effect_profile = (row.get("effect_profile") or row.get("race") or infer_race_from_name_or_zone(speaker)).strip()
            arch = (row.get("archetype") or row.get("voice_archetype") or re.sub(r"[^a-z0-9_]+", "_", speaker.lower()).strip("_") or "default_speaker").strip()
            title = (row.get("title") or row.get("quest_title_de") or f"Dialogue {eid}").strip()
            phase = (row.get("phase") or "Line").strip()

            src_path = Path(src_name)
            if not src_path.is_absolute():
                cand = orig_audio_dir / src_name
                if cand.exists():
                    src_path = cand
                elif row.get("source_ogg_path"):
                    src_path = Path(row["source_ogg_path"])

            dur = float(row.get("duration_sec") or row.get("source_duration_sec") or 0.0)
            if dur <= 0.0 and src_path.is_file():
                try:
                    import soundfile as sf
                    dur = round(sf.info(str(src_path)).duration, 2)
                except Exception:
                    dur = 6.0

            spoken_text, stage_cues = strip_stage_directions_and_placeholders(raw_text)
            if not spoken_text:
                continue

            qid_val = row.get("quest_id")
            if qid_val is None:
                m_q = re.match(r"^(\d+)", eid)
                qid_val = int(m_q.group(1)) if m_q else r_idx

            entries.append({
                "entry_id": eid,
                "kind": row.get("kind", "dialogue"),
                "quest_id": int(qid_val),
                "quest_title_de": title,
                "phase": phase,
                "npc_id": int(row.get("npc_id") or 0),
                "npc_name": speaker,
                "race": effect_profile,
                "gender": row.get("gender", "Neutral"),
                "creature_type": row.get("creature_type", "Humanoid"),
                "zone": row.get("zone", "Custom"),
                "subzone": row.get("subzone", ""),
                "model_file_id": None,
                "voice_archetype": arch,
                "role_description": f"{speaker} ({effect_profile})",
                "source_ogg_filename": Path(src_name).name,
                "source_ogg_path": str(src_path),
                "source_duration_sec": dur,
                "target_ogg_filename": tgt_name,
                "raw_de_text": raw_text,
                "clean_de_text": spoken_text,
                "stage_directions": stage_cues,
            })

    return entries


# ==============================================================================
# Optional User-Configured External Provider (Disabled by Default — BYOD)
# ==============================================================================
def sanitize_provider_text(title: str) -> str:
    """Removes markup or JSON remnants from external provider titles."""
    t = re.sub(r"\s*\|.*$", "", title)
    t = re.sub(r'\.\s*"date".*$', "", t)
    return t.strip()[:200]


def fetch_external_quest_from_template(quest_id: int, url_template: str, retries: int = 3) -> Dict:
    """
    Fetches quest data from a user-configured URL template (`QUEST_TEXT_PROVIDER_URL`),
    supporting either JSON responses (`{"title": ..., "Desc": ..., "Prog": ..., "Comp": ...}`)
    or standard HTML pages with heading sections (`Beschreibung`/`Description`,
    `Fortschritt`/`Progress`, `Vervollständigung`/`Completion`/`Abschluss`).
    """
    result = {
        "quest_id": quest_id,
        "title": f"Quest #{quest_id}",
        "Desc": "",
        "Prog": "",
        "Comp": "",
        "start_npc_id": 0,
        "start_npc_name": "NPC",
        "end_npc_id": 0,
        "end_npc_name": "NPC",
    }
    if not url_template:
        return result

    url = url_template.replace("{quest_id}", str(quest_id)).replace("{id}", str(quest_id))
    if "{quest_id}" not in url_template and "{id}" not in url_template:
        url = f"{url_template.rstrip('/')}/quest={quest_id}"

    try:
        resp = None
        for attempt in range(retries):
            try:
                resp = requests.get(
                    url,
                    headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"},
                    timeout=10,
                )
                if resp.status_code in (200, 404):
                    break
            except Exception:
                resp = None
            time.sleep(0.5 * (attempt + 1))

        if resp is None or resp.status_code != 200:
            return result

        # 1. Check if JSON API response
        ctype = (resp.headers.get("Content-Type") or "").lower()
        if "application/json" in ctype or resp.text.lstrip().startswith("{"):
            try:
                jdata = resp.json()
                for k in ("title", "Desc", "Prog", "Comp", "start_npc_id", "start_npc_name", "end_npc_id", "end_npc_name"):
                    if k in jdata and jdata[k]:
                        result[k] = jdata[k]
                return result
            except Exception:
                pass

        # 2. HTML fallback parser
        html = resp.text
        soup = BeautifulSoup(html, "lxml")

        h1 = soup.find("h1", class_="heading-size-1") or soup.find("h1")
        if h1 and h1.text:
            t_clean = sanitize_provider_text(h1.text)
            if t_clean and t_clean.lower() not in ("quests", "quest", "datenbank", "database"):
                result["title"] = t_clean

        npc_links = re.findall(r"npc=(\d+)(?:\\?/[^\]\">]+)?[\]\">]([^\[<]+)", html)
        if npc_links:
            result["start_npc_id"] = int(npc_links[0][0])
            result["start_npc_name"] = sanitize_provider_text(npc_links[0][1])
            if len(npc_links) >= 2:
                result["end_npc_id"] = int(npc_links[1][0])
                result["end_npc_name"] = sanitize_provider_text(npc_links[1][1])
            else:
                result["end_npc_id"] = result["start_npc_id"]
                result["end_npc_name"] = result["start_npc_name"]

        # Support both standard h2/h3 sections and disclosure divs
        dp = soup.find("div", id="disclosure-progress")
        if dp:
            result["Prog"] = re.sub(r"\s+", " ", dp.get_text(" ", strip=True)).strip()
        dc = soup.find("div", id="disclosure-completion")
        if dc:
            result["Comp"] = re.sub(r"\s+", " ", dc.get_text(" ", strip=True)).strip()

        section_map = {
            "Beschreibung": "Desc",
            "Fortschritt": "Prog",
            "Vervollständigung": "Comp",
            "Abschluss": "Comp",
            "Description": "Desc",
            "Progress": "Prog",
            "Completion": "Comp",
        }
        for heading in soup.find_all(["h2", "h3"]):
            sec_title = heading.get_text(strip=True)
            if sec_title not in section_map:
                continue
            phase_key = section_map[sec_title]
            if (result.get(phase_key) or "").strip():
                continue
            parts = []
            curr = heading.next_sibling
            while curr:
                if isinstance(curr, Tag) and curr.name in ("h2", "h3", "script", "style", "table", "ul", "div"):
                    if curr.name in ("h2", "h3") or (curr.name == "div" and (curr.get("id") or "").startswith("disclosure")):
                        break
                elif isinstance(curr, NavigableString):
                    t = str(curr).strip()
                    if t:
                        parts.append(t)
                elif isinstance(curr, Tag):
                    classes = curr.get("class") or []
                    if "box" not in classes and "pad" not in classes:
                        t = curr.get_text(" ", strip=True)
                        if t and "C_QuestLog.IsQuestFlaggedCompleted" not in t:
                            parts.append(t)
                curr = curr.next_sibling
            joined = re.sub(r"\s+", " ", " ".join(parts)).strip()
            if joined:
                result[phase_key] = joined
    except Exception:
        pass
    return result


def parse_saved_variables(
    sv_path: Path,
    vo_dir: Path,
    durations: Dict[str, float],
    include_gossip: bool = False,
) -> List[Dict]:
    """
    Extracts all recorded German quests, gossips, and gameobject texts from
    SavedVariables/ChattyLittleNpc.lua that have matching reference audio in `vo_dir`.
    """
    if not sv_path.exists():
        console.print(f"[yellow]Notice: SavedVariables not found at {sv_path} (skipping SavedVariables ingest)[/yellow]")
        return []

    sv_content = sv_path.read_text(encoding="utf-8")
    npc_section = sv_content.split("UnitInfoDB =")[0]
    npc_blocks = re.split(r'\n\[(\d+)\]\s*=\s*\{\s*\n\["deDE"\]', npc_section)

    entries = []
    for i in range(1, len(npc_blocks), 2):
        npc_id = int(npc_blocks[i])
        block = npc_blocks[i + 1]

        name_m = re.search(r'\["name"\]\s*=\s*"([^"]*)"', block)
        race_m = re.search(r'\["race"\]\s*=\s*"([^"]*)"', block)
        sex_m = re.search(r'\["sex"\]\s*=\s*"([^"]*)"', block)
        zone_m = re.search(r'\["zone"\]\s*=\s*"([^"]*)"', block)
        subzone_m = re.search(r'\["subzone"\]\s*=\s*"([^"]*)"', block)
        ctype_m = re.search(r'\["creatureType"\]\s*=\s*"([^"]*)"', block)
        model_m = re.search(r'\["modelFileId"\]\s*=\s*(\d+)', block)

        name = name_m.group(1) if name_m else f"NPC_{npc_id}"
        race = race_m.group(1) if race_m else ""
        sex = sex_m.group(1) if sex_m else "Neutral"
        zone = zone_m.group(1) if zone_m else "Azeroth"
        subzone = subzone_m.group(1) if subzone_m else ""
        ctype = ctype_m.group(1) if ctype_m else "Humanoid"
        model_id = int(model_m.group(1)) if model_m else None

        meta = KNOWN_NPC_ARCHETYPES.get(npc_id, {})
        if not race:
            race = meta.get(
                "race",
                "Undead" if ctype == "Untoter" else ("Entartung" if ctype == "Entartung" else infer_race_from_name_or_zone(name, zone)),
            )
        archetype = meta.get("archetype", f"{race.lower().replace(' ', '_')}_{sex.lower()}")
        role_desc = meta.get("role", f"{name} ({race}, {sex})")

        # 1. Extract Quests
        q_matches = re.finditer(
            r'\[(\d+)\]\s*=\s*\{\s*\["quest_detail"\]\s*=\s*"((?:[^"\\]|\\.)*)",\s*\["quest_progress"\]\s*=\s*"((?:[^"\\]|\\.)*)",\s*\["quest_complete"\]\s*=\s*"((?:[^"\\]|\\.)*)"',
            block,
        )
        for qm in q_matches:
            quest_id = int(qm.group(1))
            phases = {
                "Desc": qm.group(2).replace('\\"', '"'),
                "Prog": qm.group(3).replace('\\"', '"'),
                "Comp": qm.group(4).replace('\\"', '"'),
            }
            for phase, raw_text in phases.items():
                if not raw_text.strip():
                    continue
                target_filename = f"{quest_id}_{phase}.ogg"
                source_ogg = vo_dir / target_filename
                if not source_ogg.exists():
                    continue

                spoken_text, stage_cues = strip_stage_directions_and_placeholders(raw_text)
                entries.append({
                    "entry_id": f"{quest_id}_{phase}",
                    "kind": "quest",
                    "quest_id": quest_id,
                    "quest_title_de": f"Quest #{quest_id}",
                    "phase": phase,
                    "npc_id": npc_id,
                    "npc_name": name,
                    "race": race,
                    "gender": sex,
                    "creature_type": ctype,
                    "zone": zone,
                    "subzone": subzone,
                    "model_file_id": model_id,
                    "voice_archetype": archetype,
                    "role_description": role_desc,
                    "source_ogg_filename": target_filename,
                    "source_ogg_path": str(source_ogg),
                    "source_duration_sec": durations.get(target_filename, 0.0),
                    "target_ogg_filename": target_filename,
                    "raw_de_text": raw_text,
                    "clean_de_text": spoken_text,
                    "stage_directions": stage_cues,
                })

        # 2. Extract Gossip (if enabled)
        if include_gossip:
            gossip_matches = re.findall(r'\["([0-9a-f]{32}_[A-Za-z]+)"\]\s*=\s*"((?:[^"\\]|\\.)*)"', block)
            en_gossips = sorted(vo_dir.glob(f"{npc_id}_Gossip_*.ogg"))
            if gossip_matches and en_gossips:
                g_entries = []
                for g_hash_gender, raw_g_text in gossip_matches:
                    raw_g_text = raw_g_text.replace('\\"', '"')
                    if not raw_g_text.strip():
                        continue
                    target_g_filename = f"{npc_id}_Gossip_{g_hash_gender}.ogg"
                    spoken_text, stage_cues = strip_stage_directions_and_placeholders(raw_g_text)
                    g_entries.append({
                        "entry_id": f"{npc_id}_Gossip_{g_hash_gender[:8]}",
                        "kind": "gossip",
                        "quest_id": npc_id,
                        "quest_title_de": f"Gossip: {name}",
                        "phase": "Gossip",
                        "npc_id": npc_id,
                        "npc_name": name,
                        "race": race,
                        "gender": sex,
                        "creature_type": ctype,
                        "zone": zone,
                        "subzone": subzone,
                        "model_file_id": model_id,
                        "voice_archetype": archetype,
                        "role_description": role_desc,
                        "target_ogg_filename": target_g_filename,
                        "raw_de_text": raw_g_text,
                        "clean_de_text": spoken_text,
                        "stage_directions": stage_cues,
                    })
                en_sorted = sorted(en_gossips, key=lambda p: durations.get(p.name, 0.0))
                g_sorted = sorted(g_entries, key=lambda e: len(e["clean_de_text"]))
                en_padded = (en_sorted * (len(g_sorted) // max(1, len(en_sorted)) + 1))[:len(g_sorted)]
                for e, enf in zip(g_sorted, en_padded):
                    e["source_ogg_filename"] = enf.name
                    e["source_ogg_path"] = str(enf)
                    e["source_duration_sec"] = durations.get(enf.name, 0.0)
                    e["en_ref_match"] = "length_rank"
                entries.extend(g_entries)

    return entries


def resolve_missing_pack_quests_byod(
    files: List[str],
    vo_dir: Path,
    durations: Dict[str, float],
    existing_entries: List[Dict],
    cache_file: Path,
    custom_texts_dir: Path,
    max_workers: int = 4,
    ai_settings: Optional[Dict] = None,
    en_cache_file: Optional[Path] = None,
    trans_cache_file: Optional[Path] = None,
) -> List[Dict]:
    """
    Resolves quest texts (Desc, Prog, Comp) for `<QuestID>_<Phase>.ogg` files in the
    voiceover pack that aren't in SavedVariables yet, using strictly user-provided sources:
      1. Local JSON/CSV files in `data/custom_texts/` or cached JSON in `data/raw_de_text/`.
      2. Optional user-configured external endpoint (`QUEST_TEXT_PROVIDER_URL` in `.env`).
      3. Optional AI translation from user-provided English cache or `QUEST_EN_PROVIDER_URL`.
    """
    existing_keys = {e["target_ogg_filename"] for e in existing_entries}
    quest_files = [f for f in files if re.match(r"^\d+_(Desc|Prog|Comp)\.ogg$", f) and f not in existing_keys]
    needed_qids = sorted({int(f.split("_")[0]) for f in quest_files})
    needed_phases_by_qid: Dict[int, set] = {}
    for f in quest_files:
        q_str, ph_ext = f.split("_", 1)
        needed_phases_by_qid.setdefault(int(q_str), set()).add(ph_ext.replace(".ogg", ""))

    cache: Dict[str, Dict] = {}
    if cache_file.exists():
        try:
            cache = json.loads(cache_file.read_text(encoding="utf-8"))
        except Exception:
            cache = {}

    # Merge any user-provided JSON quest dictionaries in data/custom_texts/
    if custom_texts_dir.exists():
        for jf in sorted(custom_texts_dir.glob("*.json")):
            try:
                user_q = json.loads(jf.read_text(encoding="utf-8"))
                if isinstance(user_q, dict):
                    for k, v in user_q.items():
                        if isinstance(v, dict):
                            cache[str(k)] = v
            except Exception:
                pass

    provider_url = os.getenv("QUEST_TEXT_PROVIDER_URL", "").strip()
    en_provider_url = os.getenv("QUEST_EN_PROVIDER_URL", "").strip()

    if provider_url:
        to_fetch = []
        for qid in needed_qids:
            c = cache.get(str(qid))
            needed_ph = needed_phases_by_qid.get(qid, set())
            if not c or not all((c.get(p) or "").strip() for p in needed_ph):
                to_fetch.append(qid)

        if to_fetch:
            console.print(f"[cyan]• Fetching {len(to_fetch)} quests from user-configured QUEST_TEXT_PROVIDER_URL ({max_workers} threads)...[/cyan]")
            begin_step("analyze", total=len(to_fetch))
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}% ({task.completed}/{task.total})"),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                task = progress.add_task("[green]Fetching Quest Texts...", total=len(to_fetch))
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
                    future_to_qid = {
                        pool.submit(fetch_external_quest_from_template, qid, provider_url): qid
                        for qid in to_fetch
                    }
                    for idx, fut in enumerate(concurrent.futures.as_completed(future_to_qid), 1):
                        qid = future_to_qid[fut]
                        try:
                            res = fut.result()
                            existing_c = cache.get(str(qid)) or {}
                            for k, v in res.items():
                                if v and not existing_c.get(k):
                                    existing_c[k] = v
                            cache[str(qid)] = existing_c
                        except Exception:
                            pass
                        progress.advance(task)
                        update_current({
                            "entry_id": f"fetch:{qid}",
                            "npc_name": "Custom Provider",
                            "quest_title_de": (cache.get(str(qid)) or {}).get("title", f"Quest #{qid}"),
                            "phase": "Ingest",
                            "status": "fetching",
                            "cer": None,
                        })
                        increment(processed=1)
                        if idx % 100 == 0:
                            cache_file.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")
            cache_file.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")

    # --- Optional AI Translation for quests missing target-language phases ---
    en_cache: Dict[str, Dict] = {}
    trans_cache: Dict[str, Dict] = {}
    translate = bool(ai_settings and ai_is_available(ai_settings) and en_cache_file and trans_cache_file)
    if translate:
        for cf in (en_cache_file, trans_cache_file):
            if cf.exists():
                try:
                    if cf == en_cache_file:
                        en_cache = json.loads(cf.read_text(encoding="utf-8"))
                    else:
                        trans_cache = json.loads(cf.read_text(encoding="utf-8"))
                except Exception:
                    pass
        missing_de = []
        for qid in needed_qids:
            qid_str = str(qid)
            c = cache.get(qid_str) or {}
            tr = trans_cache.get(qid_str) or {}
            needed_ph = needed_phases_by_qid.get(qid, set())
            if any(not (c.get(p) or "").strip() and not (tr.get(p) or "").strip() for p in needed_ph):
                if qid_str in en_cache or en_provider_url:
                    missing_de.append(qid)
        if missing_de:
            console.print(f"[magenta]• Translating {len(missing_de)} quests with missing phases via AI ({ai_settings['provider']})...[/magenta]")
            begin_step("analyze", total=len(missing_de))
            for idx, qid in enumerate(missing_de, 1):
                qid_str = str(qid)
                needed_ph = needed_phases_by_qid.get(qid, set())
                c = cache.get(qid_str) or {}
                update_current({
                    "entry_id": f"translate:{qid}",
                    "npc_name": ai_settings["provider"],
                    "quest_title_de": c.get("title") or f"Quest #{qid}",
                    "phase": "Translation",
                    "status": "translating",
                    "cer": None,
                })
                increment(processed=1)
                en = en_cache.get(qid_str)
                missing_ph = {p for p in needed_ph if not (c.get(p) or "").strip()}
                if en is None and en_provider_url:
                    en = fetch_external_quest_from_template(qid, en_provider_url)
                    en_cache[qid_str] = en
                if not en:
                    continue
                tdata = dict(trans_cache.get(qid_str) or {
                    "title": None, "Desc": None, "Prog": None, "Comp": None,
                    "start_npc_id": en.get("start_npc_id", 0), "start_npc_name": en.get("start_npc_name", "NPC"),
                    "end_npc_id": en.get("end_npc_id", 0), "end_npc_name": en.get("end_npc_name", "NPC"),
                })
                title_de = (c.get("title") or "").strip()
                if (not title_de or title_de.startswith("Quest #")) and not (tdata.get("title") or "").strip():
                    title_en = (en.get("title") or "").strip()
                    if title_en and not title_en.startswith("Quest #"):
                        t, note, usage = translate_to_german(ai_settings, title_en)
                        if t:
                            tdata["title"] = t
                            log_ai_call(ai_settings["provider"], ai_settings["model"], f"{qid}_Title", note, ok=True, tokens=usage, cost_usd=usage.get("cost_usd", 0.0))
                for phase in sorted(missing_ph):
                    if (tdata.get(phase) or "").strip():
                        continue
                    txt = (en.get(phase) or "").strip()
                    if not txt:
                        continue
                    t, note, usage = translate_to_german(ai_settings, txt)
                    if t:
                        tdata[phase] = t
                        log_ai_call(ai_settings["provider"], ai_settings["model"], f"{qid}_{phase}", note, ok=True, tokens=usage, cost_usd=usage.get("cost_usd", 0.0))
                if any((tdata.get(k) or "").strip() for k in ("Desc", "Prog", "Comp", "title")):
                    trans_cache[qid_str] = tdata
                if idx % 50 == 0 or idx == len(missing_de):
                    trans_cache_file.write_text(json.dumps(trans_cache, indent=2, ensure_ascii=False), encoding="utf-8")
                    en_cache_file.write_text(json.dumps(en_cache, indent=2, ensure_ascii=False), encoding="utf-8")

    resolved_entries = []
    for fname in quest_files:
        qid_str, phase_ext = fname.split("_", 1)
        qid = int(qid_str)
        phase = phase_ext.replace(".ogg", "")
        c_data = cache.get(qid_str) or {}
        t_data = (trans_cache.get(qid_str) or {}) if translate else {}
        translated_by_ai = False
        raw_text = (c_data.get(phase) or "").strip()
        if not raw_text and (t_data.get(phase) or "").strip():
            raw_text = (t_data.get(phase) or "").strip()
            translated_by_ai = True
        if not raw_text:
            continue
        qdata = {
            "title": (c_data.get("title") if (c_data.get("title") and not str(c_data.get("title")).startswith("Quest #")) else None)
                     or t_data.get("title")
                     or c_data.get("title")
                     or f"Quest #{qid}",
            "Desc": c_data.get("Desc") or t_data.get("Desc") or "",
            "Prog": c_data.get("Prog") or t_data.get("Prog") or "",
            "Comp": c_data.get("Comp") or t_data.get("Comp") or "",
            "start_npc_id": c_data.get("start_npc_id") or t_data.get("start_npc_id") or 0,
            "start_npc_name": (c_data.get("start_npc_name") if c_data.get("start_npc_name") not in (None, "", "Unbekannter NPC") else None)
                              or t_data.get("start_npc_name")
                              or "NPC",
            "end_npc_id": c_data.get("end_npc_id") or t_data.get("end_npc_id") or 0,
            "end_npc_name": (c_data.get("end_npc_name") if c_data.get("end_npc_name") not in (None, "", "Unbekannter NPC") else None)
                            or t_data.get("end_npc_name")
                            or "NPC",
        }

        npc_id = qdata.get("start_npc_id", 0) if phase == "Desc" else qdata.get("end_npc_id", 0)
        npc_name = qdata.get("start_npc_name", "NPC") if phase == "Desc" else qdata.get("end_npc_name", "NPC")
        meta = KNOWN_NPC_ARCHETYPES.get(npc_id, {})
        race = meta.get("race") or infer_race_from_name_or_zone(npc_name)
        archetype = meta.get("archetype", f"npc_{npc_id}" if npc_id else f"quest_{qid}")
        spoken_text, stage_cues = strip_stage_directions_and_placeholders(raw_text)
        if not spoken_text:
            continue

        entry = {
            "entry_id": f"{qid}_{phase}",
            "kind": "quest",
            "quest_id": qid,
            "quest_title_de": qdata.get("title", f"Quest #{qid}"),
            "phase": phase,
            "npc_id": npc_id,
            "npc_name": npc_name,
            "npc_name_source": "byod_cache",
            "race": race,
            "gender": "Neutral",
            "creature_type": "Humanoid",
            "zone": "Custom",
            "subzone": "",
            "model_file_id": None,
            "voice_archetype": archetype,
            "role_description": f"{npc_name} ({race})",
            "source_ogg_filename": fname,
            "source_ogg_path": str(vo_dir / fname),
            "source_duration_sec": durations.get(fname, 0.0),
            "target_ogg_filename": fname,
            "raw_de_text": raw_text,
            "clean_de_text": spoken_text,
            "stage_directions": stage_cues,
        }
        if translated_by_ai:
            entry["translated_by_ai"] = True
        resolved_entries.append(entry)

    return resolved_entries


def select_benchmark_entries(entries: List[Dict], limit: int = 20) -> List[Dict]:
    """Selects diverse dialogue entries for a pilot run, or all if limit == 0."""
    if limit <= 0 or limit >= len(entries):
        return sorted(entries, key=lambda x: (x.get("quest_id", 0), x.get("phase", "")))

    scored = []
    seen_npcs = {}
    for e in entries:
        text_len = len(e["clean_de_text"])
        dur = e["source_duration_sec"]
        arch = e["voice_archetype"]
        npc_count = seen_npcs.get(arch, 0)
        length_score = 100 if (80 <= text_len <= 460) else (60 if text_len > 460 else 40)
        dur_score = 50 if dur >= 6.0 else 20
        diversity_bonus = max(0, 50 - npc_count * 12)
        total_score = length_score + dur_score + diversity_bonus
        scored.append((total_score, e))
        seen_npcs[arch] = npc_count + 1

    scored.sort(key=lambda x: (-x[0], x[1].get("quest_id", 0), x[1].get("phase", "")))
    selected = [item[1] for item in scored[:limit]]
    selected.sort(key=lambda x: (x.get("quest_id", 0), x.get("phase", "")))
    return selected


def main():
    parser = argparse.ArgumentParser(description="Step 1: Universal BYOD Ingest & Addon Analyzer.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--adapter", default=None, choices=["generic", "wow_chatty"], help="Ingest adapter mode (generic CSV/JSON or wow_chatty)")
    parser.add_argument("--custom-data", default=None, help="Path to custom CSV/JSON file or directory (used by generic adapter or as extra BYOD input)")
    parser.add_argument("--expansion", default=None, help="Project / Expansion name (e.g. Generic_Demo, Midnight, The_War_Within)")
    parser.add_argument("--limit", type=int, default=None, help="Override pilot_limit (0 = all entries)")
    parser.add_argument("--include-gossip", action="store_true", help="Include recorded Gossip lines from SavedVariables")
    parser.add_argument("--scrape-all-quests", action="store_true", help="Resolve additional quests from local BYOD caches or QUEST_TEXT_PROVIDER_URL")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    load_dotenv_file(project_root)
    cfg_path = project_root / args.config
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    expansion = args.expansion or cfg["project"].get("expansion", "Midnight")
    adapter = args.adapter or cfg["project"].get("adapter", "wow_chatty")
    pilot_limit = args.limit if args.limit is not None else cfg["project"].get("pilot_limit", 20)

    raw_de_dir = project_root / cfg["paths"]["data_raw_de_text"]
    raw_de_dir.mkdir(parents=True, exist_ok=True)
    custom_texts_dir = Path(args.custom_data) if args.custom_data else (project_root / cfg["paths"].get("data_custom_texts", "data/custom_texts"))
    orig_dir = project_root / cfg["paths"]["data_original"]
    orig_dir.mkdir(parents=True, exist_ok=True)

    wow_addons_dir = Path(os.getenv("WOW_ADDONS_DIR") or cfg["paths"]["wow_addons_dir"])
    source_pack_dir = wow_addons_dir / f"ChattyLittleNpc_{expansion}_voiceovers"
    if not source_pack_dir.exists():
        source_pack_dir = Path(cfg["paths"]["source_pack_dir"])

    # Auto-switch to generic BYOD mode if no WoW pack is present and custom data exists
    if adapter == "wow_chatty" and not (source_pack_dir / "VoiceoversList.lua").exists():
        if custom_texts_dir.exists() and (custom_texts_dir.is_file() or list(custom_texts_dir.glob("*.csv")) or list(custom_texts_dir.glob("*.json"))):
            adapter = "generic"

    target_addon_name = f"ChattyLittleNpc_{expansion}_deDE_voiceovers" if adapter == "wow_chatty" else f"{expansion}_deDE_voiceovers"

    console.rule(f"[bold cyan]Step 1: Universal BYOD Ingest Analyzer (Mode: {adapter} | Target: {target_addon_name})[/bold cyan]")
    begin_step("analyze", total=0, expansion=expansion)

    all_de_entries: List[Dict] = []
    files: List[str] = []
    quest_ogg_count = 0
    gossip_ogg_count = 0

    if adapter == "generic":
        all_de_entries = load_generic_custom_dataset(custom_texts_dir, orig_dir)
        files = [e["source_ogg_filename"] for e in all_de_entries]
        quest_ogg_count = len(all_de_entries)
        console.print(f"• Generic BYOD Dataset: [bold green]{len(all_de_entries)}[/bold green] dialogue entries loaded from [cyan]{custom_texts_dir}[/cyan]")
    else:
        vo_list_path = source_pack_dir / "VoiceoversList.lua"
        vo_dir = source_pack_dir / "voiceovers"
        sv_path = Path(os.getenv("WOW_SAVED_VARS") or cfg["paths"]["saved_variables_lua"])

        files, durations = parse_voiceovers_list(vo_list_path)
        quest_ogg_count = sum(1 for x in files if re.match(r"^\d+_(Desc|Prog|Comp)\.ogg$", x))
        gossip_ogg_count = sum(1 for x in files if "_Gossip_" in x)
        other_ogg_count = len(files) - quest_ogg_count - gossip_ogg_count

        console.print(f"• Source Pack: [bold]{source_pack_dir.name}[/bold]")
        console.print(f"• Total {expansion} Audio Files: [green]{len(files)}[/green] "
                      f"(Quests: {quest_ogg_count}, Gossip: {gossip_ogg_count}, Other: {other_ogg_count})")

        orig_link = orig_dir / f"{expansion.lower()}_voiceovers"
        if not orig_link.exists() and vo_dir.exists():
            os.symlink(vo_dir, orig_link)

        all_de_entries = parse_saved_variables(sv_path, vo_dir, durations, include_gossip=args.include_gossip)
        console.print(f"• Extracted [bold green]{len(all_de_entries)}[/bold green] verified entries from local SavedVariables.")

        if args.scrape_all_quests:
            quest_cache_file = raw_de_dir / f"byod_{expansion.lower()}_quests_cache.json"
            en_cache_file = raw_de_dir / f"byod_{expansion.lower()}_quests_en_cache.json"
            trans_cache_file = raw_de_dir / f"quest_translations_{expansion.lower()}_cache.json"
            try:
                ai_settings = get_ai_settings(cfg)
                if not (cfg.get("ai", {}).get("translate_missing_quests", True)):
                    ai_settings = None
            except Exception:
                ai_settings = None
            resolved = resolve_missing_pack_quests_byod(
                files, vo_dir, durations, all_de_entries, quest_cache_file,
                custom_texts_dir=custom_texts_dir,
                ai_settings=ai_settings,
                en_cache_file=en_cache_file,
                trans_cache_file=trans_cache_file,
            )
            console.print(f"• Resolved [bold green]{len(resolved)}[/bold green] additional quest phases from BYOD cache / provider!")
            all_de_entries.extend(resolved)

    selected_entries = select_benchmark_entries(all_de_entries, limit=pilot_limit)

    full_catalog_path = raw_de_dir / "midnight_de_full_catalog.json"
    pilot_catalog_path = raw_de_dir / "midnight_de_pilot_20.json"

    addon_schema_report = {
        "addon_architecture": {
            "adapter": adapter,
            "expansion": expansion,
            "pack_naming_convention": "ChattyLittleNpc_<Expansion>_<locale>_voiceovers" if adapter == "wow_chatty" else "<Project>_<locale>_voiceovers",
            "target_pack_name": target_addon_name,
            "quest_lookup_pattern": "<questId>_<Desc|Prog|Comp>.ogg",
            "gossip_lookup_pattern": "<npcId>_<Gossip|GameObject|Item>_<MD5(npcId + CleanText(de_text))>[_<Male|Female|Neutral>].ogg",
            "greeting_lookup_pattern": "<npcId>_QuestGreeting.ogg",
            "fallback_mechanism": "Per-file fallback to original reference audio when a localized file is absent.",
        },
        "statistics": {
            "total_en_files": len(files),
            "quest_en_files": quest_ogg_count,
            "gossip_en_files": gossip_ogg_count,
            "verified_de_quest_entries": len(all_de_entries),
            "selected_pilot_entries": len(selected_entries),
        },
        "entries": selected_entries,
    }

    full_catalog_path.write_text(json.dumps(all_de_entries, indent=2, ensure_ascii=False), encoding="utf-8")
    pilot_catalog_path.write_text(json.dumps(addon_schema_report, indent=2, ensure_ascii=False), encoding="utf-8")

    table = Table(title=f"Selected {len(selected_entries)} {expansion} Entries for Active Batch")
    table.add_column("#", justify="right", style="dim")
    table.add_column("File", style="cyan")
    table.add_column("Title / Scene", style="bold white")
    table.add_column("Speaker", style="green")
    table.add_column("FX Profile", style="magenta")
    table.add_column("Ref Dur", justify="right")
    table.add_column("Chars", justify="right")

    for idx, item in enumerate(selected_entries[:30], 1):
        table.add_row(
            str(idx),
            item["target_ogg_filename"],
            item["quest_title_de"],
            item["npc_name"],
            item["race"],
            f"{item['source_duration_sec']:.1f}s",
            str(len(item["clean_de_text"])),
        )

    console.print(table)
    if len(selected_entries) > 30:
        console.print(f"[dim]... and {len(selected_entries) - 30} more entries in {pilot_catalog_path.name}[/dim]")
    console.print(f"[bold green]✓ Saved mapping catalog ({len(selected_entries)} active entries) -> {pilot_catalog_path}[/bold green]")
    finish_step(ok=True)


if __name__ == "__main__":
    main()
