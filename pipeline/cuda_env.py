#!/usr/bin/env python3
"""
CUDA-Laufzeit-Vorbereitung für Faster-Whisper (pipeline/cuda_env.py)
--------------------------------------------------------------------
torch liefert CUDA-13-Bibliotheken (nvidia/cu13/lib), ctranslate2
(Faster-Whisper) linkt aber gegen libcublas.so.12. Dieses Modul legt die
NVIDIA-lib-Verzeichnisse auf LD_LIBRARY_PATH, damit der CUDA-Start klappt,
bevor ein CPU-Fallback greift.

Wichtig: VOR dem Import von faster_whisper ausführen:

    from pipeline.cuda_env import setup as setup_cuda_env
    setup_cuda_env()
"""

import os
import sysconfig
from pathlib import Path


def _prepend(path: Path) -> None:
    cur = os.environ.get("LD_LIBRARY_PATH", "")
    entry = str(path)
    if entry not in cur.split(os.pathsep):
        os.environ["LD_LIBRARY_PATH"] = entry + (os.pathsep + cur if cur else "")


def setup() -> None:
    site = Path(sysconfig.get_paths().get("purelib") or "")
    for cand in ("nvidia/cublas/lib", "nvidia/cu13/lib"):
        p = site / cand
        if p.exists():
            _prepend(p)
