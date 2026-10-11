"""Make ``gsv3`` (and, through it, ``genstudy``/``gsv2``/``src/model``) importable from the scripts."""

from __future__ import annotations

import sys
from pathlib import Path

STUDY_DIR = Path(__file__).resolve().parents[1]

if str(STUDY_DIR) not in sys.path:
    sys.path.insert(0, str(STUDY_DIR))
