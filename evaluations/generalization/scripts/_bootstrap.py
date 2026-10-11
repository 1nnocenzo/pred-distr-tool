"""Make ``genstudy`` importable when a script is run directly.

The scripts in this directory are meant to be executed as files
(``python evaluations/generalization/scripts/run_*.py``) rather than as installed
console entry points, so they prepend the study directory to ``sys.path`` before
importing the library.  Keeping that single line here avoids repeating path
surgery in every script.
"""

from __future__ import annotations

import sys
from pathlib import Path

STUDY_DIR = Path(__file__).resolve().parents[1]

if str(STUDY_DIR) not in sys.path:
    sys.path.insert(0, str(STUDY_DIR))
