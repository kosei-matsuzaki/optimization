"""Paths and GitHub constants shared across the web backend.

Importing this module also inserts the project root onto ``sys.path`` so that
``core`` and ``quick_check`` remain importable when the app is launched via
``python3 web/app.py``.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Project root = parent of web/  (this file lives at web/app_lib/config.py).
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BASE_DIR    = ROOT
RESULTS_DIR = BASE_DIR / "results"
QUICK_CHECK = BASE_DIR / "quick_check.py"
PID_FILE    = BASE_DIR / ".quick.pid"
DIR_FILE    = BASE_DIR / ".quick.dir"

GH_REPO     = "kosei-matsuzaki/optimization"
GH_WORKFLOW = "run.yml"
