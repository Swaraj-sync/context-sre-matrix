from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV_DIR = ROOT / "sre_architect_env"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ENV_DIR) not in sys.path:
    sys.path.insert(0, str(ENV_DIR))
