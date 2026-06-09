"""Make ``src/`` importable when scripts are executed from the repository root.

``python scripts/train.py`` puts ``scripts/`` on ``sys.path``, not ``src/``.
This helper keeps the repository runnable without an editable install, which
matters for reviewers who just cloned the project.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
