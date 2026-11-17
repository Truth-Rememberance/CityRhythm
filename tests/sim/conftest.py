"""Put the repo root on sys.path so `pytest` can import `backend.*`.

CI runs bare `pytest -q` (no `python -m`), which does not add the current
working directory to sys.path. Adding the repo root here (tests/sim-only,
via this conftest) makes `import backend.sim` work under both invocations.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
