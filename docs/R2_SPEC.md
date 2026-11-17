You are helping me build R2 (Simulation & Scenarios) of CityRhythm, a software-only hackathon project: a synthetic Bengaluru simulator feeding analysis modules and a React UI. I own ONLY backend/sim/, config/, data/, scripts/ (sim tools) and tests/sim/. First, save this entire message verbatim as docs/R2_SPEC.md and confirm. Do nothing else yet.

RULES
- Python 3.11, numpy (vectorized hot loop, no pandas inside step()), pydantic v2, pytest, ruff+black. Type hints, short docstrings explaining the math.
- Fully deterministic: all randomness from numpy Generators seeded per run (separate streams for process noise, sensor noise, scenario events). State must be deep-copyable (fork()).
- Never import backend/sync, threat, fix or hub. May import contracts/ and backend/common/ if they exist; otherwise use a local stub mirroring the schemas below, marked TODO(replace with contracts).
- TRUTH FIREWALL: hidden coupling params and ground truth live only in backend/sim/_truth.py. Nothing outside backend/sim may import it.
- All data is synthetic and must be labelled so. Work in small steps, run tests, show me evidence.
