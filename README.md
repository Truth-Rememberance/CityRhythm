# CityRhythm

CityRhythm is a software-only, all-free urban dynamics and threat detection platform.

## Setup

### Python Backend
Ensure Python 3.11 is installed (see `.python-version`):
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Frontend
Ensure Node 24 is active (see `.nvmrc`):
```bash
nvm use 24
cd frontend
npm install
```

## Role Ownership

| Role | Name | Responsibilities / Owned Folders |
| :--- | :--- | :--- |
| **R1** | | `contracts/`, `backend/hub/`, `backend/common/` (except `llm.py`), `scripts/`, `tests/e2e/`, `frontend/panels/replay/`, `run_all.py` |
| **R2** | | `config/`, `data/`, `backend/sim/`, `tests/sim/` |
| **R3** | | `backend/sync/`, `frontend/panels/sync/`, `tests/sync/` |
| **R4** | | `backend/threat/`, `frontend/panels/threat/`, `tests/threat/` |
| **R5** | | `backend/fix/`, `backend/common/llm.py`, `frontend/panels/fix/`, `tests/fix/` |
| **R6** | | `frontend/` (core app structure, shell, routing) |

*Pitch slides (`pitch/slides/`) are co-owned by everyone (R1–R6).*

## Repo Rules

1. **Own folder only**: Work only within your designated role folders and assigned components.
2. **Fast PR merges**: PRs into `main` must be reviewed and merged within 10 minutes.
3. **`main` must always run**: Never commit broken code to `main`. All tests and lint checks must pass.
4. **No secrets**: Do not commit secrets, credentials, or `.env` files. Use `.env.example` as reference.
5. **Naming convention**: All file, module, function, and variable names must use `snake_case`.
6. **Master runner**: Note that `run_all.py` comes from R1.

