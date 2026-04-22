# RL Therapy Lab - Hopfield Postprandial Memory

Standalone coursework/research demo built only on the top-level `OhioT1DM` XML folder.

This project is intentionally separate from Glucoscope in repository structure, backend logic, APIs, feature pipeline, memory model, and frontend code. It uses Glucoscope only as a visual reference for calm dashboard layout, card presentation, chart containers, and navigation rhythm.

## What It Does

The application stores retrospective meal windows as fixed-length feature vectors and uses a continuous Hopfield-style associative memory to retrieve similar historical postprandial cases.

Core emphasis:

- similar-case retrieval instead of a classifier zoo
- prototype memories for major response types
- feature-grounded explanations for why cases match
- honest evaluation and limitations

It is not a clinical system, not a treatment recommender, and not a dosing tool.

## Data Scope

Raw input is read only from:

```text
../OhioT1DM
```

Supported raw streams:

- CGM
- meals / carbs
- bolus
- basal / temp basal
- patient id and timestamps
- Basis heart-rate events when available

## Built Results

Current generated artifacts use the real local OhioT1DM data and produced:

- `6` patients
- `1303` extracted meal windows
- `474` usable retrospective memory windows
- `329` train memories
- `85` feature dimensions in the full dataset build
- Hopfield top-1 same-label retrieval: `41.1%`
- Hopfield top-3 hit rate: `61.6%`
- mean reciprocal rank: `0.505`

These are retrospective research metrics, not clinical validation.

## Project Layout

```text
rl-therapy-lab-hopfield-memory/
  backend/
    app/
    tests/
  frontend/
    src/
  artifacts/
    datasets/
    models/
    reports/
  docs/
    coursework_report.md
  README.md
  start.cmd
  start.sh
```

## Backend

Backend stack:

- Python
- FastAPI
- NumPy / Pandas
- scikit-learn

Key endpoints:

- `GET /api/health`
- `GET /api/dashboard`
- `GET /api/windows`
- `GET /api/windows/{window_id}`
- `POST /api/memory/retrieve`
- `POST /api/memory/custom-query`
- `GET /api/prototypes`
- `GET /api/prototypes/{label}`
- `GET /api/evaluation`
- `GET /api/evaluation/noise`
- `GET /api/about`

## Frontend

Frontend stack:

- React
- Vite
- TypeScript
- Recharts

Pages:

- Overview
- Case Explorer
- Similar Cases
- Prototype Memory
- Evaluation
- About / Methodology

## Run Locally

### Windows

```bat
start.cmd
```

If port `8000` or `5174` is already busy:

```bat
set BACKEND_PORT=8012
set FRONTEND_PORT=5182
start.cmd
```

### Linux / macOS / WSL

```bash
./start.sh
```

If the default ports are busy:

```bash
BACKEND_PORT=8012 FRONTEND_PORT=5182 ./start.sh
```

Manual run:

```bash
cd backend
python -m pip install -r requirements.txt
python -m app.build
uvicorn app.main:app --reload
```

```bash
cd frontend
npm install
npm run dev
```

The Vite dev server proxies `/api` to `http://127.0.0.1:8000`.

## Tests

From `backend/`:

```bash
python -m pytest -q
```

Current tests cover:

- feature vector shape stability
- missing heart-rate handling
- scaler fit on train only
- exact top-k retrieval count
- finite energy values
- held-out split without self-retrieval leakage
- API health endpoint

## Artifacts

Generated files include:

- `artifacts/datasets/meal_windows.csv`
- `artifacts/datasets/meal_windows.json`
- `artifacts/datasets/feature_matrix.npy`
- `artifacts/datasets/feature_metadata.json`
- `artifacts/models/runtime_bundle.pkl`
- `artifacts/models/hopfield_memory.npz`
- `artifacts/reports/latest_metrics.json`
- `artifacts/reports/latest_report.md`
- `artifacts/reports/chart_data.json`
- `docs/coursework_report.md`

## Visual Reuse vs Logic Reuse

Visual ideas reused from Glucoscope:

- left-side navigation rail
- calm dashboard composition
- card-based metric summaries
- restrained chart/table styling

Explicitly not reused from Glucoscope:

- backend modules
- XML parsing logic
- meal-processing rules
- analytics logic
- RL logic
- therapy or dose recommendation logic

## Known Limitations

- Only six OhioT1DM participants are available.
- The response labels are deterministic retrospective categories.
- The memory vectors are pre-meal and context focused, so they should not be framed as clinical prediction.
- Heart-rate coverage is incomplete for some windows.
- Retrieval is often informative even when classification-style metrics remain modest.

## Next Steps

- Add a small classical binary Hopfield educational page as a secondary demo.
- Add richer prototype-local explanations and medoid comparison views.
- Add more robust query editing around the custom-query endpoint.
