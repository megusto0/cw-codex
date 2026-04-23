#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-5174}"
HOPFIELD_BUNDLE="${PWD}/artifacts/models/runtime_bundle.pkl"
SIAMESE_BUNDLE="${PWD}/artifacts/models/siamese_runtime_bundle.pkl"
SOM_BUNDLE="${PWD}/artifacts/models/som_runtime_bundle.pkl"
COMPARISON_METRICS="${PWD}/artifacts/reports/comparison_metrics.json"

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r backend/requirements.txt

pushd frontend >/dev/null
npm install
popd >/dev/null

ARTIFACT_STATUS=1
pushd backend >/dev/null
if python -m app.artifact_status >/dev/null 2>&1; then
  ARTIFACT_STATUS=0
fi
popd >/dev/null

if [ "${FORCE_BUILD:-0}" = "1" ] || [ ! -f "$HOPFIELD_BUNDLE" ] || [ ! -f "$SIAMESE_BUNDLE" ] || [ ! -f "$SOM_BUNDLE" ] || [ ! -f "$COMPARISON_METRICS" ] || [ "$ARTIFACT_STATUS" -ne 0 ]; then
  pushd backend >/dev/null
  python -m app.build
  popd >/dev/null
else
  echo "Using existing artifacts. Set FORCE_BUILD=1 to rebuild."
fi

pushd backend >/dev/null
(uvicorn app.main:app --host 127.0.0.1 --port "$BACKEND_PORT" --reload) &
BACKEND_PID=$!
popd >/dev/null

pushd frontend >/dev/null
(VITE_API_BASE="http://127.0.0.1:${BACKEND_PORT}" BACKEND_PORT="$BACKEND_PORT" npm run dev -- --host 127.0.0.1 --port "$FRONTEND_PORT") &
FRONTEND_PID=$!
popd >/dev/null

trap 'kill $BACKEND_PID $FRONTEND_PID' EXIT
echo "Backend: http://127.0.0.1:${BACKEND_PORT}"
echo "Frontend: http://127.0.0.1:${FRONTEND_PORT}"
wait
