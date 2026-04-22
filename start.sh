#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-5174}"

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r backend/requirements.txt

pushd frontend >/dev/null
npm install
popd >/dev/null

pushd backend >/dev/null
python -m app.build
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
