@echo off
setlocal

cd /d "%~dp0"
set "ROOT=%CD%"
set "VENV_PYTHON=%ROOT%\.venv\Scripts\python.exe"
if "%BACKEND_PORT%"=="" set "BACKEND_PORT=8000"
if "%FRONTEND_PORT%"=="" set "FRONTEND_PORT=5174"

if not exist ".venv\Scripts\python.exe" (
  python -m venv .venv
)

call .venv\Scripts\activate.bat
python -m pip install --upgrade pip || exit /b 1
python -m pip install -r backend\requirements.txt || exit /b 1

pushd frontend
call npm.cmd install || exit /b 1
popd

pushd backend
python -m app.build || exit /b 1
set "BACKEND_DIR=%CD%"
popd

pushd frontend
set "FRONTEND_DIR=%CD%"
popd

start "RL Therapy Lab Backend" /D "%BACKEND_DIR%" "%VENV_PYTHON%" -m uvicorn app.main:app --host 127.0.0.1 --port %BACKEND_PORT% --reload
start "RL Therapy Lab Frontend" /D "%FRONTEND_DIR%" cmd /k "set BACKEND_PORT=%BACKEND_PORT%&& set VITE_API_BASE=http://127.0.0.1:%BACKEND_PORT%&& call npm.cmd run dev -- --host 127.0.0.1 --port %FRONTEND_PORT%"

echo Backend: http://127.0.0.1:%BACKEND_PORT%
echo Frontend: http://127.0.0.1:%FRONTEND_PORT%
