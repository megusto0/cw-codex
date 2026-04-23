@echo off
setlocal

if "%BACKEND_DIR%"=="" (
  echo BACKEND_DIR is not set.
  exit /b 1
)

if "%VENV_PYTHON%"=="" (
  echo VENV_PYTHON is not set.
  exit /b 1
)

if "%BACKEND_PORT%"=="" set "BACKEND_PORT=8000"

cd /d "%BACKEND_DIR%" || exit /b 1
set "PYTHONPATH=%BACKEND_DIR%"
"%VENV_PYTHON%" -m uvicorn app.main:app --app-dir "%BACKEND_DIR%" --host 127.0.0.1 --port %BACKEND_PORT% --reload
