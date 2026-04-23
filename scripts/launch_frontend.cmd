@echo off
setlocal

if "%FRONTEND_DIR%"=="" (
  echo FRONTEND_DIR is not set.
  exit /b 1
)

if "%BACKEND_PORT%"=="" set "BACKEND_PORT=8000"
if "%FRONTEND_PORT%"=="" set "FRONTEND_PORT=5174"

cd /d "%FRONTEND_DIR%" || exit /b 1
set "VITE_API_BASE=http://127.0.0.1:%BACKEND_PORT%"
set "BACKEND_PORT=%BACKEND_PORT%"
call npm.cmd run dev -- --host 127.0.0.1 --port %FRONTEND_PORT%
