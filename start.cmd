@echo off
setlocal

cd /d "%~dp0"
set "ROOT=%CD%"
set "VENV_PYTHON=%ROOT%\.venv\Scripts\python.exe"
set "BACKEND_DIR=%ROOT%\backend"
set "FRONTEND_DIR=%ROOT%\frontend"
set "HOPFIELD_BUNDLE=%ROOT%\artifacts\models\runtime_bundle.pkl"
set "SIAMESE_BUNDLE=%ROOT%\artifacts\models\siamese_runtime_bundle.pkl"
set "SOM_BUNDLE=%ROOT%\artifacts\models\som_runtime_bundle.pkl"
set "COMPARISON_METRICS=%ROOT%\artifacts\reports\comparison_metrics.json"
if "%BACKEND_PORT%"=="" set "BACKEND_PORT=8000"
if "%FRONTEND_PORT%"=="" set "FRONTEND_PORT=5174"

if not exist ".venv\Scripts\python.exe" (
  python -m venv .venv
)

call .venv\Scripts\activate.bat
python -m pip install --upgrade pip || exit /b 1
python -m pip install -r backend\requirements.txt || exit /b 1

pushd "%FRONTEND_DIR%"
call npm.cmd install || exit /b 1
popd

set "ARTIFACT_STATUS=1"
pushd "%BACKEND_DIR%"
python -m app.artifact_status >nul 2>&1
set "ARTIFACT_STATUS=%ERRORLEVEL%"
popd

if /I "%FORCE_BUILD%"=="1" (
  echo Rebuilding artifacts because FORCE_BUILD=1
  pushd "%BACKEND_DIR%"
  python -m app.build || exit /b 1
  popd
) else (
  if not exist "%HOPFIELD_BUNDLE%" (
    echo Hopfield runtime bundle not found. Building artifacts...
    pushd "%BACKEND_DIR%"
    python -m app.build || exit /b 1
    popd
  ) else if not exist "%SIAMESE_BUNDLE%" (
    echo Siamese runtime bundle not found. Building artifacts...
    pushd "%BACKEND_DIR%"
    python -m app.build || exit /b 1
    popd
  ) else if not exist "%SOM_BUNDLE%" (
    echo SOM runtime bundle not found. Building artifacts...
    pushd "%BACKEND_DIR%"
    python -m app.build || exit /b 1
    popd
  ) else if not exist "%COMPARISON_METRICS%" (
    echo Comparison metrics cache not found. Building artifacts...
    pushd "%BACKEND_DIR%"
    python -m app.build || exit /b 1
    popd
  ) else if not "%ARTIFACT_STATUS%"=="0" (
    echo Existing artifacts were built from another data source. Rebuilding...
    pushd "%BACKEND_DIR%"
    python -m app.build || exit /b 1
    popd
  ) else (
    echo Using existing artifacts. Set FORCE_BUILD=1 to rebuild.
  )
)

set "LAUNCH_BACKEND=%ROOT%\scripts\launch_backend.cmd"
set "LAUNCH_FRONTEND=%ROOT%\scripts\launch_frontend.cmd"

start "RL Therapy Lab Backend" cmd /k call "%LAUNCH_BACKEND%"
start "RL Therapy Lab Frontend" cmd /k call "%LAUNCH_FRONTEND%"

echo Backend: http://127.0.0.1:%BACKEND_PORT%
echo Frontend: http://127.0.0.1:%FRONTEND_PORT%
