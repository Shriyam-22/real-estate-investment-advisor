@echo off
title MLflow - Real Estate Investment Advisor
cd /d "%~dp0"
REM ---- pick a Python that has ready-made packages (Anaconda first; avoid 3.15) ----
set PY=
if exist "%USERPROFILE%\anaconda3\python.exe" set PY="%USERPROFILE%\anaconda3\python.exe"
if not defined PY (py -3.13 --version >nul 2>nul && set PY=py -3.13)
if not defined PY (py -3.12 --version >nul 2>nul && set PY=py -3.12)
if not defined PY (py -3.11 --version >nul 2>nul && set PY=py -3.11)
if not defined PY set PY=python
echo Using Python:
%PY% --version || (echo Python not found. & pause & exit /b 1)
echo === Installing packages (first run takes a few minutes) ===
%PY% -m pip install -r requirements-local.txt
if errorlevel 1 (echo Package install failed - see the error above. & pause & exit /b 1)
set MLFLOW_DISABLE_AGENT_HINT=1
if not exist mlflow.db (
  echo No MLflow logs yet - training all 6 models, about 5 minutes...
  if not exist data\processed_housing.csv %PY% src\preprocess.py
  %PY% src\train.py
  if errorlevel 1 (echo Training failed - see the error above. & pause & exit /b 1)
)
echo MLflow opens at http://localhost:5000  (close this window to stop)
start "" http://localhost:5000
%PY% -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
pause
