@echo off
title MLflow - Real Estate Investment Advisor
cd /d "%~dp0"
set PY=python
where python >nul 2>nul || set PY="%USERPROFILE%\anaconda3\python.exe"
if not exist mlflow.db (
  echo No MLflow logs yet - training all 6 models, about 5 minutes...
  if not exist data\processed_housing.csv %PY% src\preprocess.py
  %PY% src\train.py
)
echo MLflow opens at http://localhost:5000  (close this window to stop)
start "" http://localhost:5000
%PY% -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
pause
