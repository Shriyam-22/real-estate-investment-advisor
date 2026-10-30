@echo off
title Real Estate Investment Advisor
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

echo.
echo === Installing packages (first run takes a few minutes) ===
%PY% -m pip install -r requirements-local.txt
if errorlevel 1 (echo Package install failed - see the error above. & pause & exit /b 1)

echo.
echo === Checking the saved models load on this machine ===
%PY% -c "import joblib; joblib.load('models/classifier.joblib'); joblib.load('models/regressor.joblib')" 2>nul
if errorlevel 1 (
  echo Saved models don't match your library versions - retraining, about 5 minutes...
  %PY% src\preprocess.py && %PY% src\train.py
)

echo.
echo === Starting the app - it opens in your browser at http://localhost:8501 ===
echo Keep this window open while you use the app. Close it to stop.
%PY% -m streamlit run app.py
pause
