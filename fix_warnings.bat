@echo off
title Fix Anaconda warnings
cd /d "%~dp0"
set PY="%USERPROFILE%\anaconda3\python.exe"
echo Close the "MLflow - Real Estate Investment Advisor" window first if it is open.
echo.
echo === Updating bottleneck and numexpr to versions that work with NumPy 2 ===
%PY% -m pip install --upgrade bottleneck numexpr
if errorlevel 1 (echo Update failed - make sure the MLflow window is closed, then run this again. & pause & exit /b 1)
echo.
echo === Checking - no red error text should appear below ===
%PY% -c "import pandas, mlflow; print('OK - pandas', pandas.__version__, '/ mlflow', mlflow.__version__)"
echo.
echo Done. Now double-click open_mlflow.bat to reopen the dashboard.
pause
