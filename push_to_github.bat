@echo off
title Push to GitHub
cd /d "%~dp0"
set REPO=https://github.com/Shriyam-22/real-estate-investment-advisor.git

git --version || (echo Git is not installed. & pause & exit /b 1)
if not exist .git (
  git init -b main
  git config user.name "Shriyam"
  git config user.email "shriyamgopalgupta@gmail.com"
)
git add .
git commit -m "Real Estate Investment Advisor: EDA, XGBoost models, MLflow tracking, Streamlit app"
git remote remove origin 2>nul
git remote add origin %REPO%
git push -u origin main
echo.
echo If a GitHub sign-in window popped up, sign in as Shriyam-22 and it will finish.
pause
