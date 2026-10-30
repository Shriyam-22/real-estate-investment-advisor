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
echo.
echo Files changed since last push:
git status --short
echo.
set "MSG="
if exist commit_message.txt (
  set /p MSG=<commit_message.txt
  del commit_message.txt
) else (
  set /p MSG=Describe what you changed ^(then press Enter^): 
)
if "%MSG%"=="" set "MSG=Update project files"
git add .
git commit -m "%MSG%"
git remote remove origin 2>nul
git remote add origin %REPO%
git push -u origin main
echo.
echo If a GitHub sign-in window popped up, sign in as Shriyam-22 and it will finish.
pause
