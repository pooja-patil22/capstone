@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo .venv not found. Run setup_venv.bat first.
  exit /b 1
)

call ".venv\Scripts\activate.bat"
if errorlevel 1 (
  echo Failed to activate .venv.
  exit /b 1
)

python -m pip freeze > requirements.txt
if errorlevel 1 exit /b 1

echo requirements.txt updated from .venv

endlocal
