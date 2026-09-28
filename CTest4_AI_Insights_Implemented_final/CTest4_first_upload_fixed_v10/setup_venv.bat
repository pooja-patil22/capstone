@echo off
setlocal
cd /d "%~dp0"

echo Setting up local Python virtual environment in .venv ...

if exist ".venv\Scripts\python.exe" (
  echo Existing virtual environment found.
) else (
  py -3 -m venv .venv
  if errorlevel 1 (
    echo Failed to create virtual environment using py launcher.
    echo Install Python 3 and ensure the py launcher is available.
    exit /b 1
  )
)

call ".venv\Scripts\activate.bat"
if errorlevel 1 (
  echo Failed to activate .venv.
  exit /b 1
)

python -m pip install --upgrade pip
if errorlevel 1 exit /b 1

python -m pip install -r backend/requirements.txt
if errorlevel 1 exit /b 1

echo.
echo Virtual environment is ready.
echo Activate with: .venv\Scripts\activate
echo Run app with: streamlit run app.py

endlocal
