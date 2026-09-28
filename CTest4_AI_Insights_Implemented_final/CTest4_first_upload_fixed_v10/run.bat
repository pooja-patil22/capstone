@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
	echo Local .venv not found. Running setup_venv.bat...
	call setup_venv.bat
	if errorlevel 1 (
		echo Setup failed. Cannot start app.
		pause
		exit /b 1
	)
)

echo Starting QA MongoDB to ChromaDB UI using .venv...
".venv\Scripts\python.exe" -m streamlit run frontend/app.py
pause
endlocal
