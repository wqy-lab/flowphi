@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Missing Python environment. See docs\environment-setup.md.
    pause
    exit /b 1
)
if not "%~1"=="" (
    ".venv\Scripts\python.exe" main.py %*
) else if exist ".local\preview-demo.fingering.json" (
    ".venv\Scripts\python.exe" main.py --project ".local\preview-demo.fingering.json"
) else (
    ".venv\Scripts\python.exe" main.py --demo
)
if errorlevel 1 pause
