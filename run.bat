@echo off
REM Get the directory of the batch file
setlocal
set "SCRIPT_DIR=%~dp0"

REM Change to that directory
cd /d "%SCRIPT_DIR%"

REM Activate the virtual environment
call ".venv\Scripts\activate.bat"

REM Run the Python script
python main.py

pause
