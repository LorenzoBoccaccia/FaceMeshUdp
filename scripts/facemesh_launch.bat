@echo off
setlocal
cd /d "%~dp0.."

set "PYTHON=%CD%\.venv\Scripts\python.exe"
if not exist "%PYTHON%" (
    echo [facemesh] no virtualenv at "%PYTHON%"
    echo [facemesh] create it with:
    echo     python -m venv .venv
    echo     .venv\Scripts\python.exe -m pip install -e ".[dev]"
    pause
    exit /b 1
)

"%PYTHON%" -m facemesh_app.main %*
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" (
    echo [facemesh] exited with code %EXIT_CODE%
    pause
)
exit /b %EXIT_CODE%
