@echo off
setlocal
cd /d "%~dp0"

if not exist "calibration-default.json" if not exist "calibration.json" (
    echo [facemesh] no default calibration profile in "%CD%"
    echo [facemesh] run calibrate.bat first
    pause
    exit /b 1
)

call "%~dp0scripts\facemesh_launch.bat" --udp --calibration-profile default %*
exit /b %ERRORLEVEL%
