@echo off
setlocal

call "%~dp0scripts\facemesh_launch.bat" --calibrate --force-recalibrate --calibration-profile default %*
exit /b %ERRORLEVEL%
