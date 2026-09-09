@echo off
setlocal
set "PYTHONPATH="
set "PYTHONHOME="
"%~dp0runtime\python.exe" -I -B "%~dp0launch_arwen.py" --tui-only %*
if errorlevel 1 pause
