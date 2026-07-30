@echo off
setlocal
powershell.exe -NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass -File "%~dp0rw-wps.ps1" %*
exit /b %ERRORLEVEL%
