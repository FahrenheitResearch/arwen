@echo off
setlocal
set "PYTHONPATH="
set "PYTHONHOME="
rem A package that carries its own runtime starts on that runtime and takes no --python.
if exist "%~dp0runtime\python.exe" (
    "%~dp0runtime\python.exe" -I -B "%~dp0launch_arwen.py" %*
) else if defined ARWEN_PYTHON (
    "%ARWEN_PYTHON%" -I -B "%~dp0launch_arwen.py" %*
) else (
    rem No runtime in this package and no ARWEN_PYTHON: the Python launcher, or a sentence.
    where py >nul 2>nul
    if errorlevel 1 (
        echo Cannot start ArWen: no Python was found. Install Python 3.11 or newer, or set ARWEN_PYTHON to the python of the environment ArWen is installed into.
        pause
        exit /b 2
    )
    py -3 -I -B "%~dp0launch_arwen.py" %*
)
if errorlevel 1 pause
