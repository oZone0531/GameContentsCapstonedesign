@echo off
REM ===================================================================
REM  Bang RL - launch the GUI
REM
REM  Starts the PyQt6 GUI (main menu -> game / training / settings).
REM  Install the dependencies first with install_gui.bat.
REM ===================================================================

setlocal
cd /d "%~dp0"

where py >nul 2>nul
if %errorlevel%==0 (
    set "PY=py -3"
) else (
    set "PY=python"
)

%PY% -m bang_rl.gui
if errorlevel 1 (
    echo.
    echo The GUI exited with an error. If dependencies are missing, run install_gui.bat first.
    echo.
    pause
)

exit /b %errorlevel%
