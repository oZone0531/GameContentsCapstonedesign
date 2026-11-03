@echo off
REM ===================================================================
REM  Bang RL - launch the GUI
REM
REM  Starts the PyQt6 GUI (main menu -> game / training / settings).
REM  The repository virtual environment is required so the GUI loads the
REM  source files from this checkout instead of an older global install.
REM ===================================================================

setlocal
set "ROOT=%~dp0"
set "PY=%ROOT%.venv\Scripts\python.exe"

if not exist "%PY%" (
    echo.
    echo The repository virtual environment was not found:
    echo   %PY%
    echo Create it with: py -3 -m venv "%ROOT%.venv"
    echo Then run install_gui.bat.
    echo.
    pause
    endlocal
    exit /b 1
)

cd /d "%ROOT%"
"%PY%" -m bang_rl.gui
set "EXIT_CODE=%errorlevel%"
if not "%EXIT_CODE%"=="0" (
    echo.
    echo The GUI exited with an error. If dependencies are missing, run install_gui.bat first.
    echo.
    pause
)

endlocal & exit /b %EXIT_CODE%
