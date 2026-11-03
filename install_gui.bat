@echo off
REM ===================================================================
REM  Bang RL - install GUI dependencies
REM
REM  Installs the base learning/runtime packages plus the optional
REM  PyQt6 / pyqtgraph GUI packages into the repository virtual environment.
REM  Run this once before launching the GUI with run_gui.bat.
REM ===================================================================

setlocal
set "ROOT=%~dp0"
set "PY=%ROOT%.venv\Scripts\python.exe"

if not exist "%PY%" (
    echo.
    echo The repository virtual environment was not found:
    echo   %PY%
    echo Create it with: py -3 -m venv "%ROOT%.venv"
    echo.
    pause
    endlocal
    exit /b 1
)

cd /d "%ROOT%"

echo Using repository interpreter:
echo   %PY%

echo.
echo === Upgrading pip ===
"%PY%" -m pip install --upgrade pip
if errorlevel 1 goto :error

echo.
echo === Installing Bang RL with GUI extras ===
"%PY%" -m pip install -r requirements-gui.lock
if errorlevel 1 goto :error

echo.
echo === Installing the bang_rl package (editable) ===
"%PY%" -m pip install -e ".[gui]"
if errorlevel 1 goto :error

echo.
echo Done. You can now start the GUI with run_gui.bat
endlocal
exit /b 0

:error
echo.
echo Installation failed. Review the messages above.
echo.
pause
endlocal
exit /b 1
