@echo off
REM ===================================================================
REM  Bang RL - install GUI dependencies
REM
REM  Installs the base learning/runtime packages plus the optional
REM  PyQt6 / pyqtgraph GUI packages into the current Python environment.
REM  Run this once before launching the GUI with run_gui.bat.
REM ===================================================================

setlocal
cd /d "%~dp0"

REM Prefer the "py" launcher on Windows, fall back to "python".
where py >nul 2>nul
if %errorlevel%==0 (
    set "PY=py -3"
) else (
    set "PY=python"
)

echo.
echo === Upgrading pip ===
%PY% -m pip install --upgrade pip
if errorlevel 1 goto :error

echo.
echo === Installing Bang RL with GUI extras ===
%PY% -m pip install -r requirements-gui.lock
if errorlevel 1 goto :error

echo.
echo === Installing the bang_rl package (editable) ===
%PY% -m pip install -e ".[gui]"
if errorlevel 1 goto :error

echo.
echo Done. You can now start the GUI with run_gui.bat
echo.
pause
exit /b 0

:error
echo.
echo Installation failed. Review the messages above.
echo.
pause
exit /b 1
