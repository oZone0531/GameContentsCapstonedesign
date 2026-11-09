@echo off
setlocal
cd /d "%~dp0"
set "PYTHONUNBUFFERED=1"
if not exist "bang_rl\datasets\canonical" mkdir "bang_rl\datasets\canonical"
echo [Phase 1] Dummy-game canonical dataset collection (NOT PPO training)
set "GAMES=1000"
set "WORKERS=4"
set "OUT=bang_rl\datasets\canonical\parallel_%RANDOM%_%RANDOM%"
python collect_data.py --games %GAMES% --workers %WORKERS% --out "%OUT%"
if errorlevel 1 (
  echo.
  echo Collection failed. Check the error above.
  pause
  exit /b 1
)
echo.
echo Saved canonical Parquet files to: %OUT%
echo To test the speedup: python benchmark_data_collection.py --games 100 --workers 4 --out benchmark_collection
pause
