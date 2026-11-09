@echo off
setlocal
cd /d "%~dp0"
set "PYTHONUNBUFFERED=1"
set "OUT=bench_%RANDOM%_%RANDOM%"
echo CPU Data Collection Benchmark: 1 worker vs 4 workers
python benchmark_data_collection.py --games 100 --workers 4 --out "%OUT%"
if errorlevel 1 (
  echo.
  echo Benchmark failed or game-data equivalence test failed.
  pause
  exit /b 1
)
echo.
echo Result file: %OUT%\benchmark_report.json
echo Speedup ^> 1 means the parallel run was faster.
pause
