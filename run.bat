@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo [FitRoom] First run: installing. This downloads about 9 GB and takes a while...
  powershell -ExecutionPolicy Bypass -File "%~dp0setup.ps1" || goto :fail
)
set HF_HOME=%~dp0models\hf
set HF_HUB_OFFLINE=1
rem Listen on the Wi-Fi network so a phone can open the app (Phone camera button)
set VTON_HOST=0.0.0.0
start "" /b cmd /c "timeout /t 6 >nul & start http://127.0.0.1:7860"
".venv\Scripts\python.exe" -m app.server
goto :eof
:fail
echo Setup failed. See the messages above.
pause
