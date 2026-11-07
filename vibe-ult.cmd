@echo off
rem vibe-ult — launch the ULT engine server + xera-rsi harness in one go.
rem Usage:  vibe-ult.cmd [xera-rsi args...]     (from anywhere)
cd /d "%~dp0"

rem engine server in a minimized window (both models, lazy-loaded)
start "" /min python tools\ult-serve\serve.py --model all --port 8321

set XERA_BASE_URL=http://127.0.0.1:8321/v1
set XERA_API_KEY=ult
if "%XERA_MODEL%"=="" set XERA_MODEL=ult-lfm3-1.7b

zig-out\bin\xera-rsi.exe %*
