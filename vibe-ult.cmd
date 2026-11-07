@echo off
rem vibe-ult - launch the ULT engine server + xera-rsi harness in one go.
rem Usage:  vibe-ult.cmd [xera-rsi args...]     (from anywhere)
cd /d "%~dp0"

rem engine dispatcher in a minimized window; each model gets its own
rem worker subprocess so a stalled kernel kills only that worker
start "" /min python tools\ult-serve\serve.py --model all --port 8321

set XERA_BASE_URL=http://127.0.0.1:8321/v1
set XERA_API_KEY=ult
rem qwen is ready in ~5s; lfm3 compiles for ~3min in the background —
rem switch with /model once it's up
if "%XERA_MODEL%"=="" set XERA_MODEL=ult-qwen-1.5b

zig-out\bin\xera-rsi.exe %*
