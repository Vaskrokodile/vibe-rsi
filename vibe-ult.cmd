@echo off
rem vibe-ult - launch the ULT engine server + xera-rsi harness in one go.
rem Usage:  vibe-ult.cmd [xera-rsi args...]     (from anywhere)
cd /d "%~dp0"

rem reuse a live server if one is already bound — never two on :8321
curl -s -m 2 http://127.0.0.1:8321/health >nul 2>&1
if errorlevel 1 (
  rem engine dispatcher in a minimized window; each model gets its own
  rem worker subprocess so a stalled kernel kills only that worker
  start "" /min python tools\ult-serve\serve.py --model all --port 8321
) else (
  echo ult-serve already running on :8321 - reusing it
)

set XERA_BASE_URL=http://127.0.0.1:8321/v1
set XERA_API_KEY=ult
rem qwen is ready in ~5s; lfm3 compiles for ~3min in the background —
rem switch with /model once it's up
if "%XERA_MODEL%"=="" set XERA_MODEL=ult-qwen-1.5b
rem the rsi self-check is a second LLM call per turn — skip it on the
rem small local models (set XERA_NO_VERIFY=0 to re-enable)
if "%XERA_NO_VERIFY%"=="" set XERA_NO_VERIFY=1

zig-out\bin\xera-rsi.exe %*
