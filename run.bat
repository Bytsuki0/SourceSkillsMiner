@echo off
rem SourceSkillsMiner launcher — double-click me or run from any terminal.
rem Uses the project virtualenv when present, otherwise whatever python is on PATH.
setlocal
set "HERE=%~dp0"
if exist "%HERE%win_venv\Scripts\python.exe" (
    "%HERE%win_venv\Scripts\python.exe" "%HERE%miner.py" %*
) else (
    python "%HERE%miner.py" %*
)
if errorlevel 1 pause
endlocal
