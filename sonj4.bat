@echo off
rem SONJ4 launcher (Windows). Relative to this file so it works on any machine.
setlocal
set "HERE=%~dp0"
if exist "%HERE%.venv\Scripts\python.exe" (
  "%HERE%.venv\Scripts\python.exe" "%HERE%sonj.py" %*
) else (
  python "%HERE%sonj.py" %*
)
endlocal
