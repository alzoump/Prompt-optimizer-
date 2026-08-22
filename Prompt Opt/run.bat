@echo off
setlocal
cd /d "%~dp0"

where python >nul 2>&1
if errorlevel 1 (
    echo Python was not found on PATH.
    echo Install it from https://python.org and tick "Add python.exe to PATH".
    pause
    exit /b 1
)

python -c "import tiktoken" >nul 2>&1
if errorlevel 1 (
    echo Installing tiktoken for exact token counts...
    python -m pip install tiktoken --quiet
)

echo Launching Prompt Optimizer...
python "%~dp0prompt_optimizer.py"
if errorlevel 1 pause
