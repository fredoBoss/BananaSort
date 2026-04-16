@echo off
REM Install Python packages listed in requirements.txt

REM Check if requirements.txt exists
IF NOT EXIST requirements.txt (
    echo requirements.txt not found!
    exit /b 1
)

REM Install packages using pip
python -m pip install -r requirements.txt

REM Pause to show result
pause