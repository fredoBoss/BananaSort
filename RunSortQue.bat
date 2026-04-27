@echo off
title Banana Sorting Machine
cd /d "%~dp0"
"C:\Users\alfre\AppData\Local\Programs\Python\Python311\python.exe" old/SortQue.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] SortQue.py exited with an error. Check the output above.
    pause
)
