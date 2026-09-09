@echo off
setlocal
title Body posture measurement
cd /d "%~dp0"

rem Determine the best available Python interpreter.
rem Prefer the project virtualenv if present so all dependencies are guaranteed.
if exist "..\.venv\Scripts\python.exe" (
    set "PYTHON_CMD=..\.venv\Scripts\python.exe"
) else if exist ".venv\Scripts\python.exe" (
    set "PYTHON_CMD=.venv\Scripts\python.exe"
) else (
    py -3 -c "import sys" >nul 2>&1
    if not errorlevel 1 (
        set "PYTHON_CMD=py -3"
    ) else (
        python -c "import sys" >nul 2>&1
        if not errorlevel 1 (
            set "PYTHON_CMD=python"
        ) else (
            echo.
            echo ไม่พบ Python ที่พร้อมใช้งาน
            echo ติดตั้ง Python 3.11 หรือใหม่กว่า แล้วเลือก Add Python to PATH จากนั้นรัน:
            echo     py -3 -m pip install -r requirements.txt
            echo.
            pause
            exit /b 1
        )
    )
)

rem Start the integrated camera application.
%PYTHON_CMD% -m body_measure %*
set "APP_EXIT=%ERRORLEVEL%"

if not "%APP_EXIT%"=="0" (
    echo.
    echo โปรแกรมเริ่มไม่ได้ ตรวจสอบการติดตั้งไลบรารีด้วยคำสั่ง:
    echo     %PYTHON_CMD% -m pip install -r requirements.txt
    echo.
    echo หากติดตั้งแล้ว ให้ตรวจสอบว่ากล้องไม่ได้ถูกใช้งานโดยโปรแกรมอื่น
    pause
)

exit /b %APP_EXIT%
